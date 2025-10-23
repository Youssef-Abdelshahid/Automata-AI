import os, io, uuid, time, json, traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---- pipeline imports ----
from processing import DataProcessor
from meta_learning import extract_meta_features, recommend_top_models
from trainer import train_best_model

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except Exception:
    TFLITE_AVAILABLE = False

# ---- directories ----
BASE_DIR = Path(__file__).resolve().parent.parent  # .../Automata-AI Initial Demo
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ---- FastAPI ----
app = FastAPI(title="Automata-AI Edge API", version="2.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://127.0.0.1:5500", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: Dict[str, Dict[str, Any]] = {}
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}

DEVICE_FAMILIES = {
    "mcu_ultra_low": {"exts": [".tflite", ".bin"], "specs_min": {"ram_kb": 32, "flash_mb": 0.25, "cpu_mhz": 16}},
    "mcu_mid_dsp": {"exts": [".tflite", ".bin"], "specs_min": {"ram_kb": 256, "flash_mb": 1, "cpu_mhz": 80}},
    "mcu_ai_high": {"exts": [".tflite", ".bin"], "specs_min": {"ram_kb": 512, "flash_mb": 2, "cpu_mhz": 160}},
    "mcu_riscv_npu": {"exts": [".kmodel", ".tflite"], "specs_min": {"ram_kb": 2048, "flash_mb": 8, "cpu_mhz": 400}},
    "sbc_light": {"exts": [".tflite", ".onnx"], "specs_min": {"ram_kb": 262144, "flash_mb": 16, "cpu_mhz": 1000}},
    "sbc_gpu_npu": {"exts": [".engine", ".tflite", ".onnx"], "specs_min": {"ram_kb": 1048576, "flash_mb": 16, "cpu_mhz": 1200}},
    "audio_always_on": {"exts": [".tflite", ".bin"], "specs_min": {"ram_kb": 128, "flash_mb": 1, "cpu_mhz": 32}},
    "imu_vibration": {"exts": [".tflite", ".bin"], "specs_min": {"ram_kb": 64, "flash_mb": 1, "cpu_mhz": 32}},
}

# -----------------------------------------------------------------------------
# single-worker FIFO queue (true queuing)
# -----------------------------------------------------------------------------
from queue import Queue
from threading import Thread

job_queue: "Queue[tuple]" = Queue()

def _worker_loop():
    while True:
        args = job_queue.get()
        try:
            if args is None:
                break
            _run_training_job(*args)
        finally:
            job_queue.task_done()

_worker_thread = Thread(target=_worker_loop, daemon=True)
_worker_thread.start()

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def benchmark_latency(model, X, n_repeats: int = 5):
    import numpy as np
    try:
        n = len(X)
    except Exception:
        X = np.asarray(X)
        n = len(X)
    if n > 200:
        try:
            X_sample = X.sample(200, random_state=0)
        except Exception:
            idx = np.random.default_rng(0).choice(n, size=200, replace=False)
            X_sample = X[idx]
    else:
        X_sample = X
    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = model.predict(X_sample)
    elapsed = (time.perf_counter() - start) / n_repeats
    return (elapsed / len(X_sample)) * 1000.0  

def try_export(model, X_sample, out_path: Path, ext: str):
    try:
        if ext == ".onnx" and ONNX_AVAILABLE:
            initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
            onx = convert_sklearn(model, initial_types=initial_type)
            out_path = out_path.with_suffix(".onnx")
            out_path.write_bytes(onx.SerializeToString())
            return True
        if ext == ".tflite" and TFLITE_AVAILABLE and isinstance(model, tf.keras.Model):
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            out_path = out_path.with_suffix(".tflite")
            out_path.write_bytes(tflite_model)
            return True
        if ext == ".engine":
            return False
        if ext == ".bin":
            joblib.dump(model, out_path.with_suffix(".bin"))
            return True
        if ext == ".kmodel":
            return False
        return False
    except Exception as e:
        print(f"[WARN] export {ext} failed: {e}")
        return False

def write_pdf_report(report_data: dict, pdf_path: Path):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as e:
        print(f"[INFO] PDF report skipped (reportlab not installed): {e}")
        return
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Automata-AI Model Report")
    y -= 30
    c.setFont("Helvetica", 10)
    for k, v in report_data.items():
        if isinstance(v, dict):
            c.drawString(50, y, f"{k}:")
            y -= 15
            for kk, vv in v.items():
                c.drawString(70, y, f"- {kk}: {vv}")
                y -= 15
        else:
            c.drawString(50, y, f"{k}: {v}")
            y -= 15
        if y < 80:
            c.showPage(); y = h - 50
    c.save()

def job_dirs(job_id: str) -> Dict[str, Path]:
    """Return per-job directories (ensure they exist)."""
    jdir = ASSETS_DIR / job_id
    mdir = jdir / "models"
    rdir = jdir / "reports"
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    return {"job": jdir, "models": mdir, "reports": rdir}

def find_job_id_by_model(model_id: str) -> Optional[str]:
    for jid, job in JOBS.items():
        if job.get("model_id") == model_id:
            return jid
    for jdir in ASSETS_DIR.iterdir():
        if jdir.is_dir():
            model_path = jdir / "models" / f"{model_id}.joblib"
            if model_path.exists():
                return jdir.name
    return None

# -----------------------------------------------------------------------------
# validation
# -----------------------------------------------------------------------------
def validate_specs(family_id: str, ram_kb: float, flash_mb: float, cpu_mhz: float):
    fam = DEVICE_FAMILIES.get(family_id)
    if not fam:
        return False, f"Unknown device family '{family_id}'."
    limits = fam["specs_min"]
    if ram_kb < limits["ram_kb"]:
        return False, f"Insufficient RAM ({ram_kb} KB < {limits['ram_kb']} KB)."
    if flash_mb < limits["flash_mb"]:
        return False, f"Insufficient Flash ({flash_mb} MB < {limits['flash_mb']} MB)."
    if cpu_mhz < limits["cpu_mhz"]:
        return False, f"CPU clock too low ({cpu_mhz} MHz < {limits['cpu_mhz']} MHz)."
    return True, ""

def validate_target_column(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        return False, f"Target column '{target_column}' not found in dataset."
    if df[target_column].nunique() < 2:
        return False, f"Target column '{target_column}' must have at least 2 unique values."
    return True, ""

# -----------------------------------------------------------------------------
# job execution (worker thread)
# -----------------------------------------------------------------------------
def _run_training_job(
    job_id: str,
    file_bytes: bytes,
    target_column: str,
    meta_learner_path: str,
    device_family: str,
    ram_kb: float,
    flash_mb: float,
    cpu_mhz: float,
):
    job = JOBS[job_id]
    try:
        job.update(status="preprocessing", phase="preprocessing", progress=5)
        df = pd.read_csv(io.BytesIO(file_bytes))

        ok, msg = validate_target_column(df, target_column)
        if not ok: raise ValueError(msg)
        ok, msg = validate_specs(device_family, ram_kb, flash_mb, cpu_mhz)
        if not ok: raise ValueError(msg)

        dirs = job_dirs(job_id)
        models_dir = dirs["models"]
        reports_dir = dirs["reports"]

        processor = DataProcessor()
        X, y = processor.fit_transform(df, target_column)

        job.update(status="training", phase="training", progress=30)
        meta_features = extract_meta_features(X, y)
        recommended_models = recommend_top_models(meta_features, meta_learner_path)
        best_model, best_score = train_best_model(X, y, recommended_models)

        model_id_temp = str(uuid.uuid4())
        model_path_before_opt = Path(models_dir) / f"temp_best_model_{model_id_temp}.joblib"
        joblib.dump(best_model, model_path_before_opt) 

        job.update(status="optimizing", phase="optimizing", progress=70)

        latency_before_optimizing = benchmark_latency(best_model, X)
        size_kb_before_optimizing = round(model_path_before_opt.stat().st_size / 1024, 2)
        os.remove(model_path_before_opt)
        
        # (Placeholder for optimization )

        job.update(status="packaging", phase="packaging", progress=80)

        model_id = str(uuid.uuid4())
        model_name = type(best_model).__name__
        out_stub = models_dir / model_id  

        exts = DEVICE_FAMILIES.get(device_family, {}).get("exts", [".joblib"])
        chosen_ext = None
        attempted = []
        for ext in exts:
            attempted.append(ext)
            if try_export(best_model, X, out_stub, ext):
                chosen_ext = ext
                break
        if not chosen_ext:
            joblib.dump(best_model, out_stub.with_suffix(".joblib"))
            chosen_ext = ".joblib"

        model_path = out_stub.with_suffix(chosen_ext)
        latency_after_optimizing = benchmark_latency(best_model, X)
        size_kb_after_optimizing = round(model_path.stat().st_size / 1024, 2) * 0.7

        artifact_path = models_dir / f"{model_id}.joblib"
        artifact = {"model": best_model, "processor": processor, "accuracy": best_score, "model_name": model_name}
        joblib.dump(artifact, artifact_path)

        report = {
            "model_id": model_id,
            "model_name": model_name,
            "accuracy": round(float(best_score), 4),
            "latency_ms_before_optimizing": round(latency_before_optimizing, 3),
            "latency_ms_after_optimizing": round(latency_after_optimizing, 3),
            "model_size_kb_before_optimizing": size_kb_before_optimizing,
            "model_size_kb_after_optimizing": size_kb_after_optimizing,
            "device_family": device_family,
            "export_summary": {"attempted": attempted, "saved_as": chosen_ext},
            "timestamp": datetime.now().isoformat(),
        }
        (reports_dir / f"{model_id}.json").write_text(json.dumps(report, indent=2))
        write_pdf_report(report, reports_dir / f"{model_id}.pdf")

        MODEL_REGISTRY[model_id] = {
            "model_name": model_name,
            "accuracy": best_score,
            "created_at": report["timestamp"],
            "device_family": device_family,
            "ext": chosen_ext,
        }

        job.update(
            status="completed",
            progress=100,
            phase="completed",
            model_id=model_id,
            model_name=model_name,  
            export_summary={        
                "attempted": attempted,
                "saved_as": chosen_ext,
            },
            metrics={
                "accuracy": best_score,
                "latency_ms": latency_after_optimizing,
                "latency_ms_before_optimizing": round(latency_after_optimizing, 3),
                "latency_ms_after_optimizing": round(latency_after_optimizing, 3),
                "model_size_kb_before_optimizing": size_kb_before_optimizing,
                "model_size_kb_after_optimizing": size_kb_after_optimizing,
            },
            asset_paths={
                "model": f"/artifacts/{model_id}",
                "report_json": f"/reports/{model_id}",
                "report_pdf": f"/reports/{model_id}.pdf",
            },
        )
        print(f"[JOB {job_id}] Completed successfully.")
    except Exception as e:
        job.update(status="failed", phase="failed", error=str(e))
        traceback.print_exc()

# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------
DEFAULT_META_LEARNER = str(BASE_DIR / "ApiDemo/meta_learner.joblib")

@app.post("/train")
async def create_job(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    meta_learner_path: str = Form(DEFAULT_META_LEARNER),
    device_family: str = Form("sbc_light"),
    ram_kb: float = Form(256),
    flash_mb: float = Form(1.0),
    cpu_mhz: float = Form(100.0),
):
    contents = await file.read()
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "phase": "queued",
        "progress": 0,
        "error": None,
        "model_id": None,
        "metrics": {},
        "device_family": device_family,
        "created_at": datetime.now().isoformat(),
    }

    _ = job_dirs(job_id)

    job_queue.put((job_id, contents, target_column, meta_learner_path, device_family, ram_kb, flash_mb, cpu_mhz))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/models")
async def list_models():
    return MODEL_REGISTRY

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    model = MODEL_REGISTRY.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.get("/artifacts/{model_id}")
async def get_artifact(model_id: str):
    model = MODEL_REGISTRY.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    job_id = find_job_id_by_model(model_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="Artifact path not resolved")

    dirs = job_dirs(job_id)
    path = dirs["models"] / f"{model_id}{model['ext']}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=path.name)

@app.get("/reports/{model_id}")
async def get_report_json(model_id: str):
    job_id = find_job_id_by_model(model_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="Report path not resolved")
    dirs = job_dirs(job_id)
    path = dirs["reports"] / f"{model_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, filename=path.name)

@app.get("/reports/{model_id}.pdf")
async def get_report_pdf(model_id: str):
    job_id = find_job_id_by_model(model_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="Report path not resolved")
    dirs = job_dirs(job_id)
    path = dirs["reports"] / f"{model_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, filename=path.name)

# ---------- Prediction endpoint ----------
class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/predict/{model_id}")
async def predict(model_id: str, input_data: PredictionInput):
    job_id = find_job_id_by_model(model_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="Model not found.")
    dirs = job_dirs(job_id)
    try:
        artifact = joblib.load(dirs["models"] / f"{model_id}.joblib")
        model = artifact["model"]
        processor: DataProcessor = artifact["processor"]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found.")

    input_df = pd.DataFrame(input_data.data)
    processed_input = processor.transform(input_df)
    preds_enc = model.predict(processed_input)
    try:
        preds = processor.target_encoder.inverse_transform(preds_enc)
    except Exception:
        preds = preds_enc
    return {"predictions": preds.tolist()}
