from __future__ import annotations

import os
import shutil
import zipfile
import tempfile
import traceback
import time
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import asyncio

from .utils.api_util import (
    SingleUserSequentialScheduler,
    create_pipeline,
    device_families_payload,
    new_task_id,
)

# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="Automata-AI Edge API", version="3.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:53352",  
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Scheduler 
# -----------------------------

scheduler = SingleUserSequentialScheduler()


TaskType = Literal["image", "audio", "sensor"]


class SubmitTaskRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    task_type: TaskType
    dataset_path: str = Field(..., min_length=1)

    device_family_id: str = Field(..., min_length=1)
    device_specs: Dict[str, Any] = Field(default_factory=dict)

    export_ext: Optional[str] = None

    target_num_classes: Optional[int] = None
    target_name: Optional[str] = None  
    output_root: Optional[str] = None
    min_accuracy: Optional[float] = None
    seed: Optional[int] = None

    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    val_split: Optional[float] = None
    pin_memory: Optional[bool] = None
    sweep: Optional[list] = None
    epochs_per_trial: Optional[int] = None
    final_epochs: Optional[int] = None
    torch_device: Optional[str] = None

    task_id: Optional[str] = None

    run_if_idle: bool = True


class SubmitTaskResponse(BaseModel):
    task_id: str
    user_id: str
    task_type: TaskType
    sched_state: str
    status: Dict[str, Any]
    global_queue: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise HTTPException(status_code=400, detail=msg)


def _build_pipeline_kwargs(req: SubmitTaskRequest, task_id: str) -> Dict[str, Any]:
    _require(req.user_id.strip() != "", "user_id is required")
    _require(req.dataset_path.strip() != "", "dataset_path is required")
    _require(req.device_family_id.strip() != "", "device_family_id is required")

    _require(req.target_num_classes is not None, "target_num_classes is required for image/audio/sensor")

    if req.task_type == "sensor":
        _require(req.target_name is not None and str(req.target_name).strip() != "", "target_name is required for sensor tasks")

    kwargs: Dict[str, Any] = {
        "task_id": task_id,
        "user_id": req.user_id,
        "task_type": req.task_type,
        "dataset_path": req.dataset_path,
        "device_family_id": req.device_family_id,
        "device_specs": req.device_specs,
        "target_num_classes": int(req.target_num_classes),
    }

    if req.export_ext is not None:
        kwargs["export_ext"] = req.export_ext

    safe_user = "".join(ch for ch in req.user_id if ch.isalnum() or ch in ("-", "_")).strip() or "user"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    
    if req.output_root is not None:
        kwargs["output_root"] = os.path.join("runs", safe_user)
    else:
        kwargs["output_root"] = os.path.join(ASSETS_DIR, "runs", safe_user)

    if req.min_accuracy is not None:
        kwargs["min_accuracy"] = float(req.min_accuracy)
    if req.seed is not None:
        kwargs["seed"] = int(req.seed)

    if req.batch_size is not None:
        kwargs["batch_size"] = int(req.batch_size)
    if req.num_workers is not None:
        kwargs["num_workers"] = int(req.num_workers)
    if req.val_split is not None:
        kwargs["val_split"] = float(req.val_split)
    if req.pin_memory is not None:
        kwargs["pin_memory"] = bool(req.pin_memory)
    if req.sweep is not None:
        kwargs["sweep"] = req.sweep
    if req.epochs_per_trial is not None:
        kwargs["epochs_per_trial"] = int(req.epochs_per_trial)
    if req.final_epochs is not None:
        kwargs["final_epochs"] = int(req.final_epochs)
    if req.torch_device is not None:
        kwargs["torch_device"] = req.torch_device

    if req.task_type == "sensor":
        kwargs["target_name"] = str(req.target_name)

    return kwargs


def _safe_file_response(path: str, filename_fallback: str) -> FileResponse:
    if not path:
        raise HTTPException(status_code=404, detail="Artifact path not available yet.")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Artifact file not found on server.")
    return FileResponse(path, filename=os.path.basename(path) or filename_fallback)


def _find_report_by_task_id(task_id: str) -> Optional[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets", "runs")
    if not os.path.isdir(assets_dir):
        return None
    target = os.path.join(task_id, "report.pdf")
    for root, _, files in os.walk(assets_dir):
        if "report.pdf" not in files:
            continue
        if root.endswith(os.path.join(task_id)):
            return os.path.join(root, "report.pdf")

        if root.replace("/", "\\").endswith(target):
            return os.path.join(root, "report.pdf")
    return None


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/device-families")
def device_families():
    return device_families_payload()


@app.get("/queue")
def global_queue():
    return scheduler.queue_info()


class PlaceholderPipeline:
    def __init__(self, task_id, user_id, task_type):
        self.task_id = task_id
        self.user_id = user_id
        self.task_type = task_type
        self.status = "initializing"
    
    def get_status(self):
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "status": self.status,
            "message": "Initializing AI pipeline components...",
            "metrics": {},
            "errors": {}
        }
    
    def run(self):
        raise RuntimeError("Pipeline is still initializing.")

async def finalize_task_creation(
    task_id: str, 
    task_type: str, 
    kwargs: Dict[str, Any], 
    run_if_idle: bool
):
    print(f"DEBUG: Finalizing task {task_id} status...")
    
    def _create():
        print(f"DEBUG: Creating real pipeline for {task_id}...")
        return create_pipeline(task_type, **kwargs)

    try:
        loop = asyncio.get_running_loop()
        pipeline = await loop.run_in_executor(None, _create)
        print(f"DEBUG: Pipeline created for {task_id}. Updating scheduler.")
        
        scheduler.update_pipeline(task_id, pipeline)
        
        if run_if_idle:
            print(f"DEBUG: Triggering process_next for {task_id}")

            await loop.run_in_executor(None, scheduler.process_next)
            
    except Exception as e:
        print(f"ERROR: Failed to finalize task {task_id}: {e}")
        traceback.print_exc()
        scheduler.fail_task(task_id, "Pipeline initialization failed", e)


@app.post("/tasks", response_model=SubmitTaskResponse)
async def create_task(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    task_type: TaskType = Form(...),
    file: UploadFile = File(...),
    device_family: str = Form(...),
    
    title: str = Form(""),
    description: str = Form(""),
    visibility: str = Form("private"),
    task_id: Optional[str] = Form(None),
    
    ram_kb: Optional[float] = Form(None),
    flash_mb: Optional[float] = Form(None),
    cpu_mhz: Optional[float] = Form(None),
    model_extension: Optional[str] = Form(None),

    num_classes: Optional[int] = Form(None),
    target_column: Optional[str] = Form(None),

    # Power User General
    quantization: str = Form("auto"),          
    optimization_strategy: str = Form("auto"), 
    epochs: int = Form(20),
    training_speed: str = Form("auto"),        
    accuracy_tolerance: str = Form("auto"),    

    # Power User Image
    augmentation: str = Form("auto"),
    feature_handling: str = Form("auto"),
    cleaning: str = Form("auto"),
    noise_handling: str = Form("auto"),

    # Power User Sensor
    robustness: str = Form("auto"),
    outlier_removal: str = Form("auto"),
    
    # Execution
    run_if_idle: bool = Form(True),
):
    print(f"DEBUG: create_task started. User: {user_id}, Type: {task_type}")
    real_task_id = (task_id or "").strip() or new_task_id()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    
    print("DEBUG: Saving file...")
    upload_dir = os.path.join(ASSETS_DIR, "uploads", user_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    safe_filename = os.path.basename(file.filename or "dataset")
    file_path = os.path.join(upload_dir, f"{real_task_id}_{safe_filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"DEBUG: File saved to {file_path}")

    device_specs = {}
    if ram_kb is not None: device_specs["ram_kb"] = ram_kb
    if flash_mb is not None: device_specs["flash_mb"] = flash_mb
    if cpu_mhz is not None: device_specs["cpu_mhz"] = cpu_mhz

    common_args = {
        "task_id": real_task_id,
        "user_id": user_id,
        "dataset_path": file_path,
        "device_family_id": device_family,
        "device_specs": device_specs,
        "output_root": os.path.join(ASSETS_DIR, "runs", user_id),
        "export_ext": model_extension or ".h",
        "batch_size": 128,
        "final_epochs": epochs,
        "num_workers": 0,
        "title": title,
        "description": description,
        "visibility": visibility,
    }

    if task_type == "image":
        if num_classes is None:
            raise HTTPException(status_code=400, detail="num_classes is required for image tasks")
        kwargs = {
            **common_args,
            "target_num_classes": num_classes,
            "quantization": quantization,
            "optimization_strategy": optimization_strategy,
            "training_speed": training_speed,
            "accuracy_tolerance": accuracy_tolerance,
            "augmentation": augmentation,
            "feature_handling": feature_handling,
            "cleaning": cleaning,
            "noise_handling": noise_handling,
        }
    elif task_type == "audio":
        if num_classes is None:
            raise HTTPException(status_code=400, detail="num_classes is required for audio tasks")
        kwargs = {
            **common_args,
            "target_num_classes": num_classes,
            "quantization": quantization,
            "optimization_strategy": optimization_strategy,
            "training_speed": training_speed,
            "accuracy_tolerance": accuracy_tolerance,
            "noise_handling": noise_handling,
            "cleaning": cleaning,
        }
    elif task_type == "sensor":
        kwargs = {
            **common_args,
            "target_name": target_column,
            "quantization": quantization,
            "optimization_strategy": optimization_strategy,
            "training_speed": training_speed,
            "accuracy_tolerance": accuracy_tolerance,
            "robustness": robustness,
            "outlier_removal": outlier_removal,
        }
        if num_classes is not None:
            kwargs["target_num_classes"] = num_classes
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task type: {task_type}")

    print("DEBUG: Registering placeholder pipeline...")
    placeholder = PlaceholderPipeline(real_task_id, user_id, task_type)
    
    scheduler.submit(
        task_id=real_task_id,
        user_id=user_id,
        task_type=task_type,
        pipeline=placeholder
    )

    background_tasks.add_task(
        finalize_task_creation, 
        real_task_id, 
        task_type, 
        kwargs, 
        run_if_idle
    )

    print("DEBUG: Request complete. Returning response.")
    st = scheduler.get_status(real_task_id)
    return SubmitTaskResponse(
        task_id=real_task_id,
        user_id=user_id,
        task_type=task_type,
        sched_state=st["sched_state"],
        status=st["status"],
        global_queue=st["global"],
    )


@app.post("/process-next")
def process_next():

    rec = scheduler.process_next()
    if rec is None:
        return {"ran": False, "queue": scheduler.queue_info()}
    return {"ran": True, "task_id": rec.task_id, "sched_state": rec.sched_state, "queue": scheduler.queue_info()}


@app.get("/tasks/{task_id}")
def task_info(task_id: str):
    try:
        return scheduler.get_status(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task_id not found")


@app.get("/tasks/{task_id}/status")
def task_status(task_id: str):
    try:
        return scheduler.get_status(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task_id not found")


@app.get("/tasks/{task_id}/result")
def task_result(task_id: str):
    try:
        st = scheduler.get_status(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task_id not found")

    res = scheduler.get_result(task_id)
    if res is None:
        raise HTTPException(status_code=409, detail={"message": "result not ready", "status": st})

    return res


@app.get("/tasks/{task_id}/download")
def download_artifact(task_id: str, kind: Literal["model", "report", "all_models"]):

    try:
        st = scheduler.get_status(task_id)
    except KeyError:
        if kind == "report":
            fallback = _find_report_by_task_id(task_id)
            if fallback:
                return _safe_file_response(fallback, filename_fallback=f"{task_id}_report.pdf")
        raise HTTPException(status_code=404, detail="task_id not found")

    res = scheduler.get_result(task_id)
    if res is None:
        raise HTTPException(status_code=409, detail={"message": "result not ready", "status": st})

    if kind == "model":
        path = res.get("final_model_path") or ""
        return _safe_file_response(path, filename_fallback=f"{task_id}_model")

    if kind == "report":
        path = res.get("report_path") or ""
        return _safe_file_response(path, filename_fallback=f"{task_id}_report.pdf")

    exported = res.get("exported_model_paths") or res.get("metrics", {}).get("exported_model_paths") or []
    if not isinstance(exported, list) or len(exported) == 0:
        raise HTTPException(status_code=404, detail="No exported_model_paths available for this task.")

    exported = [p for p in exported if isinstance(p, str) and p and os.path.exists(p)]
    if len(exported) == 0:
        raise HTTPException(status_code=404, detail="exported_model_paths exist, but no files were found on disk.")

    task_dir = res.get("output_dir") if isinstance(res.get("output_dir"), str) else None
    base_dir = task_dir if task_dir and os.path.isdir(task_dir) else None

    if base_dir is None:
        base_dir = tempfile.gettempdir()

    zip_path = os.path.join(base_dir, f"{task_id}_all_models.zip")

    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception:
        pass

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in exported:
            zf.write(p, arcname=os.path.basename(p))

    return FileResponse(zip_path, filename=os.path.basename(zip_path))
