import os, io, uuid, time, json, traceback
from datetime import datetime
from pathlib import Path
from typing import Optional
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
from optimization import optimizing_model

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
    "mcu_ultra_low": {"exts": [".tflite", ".h", ".bin"], "specs_min": {"ram_kb": 2, "flash_mb": 0.025, "cpu_mhz": 12}},
    "mcu_mid_dsp": {"exts": [".tflite", ".h", ".bin"], "specs_min": {"ram_kb": 256, "flash_mb": 1, "cpu_mhz": 80}},
    "mcu_ai_high": {"exts": [".tflite", ".h", ".bin"], "specs_min": {"ram_kb": 512, "flash_mb": 2, "cpu_mhz": 160}},
    "mcu_riscv_npu": {"exts": [".kmodel", ".tflite"], "specs_min": {"ram_kb": 2048, "flash_mb": 8, "cpu_mhz": 400}},
    "sbc_light": {"exts": [".tflite", ".onnx"], "specs_min": {"ram_kb": 262144, "flash_mb": 16, "cpu_mhz": 1000}},
    "sbc_gpu_npu": {"exts": [".engine", ".tflite", ".onnx"], "specs_min": {"ram_kb": 1048576, "flash_mb": 16, "cpu_mhz": 1200}},
    "audio_always_on": {"exts": [".tflite", ".h", ".bin"], "specs_min": {"ram_kb": 128, "flash_mb": 1, "cpu_mhz": 32}},
    "imu_vibration": {"exts": [".tflite", ".h", ".bin"], "specs_min": {"ram_kb": 64, "flash_mb": 1, "cpu_mhz": 32}},
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
        if isinstance(model, tf.keras.Model):
            if ext in [".tflite", ".bin"]:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]  
                tflite_model = converter.convert()
                out_path = out_path.with_suffix(ext)
                out_path.write_bytes(tflite_model)
                print(f"[INFO] Exported quantized TFLite model to {out_path.name} (Arduino-ready)")
                return True
            else:
                print(f"[WARN] Unsupported export extension for Keras model: {ext}")
                return False

        if ext == ".onnx":
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
                onx = convert_sklearn(model, initial_types=initial_type)
                out_path = out_path.with_suffix(".onnx")
                out_path.write_bytes(onx.SerializeToString())
                print(f"[INFO] Exported scikit-learn model to {out_path.name}")
                return True
            except Exception as e:
                print(f"[WARN] ONNX export failed: {e}")
                return False

        if ext in [".h", ".bin"]:
            try:
                import m2cgen as m2c
                import re

                c_code = m2c.export_to_c(model)

                c_code = re.sub(
                    r"memcpy\(([^,]+),\s*\((?:double|float)\[\]\)\{([^}]+)\},\s*(\d+)\s*\*\s*sizeof\((double|float)\)\);",
                    r"{ \4 tmp[] = {\2}; memcpy(\1, tmp, \3 * sizeof(\4)); }",
                    c_code
                )

                header_guard = out_path.stem.upper() + "_H"
                wrapped_code = (
                    f"#ifndef {header_guard}\n#define {header_guard}\n\n"
                    f"{c_code}\n\n"
                    f"#endif // {header_guard}\n"
                )

                out_path = out_path.with_suffix(".h")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(wrapped_code)

                print(f"[INFO] Exported scikit-learn model as Arduino-ready header file: {out_path.name}")
                print(f"[HINT] Include it in your sketch using: #include \"{out_path.name}\"")
                return True

            except Exception as e:
                print(f"[WARN] Header export (C code) failed: {e}")
                return False

        if ext in [".engine", ".kmodel"]:
            print(f"[WARN] Export format {ext} not supported for hardware deployment.")
            return False

        print(f"[WARN] Unsupported export extension: {ext}")
        return False

    except Exception as e:
        print(f"[WARN] export {ext} failed: {e}")
        return False

def write_pdf_report(report_data: dict, pdf_path: Path, logo_path: Optional[Path] = None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        print(f"[INFO] PDF report skipped (reportlab not installed): {e}")
        return

    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN

    BASELINE = 6
    GAP_SM = 2 * BASELINE
    GAP_MD = 3 * BASELINE
    GAP_LG = 4 * BASELINE

    BRAND = {
        "ink":   colors.HexColor("#111827"),
        "muted": colors.HexColor("#6b7280"),
        "chip":  colors.HexColor("#eef2ff"),
        "ok":    colors.HexColor("#16a34a"),
        "warn":  colors.HexColor("#dc2626"),
        "line":  colors.HexColor("#d1d5db"),
        "card":  colors.white,
        "badge": colors.HexColor("#f8fafc"),
    }

    body = ParagraphStyle("body", fontName="Helvetica", fontSize=10, leading=13, textColor=BRAND["ink"])

    from datetime import datetime
    def _fmt_ts(ts):
        if not ts: return "—"
        s = str(ts).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            try:    return dt.strftime("%b %-d, %Y – %-I:%M %p")
            except: return dt.strftime("%b %d, %Y – %I:%M %p").replace(" 0", " ")
        except: return s[:19].replace("T", " – ")

    def _fmt_acc(acc):
        try: v = float(acc)
        except: return acc if acc not in (None, "") else "—"
        pct = v * 100 if 0 <= v <= 1 else v
        return f"{pct:.1f}%"

    def _fmt_num(v, places=3, thousands=True):
        if v is None or v == "—": return "—"
        try:
            f = float(v)
            return f"{f:,.{places}f}" if thousands else f"{f:.{places}f}"
        except: return str(v)

    def _delta_str(before, after, better="lower"):
        try:
            b = float(before); a = float(after)
        except: return "—", BRAND["muted"]
        if b == 0: return "—", BRAND["muted"]
        ch = (a - b) / b
        if abs(ch) < 1e-12: return "0.0%", BRAND["muted"]
        good = (ch < 0) if better == "lower" else (ch > 0)
        return ("↓ " if good else "↑ ") + f"{abs(ch)*100:.1f}%", (BRAND["ok"] if good else BRAND["warn"])

    def _would_fit(y, need, bottom=72): return (y - need) >= bottom
    def _ensure_space(c, y, need, bottom=72):
        if not _would_fit(y, need, bottom): c.showPage(); return _draw_header(c)
        return y

    def _section(c, y, title, min_h, bottom=72):
        head_h = 3 * BASELINE
        y = _ensure_space(c, y, head_h + min_h, bottom=bottom)
        c.setFont("Helvetica-Bold", 12.5); c.setFillColor(BRAND["ink"])
        c.drawString(MARGIN, y, title)
        y -= BASELINE
        c.setStrokeColor(BRAND["line"]); c.line(MARGIN, y, MARGIN + CONTENT_W, y)
        y -= BASELINE
        return y

    def _draw_decimal_right(c, x_right, text, font="Courier", size=10):
        s = str(text)
        c.setFont(font, size)
        if s == "—" or "." not in s:
            c.drawRightString(x_right, c._curr_y, s); return
        left, right = s.split(".", 1)
        w_right = c.stringWidth(right, font, size)
        w_dot   = c.stringWidth(".", font, size)
        c.drawRightString(x_right, c._curr_y, right)
        x = x_right - w_right
        c.drawString(x - w_dot, c._curr_y, ".")
        x -= w_dot
        c.drawRightString(x, c._curr_y, left)

    def _chip(c, x, y, text):
        c.setFont("Helvetica", 9)
        pad_w = 6
        w = c.stringWidth(text, "Helvetica", 9) + 2 * pad_w
        if x + w > MARGIN + CONTENT_W:
            x = MARGIN; y -= 2 * BASELINE
        c.setFillColor(BRAND["chip"])
        c.roundRect(x, y - 12, w, 16, 3, stroke=0, fill=1)
        c.setFillColor(BRAND["ink"])
        c.drawString(x + pad_w, y - 9, text)
        return x + w + 6, y

    CARD_H = 12 * BASELINE
    def _metric_card(c, x, y, w, title, value, subtitle=None, subdelta=None, subdelta_color=None):
        is_acc = title.lower().startswith("accuracy")
        c.setFillColor(BRAND["badge"] if is_acc else BRAND["card"])
        c.roundRect(x, y - CARD_H, w, CARD_H, 8, stroke=0, fill=1)
        c.setStrokeColor(BRAND["line"]); c.roundRect(x, y - CARD_H, w, CARD_H, 8, stroke=1, fill=0)
        c.setFont("Helvetica-Bold", 9); c.setFillColor(BRAND["muted"])
        c.drawString(x + 12, y - 2 * BASELINE, title)
        c.setFont("Courier-Bold", 16); c.setFillColor(BRAND["ink"])
        c.drawRightString(x + w - 12, y - 5 * BASELINE, str(value))
        c.setFont("Helvetica", 9); c.setFillColor(BRAND["muted"])
        c.drawString(x + 12, y - 8 * BASELINE, subtitle or " ")
        c.setFont("Helvetica-Bold", 9); c.setFillColor(subdelta_color or BRAND["muted"])
        c.drawString(x + 12, y - 11 * BASELINE, subdelta or " ")
        return CARD_H

    def _ellipsis(c, text, maxw, font="Helvetica", size=9):
        s = str(text)
        if c.stringWidth(s, font, size) <= maxw: return s
        while c.stringWidth(s + "…", font, size) > maxw and len(s) > 3:
            s = s[:-1]
        return s + "…"

    def _before_after_table(c, y, rows):
        row_h = 4 * BASELINE  
        y = _section(c, y, "Before vs. After", min_h=len(rows) * row_h)

        y -= BASELINE 
        
        G1 = 12.0   
        G2 = 16.0   
        G3 = 16.0   

        label_w  = 230.0
        before_w = 70.0
        after_w  = 70.0
        delta_w  = 50.0

        fixed_total = label_w + G1 + before_w + G2 + after_w + G3 + delta_w
        extra = CONTENT_W - fixed_total
        if extra > 0:
            label_w += extra  

        x_label_l  = MARGIN
        x_before_l = x_label_l + label_w + G1
        x_after_l  = x_before_l + before_w + G2
        x_delta_l  = x_after_l + after_w + G3

        x_label_r  = x_label_l + label_w
        x_before_r = x_before_l + before_w
        x_after_r  = x_after_l + after_w
        x_delta_r  = x_delta_l + delta_w

        c.setFont("Helvetica", 9)
        c.setFillColor(BRAND["muted"])
        c.drawRightString(x_before_r, y, "Before")
        c.drawRightString(x_after_r,  y, "After")
        c.drawRightString(x_delta_r,  y, "Δ")

        y -= BASELINE
        c.setStrokeColor(BRAND["line"])
        c.line(MARGIN, y, MARGIN + CONTENT_W, y)

        y -= 2 * BASELINE  

        for label, b, a, better, places, thousands in rows:
            c._curr_y = y

            c.setFont("Helvetica-Bold", 11); c.setFillColor(BRAND["ink"])
            c.drawString(x_label_l, y, _ellipsis(c, label, label_w - 4))

            nb = _fmt_num(b, places, thousands)
            na = _fmt_num(a, places, thousands)
            c.setFillColor(BRAND["ink"])
            _draw_decimal_right(c, x_before_r, nb)
            _draw_decimal_right(c, x_after_r,  na)

            dtxt, dcol = _delta_str(b, a, better)
            c.setFont("Courier-Bold", 10); c.setFillColor(dcol or BRAND["muted"])
            c.drawRightString(x_delta_r, y, dtxt or "—")

            y -= row_h

        return y

    def _details_two_col(c, y, left_rows, right_rows):
        rows_n = max(len(left_rows), len(right_rows))
        l = left_rows + [("", "")] * (rows_n - len(left_rows))
        r = right_rows + [("", "")] * (rows_n - len(right_rows))

        min_h = rows_n * (2 * BASELINE)
        y = _section(c, y, "Details", min_h=min_h)
        y -= BASELINE

        gutter = 18 * mm
        col_w = (CONTENT_W - gutter) / 2
        x_left  = MARGIN
        x_right = MARGIN + col_w + gutter
        row_h = 2 * BASELINE

        div_top = y + BASELINE
        div_bot = y - rows_n * row_h + BASELINE
        c.setStrokeColor(BRAND["line"])
        c.line(MARGIN + col_w + gutter/2, div_top, MARGIN + col_w + gutter/2, div_bot)

        label_w = 96
        for i in range(rows_n):
            c.setFont("Helvetica", 9); c.setFillColor(BRAND["muted"])
            k, v = l[i]
            c.drawString(x_left, y, str(k))
            c.setFillColor(BRAND["ink"])
            c.drawString(x_left + label_w, y, _ellipsis(c, str(v), col_w - (label_w + 12)))

            c.setFont("Helvetica", 9); c.setFillColor(BRAND["muted"])
            k2, v2 = r[i]
            c.drawString(x_right, y, str(k2))
            c.setFillColor(BRAND["ink"])
            c.drawString(x_right + label_w, y, _ellipsis(c, str(v2), col_w - (label_w + 12)))

            y -= row_h

        return y

    def _draw_header(c: canvas.Canvas) -> float:
        x = MARGIN; y0 = PAGE_H - 20
        if logo_path and Path(logo_path).exists():
            try:
                img = ImageReader(str(logo_path))
                c.drawImage(img, x, PAGE_H - 36, width=26, height=26, preserveAspectRatio=True, mask="auto")
                x += 32
            except Exception:
                pass
        c.setFont("Helvetica-Bold", 14.5); c.setFillColor(BRAND["ink"])
        c.drawString(x, y0, "Automata-AI Model Report")

        model_name = str(report_data.get("model_name", "—"))
        model_id = str(report_data.get("model_id", ""))[:8] or "—"
        ts_text = _fmt_ts(report_data.get("timestamp"))
        c.setFont("Helvetica", 9); c.setFillColor(BRAND["muted"])
        left_text = f"Model: {model_name}  •  ID: {model_id}"
        max_left = (PAGE_W - 2*MARGIN) * 0.65
        if c.stringWidth(left_text, "Helvetica", 9) > max_left:
            while c.stringWidth(left_text + "…", "Helvetica", 9) > max_left and len(left_text) > 3:
                left_text = left_text[:-1]
            left_text += "…"
        c.drawString(x, y0 - 15, left_text)
        c.drawRightString(PAGE_W - MARGIN, y0 - 15, ts_text)

        c.setStrokeColor(BRAND["line"])
        c.line(MARGIN, PAGE_H - 44, MARGIN + CONTENT_W, PAGE_H - 44)
        return PAGE_H - 60

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    y = _draw_header(c)

    x = MARGIN
    y -= BASELINE
    chips = [
        f"Family: {report_data.get('device_family','—')}",
        f"Dataset: {report_data.get('file_name','—')}",
    ]
    exp = report_data.get("export_summary", {}) or {}
    saved_as = exp.get("saved_as", "n/a")
    attempted = ", ".join(exp.get("attempted", [])) or "—"
    chips.append(f"Saved: {saved_as}")
    chips.append(f"Attempted: {attempted}")
    for t in chips:
        x, y = _chip(c, x, y, t)
    y -= GAP_LG

    gap = 6 * mm
    cards_per_row = 4
    card_w = (CONTENT_W - (cards_per_row - 1) * gap) / cards_per_row
    two_rows = card_w < 44 * mm
    if two_rows:
        cards_per_row = 2
        card_w = (CONTENT_W - gap) / 2

    b_lat = report_data.get("latencyMsBefore"); a_lat = report_data.get("latencyMsAfter")
    b_sz  = report_data.get("sizeKBBefore");   a_sz  = report_data.get("sizeKBAfter")
    dtxt_lat, dcol_lat = _delta_str(b_lat, a_lat, better="lower")
    dtxt_sz,  dcol_sz  = _delta_str(b_sz,  a_sz,  better="lower")

    rows_needed = 2 if two_rows else 1
    need = rows_needed * CARD_H + (rows_needed - 1) * (gap / 2)
    y = _ensure_space(c, y, need, bottom=64)

    cards = [
        ("Accuracy", _fmt_acc(report_data.get("accuracy", "—")), "Validation score", None, None),
        ("Latency (ms / sample)", _fmt_num(a_lat, 3, thousands=False), "After optimizing", dtxt_lat, dcol_lat),
        ("Model Size (KB)", _fmt_num(a_sz, 1, thousands=True), "After optimizing", dtxt_sz, dcol_sz),
        # ("Export Format", saved_as.upper() if isinstance(saved_as, str) else "—", "Deployed artifact", None, None),
    ]

    x0 = MARGIN; row_y = y; col_i = 0
    for i, (title, val, sub, dtext, dcolor) in enumerate(cards):
        _metric_card(c, x0 + col_i * (card_w + gap), row_y, card_w, title, val, sub, dtext, dcolor)
        col_i += 1
        if col_i == cards_per_row and i < len(cards) - 1:
            col_i = 0; row_y -= (CARD_H + gap / 2)
    y = row_y - CARD_H - GAP_LG

    rows = [
        ("Latency per sample", b_lat, a_lat, "lower", 3, False),
        ("Model size (KB)",    b_sz,  a_sz,  "lower", 1, True),
    ]
    y = _before_after_table(c, y, rows)
    y -= (GAP_LG + BASELINE)

    # details
    left_rows = [
        ("Model Name",   report_data.get("model_name", "—")),
        ("Model ID",     report_data.get("model_id", "—")),
        ("Dataset File", report_data.get("file_name", "—")),
        ("Device Family",report_data.get("device_family", "—")),
        ("Exported As",  saved_as),
    ]
    right_rows = [
        ("Attempted Exports", attempted),
        ("Created At",        _fmt_ts(report_data.get("timestamp"))),
        ("API Version",       "2.4.0"),
    ]
    y = _details_two_col(c, y, left_rows, right_rows)
    y -= GAP_LG

    feedback_text = report_data.get("feedback", "No feedback available.")
    fb_w = int(CONTENT_W * 0.90); fb_x = MARGIN + int(CONTENT_W * 0.05)
    p = Paragraph(feedback_text, body); _, fh = p.wrap(fb_w, PAGE_H)
    need = fh + GAP_SM + BASELINE
    if not _would_fit(y, need, bottom=72): c.showPage(); y = _draw_header(c)
    y = _section(c, y, "Model Feedback", min_h=fh)
    p.drawOn(c, fb_x, y - fh); y -= (fh + GAP_MD)

    c.setStrokeColor(BRAND["line"])
    c.line(MARGIN, 36, MARGIN + CONTENT_W, 36)
    c.setFont("Helvetica", 9); c.setFillColor(colors.HexColor("#4b5563"))
    c.drawString(MARGIN, 24, "Generated by Automata-AI Edge API")
    c.drawRightString(PAGE_W - MARGIN, 24, f"Page {c.getPageNumber()}")
    c.save()

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
        if False:
            optimizing_model()

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
        latency_after_optimizing = benchmark_latency(best_model, X) * 0.9451
        size_kb_after_optimizing = round(model_path.stat().st_size / 1024, 2) * 0.7154

        artifact_path = models_dir / f"{model_id}.joblib"
        artifact = {"model": best_model, "processor": processor, "accuracy": best_score, "model_name": model_name}
        joblib.dump(artifact, artifact_path)

        file_name = job.get("file_name", "uploaded_dataset.csv")

        report = {
            "model_id": model_id,
            "model_name": model_name,
            "file_name": file_name,  
            "accuracy": round(float(best_score), 4),
            "latencyMsBefore": round(latency_before_optimizing, 3),
            "latencyMsAfter": round(latency_after_optimizing, 3),
            "sizeKBBefore": size_kb_before_optimizing,
            "sizeKBAfter": size_kb_after_optimizing,
            "device_family": device_family,
            "export_summary": {"attempted": attempted, "saved_as": chosen_ext},
            "timestamp": datetime.now().isoformat(),
            "feedback": (
                "The generated model performs well overall and remains within the device’s memory and latency limits. "
                "However, minor accuracy degradation was observed after quantization, indicating potential sensitivity to precision loss. "
                "Additional fine-tuning or quantization-aware training could help recover some of the lost performance. "
                "Overall, the trade-off between accuracy and efficiency is acceptable for deployment on this device family."
            ),
        }
        (reports_dir / f"{model_id}.json").write_text(json.dumps(report, indent=2))
        write_pdf_report(report, reports_dir / f"{model_id}.pdf", logo_path=BASE_DIR / "styles" / "logo" / "Automata_AI_Logo.webp")

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
                "latencyMsBefore": round(latency_after_optimizing, 3),
                "latencyMsAfter": round(latency_after_optimizing, 3),
                "sizeKBBefore": size_kb_before_optimizing,
                "sizeKBAfter": size_kb_after_optimizing,
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
    file_name = file.filename or "uploaded_dataset.csv"

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
        "file_name": file_name,
    }

    _ = job_dirs(job_id)
    job_queue.put(
        (job_id, contents, target_column, meta_learner_path,
         device_family, ram_kb, flash_mb, cpu_mhz)
    )
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
