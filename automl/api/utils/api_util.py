from __future__ import annotations

import time
import uuid
import traceback
import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional


from automl.pipelines.utils.pipelines_utils import DEVICE_FAMILIES


def new_task_id() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


@dataclass
class TaskRecord:
    task_id: str
    user_id: str
    task_type: str
    pipeline: Any

    created_at: float = field(default_factory=_now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    sched_state: str = "queued"  # queued | running | completed | failed

    last_status: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class SingleUserSequentialScheduler:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._queue: List[str] = []
        self._running_task_id: Optional[str] = None

    # -----------------------------
    # Public API
    # -----------------------------

    def submit(self, *, task_id: str, user_id: str, task_type: str, pipeline: Any) -> TaskRecord:
        rec = TaskRecord(task_id=task_id, user_id=user_id, task_type=task_type, pipeline=pipeline)
        try:
            rec.last_status = pipeline.get_status()
        except Exception:
            rec.last_status = {}

        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"task_id already exists: {task_id}")
            self._tasks[task_id] = rec
            self._queue.append(task_id)

        return rec

    def update_pipeline(self, task_id: str, pipeline: Any) -> None:
        with self._lock:
            if task_id not in self._tasks:
                return
            rec = self._tasks[task_id]
            rec.pipeline = pipeline
            try:
                rec.last_status = pipeline.get_status()
            except Exception:
                pass

    def fail_task(self, task_id: str, reason: str, exc: Optional[BaseException] = None) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return
        
        rec.sched_state = "failed"
        err = {"reason": reason}
        if exc:
            err["exception"] = str(exc)
            err["traceback"] = traceback.format_exc()
        rec.error = err
        rec.finished_at = _now()

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> List[TaskRecord]:
        with self._lock:
            return list(self._tasks.values())

    def get_status(self, task_id: str) -> Dict[str, Any]:
        rec = self.get_task(task_id)
        if rec is None:
            raise KeyError("task_id not found")

        try:
            st = rec.pipeline.get_status()
            rec.last_status = st
        except Exception:
            st = rec.last_status or {}

        with self._lock:
            running_task_id = self._running_task_id
            queued_ids = list(self._queue)

        def _sanitize_json(obj):
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, dict):
                return {k: _sanitize_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize_json(v) for v in obj]
            if isinstance(obj, tuple):
                return [_sanitize_json(v) for v in obj]
            return obj

        st = _sanitize_json(st)
        metrics = _sanitize_json(st.get("metrics", {}))
        error = _sanitize_json(rec.error)

        return {
            "task_id": rec.task_id,
            "user_id": rec.user_id,
            "task_type": rec.task_type,
            "sched_state": rec.sched_state,
            "created_at": rec.created_at,
            "started_at": rec.started_at,
            "finished_at": rec.finished_at,
            "status": st,
            "metrics": metrics,
            "error": error,
            "global": {
                "running_task_id": running_task_id,
                "queued_task_ids": queued_ids,
            },
        }

    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        rec = self.get_task(task_id)
        if rec is None:
            raise KeyError("task_id not found")
        return rec.result

    def queue_info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running_task_id": self._running_task_id,
                "queued_task_ids": list(self._queue),
            }

    def process_next(self) -> Optional[TaskRecord]:

        with self._lock:
            if self._running_task_id is not None:
                return None
            if not self._queue:
                return None
            task_id = self._queue.pop(0)
            self._running_task_id = task_id
            rec = self._tasks.get(task_id)

        if rec is None:
            with self._lock:
                self._running_task_id = None
            return None

        rec.sched_state = "running"
        rec.started_at = _now()
        try:
            res = rec.pipeline.run()
            rec.result = res
            
            pipeline_status = res.get("status", {}).get("status")
            if pipeline_status == "failed":
                rec.sched_state = "failed"
                p_err = res.get("errors", {})
                rec.error = {
                    "reason": p_err.get("reason", "Unknown failure"),
                    "exception": p_err.get("exception"),
                    "traceback": p_err.get("traceback"),
                }
            else:
                rec.sched_state = "completed"
                rec.error = None
        except Exception as e:
            rec.sched_state = "failed"
            rec.error = {
                "reason": "pipeline.run() raised",
                "exception": str(e),
                "traceback": traceback.format_exc(),
            }
            try:
                rec.last_status = rec.pipeline.get_status()
            except Exception:
                pass
        finally:
            rec.finished_at = _now()
            try:
                rec.last_status = rec.pipeline.get_status()
            except Exception:
                pass
            with self._lock:
                self._running_task_id = None

        return rec


# -----------------------------
# Pipeline factory
# -----------------------------

def create_pipeline(task_type: str, **kwargs) -> Any:
    task_type = str(task_type).strip().lower()

    
    if task_type == "image":
        from automl.pipelines.image.image_pipeline import ImagePipeline
        return ImagePipeline(task_type=task_type, **kwargs)
    
    if task_type == "audio":
        from automl.pipelines.audio.audio_pipeline import AudioPipeline
        return AudioPipeline(task_type=task_type, **kwargs)
    
    if task_type == "sensor":
        from automl.pipelines.sensor.sensor_pipeline import SensorPipeline
        return SensorPipeline(task_type=task_type, **kwargs)

    raise ValueError(f"Unsupported task_type: {task_type}")


def device_families_payload() -> Dict[str, Any]:
    return {"device_families": DEVICE_FAMILIES}
