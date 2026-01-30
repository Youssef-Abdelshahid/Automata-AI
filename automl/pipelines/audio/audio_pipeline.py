from __future__ import annotations

import os
import time
import math
import shutil
import re
import traceback
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import torchaudio.transforms as AT

from automl.pipelines.utils.pipelines_utils import (
    validate_specs_within_family,
    validate_final_against_specs,
    get_family_name,
    family_allowed_formats,
    family_frameworks,
    normalize_device_specs,
    STATUS_ORDER,
    LOGO_PATH,
)

@dataclass
class PipelineState:
    task_id: str
    user_id: str
    task_type: str  # "audio"
    status: str = "queued"
    stage_idx: int = 0
    message: str = ""
    updated_at: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["updated_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.updated_at))
        return d


# =============================================================================
# Helper Classes
# =============================================================================

class _DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class _DSCNNPlus(nn.Module):
    def __init__(self, num_classes: int, drop=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            _DSConvBlock(48, 96,  stride=1, dropout=drop),
            _DSConvBlock(96, 192, stride=2, dropout=drop),
            _DSConvBlock(192, 192, stride=1, dropout=drop),
            _DSConvBlock(192, 256, stride=2, dropout=drop),
            _DSConvBlock(256, 256, stride=1, dropout=drop),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_drop = nn.Dropout(drop)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        x = self.pool(x).flatten(1)
        x = self.head_drop(x)
        return self.head(x)


class _InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, drop=0.0):
        super().__init__()
        assert stride in [1, 2]
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.drop(out)
        return x + out if self.use_res else out


class _MobileNetV2Spec(nn.Module):
    def __init__(self, num_classes: int, width_mult=0.75, drop=0.10):
        super().__init__()
        cfg = [
            (1,  16, 1, 1),
            (6,  24, 2, 2),
            (6,  32, 3, 2),
            (6,  64, 3, 2),
            (6,  96, 2, 1),
            (6, 160, 1, 2),
        ]

        def _make_divisible(v, divisor=8):
            return int(math.ceil(v / divisor) * divisor)

        in_ch = _make_divisible(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(1, in_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )

        blocks = []
        for t, c, n, s in cfg:
            out_ch = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(_InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=t, drop=drop))
                in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        last_ch = _make_divisible(192 * width_mult)
        self.last = nn.Sequential(
            nn.Conv2d(in_ch, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_drop = nn.Dropout(drop)
        self.head = nn.Linear(last_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        x = self.last(x)
        x = self.pool(x).flatten(1)
        x = self.head_drop(x)
        return self.head(x)


@dataclass
class _AudioPreprocessConfig:
    target_sr: int = 16000
    clip_seconds: float = 1.0
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    use_global_norm: bool = True
    use_spec_augment: bool = True
    time_mask_param: int = 8
    freq_mask_param: int = 8
    num_time_masks: int = 1
    num_freq_masks: int = 1


class _AudioPreprocessor:
    def __init__(self, config: _AudioPreprocessConfig):
        self.config = config
        self._mel = None
        self._to_db = AT.AmplitudeToDB(stype="power")
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        self._time_mask = None
        self._time_mask = None
        self._freq_mask = None

    def _pad_or_trim(self, waveform: torch.Tensor, target_len: int) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        T = waveform.shape[-1]
        if T == target_len:
            return waveform
        if T > target_len:
            return waveform[:, :target_len]
        return F.pad(waveform, (0, target_len - T))

    def _ensure_transforms(self):
        if self._mel is None:
            cfg = self.config
            self._mel = AT.MelSpectrogram(
                sample_rate=cfg.target_sr,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                n_mels=cfg.n_mels,
            )
        if self.config.use_spec_augment and self._time_mask is None:
            cfg = self.config
            self._time_mask = AT.TimeMasking(time_mask_param=cfg.time_mask_param)
            self._freq_mask = AT.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)

    def _waveform_to_logmel(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        cfg = self.config
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sr != cfg.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, cfg.target_sr)
        
        target_len = int(cfg.clip_seconds * cfg.target_sr)
        
        target_len = int(cfg.clip_seconds * cfg.target_sr)
        waveform = self._pad_or_trim(waveform, target_len)

        self._ensure_transforms()

        self._ensure_transforms()
        spec = self._mel(waveform)
        spec = self._to_db(spec)
        return spec

    def fit_global_norm(self, dataset, max_items: int = 512) -> None:
        sums, sumsq, count = 0.0, 0.0, 0
        n = len(dataset)
        idxs = list(range(n))
        random.shuffle(idxs)
        idxs = idxs[:max_items]

        with torch.no_grad():
            for i in idxs:
                waveform, sr, *_ = dataset[i]
                spec = self._waveform_to_logmel(waveform, sr)
                v = spec.reshape(-1)
                sums += float(v.sum())
                sumsq += float((v ** 2).sum())
                count += int(v.numel())

        if count > 0:
            mean = sums / count
            var = max(1e-6, (sumsq / count - mean ** 2))
            self.global_mean = mean
            self.global_std = var ** 0.5
        else:
            self.global_mean, self.global_std = 0.0, 1.0

    def transform(self, waveform: torch.Tensor, sr: int, img_size: int, augment: bool) -> torch.Tensor:
        cfg = self.config
        spec = self._waveform_to_logmel(waveform, sr)

        if cfg.use_global_norm and self.global_mean is not None:
            spec = (spec - self.global_mean) / (self.global_std + 1e-6)
        else:
            mean = spec.mean()
            std = spec.std()
            spec = (spec - mean) / (std + 1e-6)

        if augment and cfg.use_spec_augment:
            self._ensure_transforms()
            x = spec
            for _ in range(int(cfg.num_time_masks)):
                x = self._time_mask(x)
            for _ in range(int(cfg.num_freq_masks)):
                x = self._freq_mask(x)
            spec = x

        spec = spec.unsqueeze(0)
        spec = F.interpolate(spec, size=(img_size, img_size), mode="bilinear", align_corners=False)
        spec = spec.squeeze(0)
        return spec


class _AudioFolderDataset(Dataset):

    def __init__(self, root: str):
        self.root = root
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        allowed_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
            
        class_dirs = sorted([d for d in os.listdir(root) 
                           if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")])
        
        if len(class_dirs) == 0:
            raise RuntimeError(f"No class folders found in {root}")
        
        self.classes = class_dirs
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}
        
        for class_name in class_dirs:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in allowed_exts:
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fpath, class_idx = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(fpath)
        except Exception as e:
            waveform = torch.zeros(1, 16000)
            sr = 16000
        
        return waveform, sr, self.classes[class_idx]


class _AudioDatasetWrapped(Dataset):
    def __init__(self, base_ds, label2idx: Dict[str, int], tf):
        self.ds = base_ds
        self.label2idx = label2idx
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        waveform, sr, label_str = self.ds[idx]
        x = self.tf(waveform, int(sr))
        y = self.label2idx[str(label_str)]
        return x, y


class AudioPipeline:

    def __init__(
        self,
        *,
        task_id: str,
        user_id: str,
        task_type: str,
        dataset_path: str,
        device_family_id: str,
        device_specs: Dict[str, Any],
        target_num_classes: int,
        output_root: str = "runs",
        seed: int = 42,
        batch_size: int = 128,
        num_workers: int = 2,
        pin_memory: bool = True,
        torch_device: Optional[str] = None,

        sweep: Optional[list] = None,
        epochs_per_trial: int = 3,
        final_epochs: int = 10,
        label_smoothing: float = 0.08,
        mixup_alpha: float = 0.35,
        use_mixup: bool = True,

        export_ext: str = ".h",
        min_accuracy: float = 0.50,
        
        title: str = None,
        description: str = None,
        visibility: str = None,
        
        quantization: str = None,
        optimization_strategy: str = None,
        epochs: int = None,
        training_speed: str = None,
        accuracy_tolerance: str = None,
        optimization_trigger_ratio: float = 0.70,
        
        noise_handling: str = None,
        cleaning: str = None,
    ):
        if task_type != "audio":
            raise ValueError(f"AudioPipeline supports task_type='audio' only, got {task_type}")

        self.task_id = task_id
        self.user_id = user_id
        self.task_type = task_type
        
        self.title = title
        self.description = description
        self.visibility = visibility
        self.quantization = quantization
        self.optimization_strategy = optimization_strategy
        self.epochs = epochs
        self.training_speed = training_speed
        self.accuracy_tolerance = accuracy_tolerance
        self.optimization_trigger_ratio = float(optimization_trigger_ratio)
        self.noise_handling = noise_handling
        self.cleaning = cleaning

        self.dataset_path = dataset_path
        self.device_family_id = device_family_id
        self.device_specs = normalize_device_specs(device_specs)
        self.target_num_classes = int(target_num_classes)

        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.output_dir = os.path.join(output_root, task_id)
        self.dataset_extract_dir = os.path.join(self.output_dir, "dataset")

        self.export_ext = str(export_ext).strip().lower()
        if self.export_ext != "all" and not self.export_ext.startswith("."):
            self.export_ext = "." + self.export_ext
        self.min_accuracy = float(min_accuracy)

        self.sweep = sweep or [
            {"name": "dscnn_plus_full_res64",  "backbone": "dscnn_plus",        "unfreeze_blocks": 999, "img_size": 64, "lr": 1.2e-3},
            {"name": "dscnn_plus_last3_res64", "backbone": "dscnn_plus",        "unfreeze_blocks": 3,   "img_size": 64, "lr": 1.5e-3},
            {"name": "mbv2_tiny_full_res64",   "backbone": "mobilenetv2_tiny",  "unfreeze_blocks": 999, "img_size": 64, "lr": 1.0e-3},
            {"name": "mbv2_small_full_res64",  "backbone": "mobilenetv2_small", "unfreeze_blocks": 999, "img_size": 64, "lr": 9.0e-4},
        ]
        self.epochs_per_trial = int(epochs_per_trial)
        self.final_epochs = int(final_epochs)
        if self.epochs: 
             self.final_epochs = int(self.epochs)
             
        self.label_smoothing = float(label_smoothing)
        self.mixup_alpha = float(mixup_alpha)
        self.use_mixup = bool(use_mixup)

        self.state = PipelineState(task_id=task_id, user_id=user_id, task_type=task_type)

        self._data_meta: Dict[str, Any] = {}
        
        self._prep_train_loader = None
        self._prep_val_loader = None
        self._prep_test_loader = None
        
        self._final_train_loader = None
        self._final_val_loader = None
        self._final_test_loader = None
        self._trained_model = None
        self._final_model = None
        self._final_model_path: Optional[str] = None
        self._report_path: Optional[str] = None

        if torch_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(torch_device)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    # --------------------------
    # Status / results
    # --------------------------
    def update_status(self, new_status: str, message: str = "", metrics_patch: Optional[Dict] = None, errors_patch: Optional[Dict] = None) -> None:
        if new_status not in STATUS_ORDER:
            raise ValueError(f"Invalid status '{new_status}'.")
        self.state.status = new_status
        self.state.stage_idx = STATUS_ORDER.index(new_status)
        self.state.message = message
        self.state.updated_at = time.time()
        if metrics_patch:
            self.state.metrics.update(metrics_patch)
        if errors_patch:
            self.state.errors.update(errors_patch)

    def get_status(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def _fail(self, reason: str, exc: Optional[BaseException] = None) -> Dict[str, Any]:
        err = {"reason": reason}
        if exc is not None:
            err["exception"] = str(exc)
            err["traceback"] = traceback.format_exc()
        self.update_status("failed", reason, errors_patch=err)
        return self._result()

    def _result(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "status": self.state.to_dict(),
            "output_dir": self.output_dir,
            "final_model_path": self._final_model_path,
            "report_path": self._report_path,
            "metrics": dict(self.state.metrics),
            "errors": dict(self.state.errors),
            "exported_model_paths": self.state.metrics.get("exported_model_paths", []),
        }

    def _prepare_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dataset_extract_dir, exist_ok=True)

    def _unpack_and_validate_dataset(self) -> None:

        allowed_audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        
        def is_audio_file(p: str) -> bool:
            return os.path.splitext(p)[1].lower() in allowed_audio_exts
        
        def list_dir_safe(d: str):
            try:
                return [os.path.join(d, x) for x in os.listdir(d) if not x.startswith(".")]
            except Exception:
                return []
        
        def count_audio_in_dir(d: str) -> int:
            n = 0
            for r, _, files in os.walk(d):
                for f in files:
                    if is_audio_file(f):
                        n += 1
            return n
        
        def validate_audio_folder_root(root: str):
            candidates = [p for p in list_dir_safe(root) if os.path.isdir(p)]
            class_info = []
            for cdir in candidates:
                cname = os.path.basename(cdir)
                audio_count = count_audio_in_dir(cdir)
                if audio_count > 0:
                    class_info.append((cname, audio_count))
            
            if len(class_info) < 2:
                return False, f"Need >=2 class folders. Found {len(class_info)}.", {}
            
            meta = {
                "data_root": root,
                "num_classes": len(class_info),
                "classes": [c for c, _ in sorted(class_info, key=lambda x: x[0].lower())],
                "class_counts": {c: n for c, n in class_info},
            }
            return True, "OK", meta

        def find_best_root(extract_dir: str):
            ok, msg, meta = validate_audio_folder_root(extract_dir)
            if ok: return True, "OK", extract_dir, meta
            
            entries = [p for p in list_dir_safe(extract_dir) if os.path.isdir(p)]
            if len(entries) == 1:
                ok2, msg2, meta2 = validate_audio_folder_root(entries[0])
                if ok2: return True, "OK", entries[0], meta2
                
            from collections import deque
            q = deque([(extract_dir, 0)])
            best = None
            while q:
                node, depth = q.popleft()
                if depth > 3: continue
                ok3, _, meta3 = validate_audio_folder_root(node)
                if ok3:
                    total_audio = sum(meta3["class_counts"].values())
                    score = (meta3["num_classes"], total_audio)
                    if best is None or score > (best[0], best[1]):
                        best = (meta3["num_classes"], total_audio, node, meta3)
                for child in list_dir_safe(node):
                    if os.path.isdir(child):
                        q.append((child, depth+1))
            
            if best: return True, "OK", best[2], best[3]
            return False, f"No valid audio classes found in {extract_dir}", "", {}

        p = self.dataset_path
        if os.path.isdir(p):
            ok, msg, meta = validate_audio_folder_root(p)
            if not ok: raise RuntimeError(msg)
            self.dataset_path = p
            self.state.metrics.update({"dataset_mode": "folder", **meta})
            return

        if not os.path.isfile(p):
            raise FileNotFoundError(f"dataset_path not found: {p}")

        for name in os.listdir(self.dataset_extract_dir):
            full = os.path.join(self.dataset_extract_dir, name)
            if os.path.isdir(full): shutil.rmtree(full, ignore_errors=True)
            else: 
                try: os.remove(full)
                except: pass
        
        try:
            shutil.unpack_archive(p, self.dataset_extract_dir)
        except:
            raise RuntimeError(f"Failed to unpack archive: {p}")
            
        ok_root, msg_root, root_path, meta = find_best_root(self.dataset_extract_dir)
        if not ok_root: raise RuntimeError(msg_root)
        
        self.dataset_path = root_path
        self.state.metrics.update({"dataset_mode": "archive", "dataset_extracted_to": root_path, **meta})


    # ==========================================================
    # Main Entry
    # ==========================================================
    def run(self) -> Dict[str, Any]:
        try:
            self._prepare_dirs()

            self.update_status("queued", "Validating device specs.")
            ok, msg = validate_specs_within_family(self.device_family_id, self.device_specs)
            if not ok: return self._fail(f"Device specs invalid: {msg}")

            self.update_status("preprocessing", "Preparing dataset.")
            self._unpack_and_validate_dataset()
            self._phase_preprocessing()

            self.update_status("training", "Training.")
            self._phase_training()

            self.update_status("optimizing", "Optimizing.")
            self._phase_optimizing()

            self.update_status("packaging", "Exporting.")
            self._phase_packaging()
            self._phase_report()

            self._phase_validate_final()

            self.update_status("completed", "Success.")
            return self._result()

        except (RuntimeError, ValueError) as e:
            return self._fail(str(e), e)
        except Exception as e:
            return self._fail("Unhandled pipeline error", e)


    # ==========================================================
    # Phase 1: Preprocessing
    # ==========================================================
    def _phase_preprocessing(self) -> None:
        def _seed_everything(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        def _seed_worker(worker_id: int):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        _seed_everything(self.seed)
        
        img_size = 64
        t0 = time.time()
        self.update_status("preprocessing", "Analyzing dataset and building transforms.")

        full_dataset = _AudioFolderDataset(self.dataset_path)
        
        n_total = len(full_dataset)
        test_split = 0.1
        val_split = 0.1
        
        test_len = max(1, int(test_split * n_total))
        val_len = max(1, int(val_split * (n_total - test_len)))
        train_len = n_total - test_len - val_len
        
        train_base, val_base, test_base = random_split(
            full_dataset, 
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        label2idx = full_dataset.class_to_idx
        idx2label = {v: k for k, v in label2idx.items()}
        inferred = len(label2idx)
        
        if inferred != self.target_num_classes:
            raise RuntimeError(f"Found {inferred} classes, expected {self.target_num_classes} ({label2idx.keys()})")

        def profile_ds(ds, max_items=256):
            sr_counts = {}
            durations = []
            rms_values = []
            files_per_class = {}
            idxs = list(range(len(ds)))
            random.shuffle(idxs)
            idxs = idxs[:max_items]
            
            for i in idxs:
                w, sr, lbl = ds[i]
                sr = int(sr)
                sr_counts[sr] = sr_counts.get(sr, 0) + 1
                durations.append(w.shape[-1] / sr)
                rms = float(torch.sqrt((w ** 2).mean()))
                rms_values.append(rms)
                files_per_class[lbl] = files_per_class.get(lbl, 0) + 1
            
            return {
                "sr_counts": sr_counts,
                "durations": np.array(durations, dtype=np.float32),
                "rms_values": np.array(rms_values, dtype=np.float32),
                "num_files": len(ds),
                "files_per_class": files_per_class
            }
        
        prof = profile_ds(train_base)
        
        cfg = _AudioPreprocessConfig()
        if prof["sr_counts"]:
            cfg.target_sr = max(prof["sr_counts"], key=prof["sr_counts"].get)
        if len(prof["durations"]) > 0:
            p95 = np.percentile(prof["durations"], 95)
            cfg.clip_seconds = float(max(0.5, min(5.0, p95)))
            
        cfg.use_global_norm = (prof["num_files"] > 500)
        
        # Fit logic
        preproc = _AudioPreprocessor(cfg)
        if cfg.use_global_norm:
            preproc.fit_global_norm(train_base)
            
        def train_tf(w, sr): return preproc.transform(w, sr, img_size, True)
        def test_tf(w, sr): return preproc.transform(w, sr, img_size, False)
        
        train_set = _AudioDatasetWrapped(train_base, label2idx, train_tf)
        val_set = _AudioDatasetWrapped(val_base, label2idx, test_tf)
        test_set = _AudioDatasetWrapped(test_base, label2idx, test_tf)
        
        g = torch.Generator().manual_seed(self.seed)
        self._prep_train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=self.pin_memory, 
            worker_init_fn=_seed_worker, generator=g
        )
        self._prep_val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=self.pin_memory, 
            worker_init_fn=_seed_worker, generator=g
        )
        self._prep_test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=self.pin_memory, 
            worker_init_fn=_seed_worker, generator=g
        )

        elapsed = time.time() - t0
        
        self._data_meta["preprocessing_report"] = {
            "dataset": {
                "root": self.dataset_path, 
                "num_files": prof["num_files"],
                "class_counts": prof["files_per_class"], 
                "classes": list(full_dataset.class_to_idx.keys())
            },
            "audio": asdict(cfg),
            "stats": {"durations": [float(d) for d in prof["durations"]]}
        }
        self.state.metrics.update({
             "phase1_seconds": elapsed,
             "num_classes": inferred,
             "target_sr": cfg.target_sr,
             "clip_seconds": cfg.clip_seconds,
        })

    # ==========================================================
    # Phase 2: Training
    # ==========================================================

    def _phase_training(self) -> None:
        if self.target_num_classes <= 0: raise ValueError("Invalid num_classes")

        # Helpers
        def _build_backbone(backbone: str, num_classes: int) -> nn.Module:
            if backbone == "dscnn_plus":
                return _DSCNNPlus(num_classes=num_classes, drop=0.15)
            elif backbone == "mobilenetv2_tiny":
                return _MobileNetV2Spec(num_classes=num_classes, width_mult=0.50, drop=0.10)
            elif backbone == "mobilenetv2_small":
                return _MobileNetV2Spec(num_classes=num_classes, width_mult=0.75, drop=0.10)
            raise ValueError(f"Unknown backbone: {backbone}")

        def build_dataloaders(img_size: int):
            full_dataset = _AudioFolderDataset(self.dataset_path)
            n_total = len(full_dataset)
            test_len = max(1, int(0.1 * n_total))
            val_len = max(1, int(0.1 * (n_total - test_len)))
            train_len = n_total - test_len - val_len
            
            train_base, val_base, test_base = random_split(
                full_dataset, [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(self.seed)
            )
            
            saved_cfg_dict = self._data_meta.get("preprocessing_report", {}).get("audio", {})
            if saved_cfg_dict:
                cfg = _AudioPreprocessConfig(**saved_cfg_dict)
            else:
                cfg = _AudioPreprocessConfig()
            
            preproc = _AudioPreprocessor(cfg)
            if cfg.use_global_norm:
                 preproc.fit_global_norm(train_base)
                 
            def train_tf(w, sr): return preproc.transform(w, sr, img_size, True)
            def test_tf(w, sr): return preproc.transform(w, sr, img_size, False)
            
            label2idx = full_dataset.class_to_idx
            
            train_set = _AudioDatasetWrapped(train_base, label2idx, train_tf)
            val_set = _AudioDatasetWrapped(val_base, label2idx, test_tf)
            test_set = _AudioDatasetWrapped(test_base, label2idx, test_tf)
            
                
            def _seed_worker(worker_id: int):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            
            g = torch.Generator().manual_seed(self.seed)
            tl = DataLoader(train_set, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, worker_init_fn=_seed_worker, generator=g)
            vl = DataLoader(val_set, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, worker_init_fn=_seed_worker, generator=g)
            testl = DataLoader(test_set, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, worker_init_fn=_seed_worker, generator=g)
            
            return tl, vl, testl

        def smooth_one_hot(y, n_classes, smoothing):
            y_oh = torch.zeros((y.size(0), n_classes), device=y.device)
            y_oh.scatter_(1, y.unsqueeze(1), 1.0)
            if smoothing > 0:
                y_oh = y_oh * (1.0 - smoothing) + smoothing / n_classes
            return y_oh

        def mixup_batch(x, y, alpha, n_classes, smoothing):
            if alpha <= 0: return x, smooth_one_hot(y, n_classes, smoothing)
            lam = torch.distributions.Beta(alpha, alpha).sample((x.size(0),)).to(x.device)
            lam = torch.maximum(lam, 1.0-lam)
            lam_x = lam.view(-1, 1, 1, 1)
            idx = torch.randperm(x.size(0), device=x.device)
            x2, y2 = x[idx], y[idx]
            y1_sm = smooth_one_hot(y, n_classes, smoothing)
            y2_sm = smooth_one_hot(y2, n_classes, smoothing)
            x_mix = x * lam_x + x2 * (1.0-lam_x)
            lam_y = lam.view(-1, 1)
            return x_mix, y1_sm * lam_y + y2_sm * (1.0-lam_y)

        def train_epoch(model, loader, opt, scheduler, scaler):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                if self.use_mixup:
                    x, y_soft = mixup_batch(x, y, self.mixup_alpha, self.target_num_classes, self.label_smoothing)
                else:
                    y_soft = smooth_one_hot(y, self.target_num_classes, self.label_smoothing)
                    
                opt.zero_grad(set_to_none=True)
                
                with torch.autocast("cuda", enabled=(self.device.type=="cuda"), dtype=torch.float16):
                    logits = model(x)
                    loss = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                    
                if scheduler: scheduler.step()
                
                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
            return total_loss/total, correct/total

        @torch.no_grad()
        def evaluate(model, loader):
            model.eval()
            total_loss, correct, total = 0.0, 0, 0
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast("cuda", enabled=(self.device.type=="cuda"), dtype=torch.float16):
                    logits = model(x)
                    y_sm = smooth_one_hot(y, self.target_num_classes, 0.0) # no smoothing in eval metric usually
                    loss = -(y_sm * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                total_loss += loss.item()*x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
            return total_loss/total, correct/total

        results = []
        best_cfg = None
        best_val_acc = -1.0
        
        for trial in self.sweep:
            self.update_status("training", f"Sweep: {trial['name']}")
            img_size = int(trial["img_size"])
            name = trial["name"]
            
            tl, vl, testl = build_dataloaders(img_size)
            
            model = _build_backbone(trial["backbone"], self.target_num_classes).to(self.device)
            
            opt = optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
            steps = len(tl) * self.epochs_per_trial
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
            scaler = torch.amp.GradScaler(enabled=(self.device.type=="cuda"))
            
            best_trial_val = 0.0
            
            for ep in range(self.epochs_per_trial):
                tl_loss, tl_acc = train_epoch(model, tl, opt, sched, scaler)
                va_loss, va_acc = evaluate(model, vl)
                best_trial_val = max(best_trial_val, va_acc)
            
            results.append({"name": name, "val_acc": best_trial_val, "cfg": trial})
            if best_trial_val > best_val_acc:
                best_val_acc = best_trial_val
                best_cfg = trial

        self.state.metrics["sweep_results"] = results
        self.state.metrics["best_cfg"] = best_cfg
        
        if not best_cfg: raise RuntimeError("Sweep failed")
        self.update_status("training", f"Final training: {best_cfg['name']}")
        
        self._final_train_loader, self._final_val_loader, self._final_test_loader = \
            build_dataloaders(int(best_cfg["img_size"]))
            
        self._final_model = _build_backbone(best_cfg["backbone"], self.target_num_classes).to(self.device)
        
        opt = optim.AdamW(self._final_model.parameters(), lr=float(best_cfg["lr"])*0.8) # slightly lower lr
        steps = len(self._final_train_loader) * self.final_epochs
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        scaler = torch.amp.GradScaler(enabled=(self.device.type=="cuda"))
        
        best_final_val = 0.0
        best_state = None
        
        for ep in range(self.final_epochs):
            t_loss, t_acc = train_epoch(self._final_model, self._final_train_loader, opt, sched, scaler)
            v_loss, v_acc = evaluate(self._final_model, self._final_val_loader)
            if v_acc > best_final_val:
                best_final_val = v_acc
                best_state = {k:v.cpu().clone() for k,v in self._final_model.state_dict().items()}
            
            self.state.metrics["final_epoch"] = ep+1
            self.state.metrics["final_val_acc"] = v_acc
            
        if best_state:
            self._final_model.load_state_dict(best_state)
            
        self._trained_model = self._final_model

        self._final_model.to(self.device)
        test_loss, test_acc = evaluate(self._final_model, self._final_test_loader)
        self.state.metrics["final_test_acc"] = test_acc


    # ==========================================================
    # Phase 3: Optimizing
    # ==========================================================
   
    def _phase_optimizing(self) -> None:

        import copy
        import torch
        import tempfile
        import torch.nn as nn
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

        if self._trained_model is None:
            raise RuntimeError("No trained model found. Training must run before optimization.")
        
        if self._final_test_loader is None:
             raise RuntimeError("Missing test loader. Training must run before optimization.")

        self.update_status("optimizing", "Optimizing (verified) + validating accuracy/size...")
        self.state.metrics["optimization_skipped"] = False
        self.state.metrics["optimization_skip_reason"] = None
        self.state.metrics["optimization_failed"] = False
        self.state.metrics["optimization_fail_reason"] = None
        self.state.metrics["optimization_attempted_levels"] = []
        self.state.metrics.pop("optimization_attempt_accuracies", None)
        self.state.metrics["optimization_history"] = []

        def audio_fp16(model: nn.Module) -> nn.Module:
            model = model.eval()
            return model.half()

        def audio_int8_dynamic(model):
            m = copy.deepcopy(model).cpu()
            q_model = torch.quantization.quantize_dynamic(m, {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}, dtype=torch.qint8)
            return q_model

        def audio_int8_static(model, calibration_loader, backend="fbgemm"):
            if backend == "qnnpack":
                torch.backends.quantized.engine = "qnnpack"
            else:
                torch.backends.quantized.engine = "fbgemm"
            
            m = copy.deepcopy(model).cpu()
            m.eval()
            
            num_calib_batches = 8
            def get_x(batch):
                if isinstance(batch, (list, tuple)): return batch[0]
                if isinstance(batch, dict): return next(iter(batch.values()))
                return batch

            x0 = get_x(next(iter(calibration_loader))).cpu().float()
            example_inputs = (x0,)

            qconfig_mapping = get_default_qconfig_mapping(backend)
            prepared = prepare_fx(m, qconfig_mapping, example_inputs)

            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= num_calib_batches:
                        break
                    prepared(get_x(batch).cpu().float())

            return convert_fx(prepared)

        def _get_size_kb(model: nn.Module) -> float:
            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                tmp_name = tmp.name
            try:
                torch.save(model.state_dict(), tmp_name)
                size = os.path.getsize(tmp_name)
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
            return size / 1024.0

        def _estimate_peak_memory(model: nn.Module, input_shape: tuple, device: torch.device) -> float:
            peak_mem = 0
            
            def hook(module, input, output):
                nonlocal peak_mem
                if isinstance(output, torch.Tensor):
                    out_size = output.element_size() * output.nelement()
                elif isinstance(output, (tuple, list)):
                    out_size = sum(o.element_size() * o.nelement() for o in output if isinstance(o, torch.Tensor))
                else:
                    out_size = 0
                
                peak_mem = max(peak_mem, out_size)
            
            hooks = []
            for layer in model.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                     hooks.append(layer.register_forward_hook(hook))
            
            try:
                model.eval()
                with torch.no_grad():
                     dummy = torch.zeros(input_shape).to(device)
                     try:
                         p = next(model.parameters())
                         if p.dtype == torch.float16:
                             dummy = dummy.half()
                     except:
                        pass
                     model(dummy)
            except Exception:
                pass
            finally:
                for h in hooks:
                    h.remove()
            
            return peak_mem / 1024.0

        def _get_latency_ms(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
            model.eval()
            is_half = False
            try:
                p = next(model.parameters())
                if p.dtype == torch.float16:
                    is_half = True
            except:
                pass

            t0 = time.time()
            count = 0
            total_samples = 0
            
            with torch.no_grad():
                for i, (xb, _) in enumerate(loader):
                    if i > 2: break
                    xb = xb.to(device)
                    if is_half: xb = xb.half()
                    _ = model(xb)
                    if device.type == 'cuda': torch.cuda.synchronize()

            t_start = time.time()
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(device)
                    if is_half: xb = xb.half()
                    
                    if device.type == 'cuda': torch.cuda.synchronize()
                    _ = model(xb)
                    if device.type == 'cuda': torch.cuda.synchronize()
                    
                    total_samples += xb.size(0)
                    count += 1
                    if count >= max_batches:
                        break
                        
            dt = time.time() - t_start
            return (dt / max(1, total_samples)) * 1000.0

        def _evaluate_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
            model.eval()
            is_half = False
            try:
                p = next(model.parameters())
                if p.dtype == torch.float16:
                    is_half = True
            except:
                pass

            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    if is_half: xb = xb.half()
                    
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += xb.size(0)
            return float(correct / max(1, total))

        def _meets_constraints(size_kb: float, dynamic_ram_kb: float) -> bool:
            flash_limit_mb = self.device_specs.get("flash_mb")
            if flash_limit_mb is not None:
                try:
                    f_lim_kb = float(flash_limit_mb) * 1024.0
                    if f_lim_kb > 0 and size_kb > f_lim_kb:
                        return False
                except:
                    pass
            
            ram_limit_kb = self.device_specs.get("ram_kb")
            if ram_limit_kb is not None:
                try:
                    r_lim_kb = float(ram_limit_kb)
                    if r_lim_kb > 0:
                        #  total_ram_needed = size_kb + dynamic_ram_kb                        
                         total_ram_needed = dynamic_ram_kb
                         if total_ram_needed > r_lim_kb:
                            return False
                except:
                    pass
            
            return True

        def _probe_forward(model: nn.Module, loader: DataLoader, device: torch.device) -> bool:
            try:
                model.eval()
                xb, _ = next(iter(loader))
                xb = xb.to(device)
                try:
                    p = next(model.parameters())
                    if p.dtype == torch.float16:
                        xb = xb.half()
                except:
                    pass
                with torch.no_grad():
                    _ = model(xb)
                return True
            except Exception:
                return False

        # ----------------------------------------------------
        # 1. Baseline FP32
        # ----------------------------------------------------
        base_model = self._trained_model
        
        dummy_shape = (1, 3, 224, 224) 
        try:
             xb, _ = next(iter(self._final_test_loader))
             dummy_shape = (1, xb.size(1), xb.size(2), xb.size(3))
        except:
             pass

        base_acc = _evaluate_acc(base_model, self._final_test_loader, self.device)
        base_size = _get_size_kb(base_model)
        base_ram_dyn = _estimate_peak_memory(base_model, dummy_shape, self.device)
        base_lat = _get_latency_ms(base_model, self._final_test_loader, self.device)
        fits_baseline = _meets_constraints(base_size, base_ram_dyn)
        
        baseline_candidate = {
            "name": "fp32",
            "model": base_model,
            "acc": base_acc,
            "size_kb": base_size,
            "dynamic_ram_kb": base_ram_dyn,
            "latency_ms": base_lat,
            "level": "baseline",
            "valid": fits_baseline,
            "reason": None if fits_baseline else "Exceeds Flash/RAM"
        }
        
        self.state.metrics["baseline"] = {
            "acc": base_acc, 
            "size_kb": base_size, 
            "dynamic_ram_kb": base_ram_dyn,
            "latency_ms": base_lat
        }

        self.state.metrics.update({
             "accuracyBefore": base_acc,
             "latencyMsBefore": base_lat,
             "sizeKBBefore": base_size,
        })


        # ----------------------------------------------------
        # 2. Constraints Check & Skip Logic
        # ----------------------------------------------------
        quant_type = (self.quantization or "auto").lower().strip()
        strategy = (self.optimization_strategy or "balanced").lower().strip()
        
        if fits_baseline and quant_type in ["auto", "float32", "automatic"] and strategy == "balanced":
            self.state.metrics["optimization_skipped"] = True
            self.state.metrics["optimization_skip_reason"] = "baseline_within_constraints"
            self._final_model = base_model
            
            self.state.metrics.update({
                "accuracy": base_acc,
                "accuracyBefore": base_acc,
                "accuracyAfter": base_acc,
                "latencyMsBefore": base_lat,
                "latencyMsAfter": base_lat,
                "sizeKBBefore": base_size,
                "sizeKBAfter": base_size,
                "ramKBStatic": base_size,
                "ramKBDynamic": base_ram_dyn,
                "totalRAMKB": base_size + base_ram_dyn,
                "model_name": "fp32"
            })
            self.update_status("optimizing", "Baseline model fits constraints. Optimization skipped.")
            return

        # ----------------------------------------------------
        # 3. Candidate Generation
        # ----------------------------------------------------
        candidates = []
        
        candidates.append(baseline_candidate)

        self.state.metrics["optimization_history"].append({
            "name": baseline_candidate["name"], 
            "status": "success" if baseline_candidate["valid"] else "rejected",
            "acc": baseline_candidate["acc"], 
            "size_kb": baseline_candidate["size_kb"], 
            "size_mb": baseline_candidate["size_kb"] / 1024.0,
            "dynamic_ram_kb": baseline_candidate["dynamic_ram_kb"],
            "latency_ms": baseline_candidate["latency_ms"],
            "reason": baseline_candidate["reason"]
        })
        
        wants_auto = (quant_type in ["auto", "automatic"])
        wants_fp16 = (quant_type == "float16")
        wants_dyn = ("dynamic" in quant_type and "int8" in quant_type)
        wants_stat = ("static" in quant_type and "int8" in quant_type)
        
        if quant_type == "int8": 
            wants_dyn = True
        
        calib_loader = self._final_val_loader if self._final_val_loader else self._final_test_loader

        jobs = [] 
        
        if wants_auto or wants_fp16:
            if self.device.type == "cuda":
                jobs.append(("float16", lambda m: audio_fp16(m), self.device))
            elif wants_fp16:
                jobs.append(("float16", lambda m: audio_fp16(m), self.device))

        if wants_auto or wants_dyn:
            jobs.append(("dynamic_int8", lambda m: audio_int8_dynamic(m), torch.device("cpu")))

        if wants_auto or wants_stat:
            jobs.append(("static_int8", lambda m: audio_int8_static(m, calib_loader, backend="fbgemm"), torch.device("cpu")))


        for cname, func, eval_device in jobs:
            acc = None
            sz = None
            lat = None
            dyn_ram = None
            try:
                model_idx = copy.deepcopy(base_model)
                if eval_device.type == 'cpu':
                    model_idx = model_idx.cpu()
                else:
                    model_idx = model_idx.to(eval_device)

                q_model = func(model_idx)

                if not _probe_forward(q_model, self._final_test_loader, eval_device):
                    self.state.metrics["optimization_history"].append({
                        "name": cname, "status": "failed", "reason": "probe_forward_failed",
                        "acc": None, "size_kb": None, "dynamic_ram_kb": None, "latency_ms": None
                    })
                    continue

                acc = _evaluate_acc(q_model, self._final_test_loader, eval_device)
                
                lat = _get_latency_ms(q_model, self._final_test_loader, eval_device)

                sz = _get_size_kb(q_model)
                dyn_ram = _estimate_peak_memory(q_model, dummy_shape, eval_device)
                
                is_valid = _meets_constraints(sz, dyn_ram)
                
                candidates.append({
                    "name": cname,
                    "model": q_model,
                    "acc": acc,
                    "size_kb": sz,
                    "dynamic_ram_kb": dyn_ram,
                    "latency_ms": lat,
                    "level": cname,
                    "valid": is_valid,
                    "reason": None if is_valid else "Exceeds Flash/RAM"
                })
                
                status_str = "success" if is_valid else "rejected"
                
                self.state.metrics["optimization_history"].append({
                    "name": cname, "status": status_str,
                    "acc": acc, "size_kb": sz, "size_mb": sz / 1024.0 if sz else None, "dynamic_ram_kb": dyn_ram, "latency_ms": lat, 
                    "reason": None if is_valid else "Exceeds Flash/RAM"
                })

            except Exception as e:
                self.state.metrics["optimization_history"].append({
                    "name": cname, "status": "failed", "reason": str(e),
                    "acc": acc, "size_kb": sz, "size_mb": sz / 1024.0 if sz else None, "dynamic_ram_kb": dyn_ram, "latency_ms": lat
                })

        # ----------------------------------------------------
        # 4. Strategy Selection
        # ----------------------------------------------------
        valid_cands = [c for c in candidates if c.get("valid")]
        
        if not valid_cands:
            all_cands = [c for c in candidates if c.get("model") is not None]
            if all_cands:
                all_cands.sort(key=lambda c: (c.get("size_kb", float('inf')), -c.get("acc", 0)))
                best_failed = all_cands[0]
                
                self._final_model = best_failed["model"]
                self.state.metrics["optimization_failed"] = True
                self.state.metrics["optimization_fail_reason"] = "No candidates met constraints. Using best failed."
                
                self.state.metrics["accuracy"] = best_failed["acc"]
                self.state.metrics["accuracyAfter"] = best_failed["acc"]
                self.state.metrics["sizeKBAfter"] = best_failed.get("size_kb", 0)
                self.state.metrics["latencyMsAfter"] = best_failed.get("latency_ms")
                self.state.metrics["optimization_level"] = f"{best_failed['name']} (best failed)"
                
                self.state.metrics["quantization_applied"] = best_failed["name"]
                self.state.metrics["model_name"] = f"{self.model_name}_{best_failed['name'].lower().replace(' ', '_')}" if hasattr(self, 'model_name') else best_failed['name']

                return self._final_model
            
            self.state.metrics["optimization_failed"] = True
            self.state.metrics["optimization_fail_reason"] = "No candidates available."
            self.state.metrics["accuracy"] = base_acc
            self.state.metrics["accuracyAfter"] = base_acc
            self.state.metrics["sizeKBAfter"] = base_size
            self.state.metrics["latencyMsAfter"] = base_lat
            self.state.metrics["optimization_level"] = "baseline"
            self._final_model = base_model
            return self._final_model

        def _score(c):
            acc = c.get("acc", 0.0)
            lat = c.get("latency_ms") or 1e-9
            sz_mb = (c.get("size_kb") or 0.0) / 1024.0
            sz_kb = c.get("size_kb") or 0.0
            
            if "balanced" in strategy:
                return (acc * 1000.0) - (lat * 0.01) - (sz_mb * 0.01)
            if "accuracy" in strategy and "latency" in strategy:
                return (acc * 1000.0) - (lat * 0.1)
            if "ram" in strategy and "size" in strategy:
                return -(sz_kb + sz_kb)
            if "accuracy" in strategy:
                return acc
            if "latency" in strategy:
                return -lat
            if "size" in strategy:
                return -sz_kb
            if "ram" in strategy:
                return -sz_kb
            return acc  

        valid_cands.sort(key=_score, reverse=True)
        best = valid_cands[0]

        self._final_model = best["model"]
        
        self.state.metrics.update({
            "accuracy": best["acc"],
            "accuracyBefore": base_acc,
            "accuracyAfter": best["acc"],
            "latencyMsBefore": base_lat,
            "latencyMsAfter": best["latency_ms"],
            "sizeKBBefore": base_size,
            "sizeKBAfter": best["size_kb"],
            "ramKBStatic": best["size_kb"],
            "ramKBDynamic": best["dynamic_ram_kb"],
            "totalRAMKB": best["size_kb"] + best["dynamic_ram_kb"],
            "model_name": best["name"], 
            "quantization_applied": best["name"],
            "optimization_search_candidates": [c["name"] for c in candidates],
            "optimization_valid_candidates": [c["name"] for c in valid_cands]
        })

        self.update_status(
            "optimizing",
            f"Optimization Selected: {best['name']} (Acc: {best['acc']:.4f}, Size: {best['size_kb']:.2f}KB)"
        )

        return self._final_model
   
    # ==========================================================
    # Phase 4: Packaging
    # ==========================================================

    def _phase_packaging(self) -> None:
        if self._final_model is None:
            raise RuntimeError("No final model available for packaging/export.")

        import os
        import sys
        import io
        import types
        import shutil
        import tempfile
        import warnings
        from pathlib import Path
        from contextlib import contextmanager

        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
        os.environ.setdefault("TF_MLIR_ENABLE_DEBUG_INFO", "0")
        os.environ.setdefault("TF_ENABLE_MLIR", "0")

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r".*deprecated.*", module=r"tensorflow(\..*)?$")
        warnings.filterwarnings("ignore", message=r".*tf\.losses\..*deprecated.*")
        warnings.filterwarnings("ignore", message=r".*tf\.executing_eagerly_outside_functions.*deprecated.*")
        warnings.filterwarnings("ignore", message=r".*tf\.logging\..*deprecated.*")
        warnings.filterwarnings("ignore", message=r".*tf\.control_flow_v2_enabled.*deprecated.*")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"tensorflow_probability(\..*)?$")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"keras(\..*)?$")
        warnings.filterwarnings("ignore",message=r".*Exporting a model to ONNX.*GRU.*batch_size.*",category=UserWarning)
        warnings.filterwarnings("ignore",message=r".*tf\.logging\.TaskLevelStatusMessage.*deprecated.*",category=UserWarning)
        warnings.filterwarnings("ignore",message=r".*tf\.control_flow_v2_enabled.*deprecated.*",category=UserWarning)
        warnings.filterwarnings("ignore",message=r".*TensorFlow Addons \(TFA\) has ended development.*",category=UserWarning)
        warnings.filterwarnings("ignore",message=r".*Tensorflow Addons supports.*strictly below 2\.15\.0.*",category=UserWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module=r"tensorflow_addons(\..*)?$")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"tensorflow_addons\.utils(\..*)?$")

        @contextmanager
        def _suppress_output():
            devnull = open(os.devnull, "w")
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                devnull.close()

        def _infer_model_name() -> str:
            for attr in ("model_name", "model_id", "final_model_name", "selected_model_name"):
                try:
                    v = getattr(self, attr)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                except Exception:
                    pass
            try:
                st = getattr(self, "state", None)
                if st is not None and hasattr(st, "metrics") and isinstance(st.metrics, dict):
                    for k in ("model_name", "selected_model", "final_model_name", "best_model_name"):
                        v = st.metrics.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
            except Exception:
                pass
            return "unknown_model"

        MODEL_NAME = _infer_model_name()

        def _is_torch_model(model) -> bool:
            try:
                import torch
                return isinstance(model, torch.nn.Module)
            except Exception:
                return False

        def _is_tf_model(model) -> bool:
            try:
                import tensorflow as tf
                return isinstance(model, tf.keras.Model)
            except Exception:
                return False

        def _write_bytes(out_base: str | Path, data: bytes, suffix: str) -> Path:
            p = Path(out_base).with_suffix(suffix)
            p.write_bytes(data)
            return p

        def _bytes_to_c_header(data: bytes, out_base: str | Path, var_name="model_data") -> Path:
            p = Path(out_base).with_suffix(".h")
            hex_array = ", ".join(f"0x{b:02x}" for b in data)
            header = (
                "#ifndef MODEL_DATA_H\n"
                "#define MODEL_DATA_H\n\n"
                f"const unsigned char {var_name}[] = {{\n    {hex_array}\n}};\n"
                f"const unsigned int {var_name}_len = {len(data)};\n\n"
                "#endif\n"
            )
            p.write_text(header)
            return p

        def _get_sample_batch_for_export():
            import torch

            loader = getattr(self, "_final_train_loader", None)
            if loader is None:
                raise RuntimeError("train_loader missing; cannot create export input.")

            try:
                xb, _ = next(iter(loader))
            except Exception as e:
                raise RuntimeError(f"Could not fetch batch for export: {e}")

            if not isinstance(xb, torch.Tensor):
                raise RuntimeError("Export sample is not a torch.Tensor")

            x = xb[:1]

            try:
                model = self._final_model
                dev = next(model.parameters()).device
                dt = next(model.parameters()).dtype
            except Exception:
                dev = getattr(self, "device", torch.device("cpu"))
                dt = None

            x = x.to(dev)
            if dt is not None and dt == torch.float16:
                x = x.half()

            return x

        def export_onnx_torch(model, sample_input, out_base, opset: int = 13) -> Path:
            import torch
            out_path = Path(out_base).with_suffix(".onnx")
            
            model = model.cpu()
            sample_input = sample_input.cpu()
            model.eval()
            
            torch.onnx.export(
                model,
                sample_input,
                str(out_path),
                opset_version=int(opset),
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
            return out_path

        def _patch_tf_compat_modules() -> None:
            try:
                import tensorflow as tf
            except Exception:
                return

            if "tensorflow.compat" not in sys.modules:
                sys.modules["tensorflow.compat"] = types.ModuleType("tensorflow.compat")
            if "tensorflow.compat.v2" not in sys.modules:
                sys.modules["tensorflow.compat.v2"] = tf

            try:
                sys.modules["tensorflow.compat"].v2 = sys.modules["tensorflow.compat.v2"]
            except Exception:
                pass
            try:
                keras_obj = tf.keras
            except Exception:
                return

            keras_mod = sys.modules.get("tensorflow.compat.v2.keras")
            if keras_mod is None or not isinstance(keras_mod, types.ModuleType):
                keras_mod = types.ModuleType("tensorflow.compat.v2.keras")
                sys.modules["tensorflow.compat.v2.keras"] = keras_mod

            try:
                for k in dir(keras_obj):
                    if not k.startswith("_"):
                        setattr(keras_mod, k, getattr(keras_obj, k))
            except Exception:
                pass

            try:
                setattr(sys.modules["tensorflow.compat.v2"], "keras", keras_obj)
            except Exception:
                pass

        def _ensure_onnx_mapping_shim(onnx_mod):
            try:
                import onnx.helper
            except ImportError:
                pass

            import numpy as np
            from onnx import TensorProto

            class MappingShim:
                TENSOR_TYPE_TO_NP_TYPE = {
                    TensorProto.FLOAT: np.dtype("float32"),
                    TensorProto.BOOL: np.dtype("bool"),
                    TensorProto.INT32: np.dtype("int32"),
                    TensorProto.INT64: np.dtype("int64"),
                    TensorProto.STRING: np.dtype("object"),
                    TensorProto.UINT8: np.dtype("uint8"),
                    TensorProto.UINT64: np.dtype("uint64"),
                    TensorProto.INT8: np.dtype("int8"),
                    TensorProto.INT16: np.dtype("int16"),
                    TensorProto.UINT16: np.dtype("uint16"),
                    TensorProto.FLOAT16: np.dtype("float16"),
                    TensorProto.DOUBLE: np.dtype("float64"),
                }
                NP_TYPE_TO_TENSOR_TYPE = {
                    "float32": TensorProto.FLOAT,
                    "bool": TensorProto.BOOL,
                    "int32": TensorProto.INT32,
                    "int64": TensorProto.INT64,
                    "uint8": TensorProto.UINT8,
                    "uint64": TensorProto.UINT64,
                    "int8": TensorProto.INT8,
                    "int16": TensorProto.INT16,
                    "uint16": TensorProto.UINT16,
                    "float16": TensorProto.FLOAT16,
                    "float64": TensorProto.DOUBLE,
                }

            for k, v in list(MappingShim.NP_TYPE_TO_TENSOR_TYPE.items()):
                try:
                    MappingShim.NP_TYPE_TO_TENSOR_TYPE[np.dtype(k)] = v
                except Exception:
                    pass

            if not hasattr(onnx_mod, "mapping"):
                onnx_mod.mapping = MappingShim
            
            if hasattr(onnx_mod, "helper") and not hasattr(onnx_mod.helper, "mapping"):
                onnx_mod.helper.mapping = MappingShim

        def _torch_to_tflite_via_onnx(model, out_base: str, quantize: bool) -> Path:
            import copy
            model = copy.deepcopy(model).cpu().float()
            
            sample_input = _get_sample_batch_for_export()
            if hasattr(sample_input, "float"): 
                 sample_input = sample_input.cpu().float()
            else:
                 sample_input = sample_input.cpu()
                 
            onnx_path = export_onnx_torch(model, sample_input, out_base, opset=13)

            try:
                import onnx
                _ensure_onnx_mapping_shim(onnx)
                from onnx_tf.backend import prepare
            except Exception as e:
                raise RuntimeError(f"onnx / onnx-tf not available for torch->tflite conversion: {e}")

            _patch_tf_compat_modules()

            tmp_dir = tempfile.mkdtemp(prefix="onnx_tf_savedmodel_")
            try:
                with _suppress_output():
                    onnx_model = onnx.load(str(onnx_path))
                    tf_rep = prepare(onnx_model)
                    tf_rep.export_graph(tmp_dir)

                import tensorflow as tf
                try:
                    tf.get_logger().setLevel("ERROR")
                except Exception:
                    pass
                try:
                    tf.autograph.set_verbosity(0)
                except Exception:
                    pass

                converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
                if quantize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

                with _suppress_output():
                    tflite_model = converter.convert()

                tflite_path = Path(out_base).with_suffix(".tflite")
                tflite_path.write_bytes(tflite_model)
                return tflite_path

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        def export_tflite_from_keras(model, out_base, quantize: bool = True) -> Path:
            import tensorflow as tf
            if not isinstance(model, tf.keras.Model):
                raise TypeError("export_tflite_from_keras requires a tf.keras.Model")

            try:
                tf.get_logger().setLevel("ERROR")
            except Exception:
                pass
            try:
                tf.autograph.set_verbosity(0)
            except Exception:
                pass

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

            with _suppress_output():
                tflite_model = converter.convert()

            out_path = Path(out_base).with_suffix(".tflite")
            out_path.write_bytes(tflite_model)
            return out_path

        def export_bin_from_tflite(tflite_path, out_base) -> Path:
            data = Path(tflite_path).read_bytes()
            return _write_bytes(out_base, data, ".bin")

        def export_h_from_bin(bin_path, out_base) -> Path:
            data = Path(bin_path).read_bytes()
            return _bytes_to_c_header(data, out_base, var_name="model_data")

        def _should_quantize() -> bool:
            st = getattr(self, "state", None)
            metrics = getattr(st, "metrics", {}) if st is not None else {}
            q = metrics.get("quantization_applied")
            if q is None and hasattr(self, "quantization"):
                q = getattr(self, "quantization")
            if q is None:
                return True
            qs = str(q).strip().lower()
            return qs not in ("none", "false", "0", "fp32", "float32", "no")

        def _normalize_ext(ext: str) -> str:
            ext = str(ext).strip().lower()
            if ext == "all":
                return "all"
            if not ext.startswith("."):
                ext = "." + ext
            return ext

        def _family_allowed_exts() -> list[str]:
            exts = family_allowed_formats(self.device_family_id)
            if not isinstance(exts, (list, tuple)):
                return []
            return [str(e).strip().lower() for e in exts]

        def _export_ext(model, out_base: str, ext: str) -> tuple[bool, str, str]:
            ext = _normalize_ext(ext)

            def _reason(msg: str) -> tuple[bool, str, str]:
                return False, msg, ""

            try:
                if ext == ".onnx":
                    if not _is_torch_model(model):
                        return _reason("ONNX export requires a torch.nn.Module (image pipeline final model is expected to be torch).")
                    sample_input = _get_sample_batch_for_export()
                    p = export_onnx_torch(model, sample_input, out_base, opset=13)
                    return True, "", str(p)

                if ext == ".tflite":
                    q = _should_quantize()

                    if _is_tf_model(model):
                        p = export_tflite_from_keras(model, out_base, quantize=q)
                        return True, "", str(p)

                    if _is_torch_model(model):
                        try:
                            p = _torch_to_tflite_via_onnx(model, out_base, quantize=q)
                            return True, "", str(p)
                        except Exception as e:
                            reason = f"torch->tflite conversion failed: {type(e).__name__}: {e}"
                            try:
                                self.state.metrics["tflite_export_failed"] = True
                                self.state.metrics["tflite_export_failure_reason"] = reason
                            except Exception:
                                pass
                            return _reason(reason)

                    return _reason(f".tflite export not supported for model type: {type(model)}")

                if ext == ".bin":
                    ok, reason, tflite_path = _export_ext(model, out_base, ".tflite")
                    if not ok:
                        return _reason(f".bin requires .tflite first; reason: {reason}")
                    p = export_bin_from_tflite(Path(tflite_path), out_base)
                    return True, "", str(p)

                if ext == ".h":
                    ok, reason, bin_path = _export_ext(model, out_base, ".bin")
                    if not ok:
                        return _reason(f".h requires .bin first; reason: {reason}")
                    p = export_h_from_bin(Path(bin_path), out_base)
                    return True, "", str(p)

                if ext == ".kmodel":
                    return _reason("not supported right now (disabled by request)")
                if ext == ".engine":
                    return _reason("not supported right now (disabled by request)")

                return _reason("extension not supported")

            except Exception as e:
                return False, f"{type(e).__name__}: {e}", ""

        requested = _normalize_ext(getattr(self, "export_ext", ".h"))
        allowed = _family_allowed_exts()

        if not allowed:
            raise RuntimeError(f"DEVICE_FAMILIES has no model_exts for family_id={self.device_family_id}")

        if requested != "all" and requested not in allowed:
            raise RuntimeError(
                f"Requested export_ext={requested} not allowed for family={self.device_family_id}. "
                f"Allowed={allowed}"
            )

        os.makedirs(self.output_dir, exist_ok=True)
        out_base = os.path.join(self.output_dir, "final_model")

        exported_paths: list[str] = []
        export_errors: dict[str, str] = {}

        exts_to_try = allowed if requested == "all" else [requested]
        try:
            self.state.metrics["export_ext_attempted"] = list(exts_to_try)
            self.state.metrics["packaging_model_name"] = MODEL_NAME
        except Exception:
            pass

        for ext in exts_to_try:
            ok, reason, produced = _export_ext(self._final_model, out_base, ext)

            if ok and produced and os.path.isfile(produced):
                exported_paths.append(produced)
            elif ok and produced and not os.path.isfile(produced):
                export_errors[ext] = f"export reported success but file missing: {produced}"
            elif ok and not produced:
                export_errors[ext] = "export reported success but produced path unknown"
            else:
                export_errors[ext] = reason or "export failed or not supported"

        if not exported_paths:
            raise RuntimeError(
                f"No exports succeeded. "
                f"family={self.device_family_id}, "
                f"requested={requested}, "
                f"errors={export_errors}"
            )

        self._final_model_path = exported_paths[0]

        try:
            self.state.metrics["exported_exts"] = sorted({Path(p).suffix.lower() for p in exported_paths})
        except Exception:
            pass

        sizes_mb: dict[str, float] = {}
        for p in exported_paths:
            try:
                sizes_mb[os.path.basename(p)] = float(os.path.getsize(p) / (1024 * 1024))
            except Exception:
                pass

        if requested == "all":
            try:
                import zipfile
                zip_path = os.path.join(self.output_dir, "export_all.zip")
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in exported_paths:
                        if os.path.isfile(p):
                            zf.write(p, arcname=os.path.basename(p))
                try:
                    self.state.metrics["export_zip_path"] = zip_path
                except Exception:
                    pass
                self._final_model_path = zip_path

                try:
                    zsize = float(os.path.getsize(self._final_model_path) / (1024 * 1024))
                    sizes_mb[os.path.basename(self._final_model_path)] = zsize
                except Exception:
                    pass

            except Exception as e:
                try:
                    self.state.metrics["export_zip_error"] = str(e)
                except Exception:
                    pass

        final_name = os.path.basename(self._final_model_path)
        final_size = sizes_mb.get(final_name)

        try:
            self.state.metrics.update(
                {
                    "exported_model_paths": exported_paths,
                    "export_errors": export_errors,
                    "export_ext_requested": requested,
                    "export_ext_allowed": allowed,
                    "exported_sizes_mb": sizes_mb,
                    "final_model_size_mb": final_size,
                    # "sizeKBAfter": float(final_size or 0.0) * 1024.0,
                    "packaging": {
                        "final_model_path": self._final_model_path,
                        "exported_paths": exported_paths,
                        "sizes_mb": sizes_mb,
                    },
                    "final_model_path": self._final_model_path,
                }
            )
        except Exception:
            pass

    # ==========================================================
    # Phase 5: Report
    # ==========================================================
    def _phase_report(self) -> None:

        prep = self._data_meta.get("preprocessing_report", {})
        if not prep: 
            return

        ds = prep.get("dataset", {})
        audio_cfg = prep.get("audio", {})
        stats = prep.get("stats", {})

        flat_report = {
            "task_id": self.task_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "dataset_mode": self.state.metrics.get("dataset_mode", "audio_folder"),
            "data_root": os.path.basename(str(ds.get("root", self.dataset_path))) if ds.get("root", self.dataset_path) else "",
            "num_files": int(ds.get("num_files", 0)),
            "num_classes": self.state.metrics.get("num_classes", 0),
            "class_counts": ds.get("class_counts", {}), # dict

            "audio": {
                "sample_rate": int(audio_cfg.get("target_sr", 16000)),
                "clip_seconds": float(audio_cfg.get("clip_seconds", 1.0)),
                "n_mels": int(audio_cfg.get("n_mels", 64)),
                "global_norm": bool(audio_cfg.get("use_global_norm", False)),
                "specaug": bool(audio_cfg.get("use_spec_augment", False)),
            },
            
            "stats": stats,
            
            "loader": {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
            },
        }

        flat_report["training"] = {
            "sweep_results": self.state.metrics.get("sweep_results"),
            "best_cfg": self.state.metrics.get("best_cfg"),
            "final_best_val_acc": self.state.metrics.get("final_val_acc"),
            "final_test_acc": self.state.metrics.get("final_test_acc"),
        }
        flat_report["optimization"] = self.state.metrics.get("optimization", {})
        flat_report["packaging"] = {
            "final_model_path": self._final_model_path,
            "exported_model_paths": self.state.metrics.get("exported_model_paths"),
        }
        flat_report["device"] = {
             "family_id": self.device_family_id,
             "family_name": get_family_name(self.device_family_id),
             "device_specs": self.device_specs,
        }

        report_path = os.path.join(self.output_dir, "report.pdf")
        def _clean_dataset_name(name: str) -> str:
            s = str(name or "")
            if not s:
                return s
            m = re.match(r"^[0-9a-f]{16,}_(.+)$", s)
            return m.group(1) if m else s
        opt_block = self.state.metrics.get("optimization", {}) or {}
        report_data = {
            "model_name": self.state.metrics.get("model_name", ""),
            "accuracy": self.state.metrics.get("accuracy"),
            "accuracyBefore": self.state.metrics.get("accuracyBefore"),
            "accuracyAfter": self.state.metrics.get("accuracyAfter"),
            "latencyMsBefore": self.state.metrics.get("latencyMsBefore"),
            "latencyMsAfter": self.state.metrics.get("latencyMsAfter"),
            "sizeKBBefore": self.state.metrics.get("sizeKBBefore"),
            "sizeKBAfter": self.state.metrics.get("sizeKBAfter"),
            "dataset_name": _clean_dataset_name(os.path.basename(str(self.dataset_path))) if self.dataset_path else "",
            "device_family_name": get_family_name(self.device_family_id),
            "device_family_id": self.device_family_id,
            "export_ext_requested": self.state.metrics.get("export_ext_requested", self.export_ext),
            "export_ext_allowed": self.state.metrics.get("export_ext_allowed", family_allowed_formats(self.device_family_id)),
            "saved_as": (os.path.splitext(self._final_model_path)[1].lower() if self._final_model_path else ""),
            "attempted_exts": self.state.metrics.get("export_ext_attempted", self.state.metrics.get("export_ext_allowed", family_allowed_formats(self.device_family_id))),
            "min_accuracy": self.min_accuracy,
            "exported_paths": self.state.metrics.get("exported_model_paths", []),
            "quantization_applied": self.state.metrics.get("quantization_applied", ""),
            "class_labels": list((flat_report.get("class_counts") or {}).keys()),
            "target_name": None,
            "target_num_classes": self.target_num_classes,
            "optimization_strategy": self.optimization_strategy,
            "optimization_level": self.state.metrics.get("optimization_level"),
            "accuracy_tolerance": self.accuracy_tolerance,
            "accuracy_drop_cap": opt_block.get("max_abs_drop"),
            "accuracy_drop_allowed": opt_block.get("max_abs_drop"),
            "optimization_trigger_ratio": self.optimization_trigger_ratio,
            "quantization_requested": self.quantization,
            "optimization_skipped": self.state.metrics.get("optimization_skipped", False),
            "optimization_skip_reason": self.state.metrics.get("optimization_skip_reason"),
            "optimization_failed": self.state.metrics.get("optimization_failed", False),
            "optimization_fail_reason": self.state.metrics.get("optimization_fail_reason"),
        }
        try:
            report_data["accuracy_drop_cap_pct"] = f"{float(report_data.get('accuracy_drop_cap', 0.0)) * 100.0:.1f}%"
        except Exception:
            report_data["accuracy_drop_cap_pct"] = ""
        try:
            report_data["accuracy_drop_allowed_pct"] = f"{float(report_data.get('accuracy_drop_allowed', 0.0)) * 100.0:.1f}%"
        except Exception:
            report_data["accuracy_drop_allowed_pct"] = ""
        try:
            report_data["optimization_trigger_ratio_pct"] = f"{float(self.optimization_trigger_ratio) * 100.0:.0f}%"
        except Exception:
            report_data["optimization_trigger_ratio_pct"] = ""
        report_data["optimization_skipped_display"] = "Yes" if report_data.get("optimization_skipped") else "No"
        report_data["optimization_failed_display"] = "Yes" if report_data.get("optimization_failed") else "No"
        report_data["quantization_applied_display"] = report_data.get("quantization_applied") or "None"
        
        attempt_rows = self.state.metrics.get("optimization_attempt_accuracies")
        attempt_levels = self.state.metrics.get("optimization_attempted_levels")
        if not attempt_rows:
            attempt_rows = []
            for key in ("fp32", "fp16", "int8_dynamic"):
                blk = opt_block.get(key)
                if not isinstance(blk, dict):
                    continue
                acc = blk.get("test_acc")
                size_mb = blk.get("state_dict_mb")
                size_kb = None
                try:
                    if size_mb is not None:
                        size_kb = float(size_mb) * 1024.0
                except Exception:
                    size_kb = None
                attempt_rows.append((key, acc, size_kb))
        report_data["optimization_attempt_accuracies"] = attempt_rows
        report_data["optimization_history"] = self.state.metrics.get("optimization_history", [])
        if attempt_levels:
            report_data["optimization_attempted_levels"] = list(attempt_levels)
        else:
            report_data["optimization_attempted_levels"] = [r[0] for r in attempt_rows]
        report_data["optimization_attempted_levels_display"] = " -> ".join(report_data["optimization_attempted_levels"]) if report_data["optimization_attempted_levels"] else ""
        attempt_parts = []
        for lvl, accv, sizev in attempt_rows:
            acc_txt = "n/a"
            size_txt = "n/a"
            if accv is not None:
                try:
                    acc_txt = f"{float(accv) * 100.0:.1f}%"
                except Exception:
                    acc_txt = str(accv)
            if sizev is not None:
                try:
                    size_txt = f"{float(sizev):.2f} KB"
                except Exception:
                    size_txt = str(sizev)
            attempt_parts.append(f"{lvl}: {acc_txt}, {size_txt}")
        report_data["optimization_attempt_accuracies_display"] = "; ".join(attempt_parts)
        reason_map = {
            "within_specs_balanced": "Within specs",
            "strategy_accuracy": "Accuracy prioritized",
            "exceeds_flash_limit_before_optimization": "Over flash limit before optimization",
            "optimization_no_viable_candidate": "No viable optimization candidate",
            "no_candidate_within_accuracy_tolerance": "No candidate within accuracy tolerance",
            "no_candidate_meets_specs": "No candidate met device specs",
            "int8_not_supported_for_model": "Int8 not supported for this model",
            "float16_not_supported": "Float16 not supported for this device/model",
            "accuracy_drop_exceeds_tolerance": "Accuracy drop exceeds tolerance",
        }
        raw_reason = report_data.get("optimization_skip_reason")
        report_data["optimization_skip_reason_display"] = reason_map.get(raw_reason, raw_reason) or "None"
        fail_reason = report_data.get("optimization_fail_reason")
        report_data["optimization_fail_reason_display"] = reason_map.get(fail_reason, fail_reason) or "None"
        deployment_data = {
            "family_id": self.device_family_id,
            "family_name": get_family_name(self.device_family_id),
            "frameworks": family_frameworks(self.device_family_id),
            "allowed_exts": family_allowed_formats(self.device_family_id),
            "exported_paths": self.state.metrics.get("exported_model_paths", []),
        }
        self._generate_audio_report(
            report=flat_report,
            train_loader=self._prep_train_loader,
            path=report_path,
            project_name="Automata-AI Audio Report",
            logo_path=LOGO_PATH,
            report_data=report_data,
            deployment_data=deployment_data,
        )
        self._report_path = report_path
        self.state.metrics["report_path"] = report_path

    def _generate_audio_report(
        self,
        report: dict,
        train_loader=None,
        path: str = "audio_prep_report.pdf",
        project_name: str = "Automata AI - Audio Preprocessing Report",
        logo_path: Optional[str] = None,
        report_data: Optional[Dict[str, Any]] = None,
        deployment_data: Optional[Dict[str, Any]] = None,
    ):

        import os, tempfile, datetime
        import numpy as np
        import torch

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
            Image as RLImage, Preformatted
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm

        report_data = report_data or {}
        deployment_data = deployment_data or {}
        styles = getSampleStyleSheet()
        H1 = ParagraphStyle("H1", parent=styles["Heading1"], alignment=1, spaceAfter=10)
        H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=6, spaceAfter=8)
        body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=6, leading=13)
        small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=6)
        code_style = ParagraphStyle("code", parent=styles["Code"], fontSize=8, leading=10, spaceAfter=6, leftIndent=10)
        caption = ParagraphStyle("cap", parent=styles["BodyText"], fontSize=9, leading=11, alignment=1, spaceAfter=10)

        cover_title = ParagraphStyle("cover_title", parent=styles["Title"], alignment=1, fontSize=24, spaceAfter=18, leading=30)
        cover_subtitle = ParagraphStyle("cover_subtitle", parent=styles["Heading2"], alignment=1, fontSize=16, spaceAfter=12, leading=20)
        cover_meta = ParagraphStyle("cover_meta", parent=styles["Normal"], alignment=1, fontSize=10, textColor=colors.gray, spaceAfter=24)
        cover_desc = ParagraphStyle("cover_desc", parent=styles["Normal"], alignment=1, fontSize=12, leading=16, spaceAfter=0)

        tmp_files = []

        def _convert_logo_to_png(path: str) -> Optional[str]:
            from PIL import Image as PILImage
            try:
                img = PILImage.open(path).convert("RGBA")
                fd, tmp = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                img.save(tmp, format="PNG")
                return tmp
            except Exception:
                return None

        logo_png = None
        if logo_path and os.path.exists(logo_path):
            logo_png = _convert_logo_to_png(logo_path)
            if logo_png: tmp_files.append(logo_png)

        def _set_pub_rcparams():
            plt.rcParams.update({
                "font.size": 9,
                "axes.titlesize": 11,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
            })

        def _save_fig_tmp(fig) -> str:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            fig.savefig(tmp_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            tmp_files.append(tmp_path)
            return tmp_path

        def _fit_rl_image(img_path: str, max_w: float, max_h: float):
            from PIL import Image as PILImage
            try:
                im = PILImage.open(img_path)
                w, h = im.size
                scale = min(max_w / float(w), max_h / float(h))
                return RLImage(img_path, width=w * scale, height=h * scale)
            except Exception:
                return Paragraph("[Image Error]", small)

        def _fmt_num(v, places=3):
            try:
                return f"{float(v):.{places}f}"
            except Exception:
                return str(v) if v not in (None, "") else ""

        def _fmt_acc(v):
            try:
                f = float(v)
            except Exception:
                return str(v) if v not in (None, "") else ""
            pct = f * 100.0 if 0 <= f <= 1 else f
            return f"{pct:.1f}%"

        def _delta(before, after):
            try:
                b = float(before)
                a = float(after)
            except Exception:
                return "", colors.grey
            if b == 0:
                return "", colors.grey
            change = (a - b) / b * 100.0
            if abs(change) < 1e-9:
                return "0.0%", colors.grey
            return (f"{change:+.1f}%", colors.red if change > 0 else colors.green)

        def _exported_exts_from_paths(paths: list) -> list:
            exts = []
            for p in paths or []:
                ext = os.path.splitext(str(p))[1].lower()
                if ext:
                    exts.append(ext)
            return sorted(set(exts))

        def _kv_table(rows, col_widths):
            t = Table(rows, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            return t

        def _chart_class_distribution(class_counts: dict):
            _set_pub_rcparams()
            names = list(class_counts.keys())
            vals = [class_counts[k] for k in names]
            
            xy = sorted(zip(names, vals), key=lambda x: x[1], reverse=True)
            names = [x[0] for x in xy]
            vals = [x[1] for x in xy]
            
            if len(names) > 25:
                names = names[:25]
                vals = vals[:25]

            fig = plt.figure(figsize=(6.2, 3.4))
            ax = fig.add_subplot(111)
            ax.bar(range(len(names)), vals)
            ax.set_title("Class Distribution (top 25)")
            ax.set_ylabel("Files")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=60, ha="right")
            return _save_fig_tmp(fig)

        def _chart_durations(stats: dict):
            _set_pub_rcparams()
            durs = stats.get("durations", None)
            if not isinstance(durs, list) or len(durs) == 0:
                return None
            fig = plt.figure(figsize=(6.2, 3.0))
            ax = fig.add_subplot(111)
            d = np.array(durs)
            p99 = np.percentile(d, 99)
            d = d[d <= p99]
            ax.hist(d, bins=30)
            ax.set_title(f"Clip Duration Distribution (<= {p99:.2f}s)")
            ax.set_xlabel("seconds")
            ax.set_ylabel("count")
            return _save_fig_tmp(fig)

        def _plot_sample_mel(train_loader):
            if train_loader is None: return None
            try:
                xb, yb = next(iter(train_loader))
            except Exception: return None
            if not isinstance(xb, torch.Tensor) or xb.ndim != 4: return None
            mel = xb[0, 0].detach().cpu().float().numpy()
            
            _set_pub_rcparams()
            fig = plt.figure(figsize=(6.2, 3.0))
            ax = fig.add_subplot(111)
            ax.imshow(mel, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title("Sample Mel-Spectrogram (Post-Process)")
            ax.set_xlabel("time frames")
            ax.set_ylabel("mel bins")
            return _save_fig_tmp(fig)

        # ----------------------------
        # Document
        # ----------------------------
        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=3.1 * cm,
            bottomMargin=2.0 * cm,
        )

        story = []

        def draw_header_footer(canvas, doc_):
            canvas.saveState()
            header_bottom = doc_.pagesize[1] - doc_.topMargin + 0.25 * cm
            
            if logo_png:
                try:
                    iw, ih = 0, 0
                    from PIL import Image as PILImage
                    with PILImage.open(logo_png) as im:
                        iw, ih = im.size
                    aspect = iw / float(ih)
                    h_draw = 1.2 * cm
                    w_draw = h_draw * aspect
                    canvas.drawImage(
                        logo_png,
                        doc_.leftMargin,
                        doc_.pagesize[1] - 2.0 * cm,
                        width=w_draw,
                        height=h_draw,
                        mask="auto"
                    )
                except: pass

            title_y = doc_.pagesize[1] - 1.35 * cm
            if logo_png:
                title_y = doc_.pagesize[1] - 2.0 * cm + (1.2 * cm / 2.0) - (0.25 * cm)
            canvas.setFont("Helvetica-Bold", 12)
            canvas.drawCentredString(
                doc_.pagesize[0] / 2.0,
                title_y,
                project_name,
            )

            canvas.setLineWidth(0.4)
            canvas.setStrokeColor(colors.grey)
            canvas.line(doc_.leftMargin, header_bottom, doc_.pagesize[0] - doc_.rightMargin, header_bottom)

            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.black)
            canvas.drawString(doc_.leftMargin, 1.15 * cm, f"Page {doc_.page}")
            canvas.drawRightString(
                doc_.pagesize[0] - doc_.rightMargin,
                1.15 * cm,
                f"© {datetime.datetime.now().year} Automata AI — All rights reserved",
            )
            canvas.restoreState()

        story.append(Spacer(1, 2.6 * cm))
        if logo_png:
            cover_logo = _fit_rl_image(logo_png, max_w=6.5 * cm, max_h=6.5 * cm)
            cover_logo.hAlign = "CENTER"
            story.append(cover_logo)
            story.append(Spacer(1, 1.2 * cm))
        story.append(Paragraph(project_name, cover_title))
        story.append(Paragraph("Automated Pipeline Report", cover_subtitle))
        story.append(Paragraph(f"Generated on {report.get('timestamp','')}", cover_meta))
        story.append(Spacer(1, 1.2 * cm))
        story.append(Paragraph(
            "This report summarizes dataset characteristics, preprocessing steps, "
            "training results, optimization decisions, and export artifacts for audio models.",
            cover_desc
        ))
        story.append(PageBreak())

        story.append(Paragraph("1. Dataset Overview", H2))
        
        stats = report.get("stats", {})
        ds_info = report.get("dataset", {})
        audio = report.get("audio", {})
        class_counts = report.get("class_counts", {})
        
        durs = np.array(stats.get("durations", []))
        rms = np.array(stats.get("rms_values", []))
        sr_counts = stats.get("sr_counts", {})
        
        files_per_class = class_counts
        
        dur_mean, dur_p5, dur_p50, dur_p95 = 0,0,0,0
        if len(durs) > 0:
            dur_mean = float(np.mean(durs))
            dur_p5 = float(np.percentile(durs, 5))
            dur_p50 = float(np.percentile(durs, 50))
            dur_p95 = float(np.percentile(durs, 95))
            
        rms_mean, rms_std = 0,0
        if len(rms) > 0:
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
        sr_str = "N/A"
        if sr_counts:
            sr_items = sorted(sr_counts.items(), key=lambda kv: -kv[1])
            sr_str = ", ".join(f"{sr}Hz({cnt})" for sr, cnt in sr_items)
            
        num_files = report.get("num_files", 0)
        num_classes = report.get("num_classes", 0)
        min_per_class = min(files_per_class.values()) if files_per_class else 0
        median_per_class = int(np.median(list(files_per_class.values()))) if files_per_class else 0
        
        target_sr = int(audio.get("target_sr", 16000))
        clip_s = float(audio.get("clip_seconds", 1.0))
        n_mels = int(audio.get("n_mels", 64))
        win_len = int(audio.get("win_length", 400))
        hop_len = int(audio.get("hop_length", 160))
        img_size = int(report.get("img_size", 64))
        
        target_samples = int(clip_s * target_sr)
        est_frames = max(1, 1 + (target_samples - win_len) // hop_len)
        spec_before = f"(1, {n_mels}, {est_frames})"
        spec_after = f"(1, {img_size}, {img_size})"

        gn = "ON" if audio.get("global_norm") else "OFF"
        sa = "ON" if audio.get("specaug") else "OFF"

        summ_lines = []
        summ_lines.append(f"[Audio Preproc] SR={audio.get('target_sr')} | Clip={audio.get('clip_seconds',0):.2f}s | Mels={audio.get('n_mels')} | GlobalNorm={gn} | SpecAug={sa}")
        summ_lines.append(f"               (Tmask={audio.get('time_mask_param')}x{audio.get('num_time_masks')}, Fmask={audio.get('freq_mask_param')}x{audio.get('num_freq_masks')})")
        summ_lines.append(f"               Files={num_files}, Classes={num_classes}, Min/class={min_per_class}")
        summ_lines.append("")
        summ_lines.append(f"  Durations (s): mean={dur_mean:.3f}, p5={dur_p5:.3f}, p50={dur_p50:.3f}, p95={dur_p95:.3f}")
        summ_lines.append(f"  RMS:           mean={rms_mean:.5f}, std={rms_std:.5f} | Files/class median={median_per_class}")
        summ_lines.append(f"  Sample rates:  {sr_str}")
        summ_lines.append(f"  Spectrogram shapes: raw={spec_before} -> resized={spec_after}")
        
        story.append(Paragraph("<b>Detailed Preprocessing Statistics:</b>", body))
        story.append(Preformatted("\n".join(summ_lines), code_style))
        story.append(Spacer(1, 0.3 * cm))
        ds_root = str(ds_info.get("root", ""))
        ds_name = os.path.basename(ds_root) if ds_root else ""
        summary_rows = [["Field", "Value"]]
        if ds_name:
            summary_rows.append(["Dataset", ds_name])
        if num_files:
            summary_rows.append(["Num files", str(num_files)])
        if num_classes:
            summary_rows.append(["Num classes", str(num_classes)])
        if audio.get("target_sr") is not None:
            summary_rows.append(["Target SR", str(audio.get("target_sr"))])
        if audio.get("clip_seconds") is not None:
            summary_rows.append(["Clip seconds", str(audio.get("clip_seconds"))])
        if audio.get("n_mels") is not None:
            summary_rows.append(["Mel bins", str(audio.get("n_mels"))])
        fft = audio.get("n_fft")
        hop = audio.get("hop_length")
        win = audio.get("win_length")
        if any(v is not None for v in (fft, hop, win)):
            summary_rows.append(["FFT / Hop / Win", f"{fft or ''} / {hop or ''} / {win or ''}"])
        summary_rows.append(["Global norm", "ON" if audio.get("global_norm") else "OFF"])
        summary_rows.append(["SpecAug", "ON" if audio.get("specaug") else "OFF"])
        story.append(_kv_table(summary_rows, [5.2 * cm, 10.0 * cm]))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("<b>Preprocessing Steps Applied:</b>", body))
        steps = []
        if audio.get("target_sr"):
            steps.append(f"Resample to {audio.get('target_sr')} Hz.")
        if audio.get("clip_seconds"):
            steps.append(f"Clip/Pad audio to {audio.get('clip_seconds')} seconds.")
        if audio.get("n_mels"):
            steps.append(f"Compute mel-spectrogram with {audio.get('n_mels')} mel bins.")
        if audio.get("n_fft") and audio.get("hop_length") and audio.get("win_length"):
            steps.append(
                f"STFT params: n_fft={audio.get('n_fft')}, hop_length={audio.get('hop_length')}, "
                f"win_length={audio.get('win_length')}."
            )
        steps.append("Global normalization." if audio.get("global_norm") else "No global normalization.")
        if audio.get("specaug"):
            steps.append(
                f"SpecAugment enabled: time_mask={audio.get('time_mask_param')} × {audio.get('num_time_masks')}, "
                f"freq_mask={audio.get('freq_mask_param')} × {audio.get('num_freq_masks')}."
            )
        else:
            steps.append("SpecAugment disabled.")
        for s in steps:
            story.append(Paragraph(f"• {s}", body))
        story.append(Spacer(1, 0.3 * cm))
        
        story.append(Paragraph("<b>Class Distribution Chart:</b>", body))
        if class_counts:
            p = _chart_class_distribution(class_counts)
            img = _fit_rl_image(p, max_w=16.0 * cm, max_h=6.0 * cm)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 0.2 * cm))
        story.append(PageBreak())

        story.append(Paragraph("3. Dataset Statistics (Visuals)", H2))
        dur_fig = _chart_durations({"durations": durs})
        if dur_fig:
            img = _fit_rl_image(dur_fig, max_w=16.0 * cm, max_h=6.2 * cm)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Paragraph("Figure: Duration distribution (seconds).", caption))
            story.append(Spacer(1, 0.2 * cm))
            
        sample_mel = _plot_sample_mel(train_loader)
        if sample_mel:
            img = _fit_rl_image(sample_mel, max_w=16.0 * cm, max_h=6.2 * cm)
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Paragraph("Figure: Example mel-spectrogram (Training Batch).", caption))
        story.append(PageBreak())

        if report_data:
            story.append(Paragraph("4. Model Report", H2))
            ds_name = str(report_data.get("dataset_name", ""))
            fam_name = str(report_data.get("device_family_name", ""))
            fam_id = str(report_data.get("device_family_id", ""))
            saved_as = str(report_data.get("saved_as", ""))
            attempted = report_data.get("attempted_exts", []) or []
            exported_paths = report_data.get("exported_paths", []) or []
            exported_exts = _exported_exts_from_paths(exported_paths)

            acc = report_data.get("accuracy")
            acc_b = report_data.get("accuracyBefore")
            acc_a = report_data.get("accuracyAfter")
            lat_b = report_data.get("latencyMsBefore")
            lat_a = report_data.get("latencyMsAfter")
            sz_b = report_data.get("sizeKBBefore")
            sz_a = report_data.get("sizeKBAfter")
            optimized_flag = not bool(report_data.get("optimization_skipped"))
            failed_flag = bool(report_data.get("optimization_failed"))
            if not optimized_flag:
                acc_a = acc_b
                lat_a = lat_b
                sz_a = sz_b

            if not optimized_flag:
                dlat_txt, dlat_col = "+0.0%", colors.green
                dsz_txt, dsz_col = "+0.0%", colors.green
            else:
                dlat_txt, dlat_col = _delta(lat_b, lat_a)
                dsz_txt, dsz_col = _delta(sz_b, sz_a)

            def _delta_acc(before, after):
                try:
                    b = float(before)
                    a = float(after)
                except Exception:
                    return ""
                if b == 0:
                    return ""
                return f"{((a - b) / b) * 100.0:+.1f}%"
            acc_delta = "+0.0%" if (not optimized_flag) else _delta_acc(acc_b, acc_a)

            cards = [
                ["Accuracy", _fmt_acc(acc), "", ""],
                ["Latency (ms / sample)", _fmt_num(lat_a, 3), "After optimizing", dlat_txt],
                ["Model Size (KB)", _fmt_num(sz_a, 2), "After optimizing", dsz_txt],
            ]

            card_rows = []
            for ctitle, cval, csub, cdelta in cards:
                color = colors.grey
                if ctitle.startswith("Latency"):
                    color = dlat_col
                elif ctitle.startswith("Model Size"):
                    color = dsz_col
                card_rows.append([
                    Paragraph(f"<b>{ctitle}</b>", small),
                    Paragraph(f"<b>{cval}</b>", H2),
                    Paragraph(csub, small),
                    Paragraph(cdelta, ParagraphStyle("delta", parent=small, textColor=color)),
                ])

            card_tbl = Table(card_rows, colWidths=[doc.width * 0.33, doc.width * 0.22, doc.width * 0.25, doc.width * 0.20])
            card_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(card_tbl)
            story.append(Spacer(1, 0.35 * cm))

            # Before vs After
            story.append(Paragraph("Before vs. After", H2))

            ba_rows = [
                ["Metric", "Before", "After", "Delta"],
                ["Accuracy", _fmt_acc(acc_b), _fmt_acc(acc_a), acc_delta],
                ["Latency per sample", _fmt_num(lat_b, 3), _fmt_num(lat_a, 3), dlat_txt],
                ["Model size (KB)", _fmt_num(sz_b, 2), _fmt_num(sz_a, 2), dsz_txt],
            ]
            ba_tbl = Table(ba_rows, colWidths=[doc.width * 0.45, doc.width * 0.18, doc.width * 0.18, doc.width * 0.19])
            ba_tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(ba_tbl)
            story.append(Spacer(1, 0.35 * cm))

            story.append(Paragraph("Details", H2))
            def _p(v):
                return Paragraph(str(v) if v is not None else "", small)
            details_rows = [
                [_p("Field"), _p("Value"), _p("Field"), _p("Value")],
                [_p("Model name"), _p("Model"), _p("Device family"), _p(f"{fam_name} ({fam_id})")],
                [_p("Dataset"), _p(ds_name), _p("Classes"), _p(report_data.get("target_num_classes", ""))],
                [_p("Optimization strategy"), _p(report_data.get("optimization_strategy", "")), _p("Optimize if size >"), _p(report_data.get("optimization_trigger_ratio_pct", ""))],
                [_p("Accuracy tolerance"), _p(report_data.get("accuracy_tolerance", "")), _p("Quantization requested"), _p(report_data.get("quantization_requested", ""))],
                [_p("Quantization applied"), _p(report_data.get("quantization_applied_display", "")), _p("Export requested"), _p(report_data.get("export_ext_requested", ""))],
                [_p("Export allowed"), _p(", ".join(report_data.get("export_ext_allowed", []) or [])), _p("Saved as"), _p(saved_as)],
                [_p("Optimization skipped"), _p(report_data.get("optimization_skipped_display", "")), _p("Skip reason"), _p(report_data.get("optimization_skip_reason_display", ""))],
                [_p("Optimization failed"), _p(report_data.get("optimization_failed_display", "")), _p("Failure reason"), _p(report_data.get("optimization_fail_reason_display", ""))]
            ]
            det_tbl = Table(details_rows, colWidths=[doc.width * 0.20, doc.width * 0.30, doc.width * 0.20, doc.width * 0.30])
            det_tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(det_tbl)
            story.append(Spacer(1, 0.35 * cm))

            opt_history = report_data.get("optimization_history", [])
            if opt_history:
                story.append(Paragraph("Optimization Details", H2))
                
                hist_data = [["Candidate", "Status", "Accuracy", "Size (KB)", "Latency (ms)", "Reason"]]
                
                for item in opt_history:
                    c_name = str(item.get("candidate", item.get("name", "n/a")))
                    c_status = str(item.get("status", "n/a"))
                    c_reason = str(item.get("reason", ""))
                    
                    acc_val = item.get("acc")
                    size_val = item.get("size_kb")
                    lat_val = item.get("latency_ms")
                    
                    acc_str = f"{float(acc_val)*100:.1f}%" if acc_val is not None else "-"
                    size_str = f"{float(size_val):.2f}" if size_val is not None else "-"
                    lat_str = f"{float(lat_val):.2f}" if lat_val is not None else "-"
                    
                    reason_para = Paragraph(c_reason, small)
                    
                    status_style = small
                    s_lower = c_status.lower()
                    if s_lower == "success":
                        status_style = ParagraphStyle("s_ok", parent=small, textColor=colors.green)
                    elif s_lower == "failed":
                        status_style = ParagraphStyle("s_fail", parent=small, textColor=colors.red)
                    elif s_lower == "rejected":
                        status_style = ParagraphStyle("s_rej", parent=small, textColor=colors.orange)
                    elif s_lower == "skipped":
                        status_style = ParagraphStyle("s_skip", parent=small, textColor=colors.gray)

                    hist_data.append([c_name, Paragraph(c_status, status_style), acc_str, size_str, lat_str, reason_para])

                t_width = doc.width
                col_widths = [t_width*0.14, t_width*0.09, t_width*0.10, t_width*0.10, t_width*0.12, t_width*0.45]
                
                hist_tbl = Table(hist_data, colWidths=col_widths)
                hist_tbl.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(hist_tbl)
                story.append(Spacer(1, 0.3 * cm))

            # Model Feedback
            min_acc = report_data.get("min_accuracy")
            meets_acc = ""
            try:
                acc_val = float(acc)
                min_val = float(min_acc) if min_acc is not None else None
                if min_val is not None:
                    meets_acc = "meets" if acc_val >= min_val else "does not meet"
            except Exception:
                meets_acc = ""

            def _pct_change(b, a):
                try:
                    b = float(b)
                    a = float(a)
                    if b == 0:
                        return ""
                    return f"{((a - b) / b) * 100.0:+.1f}%"
                except Exception:
                    return ""

            lat_delta = _pct_change(lat_b, lat_a)
            size_delta = _pct_change(sz_b, sz_a)

            fb_lines = []
            
            fb_lines.append(f"The final model accuracy is {_fmt_acc(acc)}, which {meets_acc} the minimum requirement.")

            if optimized_flag and not failed_flag:
                applied = report_data.get("quantization_applied_display", "unknown")
                fb_lines.append(f"Optimization was successful using {applied} quantization.")
            elif failed_flag:
                fb_lines.append("Optimization failed, so the pipeline reverted to the baseline FP32 model.")
            else:
                fb_lines.append("Optimization was skipped based on the configured strategy.")

            if sz_b and sz_a:
                fb_lines.append(f"Model size moved from {_fmt_num(sz_b, 1)} KB to {_fmt_num(sz_a, 1)} KB ({size_delta}).")
            if lat_b and lat_a:
                fb_lines.append(f"Latency moved from {_fmt_num(lat_b, 2)} ms to {_fmt_num(lat_a, 2)} ms per sample ({lat_delta}).")
            
            feedback = " ".join(fb_lines)

            attempts_sentence = ""
            attempts_disp = report_data.get("optimization_attempted_levels_display", "")
            attempts_acc = report_data.get("optimization_attempt_accuracies_display", "")
            if attempts_disp:
                if attempts_acc:
                    attempts_sentence = f"Candidates evaluated: {attempts_disp} (details: {attempts_acc})."
                else:
                    attempts_sentence = f"Candidates evaluated: {attempts_disp}."
            if attempts_sentence:
                feedback = f"{feedback} {attempts_sentence}"

            story.append(PageBreak())
            story.append(Paragraph("Model Feedback", H2))
            story.append(Paragraph(feedback, body))
            story.append(Spacer(1, 0.3 * cm))

        if deployment_data:
            story.append(PageBreak())
            story.append(Paragraph("5. Deployment Guidance", H2))
            fam_name = deployment_data.get("family_name", "")
            fam_id = deployment_data.get("family_id", "")
            frameworks = deployment_data.get("frameworks", []) or []
            allowed_exts = deployment_data.get("allowed_exts", []) or []
            exported_paths = deployment_data.get("exported_paths", []) or []
            exported_exts = _exported_exts_from_paths(exported_paths)

            story.append(Paragraph(
                f"<b>Device family:</b> {fam_name} ({fam_id})",
                body
            ))
            if frameworks:
                story.append(Paragraph(
                    f"<b>Supported frameworks:</b> {', '.join(map(str, frameworks))}",
                    body
                ))
            if allowed_exts:
                story.append(Paragraph(
                    f"<b>Allowed export formats:</b> {', '.join(map(str, allowed_exts))}",
                    body
                ))
            if exported_exts:
                story.append(Paragraph(
                    f"<b>Exported format(s):</b> {', '.join(map(str, exported_exts))}",
                    body
                ))

            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("<b>General Deployment Steps</b>", body))
            steps = [
                "Copy the exported model into your firmware/application project.",
                "Apply the same preprocessing steps used during training (see this report).",
                "Load the model with a runtime supported by your device family.",
                "Run a single inference to validate end-to-end integration.",
                "Map output indices/probabilities to target labels.",
            ]
            for i, s in enumerate(steps, 1):
                story.append(Paragraph(f"{i}. {s}", body))

            story.append(Spacer(1, 0.25 * cm))
            story.append(Paragraph("<b>Minimal Example Usage</b>", body))
            code_lines = [
                "# adapt to your runtime/framework",
                "model = load_model(\"exported_model\")",
                "raw_input = read_audio_clip()",
                "input_data = preprocess(raw_input)  # same steps as in report",
                "output = model.infer(input_data)",
                "label = labels[argmax(output)]",
                "print(label)",
            ]
            story.append(Preformatted("\n".join(code_lines), code_style))

            labels = report_data.get("class_labels", []) if report_data else []
            if labels:
                story.append(Paragraph(f"<b>Target labels:</b> {', '.join(map(str, labels))}", body))

        try:
            doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
            print(f"Report generated: {path}")
        except Exception as e:
            print(f"Failed to build report: {e}")
            traceback.print_exc()

        for p in tmp_files:
            try: os.remove(p)
            except: pass

    # ==========================================================
    # Validation
    # ==========================================================
    def _phase_validate_final(self) -> None:
        if self._final_model is None:
             raise RuntimeError("No final model available for validation.")

        test_acc = self.state.metrics.get("final_test_acc", None)
        if test_acc is None:
             raise RuntimeError("Missing final_test_acc metric; cannot validate minimum accuracy.")

        if "final_model_size_mb" not in self.state.metrics:
             opt = self.state.metrics.get("optimization", {})
             size = opt.get("final_size", 0.0)
        else:
             size = self.state.metrics["final_model_size_mb"]

        ok, reason, metrics_patch = validate_final_against_specs(
            test_acc=float(test_acc),
            min_accuracy=float(self.min_accuracy),
            model_file_size_mb=float(size),
            torch_model_obj=self._final_model,
            device_specs=self.device_specs,
            overhead_factor=1.20,
        )

        self.state.metrics.update(metrics_patch)

        if not ok:
            if self.state.metrics.get("optimization_failed"):
                self.state.metrics["validation_note"] = f"Skipped constraint check (optimization_failed=True): {reason}"
            else:
                raise RuntimeError(reason)
