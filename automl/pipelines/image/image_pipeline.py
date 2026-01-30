from __future__ import annotations

import os
import time
import shutil
import traceback
import re
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
from dataclasses import field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, random_split

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
    task_type: str  # "image"
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


class ImagePipeline:

    def __init__(
        self,
        *,
        task_id: str,
        user_id: str,
        task_type: str,
        dataset_path: str,
        device_family_id: str,
        device_specs: Dict[str, Any],
        export_ext: str = ".h",
        target_num_classes: int,
        output_root: str = "runs",
        min_accuracy: float = 0.50,
        seed: int = 42,
        batch_size: int = 128,
        num_workers: int = 2,
        val_split: float = 0.1,
        pin_memory: bool = True,
        sweep: Optional[list] = None,
        epochs_per_trial: int = 3,
        final_epochs: int = 5,
        torch_device: Optional[str] = None,

        title: str = None,
        description: str = None,
        visibility: str = None,

        quantization: str = None,
        optimization_strategy: str = None,
        training_speed: str = None,
        accuracy_tolerance: str = None,
        optimization_trigger_ratio: float = 0.70,
        augmentation: str = None,
        feature_handling: str = None,
        cleaning: str = None,
        noise_handling: str = None,
    ):
        if task_type != "image":
            raise ValueError(f"ImagePipeline supports task_type='image' only, got {task_type}")

        self.task_id = task_id
        self.user_id = user_id
        self.task_type = task_type

        self.title = title
        self.description = description
        self.visibility = visibility

        self.quantization = quantization
        self.optimization_strategy = optimization_strategy
        self.training_speed = training_speed
        self.accuracy_tolerance = accuracy_tolerance
        self.optimization_trigger_ratio = float(optimization_trigger_ratio)
        self.augmentation = augmentation
        self.feature_handling = feature_handling
        self.cleaning = cleaning
        self.noise_handling = noise_handling

        self.dataset_path = dataset_path  
        self.device_family_id = device_family_id
        self.device_specs = normalize_device_specs(device_specs)
        self.target_num_classes = int(target_num_classes)

        self.export_ext = str(export_ext).strip().lower()
        if self.export_ext != "all" and not self.export_ext.startswith("."):
            self.export_ext = "." + self.export_ext

        self.output_dir = os.path.join(output_root, task_id)
        self.dataset_extract_dir = os.path.join(self.output_dir, "dataset")

        self.min_accuracy = float(min_accuracy)

        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_split = float(val_split)
        self.pin_memory = bool(pin_memory)

        self.epochs_per_trial = int(epochs_per_trial)
        self.final_epochs = int(final_epochs)

        self.sweep = sweep if sweep is not None else [
            {"name": "mnetv3_small_unfreeze0_res160",      "backbone": "mnetv3_small",      "unfreeze_blocks": 0, "img_size": 160, "lr": 3e-3},
            {"name": "mnetv3_small_unfreeze1_res160",      "backbone": "mnetv3_small",      "unfreeze_blocks": 1, "img_size": 160, "lr": 2e-3},
            {"name": "shufflenetv2_x0_5_unfreeze1_res160", "backbone": "shufflenetv2_x0_5", "unfreeze_blocks": 1, "img_size": 160, "lr": 2e-3},
            {"name": "squeezenet1_1_unfreeze1_res160",     "backbone": "squeezenet1_1",     "unfreeze_blocks": 1, "img_size": 160, "lr": 2e-3},
        ]

        self.STRONG_AUG_THRESHOLD = 5_000
        self.MODERATE_AUG_THRESHOLD = 50_000
        self.RAND_N = 2
        self.RAND_M_STRONG = 18
        self.RAND_M_MODERATE = 10
        self.RAND_M_LIGHT = 6

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device(torch_device) if torch_device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.state = PipelineState(
            task_id=self.task_id,
            user_id=self.user_id,
            task_type=self.task_type,
            status="queued",
            stage_idx=0,
            message="",
            updated_at=time.time(),
            metrics={},
            errors={},
        )

        self._data_meta: Dict[str, Any] = {}

        self._prep_train_loader = None
        self._prep_val_loader = None
        self._prep_test_loader = None

        self._final_train_loader = None
        self._final_val_loader = None
        self._final_test_loader = None

        self._trained_model: Optional[nn.Module] = None
        self._final_model: Optional[nn.Module] = None
        self._final_model_path: Optional[str] = None
        self._report_path: Optional[str] = None

    # --------------------------
    # Status helpers
    # --------------------------

    def update_status(
        self,
        new_status: str,
        message: str = "",
        metrics_patch: Optional[Dict[str, Any]] = None,
        errors_patch: Optional[Dict[str, Any]] = None,
    ) -> None:
        if new_status not in STATUS_ORDER:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {STATUS_ORDER}")
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
            "status": self.get_status(),
            "output_dir": self.output_dir,
            "final_model_path": self._final_model_path,
            "report_path": self._report_path,
            "metrics": dict(self.state.metrics),
            "errors": dict(self.state.errors),
            "exported_model_paths": self.state.metrics.get("exported_model_paths", []),
        }

    # --------------------------
    # Main entry
    # --------------------------

    def run(self) -> Dict[str, Any]:
        try:
            self._prepare_dirs()

            self.update_status("queued", "Validating device specs with family...")
            ok, msg = validate_specs_within_family(self.device_family_id, self.device_specs)
            if not ok:
                return self._fail(f"Device specs invalid for selected family: {msg}")

            self.update_status("preprocessing", "Preparing dataset + preprocessing...")
            self._unpack_and_validate_dataset()
            self._phase_preprocessing()


            self.update_status("training", "Training (sweep + final)...")
            self._phase_training()

            self.update_status("optimizing", "Optimizing...")
            self._phase_optimizing()

            self.update_status("packaging", "Saving final model + generating report...")
            self._phase_packaging()
            self._phase_report()

            self._phase_validate_final()

            self.update_status("completed", "Completed successfully.")
            return self._result()

        except (RuntimeError, ValueError) as e:
            return self._fail(str(e), e)
        except Exception as e:
            return self._fail("Unhandled pipeline error", e)

    # --------------------------
    # Directories
    # --------------------------

    def _prepare_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _unpack_and_validate_dataset(self) -> None:

        allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def is_image_file(p: str) -> bool:
            return os.path.splitext(p)[1].lower() in allowed_exts

        def list_dir_safe(d: str):
            try:
                return [os.path.join(d, x) for x in os.listdir(d) if not x.startswith(".")]
            except Exception:
                return []

        def count_images_in_dir(d: str) -> int:
            n = 0
            for r, _, files in os.walk(d):
                for f in files:
                    if is_image_file(f):
                        n += 1
            return n

        def validate_imagefolder_root(root: str) -> Tuple[bool, str, Dict[str, Any]]:

            candidates = [p for p in list_dir_safe(root) if os.path.isdir(p)]

            class_info = []
            for cdir in candidates:
                cname = os.path.basename(cdir)
                img_count = count_images_in_dir(cdir)
                if img_count > 0:
                    class_info.append((cname, img_count))

            if len(class_info) < 2:
                return False, (
                    f"Invalid ImageFolder structure at '{root}'. "
                    f"Need >=2 class folders with images. Found {len(class_info)} valid class folders."
                ), {"found_classes": class_info}

            meta = {
                "data_root": root,
                "num_classes": len(class_info),
                "classes": [c for c, _ in sorted(class_info, key=lambda x: x[0].lower())],
                "class_image_counts": {c: n for c, n in class_info},
                "allowed_exts": sorted(list(allowed_exts)),
            }
            return True, "OK", meta

        def find_best_root(extract_dir: str) -> Tuple[bool, str, str, Dict[str, Any]]:
            ok, msg, meta = validate_imagefolder_root(extract_dir)
            if ok:
                return True, "OK", extract_dir, meta
            entries = [p for p in list_dir_safe(extract_dir) if os.path.isdir(p)]
            if len(entries) == 1:
                ok2, msg2, meta2 = validate_imagefolder_root(entries[0])
                if ok2:
                    return True, "OK", entries[0], meta2

            from collections import deque
            q = deque([(extract_dir, 0)])
            best = None  
            while q:
                node, depth = q.popleft()
                if depth > 3:
                    continue

                ok3, _, meta3 = validate_imagefolder_root(node)
                if ok3:
                    total_images = sum(meta3["class_image_counts"].values())
                    score = (meta3["num_classes"], total_images)
                    if best is None or score > (best[0], best[1]):
                        best = (meta3["num_classes"], total_images, node, meta3)

                for child in list_dir_safe(node):
                    if os.path.isdir(child):
                        q.append((child, depth + 1))

            if best is not None:
                return True, "OK", best[2], best[3]

            return False, f"Could not find a valid ImageFolder root inside extracted data: {msg}", "", {}

        p = self.dataset_path
        if not isinstance(p, str) or len(p) == 0:
            raise ValueError("dataset_path must be a non-empty string.")

        if os.path.isdir(p):
            ok, msg, meta = validate_imagefolder_root(p)
            if not ok:
                raise RuntimeError(msg)
            self.dataset_path = p
            self.state.metrics.update({"dataset_mode": "folder", **meta})
            return

        if not os.path.isfile(p):
            raise FileNotFoundError(f"dataset_path does not exist: {p}")

        os.makedirs(self.dataset_extract_dir, exist_ok=True)

        for name in os.listdir(self.dataset_extract_dir):
            full = os.path.join(self.dataset_extract_dir, name)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                try:
                    os.remove(full)
                except OSError:
                    pass

        try:
            shutil.unpack_archive(p, self.dataset_extract_dir)
        except shutil.ReadError:
            raise RuntimeError(f"Unsupported archive format or corrupted archive: {p}")

        ok_root, msg_root, root_path, meta = find_best_root(self.dataset_extract_dir)
        if not ok_root:
            raise RuntimeError(msg_root)

        self.dataset_path = root_path
        self.state.metrics.update({"dataset_mode": "archive", "dataset_extracted_to": root_path, **meta})

    # ==========================================================
    # Phase 1: preprocessing 
    # ==========================================================

    def _phase_preprocessing(self) -> None:
        def pick_img_size() -> int:
            return 64

        def decide_strategy_b(num_images: int):
            if num_images < self.STRONG_AUG_THRESHOLD:
                level = "strong"
                family = "trivialaugment"
                randM = self.RAND_M_STRONG
            elif num_images <= self.MODERATE_AUG_THRESHOLD:
                level = "moderate"
                family = "randaugment"
                randM = self.RAND_M_MODERATE
            else:
                level = "light"
                family = "basic"
                randM = self.RAND_M_LIGHT
            return level, family, self.RAND_N, randM

        def make_transforms(img_size: int, family: str, rand_n: int, rand_m: int):
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

            train_ops = [T.Resize((img_size, img_size))]

            if family == "trivialaugment":
                train_ops.append(T.TrivialAugmentWide())
            elif family == "randaugment":
                train_ops.append(T.RandAugment(num_ops=rand_n, magnitude=rand_m))
            elif family == "basic":
                train_ops.extend([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                ])
            elif family == "none":
                pass

            train_ops.extend([T.ToTensor(), normalize])

            test_ops = [T.Resize((img_size, img_size)), T.ToTensor(), normalize]
            return T.Compose(train_ops), T.Compose(test_ops)

        def class_counts_from_dataset(dataset, num_classes: int):
            counts = [0] * int(num_classes)
            for _, y in dataset:
                counts[int(y)] += 1
            return counts

        def sample_resolution_stats(dataset, k: int = 128):
            n = len(dataset)
            k = min(int(k), n)
            if k <= 0:
                return {"w": [], "h": []}

            idxs = np.random.choice(n, size=k, replace=False)
            ws, hs = [], []
            for i in idxs:
                x, _ = dataset[i]
                if isinstance(x, Image.Image):
                    w, h = x.size
                else:
                    try:
                        h = int(x.shape[-2])
                        w = int(x.shape[-1])
                    except Exception:
                        continue
                ws.append(w)
                hs.append(h)
            return {"w": ws, "h": hs}

        def get_dataloaders_imagefolder(img_size: int):
            if not os.path.isdir(self.dataset_path):
                raise FileNotFoundError(f"dataset_path not found: {self.dataset_path}")

            full_ds = torchvision.datasets.ImageFolder(self.dataset_path, transform=None)
            num_classes = len(full_ds.classes)

            n_images = int(len(full_ds))
            counts = class_counts_from_dataset(full_ds, num_classes)
            res_stats = sample_resolution_stats(full_ds, k=128)

            level, family, rand_n, rand_m = decide_strategy_b(n_images)
            train_tf, test_tf = make_transforms(img_size, family, rand_n, rand_m)

            TEST_SPLIT = 0.1
            test_len = max(1, int(TEST_SPLIT * len(full_ds)))
            remain_len = len(full_ds) - test_len

            remain_set, test_set = random_split(
                full_ds, [remain_len, test_len],
                generator=torch.Generator().manual_seed(self.seed)
            )

            val_len = max(1, int(remain_len * self.val_split))
            train_len = remain_len - val_len

            train_set, val_set = random_split(
                remain_set, [train_len, val_len],
                generator=torch.Generator().manual_seed(self.seed)
            )

            base_train = torchvision.datasets.ImageFolder(self.dataset_path, transform=train_tf)
            base_test = torchvision.datasets.ImageFolder(self.dataset_path, transform=test_tf)

            train_set = torch.utils.data.Subset(base_train, train_set.indices)
            val_set = torch.utils.data.Subset(base_test, val_set.indices)
            test_set = torch.utils.data.Subset(base_test, test_set.indices)

            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            val_loader = DataLoader(
                val_set, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
            test_loader = DataLoader(
                test_set, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )

            prep_report = {
                "augmentation": {
                    "strategy": "Strategy B",
                    "level": level,
                    "family": family,
                    "rand_n": int(rand_n),
                    "rand_m": int(rand_m),
                    "img_size": int(img_size),
                },
                "dataset": {
                    "mode": "imagefolder",
                    "root": self.dataset_path,
                    "num_images": int(n_images),
                    "num_classes": int(num_classes),
                    "classes": list(full_ds.classes),
                    "class_counts": list(counts),
                    "resolution_sample": {
                        "n": int(min(128, n_images)),
                        "w": list(res_stats["w"]),
                        "h": list(res_stats["h"]),
                    },
                    "split": {"test_split": float(TEST_SPLIT), "val_split": float(self.val_split)},
                    "train_samples": int(len(train_set)),
                    "val_samples": int(len(val_set)),
                    "test_samples": int(len(test_set)),
                    "train_batches": int(len(train_loader)),
                    "val_batches": int(len(val_loader)),
                    "test_batches": int(len(test_loader)),
                    "seed": int(self.seed),
                },
                "loader": {
                    "batch_size": int(self.batch_size),
                    "num_workers": int(self.num_workers),
                    "pin_memory": bool(self.pin_memory),
                }
            }

            return train_loader, val_loader, test_loader, prep_report

        img_size = pick_img_size()
        train_loader, val_loader, test_loader, prep_report = get_dataloaders_imagefolder(img_size)

        self._prep_train_loader = train_loader
        self._prep_val_loader = val_loader
        self._prep_test_loader = test_loader  
        self._data_meta["preprocessing_report"] = prep_report

        inferred = int(prep_report["dataset"]["num_classes"])
        self.state.metrics["inferred_num_classes"] = inferred
        if inferred != int(self.target_num_classes):
            raise RuntimeError(f"Dataset mismatch: Found {inferred} classes but you specified {self.target_num_classes}. Please check your dataset input.")

        self.update_status(
            "preprocessing",
            "Preprocessing complete (dataloaders ready).",
            metrics_patch={
                "dataset_samples": prep_report["dataset"]["num_images"],
                "train_samples": prep_report["dataset"]["train_samples"],
                "val_samples": prep_report["dataset"]["val_samples"],
                "test_samples": prep_report["dataset"]["test_samples"],
                "augment_level": prep_report["augmentation"]["level"],
                "augment_family": prep_report["augmentation"]["family"],
                "img_size": prep_report["augmentation"]["img_size"],
            },
        )

    # ==========================================================
    # Phase 2: training 
    # ==========================================================

    def _phase_training(self) -> None:
        if self.target_num_classes <= 0:
            raise ValueError("target_num_classes must be > 0")

        criterion = nn.CrossEntropyLoss()

        def build_backbone(backbone: str, num_classes: int) -> nn.Module:
            if backbone == "mnetv3_small":
                weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                model = torchvision.models.mobilenet_v3_small(weights=weights)
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
                return model

            if backbone == "shufflenetv2_x0_5":
                weights = torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
                model = torchvision.models.shufflenet_v2_x0_5(weights=weights)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                return model

            if backbone == "squeezenet1_1":
                weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT
                model = torchvision.models.squeezenet1_1(weights=weights)
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
                model.num_classes = num_classes
                return model

            raise ValueError(f"Unknown backbone: {backbone}")

        def unfreeze_module(module: nn.Module) -> None:
            for p in module.parameters():
                p.requires_grad = True

        def get_head_module(model: nn.Module, backbone: str) -> nn.Module:
            if backbone == "mnetv3_small":
                return model.classifier
            if backbone == "shufflenetv2_x0_5":
                return model.fc
            if backbone == "squeezenet1_1":
                return model.classifier
            raise ValueError(f"Unknown backbone: {backbone}")

        def get_block_list(model: nn.Module, backbone: str):
            if backbone == "mnetv3_small":
                return list(model.features)
            if backbone == "shufflenetv2_x0_5":
                blocks = [model.conv1, model.maxpool, model.stage2, model.stage3, model.stage4, model.conv5]
                return blocks
            if backbone == "squeezenet1_1":
                return list(model.features)
            raise ValueError(f"Unknown backbone: {backbone}")

        def freeze_all_but_head(model: nn.Module, backbone: str) -> None:
            for p in model.parameters():
                p.requires_grad = False
            head = get_head_module(model, backbone)
            unfreeze_module(head)

        def unfreeze_last_n_blocks(model: nn.Module, backbone: str, n_blocks: int) -> None:
            freeze_all_but_head(model, backbone)
            if n_blocks <= 0:
                return
            blocks = get_block_list(model, backbone)
            for b in blocks[-int(n_blocks):]:
                unfreeze_module(b)

        def count_trainable_params(model: nn.Module) -> int:
            return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        def make_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
            params = [p for p in model.parameters() if p.requires_grad]
            return optim.AdamW(params, lr=float(lr), weight_decay=1e-4)

        @torch.no_grad()
        def evaluate(model: nn.Module, loader) -> Tuple[float, float]:
            model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * x.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(x.size(0))
            return total_loss / max(1, total), correct / max(1, total)

        def train_one_epoch(model: nn.Module, loader, optimizer_: optim.Optimizer) -> Tuple[float, float]:
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                optimizer_.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer_.step()
                total_loss += float(loss.item()) * x.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(x.size(0))
            return total_loss / max(1, total), correct / max(1, total)

        def build_trial_loaders(img_size: int):
            full_ds = torchvision.datasets.ImageFolder(self.dataset_path, transform=None)
            n_images = int(len(full_ds))

            if n_images < self.STRONG_AUG_THRESHOLD:
                family = "trivialaugment"
                rand_m = self.RAND_M_STRONG
            elif n_images <= self.MODERATE_AUG_THRESHOLD:
                family = "randaugment"
                rand_m = self.RAND_M_MODERATE
            else:
                family = "basic"
                rand_m = self.RAND_M_LIGHT

            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_ops = [T.Resize((img_size, img_size))]
            if family == "trivialaugment":
                train_ops.append(T.TrivialAugmentWide())
            elif family == "randaugment":
                train_ops.append(T.RandAugment(num_ops=self.RAND_N, magnitude=rand_m))
            elif family == "basic":
                train_ops.extend([T.RandomHorizontalFlip(p=0.5), T.RandomResizedCrop(img_size, scale=(0.85, 1.0))])
            train_ops.extend([T.ToTensor(), normalize])

            test_ops = [T.Resize((img_size, img_size)), T.ToTensor(), normalize]

            base_train = torchvision.datasets.ImageFolder(self.dataset_path, transform=T.Compose(train_ops))
            base_test = torchvision.datasets.ImageFolder(self.dataset_path, transform=T.Compose(test_ops))

            TEST_SPLIT = 0.1
            test_len = max(1, int(TEST_SPLIT * len(full_ds)))
            remain_len = len(full_ds) - test_len

            remain_set, test_set = random_split(
                full_ds, [remain_len, test_len],
                generator=torch.Generator().manual_seed(self.seed)
            )

            val_len = max(1, int(remain_len * self.val_split))
            train_len = remain_len - val_len

            train_set, val_set = random_split(
                remain_set, [train_len, val_len],
                generator=torch.Generator().manual_seed(self.seed)
            )

            train_set = torch.utils.data.Subset(base_train, train_set.indices)
            val_set = torch.utils.data.Subset(base_test, val_set.indices)
            test_set = torch.utils.data.Subset(base_test, test_set.indices)

            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, pin_memory=self.pin_memory)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers, pin_memory=self.pin_memory)
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, pin_memory=self.pin_memory)
            return train_loader, val_loader, test_loader

        results = []
        best_cfg = None
        best_cfg_val = -1.0

        # --- SWEEP trials ---
        for trial in self.sweep:
            self.update_status("training", f"SWEEP trial: {trial.get('name','(unnamed)')}")
            t0 = time.time()

            train_loader, val_loader, test_loader = build_trial_loaders(int(trial["img_size"]))

            model = build_backbone(str(trial["backbone"]), num_classes=self.target_num_classes).to(self.device)
            unfreeze_last_n_blocks(model, str(trial["backbone"]), int(trial["unfreeze_blocks"]))

            trainable = count_trainable_params(model)
            total_params = int(sum(p.numel() for p in model.parameters()))
            optimizer_ = make_optimizer(model, lr=float(trial["lr"]))

            best_val = 0.0
            best_state = None

            for epoch in range(1, self.epochs_per_trial + 1):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer_)
                va_loss, va_acc = evaluate(model, val_loader)

                if va_acc > best_val:
                    best_val = float(va_acc)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                self.state.metrics["last_trial"] = trial.get("name", "")
                self.state.metrics["last_epoch"] = epoch
                self.state.metrics["last_train_acc"] = float(tr_acc)
                self.state.metrics["last_val_acc"] = float(va_acc)

            if best_state is not None:
                model.load_state_dict(best_state)

            te_loss, te_acc = evaluate(model, test_loader)
            dt = time.time() - t0

            row = {
                "name": trial.get("name", ""),
                "cfg": dict(trial),
                "best_val_acc": float(best_val),
                "test_acc": float(te_acc),
                "test_loss": float(te_loss),
                "trainable_params": int(trainable),
                "total_params": int(total_params),
                "trainable_pct": float(100.0 * trainable / max(1, total_params)),
                "seconds": float(dt),
            }
            results.append(row)

            if float(best_val) > float(best_cfg_val):
                best_cfg_val = float(best_val)
                best_cfg = dict(trial)

        if best_cfg is None:
            raise RuntimeError("Sweep produced no valid configuration.")

        self.state.metrics["sweep_results"] = results
        self.state.metrics["best_cfg"] = best_cfg
        self.state.metrics["best_cfg_val_acc"] = float(best_cfg_val)

        # --- FINAL training with best cfg ---
        self.update_status("training", f"Final training with best cfg: {best_cfg.get('name','')}")

        self._final_train_loader, self._final_val_loader, self._final_test_loader = build_trial_loaders(int(best_cfg["img_size"]))

        best_model = build_backbone(str(best_cfg["backbone"]), num_classes=self.target_num_classes).to(self.device)
        unfreeze_last_n_blocks(best_model, str(best_cfg["backbone"]), int(best_cfg["unfreeze_blocks"]))

        optimizer_ = make_optimizer(best_model, lr=float(best_cfg["lr"]))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_, T_max=self.final_epochs)

        best_val = 0.0
        best_state = None

        for epoch in range(1, self.final_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(best_model, self._final_train_loader, optimizer_)
            va_loss, va_acc = evaluate(best_model, self._final_val_loader )
            scheduler.step()

            if va_acc > best_val:
                best_val = float(va_acc)
                best_state = {k: v.detach().cpu().clone() for k, v in best_model.state_dict().items()}

            self.state.metrics["final_epoch"] = epoch
            self.state.metrics["final_train_acc"] = float(tr_acc)
            self.state.metrics["final_val_acc"] = float(va_acc)

        if best_state is not None:
            best_model.load_state_dict(best_state)

        final_test_loss, final_test_acc = evaluate(best_model, self._final_test_loader)

        self._trained_model = best_model
        self._final_model = best_model

        self.state.metrics["final_best_val_acc"] = float(best_val)
        self.state.metrics["final_test_loss"] = float(final_test_loss)
        self.state.metrics["test_acc"] = float(final_test_acc)

    # ==========================================================
    # Phase 3: optimizing 
    # ==========================================================

    def _phase_optimizing(self) -> None:
        import copy
        import tempfile

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

        def image_fp16(model: nn.Module) -> nn.Module:
            model = model.eval()
            return model.half()

        def image_int8_dynamic(model):
            m = copy.deepcopy(model).cpu()
            q_model = torch.quantization.quantize_dynamic(m, {nn.Linear, nn.LSTM, nn.GRU, nn.RNN}, dtype=torch.qint8)
            return q_model

        def image_int8_static(model, calibration_loader, backend="fbgemm"):
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
                jobs.append(("float16", lambda m: image_fp16(m), self.device))
            elif wants_fp16:
                jobs.append(("float16", lambda m: image_fp16(m), self.device))

        if wants_auto or wants_dyn:
            jobs.append(("dynamic_int8", lambda m: image_int8_dynamic(m), torch.device("cpu")))

        if wants_auto or wants_stat:
            jobs.append(("static_int8", lambda m: image_int8_static(m, calib_loader, backend="fbgemm"), torch.device("cpu")))


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

        valid_cands = [c for c in candidates if c.get("valid", False)]
        
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
    # Phase 4: packaging 
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
    # Phase 5: report 
    # ==========================================================

    def _phase_report(self) -> None:

        prep = self._data_meta.get("preprocessing_report")
        if not isinstance(prep, dict):
            raise RuntimeError("preprocessing_report missing; cannot generate report.")

        aug = prep.get("augmentation", {}) or {}
        ds = prep.get("dataset", {}) or {}
        loader_cfg = prep.get("loader", {}) or {}
        classes = ds.get("classes", []) or []
        class_counts = ds.get("class_counts", None)

        class_counts_dict = {}
        if isinstance(class_counts, dict):
            class_counts_dict = dict(class_counts)
        elif isinstance(class_counts, list) and classes:
            m = min(len(classes), len(class_counts))
            class_counts_dict = {str(classes[i]): int(class_counts[i]) for i in range(m)}
        else:
            class_counts_dict = {}

        splits = {
            "seed": int(ds.get("seed", self.seed)),
            "val_split": float(ds.get("split", {}).get("val_split", self.val_split)) if isinstance(ds.get("split", {}), dict) else float(self.val_split),
            "train_samples": int(ds.get("train_samples", 0)),
            "val_samples": int(ds.get("val_samples", 0)),
            "test_samples": int(ds.get("test_samples", 0)),
        }

        strategy_b = {
            "strong_aug_threshold": int(getattr(self, "STRONG_AUG_THRESHOLD", 5000)),
            "moderate_aug_threshold": int(getattr(self, "MODERATE_AUG_THRESHOLD", 50000)),
            "randaugment_N": int(aug.get("rand_n", getattr(self, "RAND_N", 2))),
            "randaugment_M": int(aug.get("rand_m", 0)),
            "family": str(aug.get("family", "")),
            "level": str(aug.get("level", "")),
        }

        resolution_stats = {}
        res_sample = ds.get("resolution_sample", {}) or {}
        ws = res_sample.get("w", []) or []
        hs = res_sample.get("h", []) or []
        if isinstance(ws, list) and isinstance(hs, list):
            m = min(len(ws), len(hs))
            if m > 0:
                pairs = [(int(ws[i]), int(hs[i])) for i in range(m)]
                resolution_stats["sample_pairs"] = pairs

                w_arr = np.array([p[0] for p in pairs], dtype=float)
                h_arr = np.array([p[1] for p in pairs], dtype=float)
                resolution_stats.update({
                    "width_mean": float(w_arr.mean()),
                    "height_mean": float(h_arr.mean()),
                    "width_median": float(np.median(w_arr)),
                    "height_median": float(np.median(h_arr)),
                    "width_min": int(w_arr.min()),
                    "width_max": int(w_arr.max()),
                    "height_min": int(h_arr.min()),
                    "height_max": int(h_arr.max()),
                    "resolution_bad_count": 0,
                    "resolution_scan_limit": int(m),
                })

        flat_report = {
            "task_id": self.task_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),

            "dataset_mode": self.state.metrics.get("dataset_mode", ds.get("mode", "")),
            "data_root": os.path.basename(str(ds.get("root", self.dataset_path))) if ds.get("root", self.dataset_path) else "",
            "num_images": int(ds.get("num_images", 0)),
            "num_classes": int(ds.get("num_classes", 0)),

            "class_counts": class_counts_dict,
            "imbalance_ratio": None,

            "img_size": int(aug.get("img_size", 0)),
            "normalization": "ImageNet mean/std",

            "resolution_stats": resolution_stats,
            "strategy_b": strategy_b,
            "splits": splits,
            "loader": {
                "batch_size": int(loader_cfg.get("batch_size", self.batch_size)),
                "num_workers": int(loader_cfg.get("num_workers", self.num_workers)),
                "pin_memory": bool(loader_cfg.get("pin_memory", self.pin_memory)),
            },
        }

        flat_report["training"] = {
            "sweep_results": self.state.metrics.get("sweep_results"),
            "best_cfg": self.state.metrics.get("best_cfg"),
            "best_cfg_val_acc": self.state.metrics.get("best_cfg_val_acc"),
            "final_best_val_acc": self.state.metrics.get("final_best_val_acc"),
            "final_test_loss": self.state.metrics.get("final_test_loss"),
            "final_test_acc": self.state.metrics.get("test_acc"),
        }
        flat_report["optimization"] = self.state.metrics.get("optimization", {})
        flat_report["packaging"] = {
            "final_model_path": self._final_model_path,
            "final_model_size_mb": self.state.metrics.get("final_model_size_mb"),
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
            "class_labels": list((flat_report.get("class_counts") or {}).keys()),
            "exported_paths": self.state.metrics.get("exported_model_paths", []),
            "quantization_applied": self.state.metrics.get("quantization_applied", ""),
            "target_name": None,
            "target_num_classes": self.target_num_classes,
            "optimization_strategy": self.optimization_strategy,
            "accuracy_tolerance": self.accuracy_tolerance,
            "accuracy_drop_cap": opt_block.get("max_abs_drop"),
            "accuracy_drop_allowed": opt_block.get("max_abs_drop"),
            "optimization_trigger_ratio": self.optimization_trigger_ratio,
            "quantization_requested": self.quantization,
            "optimization_skipped": self.state.metrics.get("optimization_skipped", False),
            "optimization_skip_reason": self.state.metrics.get("optimization_skip_reason"),
            "optimization_failed": self.state.metrics.get("optimization_failed", False),
            "optimization_fail_reason": self.state.metrics.get("optimization_fail_reason"),
            "optimization_history": self.state.metrics.get("optimization_history", []),
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
        self. _generate_image_report(
            report=flat_report,
            train_loader=self._prep_train_loader,
            path=report_path,
            project_name="Automata-AI Image Report",
            logo_path=LOGO_PATH,
            report_data=report_data,
            deployment_data=deployment_data,
        )
        self._report_path = report_path
        self.state.metrics["report_path"] = report_path

    def  _generate_image_report(
        self,
        report: dict,
        train_loader=None,
        path: str = "image_prep_report.pdf",
        project_name: str = "Automata AI - Image Preprocessing Report",
        logo_path: Optional[str] = None,
        report_data: Optional[Dict[str, Any]] = None,
        deployment_data: Optional[Dict[str, Any]] = None,
    ):
        import os, tempfile, datetime
        from typing import Optional, List, Dict, Any

        import numpy as np
        import torch

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from PIL import Image as PILImage

        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
            Image as RLImage
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm

        styles = getSampleStyleSheet()

        H1 = ParagraphStyle("H1", parent=styles["Heading1"], alignment=1, spaceAfter=10)
        H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=6, spaceAfter=8)
        body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=6, leading=13)
        small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=6)
        caption = ParagraphStyle("cap", parent=styles["BodyText"], fontSize=9, leading=11, alignment=1, spaceAfter=10)
        mono_wrap = ParagraphStyle(
            "mono_wrap",
            parent=small,
            fontName="Courier",
            fontSize=7.5,
            leading=9,
            wordWrap="CJK",
            splitLongWords=1,
        )

        cover_title = ParagraphStyle("cover_title", parent=styles["Title"], alignment=1, fontSize=24, spaceAfter=18, leading=30)
        cover_subtitle = ParagraphStyle("cover_subtitle", parent=styles["Heading2"], alignment=1, fontSize=16, spaceAfter=12, leading=20)
        cover_meta = ParagraphStyle("cover_meta", parent=styles["Normal"], alignment=1, fontSize=10, textColor=colors.gray, spaceAfter=24)
        cover_desc = ParagraphStyle("cover_desc", parent=styles["Normal"], alignment=1, fontSize=12, leading=16, spaceAfter=0)

        report_data = report_data or {}
        deployment_data = deployment_data or {}
        report_data = report_data or {}
        deployment_data = deployment_data or {}
        def _set_pub_rcparams():
            plt.rcParams.update({
                "font.size": 9,
                "axes.titlesize": 11,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.linewidth": 0.8,
            })

        _set_pub_rcparams()

        def _style_axes(ax, grid_axis="y"):
            ax.set_axisbelow(True)
            if grid_axis in ("y", "both"):
                ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
            if grid_axis in ("x", "both"):
                ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_linewidth(0.8)
                ax.spines[s].set_alpha(0.7)
            ax.tick_params(axis="both", which="both", length=3, width=0.8)
            return ax

        def _save_fig_to_png(fig) -> str:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.close()
            fig.savefig(tmp.name, dpi=300, bbox_inches="tight", pad_inches=0.06, facecolor="white")
            plt.close(fig)
            return tmp.name

        def _fit_rl_image(img_path: str, max_w: float, max_h: float) -> RLImage:
            im = PILImage.open(img_path)
            w, h = im.size
            im.close()
            scale = min(max_w / float(w), max_h / float(h))
            return RLImage(img_path, width=w * scale, height=h * scale)

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

        def _convert_logo_to_png(logo_path: str) -> Optional[str]:
            try:
                img = PILImage.open(logo_path).convert("RGBA")
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.close()
                img.save(tmp.name, format="PNG")
                return tmp.name
            except Exception:
                return None

        def _chart_class_counts(class_counts: dict) -> Optional[str]:
            if not class_counts:
                return None

            keys = list(class_counts.keys())

            def _is_intlike(s):
                try:
                    int(str(s))
                    return True
                except Exception:
                    return False

            if all(_is_intlike(k) for k in keys):
                items = sorted(class_counts.items(), key=lambda x: int(str(x[0])))
            else:
                items = sorted(class_counts.items(), key=lambda x: str(x[0]).lower())

            labels = [str(k) for k, _ in items]
            values = [int(v) for _, v in items]
            total = sum(values)
            if total <= 0:
                return None

            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            x = np.arange(len(labels))
            bars = ax.bar(x, values, width=0.65)

            ax.set_title("Class distribution")
            ax.set_ylabel("Images")
            ax.set_xticks(x)

            max_len = max(len(l) for l in labels) if labels else 0
            rot = 0 if max_len <= 10 and len(labels) <= 12 else 30
            ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")

            _style_axes(ax, grid_axis="y")

            try:
                ax.bar_label(
                    bars,
                    labels=[f"{v} ({(v/total)*100:.1f}%)" for v in values],
                    padding=3,
                    fontsize=8
                )
            except Exception:
                pass

            fig.tight_layout()
            return _save_fig_to_png(fig)

        def _sample_resolution_pairs_for_report(train_loader, max_images=256):
            try:
                xb, _ = next(iter(train_loader))
                h = int(xb.shape[2])
                w = int(xb.shape[3])
                return {"post_transform_shape": f"{w}×{h}"}
            except Exception:
                return {}

        def _chart_resolution_scatter(res_stats: dict) -> Optional[str]:
            pairs = res_stats.get("sample_pairs", None) if isinstance(res_stats, dict) else None
            if not pairs:
                return None

            ws = np.array([p[0] for p in pairs], dtype=float)
            hs = np.array([p[1] for p in pairs], dtype=float)

            if ws.min() == ws.max() and hs.min() == hs.max():
                return None

            fig, ax = plt.subplots(figsize=(6.9, 4.0))
            ax.scatter(ws, hs, s=14, alpha=0.6)
            ax.set_title("Resolution scatter (sampled)")
            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Height (px)")
            _style_axes(ax, grid_axis="both")

            try:
                ax.scatter([ws.mean()], [hs.mean()], s=60, marker="x")
                ax.text(ws.mean(), hs.mean(), f"  mean≈{ws.mean():.0f}×{hs.mean():.0f}", va="center", fontsize=8)
            except Exception:
                pass

            fig.tight_layout()
            return _save_fig_to_png(fig)

        def _make_aug_grid_png(train_loader, max_images=12) -> Optional[str]:
            try:
                xb, yb = next(iter(train_loader))
            except Exception:
                return None

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            x = xb[:max_images].detach().cpu()
            if x.shape[1] == 3:
                x = x * std + mean
            x = x.clamp(0, 1)

            n = x.size(0)
            cols = 6
            rows = int(np.ceil(n / cols))

            fig = plt.figure(figsize=(cols * 2.05, rows * 2.05))
            for i in range(n):
                ax = plt.subplot(rows, cols, i + 1)
                img = x[i].permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.set_title(str(int(yb[i])), fontsize=9)
                ax.axis("off")

            fig.suptitle("Augmented training samples (Strategy B)", fontsize=12)
            fig.tight_layout()
            return _save_fig_to_png(fig)

        def _augmentation_explanation_paragraphs(strategy_b: dict) -> List[str]:
            chosen_family = str(strategy_b.get("family", "")).lower()
            chosen_level = strategy_b.get("level", "")

            mode_text = []
            mode_text.append(
                "<b>Augmentation modes (simple explanation):</b> "
                "Augmentations apply random, label-preserving changes to training images to reduce overfitting. "
                "They are <b>never</b> applied to validation/test to keep evaluation fair."
            )
            mode_text.append(
                "<b>• None:</b> No random changes. Useful when data is abundant or when you want maximum determinism, "
                "but it can overfit on small datasets."
            )
            mode_text.append(
                "<b>• Basic:</b> A light set of common transforms (typically horizontal flips and small random crops). "
                "Think “minor camera framing changes”."
            )
            mode_text.append(
                "<b>• TrivialAugmentWide:</b> Applies <b>one</b> randomly chosen transform (e.g., rotate, brightness, contrast) "
                "with random strength. Strong regularization without any tuning."
            )
            mode_text.append(
                "<b>• RandAugment:</b> Applies <b>N</b> random transforms sequentially, all with a shared strength <b>M</b>. "
                "This is stronger than Basic, but still cheap because it does not perform any search."
            )

            if chosen_family == "randaugment":
                mode_text.append(
                    f"<b>Chosen in this run:</b> <b>RandAugment</b> (level=<b>{chosen_level}</b>, "
                    f"N=<b>{strategy_b.get('randaugment_N')}</b>, M=<b>{strategy_b.get('randaugment_M')}</b>)."
                )
            elif chosen_family == "trivialaugment":
                mode_text.append(
                    f"<b>Chosen in this run:</b> <b>TrivialAugmentWide</b> (level=<b>{chosen_level}</b>)."
                )
            elif chosen_family == "basic":
                mode_text.append(
                    f"<b>Chosen in this run:</b> <b>Basic</b> (level=<b>{chosen_level}</b>)."
                )
            else:
                mode_text.append(
                    f"<b>Chosen in this run:</b> <b>{strategy_b.get('family','')}</b> (level=<b>{chosen_level}</b>)."
                )

            return mode_text

        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=3.1 * cm,
            bottomMargin=2.0 * cm
        )

        story = []
        tmp_files = []

        logo_png = None
        if logo_path and os.path.exists(logo_path):
            logo_png = _convert_logo_to_png(logo_path)
            if logo_png:
                tmp_files.append(logo_png)

        def draw_header_footer(canvas, doc_):
            canvas.saveState()
            header_top = doc_.pagesize[1] - 1.0 * cm
            header_bottom = doc_.pagesize[1] - doc_.topMargin + 0.25 * cm

            if logo_png:
                lw, lh = (1.25 * cm, 1.25 * cm)
                x = doc_.leftMargin
                y = header_top - lh
                canvas.drawImage(logo_png, x, y, width=lw, height=lh, preserveAspectRatio=True, mask="auto")

            title_y = doc_.pagesize[1] - 1.35 * cm
            if logo_png:
                title_y = header_top - (lh / 2.0) - (0.25 * cm)
            canvas.setFont("Helvetica-Bold", 12)
            canvas.drawCentredString(
                doc_.pagesize[0] / 2.0,
                title_y,
                project_name
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
                f"© {datetime.datetime.now().year} Automata AI — All rights reserved"
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
            "training results, optimization decisions, and export artifacts for image models.",
            cover_desc
        ))
        story.append(PageBreak())

        story.append(Paragraph("1. Dataset Overview", H2))

        class_counts = report.get("class_counts", {}) or {}
        counts_only = list(class_counts.values()) if class_counts else []
        imbalance_ratio = report.get("imbalance_ratio", None)
        if imbalance_ratio is None and counts_only:
            imbalance_ratio = (max(counts_only) / max(1, min(counts_only)))

        res = report.get("resolution_stats", {}) or {}
        sb = report.get("strategy_b", {}) or {}
        splits = report.get("splits", {}) or {}
        loader_cfg = report.get("loader", {}) or {}

        dataset_mode = report.get("dataset_mode", "")
        data_root = report.get("data_root", "")
        n_images = report.get("num_images", "")
        n_classes = report.get("num_classes", "")

        overview_rows = [
            ["Dataset mode", str(dataset_mode)],
            ["# Images", str(n_images)],
            ["# Classes", str(n_classes)],
            ["Imbalance ratio", f"{float(imbalance_ratio):.3f}" if imbalance_ratio is not None else ""],
            ["Chosen img_size", str(report.get("img_size", ""))],
            ["Normalization", str(report.get("normalization", ""))],
            ["Batch size", str(loader_cfg.get("batch_size", ""))],
            ["Num workers", str(loader_cfg.get("num_workers", ""))],
            ["Seed", str(splits.get("seed", report.get("seed", "")))],
        ]

        if "width_mean" in res:
            overview_rows += [
                ["Mean resolution (sampled)", f"{res.get('width_mean',0):.1f} × {res.get('height_mean',0):.1f}"],
                ["Median resolution (sampled)", f"{res.get('width_median',0):.1f} × {res.get('height_median',0):.1f}"],
                ["Resolution min/max (sampled)", f"{res.get('width_min','?')}–{res.get('width_max','?')} × {res.get('height_min','?')}–{res.get('height_max','?')}"],
                ["Unreadable in scan", f"{res.get('resolution_bad_count',0)} / {res.get('resolution_scan_limit',0)}"],
            ]

        if train_loader is not None:
            post_shape = _sample_resolution_pairs_for_report(train_loader)
            if post_shape.get("post_transform_shape"):
                overview_rows.append(["Post-transform tensor shape", post_shape["post_transform_shape"]])

        t = Table(overview_rows, colWidths=[7.5 * cm, 8.5 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

        chart_paths = []
        p1 = _chart_class_counts(class_counts)
        if p1:
            chart_paths.append(("Target class distribution", p1))

        p2 = _chart_resolution_scatter(res)
        if p2:
            chart_paths.append(("Resolution scatter (sampled)", p2))

        for title, p in chart_paths:
            tmp_files.append(p)
            story.append(_fit_rl_image(p, max_w=doc.width, max_h=8.2 * cm))
            story.append(Paragraph(title, caption))

        if "width_min" in res and "width_max" in res and "height_min" in res and "height_max" in res:
            if res["width_min"] == res["width_max"] and res["height_min"] == res["height_max"]:
                story.append(Paragraph(
                    f"All sampled images share the same resolution: <b>{res['width_min']}×{res['height_min']}</b>. "
                    "A scatter plot is omitted because it would be uninformative.",
                    small
                ))

        story.append(PageBreak())

        story.append(Paragraph("Configuration Snapshot", H2))

        cfg_rows = [
            ["STRONG_AUG_THRESHOLD", str(sb.get("strong_aug_threshold", ""))],
            ["MODERATE_AUG_THRESHOLD", str(sb.get("moderate_aug_threshold", ""))],
            ["RandAugment N", str(sb.get("randaugment_N", ""))],
            ["RandAugment M", str(sb.get("randaugment_M", ""))],
            ["Augmentation family", str(sb.get("family", ""))],
            ["Augmentation level", str(sb.get("level", ""))],
            ["Validation split", str(splits.get("val_split", ""))],
            ["Train / Val / Test samples", f"{splits.get('train_samples','')} / {splits.get('val_samples','')} / {splits.get('test_samples','')}"],
            ["Batch size", str(loader_cfg.get("batch_size", ""))],
            ["Num workers", str(loader_cfg.get("num_workers", ""))],
            ["Pin memory", str(loader_cfg.get("pin_memory", ""))],
        ]

        tc = Table(cfg_rows, colWidths=[7.5 * cm, 8.5 * cm])
        tc.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.97)]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tc)
        story.append(PageBreak())

        story.append(Paragraph("2. Preprocessing Steps Applied", H2))

        story.append(Paragraph(
            "<b>• Input decoding & integrity:</b> Images are loaded and decoded into a consistent in-memory representation. "
            "For file-based datasets, unreadable images can silently break training; therefore, the pipeline optionally performs "
            "a sampled scan to detect decode failures early.",
            body
        ))

        story.append(Paragraph(
            f"<b>• Standardization (shape):</b> All images are resized to <b>{report.get('img_size','')}</b> × "
            f"<b>{report.get('img_size','')}</b>. This produces fixed tensor shapes, stable batching, and predictable compute cost "
            "across datasets—important when comparing architectures fairly in NAS.",
            body
        ))

        story.append(Paragraph(
            f"<b>• Tensor conversion & normalization:</b> Images are converted to floating-point tensors and normalized using "
            f"<b>{report.get('normalization','')}</b> statistics. Normalization stabilizes optimization and makes training behavior "
            "more consistent across datasets and architectures (especially when using pretrained backbones).",
            body
        ))

        for p in _augmentation_explanation_paragraphs(sb):
            story.append(Paragraph(p, body if p.startswith("<b>•") else small))

        story.append(Paragraph(
            f"<b>• Splitting & evaluation fairness:</b> The dataset is split into train/validation/test "
            f"(<b>{splits.get('train_samples','')}</b> / <b>{splits.get('val_samples','')}</b> / <b>{splits.get('test_samples','')}</b>). "
            "Augmentations are applied to training batches only. Validation and test pipelines remain deterministic to ensure "
            "fair comparison between candidate architectures.",
            body
        ))

        story.append(Paragraph(
            f"<b>• DataLoader settings:</b> batch_size=<b>{loader_cfg.get('batch_size','')}</b>, "
            f"num_workers=<b>{loader_cfg.get('num_workers','')}</b>, pin_memory=<b>{loader_cfg.get('pin_memory','')}</b>. "
            "These settings control input throughput and help keep the GPU utilized during training.",
            body
        ))

        story.append(PageBreak())

        story.append(Paragraph("3. Visual Sanity Check", H2))
        story.append(Paragraph(
            "The grid below shows a sample of training images after the selected Strategy B augmentation policy. "
            "This is a quick sanity check that augmentations are label-preserving and not overly destructive.",
            body
        ))

        if train_loader is not None:
            grid = _make_aug_grid_png(train_loader, max_images=12)
            if grid:
                tmp_files.append(grid)
                story.append(_fit_rl_image(grid, max_w=doc.width, max_h=16 * cm))
                story.append(Paragraph("Augmented samples (train loader)", caption))
            else:
                story.append(Paragraph("Could not render augmented sample grid (train_loader unavailable or empty).", small))
        else:
            story.append(Paragraph("No train_loader provided; skipping sample grid.", small))

        if report_data:
            story.append(PageBreak())
            story.append(Paragraph("4. Model Report", H2))

            ds_name = str(report_data.get("dataset_name", ""))
            fam_name = str(report_data.get("device_family_name", ""))
            fam_id = str(report_data.get("device_family_id", ""))
            saved_as = str(report_data.get("saved_as", ""))
            attempted = report_data.get("attempted_exts", []) or []
            exported_paths = report_data.get("exported_paths", []) or []
            exported_exts = sorted({os.path.splitext(str(p))[1].lower() for p in exported_paths if p})

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
                [_p("Quantization applied"), _p(report_data.get("quantization_applied", "")), _p("Export requested"), _p(report_data.get("export_ext_requested", ""))],
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
            exported_exts = sorted({os.path.splitext(str(p))[1].lower() for p in exported_paths if p})

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
            story.append(Paragraph("General Deployment Steps", H2))
            steps = [
                "Copy the exported model into your firmware/application project.",
                "Apply the same preprocessing steps used during training (see this report).",
                "Load the model with a runtime supported by your device family.",
                "Run a single inference to validate end-to-end integration.",
                "Map output indices/probabilities to target labels.",
            ]
            for i, s in enumerate(steps, 1):
                story.append(Paragraph(f"<b>{i}.</b> {s}", body))

            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Minimal Example Usage", H2))
            code_lines = [
                "# adapt to your runtime/framework",
                "model = load_model(\"exported_model\")",
                "raw_input = read_image()",
                "input_data = preprocess(raw_input)  # same steps as in report",
                "output = model.infer(input_data)",
                "label = labels[argmax(output)]",
                "print(label)",
            ]
            code_txt = "<br/>".join(code_lines)
            code_block = Table([[Paragraph(code_txt, mono_wrap)]], colWidths=[doc.width])
            code_block.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), colors.Color(0.96, 0.96, 0.96)),
                ("BOX", (0,0), (-1,-1), 0.3, colors.grey),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.append(code_block)

            labels = report_data.get("class_labels", []) or []
            if labels:
                label_preview = ", ".join([str(x) for x in labels[:30]])
                if len(labels) > 30:
                    label_preview += " ..."
                story.append(Spacer(1, 0.2 * cm))
                story.append(Paragraph(f"<b>Target labels:</b> {label_preview}", body))


        doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)

        for f in tmp_files:
            try:
                os.remove(f)
            except Exception:
                pass

        print(f"[INFO] Image preprocessing report saved to {path}")

    # ==========================================================
    # Final validation 
    # ==========================================================

    def _phase_validate_final(self) -> None:
        if self._final_model is None:
            raise RuntimeError("No final model available for validation.")
        if not self._final_model_path or not os.path.isfile(self._final_model_path):
            raise RuntimeError("Final model file not found for validation.")

        test_acc = self.state.metrics.get("test_acc", None)
        if test_acc is None:
            raise RuntimeError("Missing test_acc metric; cannot validate minimum accuracy.")

        size_mb = float(self.state.metrics.get("final_model_size_mb") or 1e9)

        ok, reason, metrics_patch = validate_final_against_specs(
            test_acc=float(test_acc),
            min_accuracy=float(self.min_accuracy),
            model_file_size_mb=float(size_mb),
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
