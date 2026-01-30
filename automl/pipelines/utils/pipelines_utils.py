from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple



DEVICE_FAMILIES: List[Dict[str, Any]] = [
    {
        "id": "mcu_ultra_low",
        "name": "Ultra-Low-Power MCUs",
        "note": "≤256 KB RAM, ≤2 MB Flash; always-on sensing",
        "specs_hint": {
            "ramKB": [2, 256],
            "flashMB": [0.025, 2],
            "cpuMHz": [12, 120],
        },
        "frameworks": ["TFLite-Micro", "CMSIS-NN"],
        "model_exts": [".tflite", ".bin", ".h"],
    },
    {
        "id": "mcu_mid_dsp",
        "name": "Mid-Range MCUs (DSP/SIMD)",
        "note": "Up to 1 MB RAM, 8 MB Flash",
        "specs_hint": {
            "ramKB": [256, 1024],
            "flashMB": [1, 8],
            "cpuMHz": [80, 240],
        },
        "frameworks": ["TFLite-Micro", "CMSIS-NN"],
        "model_exts": [".tflite", ".bin", ".h"],
    },
    {
        "id": "mcu_ai_high",
        "name": "High-End / AI MCUs",
        "note": "Dual-core options; up to ~2 MB RAM",
        "specs_hint": {
            "ramKB": [512, 2048],
            "flashMB": [2, 16],
            "cpuMHz": [160, 600],
        },
        "frameworks": ["TFLite-Micro", "CMSIS-NN", "microTVM"],
        "model_exts": [".tflite", ".bin", ".h"],
    },
    {
        "id": "mcu_riscv_npu",
        "name": "MCUs with NPU (RISC-V class)",
        "note": "On-die KPU/NPU; SRAM in MBs",
        "specs_hint": {
            "ramKB": [2048, 8192],
            "flashMB": [8, 16],
            "cpuMHz": [400, 800],
        },
        "frameworks": ["TFLite-Micro", "uTensor"],
        "model_exts": [".kmodel", ".tflite"],
    },
    {
        "id": "sbc_light",
        "name": "Lightweight SBCs",
        "note": "Linux-capable; up to 4 GB RAM",
        "specs_hint": {
            "ramKB": [262144, 4194304],
            "flashMB": [16, 64],
            "cpuMHz": [1000, 1800],
        },
        "frameworks": ["TFLite", "ONNX Runtime"],
        "model_exts": [".tflite", ".onnx"],
    },
    {
        "id": "sbc_gpu_npu",
        "name": "SBCs with GPU/NPU",
        "note": "Jetson/Coral class edge AI",
        "specs_hint": {
            "ramKB": [1048576, 8388608],
            "flashMB": [16, 128],
            "cpuMHz": [1200, 2500],
        },
        "frameworks": ["TensorRT", "TFLite", "ONNX Runtime"],
        "model_exts": [".engine", ".tflite", ".onnx"],
    },
    {
        "id": "audio_always_on",
        "name": "Always-On Audio MCUs",
        "note": "Keyword spotting; ultra low power",
        "specs_hint": {
            "ramKB": [128, 512],
            "flashMB": [1, 4],
            "cpuMHz": [32, 160],
        },
        "frameworks": ["TFLite-Micro", "CMSIS-NN"],
        "model_exts": [".tflite", ".bin", ".h"],
    },
    {
        "id": "imu_vibration",
        "name": "IMU/Vibration Sensing MCUs",
        "note": "Predictive maintenance; tiny models",
        "specs_hint": {
            "ramKB": [64, 512],
            "flashMB": [1, 4],
            "cpuMHz": [32, 240],
        },
        "frameworks": ["TFLite-Micro", "CMSIS-NN"],
        "model_exts": [".tflite", ".bin", ".h"],
    },
]

DEVICE_FAMILY_BY_ID: Dict[str, Dict[str, Any]] = {f["id"]: f for f in DEVICE_FAMILIES}

STATUS_ORDER = ["queued", "preprocessing", "training", "optimizing", "packaging", "completed", "failed"]

LOGO_PATH = r"automl\pipelines\utils\logo\Automata_AI_Logo.webp"

# ============================================================================
# Family getters
# ============================================================================

def get_family(device_family_id: str) -> Optional[Dict[str, Any]]:
    return DEVICE_FAMILY_BY_ID.get(device_family_id)


def get_family_name(device_family_id: str) -> str:
    fam = get_family(device_family_id)
    return fam["name"] if fam else device_family_id


def family_allowed_formats(device_family_id: str) -> List[str]:
    fam = get_family(device_family_id) or {}
    exts = fam.get("model_exts", [])
    return list(exts) if isinstance(exts, list) else []


def family_frameworks(device_family_id: str) -> List[str]:
    fam = get_family(device_family_id) or {}
    fw = fam.get("frameworks", [])
    return list(fw) if isinstance(fw, list) else []


# ============================================================================
# Device specs normalization + validation 
# ============================================================================

def normalize_device_specs(device_specs: Dict[str, Any]) -> Dict[str, Any]:

    if device_specs is None:
        device_specs = {}

    out = dict(device_specs)

    if "ramKB" in out and "ram_kb" not in out:
        out["ram_kb"] = out["ramKB"]
    if "flashMB" in out and "flash_mb" not in out:
        out["flash_mb"] = out["flashMB"]
    if "cpuMHz" in out and "cpu_mhz" not in out:
        out["cpu_mhz"] = out["cpuMHz"]

    if "ram_kb" in out:
        out["ram_kb"] = int(out["ram_kb"])
    if "flash_mb" in out:
        out["flash_mb"] = float(out["flash_mb"])
    if "cpu_mhz" in out:
        out["cpu_mhz"] = int(out["cpu_mhz"])

    return out


def validate_specs_within_family(device_family_id: str, device_specs: Dict[str, Any]) -> Tuple[bool, str]:

    fam = get_family(device_family_id)
    if fam is None:
        return False, f"Unknown device_family_id='{device_family_id}'."

    specs = normalize_device_specs(device_specs)

    for k in ("ram_kb", "flash_mb", "cpu_mhz"):
        if k not in specs:
            return False, f"device_specs missing required key '{k}'."

    ram_kb = int(specs["ram_kb"])
    flash_mb = float(specs["flash_mb"])
    cpu_mhz = int(specs["cpu_mhz"])

    if ram_kb <= 0:
        return False, "device_specs.ram_kb must be > 0."
    if flash_mb <= 0:
        return False, "device_specs.flash_mb must be > 0."
    if cpu_mhz <= 0:
        return False, "device_specs.cpu_mhz must be > 0."

    hint = fam.get("specs_hint", {})
    ram_rng = hint.get("ramKB")
    flash_rng = hint.get("flashMB")
    cpu_rng = hint.get("cpuMHz")

    if isinstance(ram_rng, list) and len(ram_rng) == 2:
        lo, hi = float(ram_rng[0]), float(ram_rng[1])
        if not (lo <= float(ram_kb) <= hi):
            return False, f"RAM {ram_kb}KB outside family range [{lo}, {hi}]KB."

    if isinstance(flash_rng, list) and len(flash_rng) == 2:
        lo, hi = float(flash_rng[0]), float(flash_rng[1])
        if not (lo <= float(flash_mb) <= hi):
            return False, f"Flash {flash_mb}MB outside family range [{lo}, {hi}]MB."

    if isinstance(cpu_rng, list) and len(cpu_rng) == 2:
        lo, hi = float(cpu_rng[0]), float(cpu_rng[1])
        if not (lo <= float(cpu_mhz) <= hi):
            return False, f"CPU {cpu_mhz}MHz outside family range [{lo}, {hi}]MHz."

    return True, "OK"


# ============================================================================
# Shared final validation helpers 
# ============================================================================

def bytes_to_kb(x: int) -> float:
    return float(x) / 1024.0


def estimate_param_bytes_torch(model) -> int:

    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def ram_proxy_kb_from_torch_model(model, overhead_factor: float = 1.20) -> Tuple[float, float]:

    param_bytes = estimate_param_bytes_torch(model)
    param_kb = bytes_to_kb(param_bytes)
    total_proxy_kb = float(param_kb) * float(overhead_factor)
    return float(param_kb), float(total_proxy_kb)


def validate_final_against_specs(
    *,
    test_acc: float,
    min_accuracy: float,
    model_file_size_mb: float,
    torch_model_obj,
    device_specs: Dict[str, Any],
    overhead_factor: float = 1.20,
) -> Tuple[bool, str, Dict[str, Any]]:

    specs = normalize_device_specs(device_specs)
    ram_kb_limit = float(specs["ram_kb"])
    flash_mb_limit = float(specs["flash_mb"])

    metrics_patch: Dict[str, Any] = {
        "test_acc": float(test_acc),
        "min_accuracy": float(min_accuracy),
        "final_model_size_mb": float(model_file_size_mb),
        "flash_mb_limit": float(flash_mb_limit),
        "ram_kb_limit": float(ram_kb_limit),
        "ram_overhead_factor": float(overhead_factor),
    }

    if float(test_acc) < float(min_accuracy):
        return False, f"Accuracy {test_acc:.4f} < min_accuracy {min_accuracy:.2f}", metrics_patch

    # if float(model_file_size_mb) > float(flash_mb_limit):
    #     return (
    #         False,
    #         f"Model size ({model_file_size_mb:.3f} MB) is too large for the selected device's flash memory ({flash_mb_limit:.3f} MB).",
    #         metrics_patch,
    #     )

    param_kb, total_proxy_kb = ram_proxy_kb_from_torch_model(torch_model_obj, overhead_factor=overhead_factor)
    metrics_patch["estimated_param_ram_kb"] = float(param_kb)
    metrics_patch["estimated_total_ram_kb_proxy"] = float(total_proxy_kb)

    if float(total_proxy_kb) > float(ram_kb_limit):
        return (
            False,
            f"Estimated RAM proxy {total_proxy_kb:.1f}KB exceeds RAM limit {ram_kb_limit:.1f}KB "
            f"(param_kb={param_kb:.1f}, overhead_factor={overhead_factor})",
            metrics_patch,
        )

    return True, "OK", metrics_patch


# ============================================================================
# SENSOR PIPELINE CONFIGURATION
# ============================================================================

MODEL_IDS = {
    "logreg": 1,
    "rf": 2,
    "xgboost": 3,
    "cnn1d": 4,
    "tiny_rnn": 5,
    "mlp": 6,
    "tinyconv": 7,
}

DL_MAX_EPOCHS = 20
DL_BATCH_SIZE = 128

MODEL_CAPABILITIES = {
    "logreg": {
        "model_id": MODEL_IDS["logreg"],
        "model_name": "logreg",
        "is_deep_learning": False,
        "is_tree_based": False,
        "is_linear": True,
        "model_family": "Linear",
        "parameterization_type": "linear-in-features",
        "complexity_training_big_o": "O(n · d)",
        "complexity_inference_big_o": "O(d)",
        "is_probabilistic": True,
        "is_ensemble_model": False,
        "regularization_supported": "L2",
        "supports_multiclass_natively": True,
        "supports_online_learning": False,
        "supports_multiple_trees": False,
        "tree_growth_strategy": "none",
        "default_max_depth": 0,
        "supports_pruning": False,
        "splitting_criterion": "none",
        "architecture_type": "none",
        "supports_dropout": False,
        "supports_batchnorm": False,
        "default_activation": "none",
        "supports_cuda_acceleration": False,
        "supports_non_linearity": False,
        "supports_categorical_directly": False,
        "supports_missing_values": False,
        "supports_gpu": False,
        "n_estimators": 0,
        "avg_tree_depth": 0.0,
        "max_tree_depth": 0,
        "n_leaves_mean": 0.0,
        "n_layers": 0,
        "hidden_units_mean": 0.0,
        "dropout_rate_mean": 0.0,
        "activation_type": "none",
        "batch_size": 0,
        "epochs": 0,
    },
    "rf": {
        "model_id": MODEL_IDS["rf"],
        "model_name": "rf",
        "is_deep_learning": False,
        "is_tree_based": True,
        "is_linear": False,
        "model_family": "TreeEnsemble",
        "parameterization_type": "fixed-per-estimator",
        "complexity_training_big_o": "O(n · log n · trees)",
        "complexity_inference_big_o": "O(trees · depth)",
        "is_probabilistic": True,
        "is_ensemble_model": True,
        "regularization_supported": "None",
        "supports_multiclass_natively": True,
        "supports_online_learning": False,
        "supports_multiple_trees": True,
        "tree_growth_strategy": "depth-based",
        "default_max_depth": 0,
        "supports_pruning": False,
        "splitting_criterion": "gini",
        "architecture_type": "none",
        "supports_dropout": False,
        "supports_batchnorm": False,
        "default_activation": "none",
        "supports_cuda_acceleration": False,
        "supports_non_linearity": True,
        "supports_categorical_directly": False,
        "supports_missing_values": False,
        "supports_gpu": False,
        "n_estimators": 200,
        "avg_tree_depth": 0.0,
        "max_tree_depth": 0,
        "n_leaves_mean": 0.0,
        "n_layers": 0,
        "hidden_units_mean": 0.0,
        "dropout_rate_mean": 0.0,
        "activation_type": "none",
        "batch_size": 0,
        "epochs": 0,
    },
    "xgboost": {
        "model_id": MODEL_IDS["xgboost"],
        "model_name": "xgboost",
        "is_deep_learning": False,
        "is_tree_based": True,
        "is_linear": False,
        "model_family": "BoostedTrees",
        "parameterization_type": "fixed-per-estimator",
        "complexity_training_big_o": "O(n · log n · trees)",
        "complexity_inference_big_o": "O(trees · depth)",
        "is_probabilistic": True,
        "is_ensemble_model": True,
        "regularization_supported": "L1/L2",
        "supports_multiclass_natively": True,
        "supports_online_learning": False,
        "supports_multiple_trees": True,
        "tree_growth_strategy": "leaf-based",
        "default_max_depth": 6,
        "supports_pruning": True,
        "splitting_criterion": "gain",
        "architecture_type": "none",
        "supports_dropout": False,
        "supports_batchnorm": False,
        "default_activation": "none",
        "supports_cuda_acceleration": True,
        "supports_non_linearity": True,
        "supports_categorical_directly": False,
        "supports_missing_values": True,
        "supports_gpu": True,
        "n_estimators": 200,
        "avg_tree_depth": 6.0,
        "max_tree_depth": 6,
        "n_leaves_mean": 0.0,
        "n_layers": 0,
        "hidden_units_mean": 0.0,
        "dropout_rate_mean": 0.0,
        "activation_type": "none",
        "batch_size": 0,
        "epochs": 0,
    },
    "cnn1d": {
        "model_id": MODEL_IDS["cnn1d"],
        "model_name": "cnn1d",
        "is_deep_learning": True,
        "is_tree_based": False,
        "is_linear": False,
        "model_family": "CNN",
        "parameterization_type": "linear-in-features",
        "complexity_training_big_o": "O(n · d · epochs)",
        "complexity_inference_big_o": "O(d · filters)",
        "is_probabilistic": True,
        "is_ensemble_model": False,
        "regularization_supported": "L2",
        "supports_multiclass_natively": True,
        "supports_online_learning": False,
        "supports_multiple_trees": False,
        "tree_growth_strategy": "none",
        "default_max_depth": 0,
        "supports_pruning": False,
        "splitting_criterion": "none",
        "architecture_type": "CNN1D",
        "supports_dropout": False,
        "supports_batchnorm": False,
        "default_activation": "relu",
        "supports_cuda_acceleration": True,
        "supports_non_linearity": True,
        "supports_categorical_directly": False,
        "supports_missing_values": False,
        "supports_gpu": True,
        "n_estimators": 0,
        "avg_tree_depth": 0.0,
        "max_tree_depth": 0,
        "n_leaves_mean": 0.0,
        "n_layers": 2,
        "hidden_units_mean": 8.0,
        "dropout_rate_mean": 0.0,
        "activation_type": "relu",
        "batch_size": DL_BATCH_SIZE,
        "epochs": DL_MAX_EPOCHS,
    },
    "tiny_rnn": {
        "model_id": MODEL_IDS["tiny_rnn"],
        "model_name": "tiny_rnn",
        "is_deep_learning": True,
        "is_tree_based": False,
        "is_linear": False,
        "model_family": "RNN",
        "parameterization_type": "linear-in-features",
        "complexity_training_big_o": "O(n · d · hidden_dim · epochs)",
        "complexity_inference_big_o": "O(d · hidden_dim)",
        "is_probabilistic": True,
        "is_ensemble_model": False,
        "regularization_supported": "L2",
        "supports_multiclass_natively": True,
        "supports_online_learning": False,
        "supports_multiple_trees": False,
        "tree_growth_strategy": "none",
        "default_max_depth": 0,
        "supports_pruning": False,
        "splitting_criterion": "none",
        "architecture_type": "RNN-GRU",
        "supports_dropout": False,
        "supports_batchnorm": False,
        "default_activation": "tanh",
        "supports_cuda_acceleration": True,
        "supports_non_linearity": True,
        "supports_categorical_directly": False,
        "supports_missing_values": False,
        "supports_gpu": True,
        "n_estimators": 0,
        "avg_tree_depth": 0.0,
        "max_tree_depth": 0,
        "n_leaves_mean": 0.0,
        "n_layers": 2,
        "hidden_units_mean": 32.0,
        "dropout_rate_mean": 0.0,
        "activation_type": "tanh",
        "batch_size": DL_BATCH_SIZE,
        "epochs": DL_MAX_EPOCHS,
    },
    "mlp": {
        "model_id": MODEL_IDS["mlp"],
        "model_name": "mlp",
        "is_deep_learning": True,
        "is_tree_based": False,
        "is_linear": False,
        "model_family": "MLP",
        "parameterization_type": "linear-in-features",
        "complexity_training_big_o": "O(n · Σ(layer_dims) · epochs)",
        "complexity_inference_big_o": "O(Σ(layer_dims))",
        "is_probabilistic": True,
        "is_ensemble_model": False,
        "regularization_supported": "L2",
        "supports_multiclass_natively": True,
        "supports_online_learning": False,
        "supports_multiple_trees": False,
        "tree_growth_strategy": "none",
        "default_max_depth": 0,
        "supports_pruning": False,
        "splitting_criterion": "none",
        "architecture_type": "MLP",
        "supports_dropout": False,
        "supports_batchnorm": False,
        "default_activation": "relu",
        "supports_cuda_acceleration": True,
        "supports_non_linearity": True,
        "supports_categorical_directly": False,
        "supports_missing_values": False,
        "supports_gpu": True,
        "n_estimators": 0,
        "avg_tree_depth": 0.0,
        "max_tree_depth": 0,
        "n_leaves_mean": 0.0,
        "n_layers": 3,
        "hidden_units_mean": (128.0 + 64.0) / 2.0,
        "dropout_rate_mean": 0.0,
        "activation_type": "relu",
        "batch_size": DL_BATCH_SIZE,
        "epochs": DL_MAX_EPOCHS,
    },
    "tinyconv": {
        "model_id": MODEL_IDS["tinyconv"],
        "model_name": "tinyconv",
        "is_deep_learning": True,
        "is_tree_based": False,
        "is_linear": False,
        "model_family": "CNN",
        "parameterization_type": "linear-in-features",
        "complexity_training_big_o": "O(n · d · epochs)",
        "complexity_inference_big_o": "O(d · filters)",
        "is_probabilistic": True,
        "is_ensemble_model": False,
        "regularization_supported": "L2",
        "supports_multiclass_natively": True,
        "supports_online_learning": False,
        "supports_multiple_trees": False,
        "tree_growth_strategy": "none",
        "default_max_depth": 0,
        "supports_pruning": False,
        "splitting_criterion": "none",
        "architecture_type": "CNN1D",
        "supports_dropout": False,
        "supports_batchnorm": False,
        "default_activation": "relu",
        "supports_cuda_acceleration": True,
        "supports_non_linearity": True,
        "supports_categorical_directly": False,
        "supports_missing_values": False,
        "supports_gpu": True,
        "n_estimators": 0,
        "avg_tree_depth": 0.0,
        "max_tree_depth": 0,
        "n_leaves_mean": 0.0,
        "n_layers": 2,
        "hidden_units_mean": 4.0,
        "dropout_rate_mean": 0.0,
        "activation_type": "relu",
        "batch_size": DL_BATCH_SIZE,
        "epochs": DL_MAX_EPOCHS,
    },
}

KEY_ORDER = [
    "Task_id",
    "dataset_id",
    "dataset_name",
    "n_samples",
    "n_features",
    "n_numeric_features",
    "n_categorical_features",
    "n_binary_features",
    "n_classes",
    "class_balance_std",
    "class_entropy",
    "mean_feature_variance",
    "median_feature_variance",
    "mean_corr_abs",
    "max_corr_abs",
    "feature_skewness_mean",
    "feature_kurtosis_mean",
    "missing_percentage",
    "avg_cardinality_categorical",
    "complexity_ratio",
    "intrinsic_dim_estimate",
    "landmark_lr_accuracy",
    "landmark_dt_depth3_accuracy",
    "landmark_knn3_accuracy",
    "landmark_random_noise_accuracy",
    "fisher_discriminant_ratio",
    "model_id",
    "model_name",
    "model_family",
    "is_deep_learning",
    "is_tree_based",
    "is_linear",
    "parameterization_type",
    "complexity_training_big_o",
    "complexity_inference_big_o",
    "is_probabilistic",
    "is_ensemble_model",
    "regularization_supported",
    "supports_multiclass_natively",
    "supports_online_learning",
    "supports_multiple_trees",
    "tree_growth_strategy",
    "default_max_depth",
    "supports_pruning",
    "splitting_criterion",
    "architecture_type",
    "supports_dropout",
    "supports_batchnorm",
    "default_activation",
    "supports_cuda_acceleration",
    "supports_non_linearity",
    "supports_categorical_directly",
    "supports_missing_values",
    "supports_gpu",
    "n_estimators",
    "avg_tree_depth",
    "max_tree_depth",
    "n_leaves_mean",
    "n_layers",
    "hidden_units_mean",
    "dropout_rate_mean",
    "activation_type",
    "batch_size",
    "epochs",
    "accuracy",
    "f1_macro",
    "precision_macro",
    "trained_model_size_kb",
    "inference_speed_ms",
    "static_usage_ram_kb",
    "dynamic_usage_ram_kb",
    "full_ram_usage_kb",
    "model_n_parameters",
]

STRING_KEYS = {
    "dataset_name",
    "model_name",
    "model_family",
    "parameterization_type",
    "complexity_training_big_o",
    "complexity_inference_big_o",
    "regularization_supported",
    "tree_growth_strategy",
    "splitting_criterion",
    "architecture_type",
    "default_activation",
    "activation_type",
}

LANDMARK_KEYS = [
    "landmark_lr_accuracy",
    "landmark_dt_depth3_accuracy",
    "landmark_knn3_accuracy",
    "landmark_random_noise_accuracy",
    "fisher_discriminant_ratio",
]
