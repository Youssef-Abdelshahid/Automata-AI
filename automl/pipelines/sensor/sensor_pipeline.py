from __future__ import annotations

import copy
import os
import time
import json
import shutil
import random
import tempfile
import traceback
import warnings
import logging
import re
import joblib
import importlib
import datetime
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple, List, Callable
import m2cgen
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image as PILImage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, LongTable
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from automl.pipelines.utils.pipelines_utils import (
    validate_specs_within_family,
    validate_final_against_specs,
    get_family_name,
    family_allowed_formats,
    family_frameworks,
    normalize_device_specs,
    MODEL_CAPABILITIES,
    KEY_ORDER,
    STRING_KEYS,
    STATUS_ORDER,
    LOGO_PATH,
)

logger = logging.getLogger("AutomataPreprocessor")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


@dataclass
class PreprocessingConfig:
    drop_missing_threshold: float = 0.8
    high_cardinality_threshold: float = 0.2  
    high_cardinality_min_unique: int = 15

    numeric_imputer: str = "median"          # 'mean','median','most_frequent'
    numeric_scaler: str = "standard"         # 'standard','minmax','robust','none'

    categorical_imputer: str = "most_frequent"  # 'most_frequent','constant'

    ohe_min_frequency: Optional[float] = None   
    ohe_max_categories: Optional[int] = None

    high_cardinality_encoder: str = "frequency" # currently: 'frequency'

    datetime_handling: str = "drop"  # 'drop' | 'extract'
    datetime_extract_parts: Tuple[str, ...] = ("year", "month", "day", "dayofweek")

    feature_selection: str = "auto"          # 'auto','none','mutual_info'
    feature_fraction: float = 0.75

    balancing: str = "class_weight"          # 'none','class_weight'
    imbalance_threshold: float = 1.5         # apply class_weight only if imbalance_ratio > this


class DatasetAnalyzer:
    DATE_REGEXES = [
        re.compile(r"^\d{4}-\d{2}-\d{2}"),        # YYYY-MM-DD
        re.compile(r"^\d{2}/\d{2}/\d{4}"),        # MM/DD/YYYY
        re.compile(r"^\d{4}/\d{2}/\d{2}"),        # YYYY/MM/DD
        re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}"),  # 01-Jan-2020
        re.compile(r"^\d{8}$"),                   # YYYYMMDD
    ]

    def _looks_like_datetime(self, series: pd.Series, sample_n: int = 50) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        s = series.dropna().astype(str)
        if s.shape[0] == 0:
            return False

        sample = s.head(sample_n)

        hit = 0
        for v in sample:
            for rx in self.DATE_REGEXES:
                if rx.search(v):
                    hit += 1
                    break
        if (hit / max(len(sample), 1)) > 0.6:
            return True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format")
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.8

    def infer_column_types(self, X: pd.DataFrame) -> Dict[str, str]:
        types: Dict[str, str] = {}
        for col in X.columns:
            s = X[col]
            if pd.api.types.is_bool_dtype(s):
                types[col] = "categorical"
            elif pd.api.types.is_numeric_dtype(s):
                types[col] = "numeric"
            elif self._looks_like_datetime(s):
                types[col] = "datetime"
            else:
                types[col] = "categorical"
        return types

    def compute_meta(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        n_rows, n_cols = X.shape
        col_types = self.infer_column_types(X)
        missing_frac = X.isna().mean().to_dict()
        cardinality = X.nunique(dropna=True).to_dict()

        meta: Dict[str, Any] = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "col_types": col_types,
            "missing_frac": missing_frac,
            "cardinality": cardinality,
        }

        if y is not None:
            y_s = pd.Series(y)
            counts = y_s.value_counts().to_dict()
            if counts:
                majority = max(counts.values())
                minority = min(counts.values())
                imbalance_ratio = majority / max(1, minority)
            else:
                counts = {}
                imbalance_ratio = 1.0
            meta["class_counts"] = counts
            meta["imbalance_ratio"] = imbalance_ratio

        return meta


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, unseen_value: float = 0.0):
        self.unseen_value = unseen_value
        self.mappings_: List[Dict[Any, float]] = []
        self.n_features_in_: Optional[int] = None

    def fit(self, X, y=None):
        arr = self._to_2d(X)
        self.n_features_in_ = arr.shape[1]
        self.mappings_ = []
        for j in range(arr.shape[1]):
            col = pd.Series(arr[:, j])
            self.mappings_.append(col.value_counts(normalize=True).to_dict())
        return self

    def transform(self, X):
        arr = self._to_2d(X)
        out = np.zeros((arr.shape[0], arr.shape[1]), dtype=float)
        for j in range(arr.shape[1]):
            mapping = self.mappings_[j]
            out[:, j] = pd.Series(arr[:, j]).map(mapping).fillna(self.unseen_value).to_numpy(dtype=float)
        return out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            n = self.n_features_in_ or 0
            return [f"x{j}_freq" for j in range(n)]
        return [f"{name}_freq" for name in input_features]

    @staticmethod
    def _to_2d(X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.Series):
            return X.to_numpy().reshape(-1, 1)
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


def _make_ohe(cfg: PreprocessingConfig):
    base = {"handle_unknown": "ignore"}
    candidates = []

    kw = dict(base)
    kw["sparse_output"] = False
    if cfg.ohe_min_frequency is not None:
        kw["min_frequency"] = cfg.ohe_min_frequency
    if cfg.ohe_max_categories is not None:
        kw["max_categories"] = cfg.ohe_max_categories
    candidates.append(kw)

    kw = dict(base)
    kw["sparse"] = False
    if cfg.ohe_min_frequency is not None:
        kw["min_frequency"] = cfg.ohe_min_frequency
    if cfg.ohe_max_categories is not None:
        kw["max_categories"] = cfg.ohe_max_categories
    candidates.append(kw)

    candidates.append({**base, "sparse_output": False})
    candidates.append({**base, "sparse": False})
    candidates.append(base)

    last_err = None
    for c in candidates:
        try:
            return OneHotEncoder(**c)
        except TypeError as e:
            last_err = e
    raise last_err


def build_rule_based_config(meta: Dict[str, Any], cfg: Optional[PreprocessingConfig] = None) -> Tuple[PreprocessingConfig, Dict[str, Any]]:
    cfg = cfg or PreprocessingConfig()
    cfg = PreprocessingConfig(**asdict(cfg))

    if cfg.feature_selection == "auto":
        cfg.feature_selection = "mutual_info" if meta.get("n_cols", 0) > 30 else "none"

    drop_cols: List[str] = []
    for col, frac in meta.get("missing_frac", {}).items():
        if frac > cfg.drop_missing_threshold:
            drop_cols.append(col)
    for col, c in meta.get("cardinality", {}).items():
        if c <= 1 and col not in drop_cols:
            drop_cols.append(col)

    return cfg, {"drop_cols": drop_cols}


class AutomataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        strategy: str = "auto",
        verbose: bool = False,

        report: bool = False,
        report_path: str = "prep_report.pdf",
        project_name: str = "Automata AI Project",
        author: str = "",
        logo_path: Optional[str] = None,
    ):
        self.config = config
        self.strategy = strategy
        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.DEBUG)

        self.report = report
        self.report_path = report_path
        self.project_name = project_name
        self.author = author
        self.logo_path = logo_path

        self.report_generator_ref_: Optional[Tuple[str, str]] = None

        self.analyzer_ = DatasetAnalyzer()

        self.meta_: Optional[Dict[str, Any]] = None
        self.meta_used_: Optional[Dict[str, Any]] = None

        self.config_: Optional[PreprocessingConfig] = None
        self.aux_: Optional[Dict[str, Any]] = None
        self.pipeline_: Optional[Pipeline] = None

        self.feature_names_in_: Optional[List[str]] = None
        self.output_feature_names_: Optional[List[str]] = None

        self.class_weights_: Optional[Dict[Any, float]] = None
        self.applied_: Dict[str, Any] = {}
        self.fs_k_: Optional[int] = None

        self.drop_cols_: List[str] = []
        self.datetime_cols_: List[str] = []
        self.datetime_generated_cols_: List[str] = []

    def set_report_generator(self, fn: Callable):
        self.report_generator_ref_ = (fn.__module__, fn.__name__)
        return self

    def fit(self, X, y=None):
        X = self._ensure_df(X)
        self.feature_names_in_ = list(X.columns)

        self.meta_ = self.analyzer_.compute_meta(X, y)
        if self.strategy == "auto" or self.config is None:
            self.config_, self.aux_ = build_rule_based_config(self.meta_, None)
        else:
            self.config_, self.aux_ = build_rule_based_config(self.meta_, self.config)

        self.drop_cols_ = list((self.aux_ or {}).get("drop_cols", []))
        X_used = X.drop(columns=self.drop_cols_, errors="ignore")

        X_used = self._handle_datetime_fit(X_used)

        col_types_used = self.analyzer_.infer_column_types(X_used)
        numeric_cols = [c for c, t in col_types_used.items() if t == "numeric"]
        categorical_cols = [c for c, t in col_types_used.items() if t == "categorical"]

        missing_frac_used = X_used.isna().mean().to_dict()
        cardinality_used = X_used.nunique(dropna=True).to_dict()
        self.meta_used_ = {
            "n_rows": int(X_used.shape[0]),
            "n_cols": int(X_used.shape[1]),
            "col_types": col_types_used,
            "missing_frac": missing_frac_used,
            "cardinality": cardinality_used,
        }

        numeric_missing_cols = [c for c in numeric_cols if missing_frac_used.get(c, 0.0) > 0.0]

        card = X_used[categorical_cols].nunique(dropna=True) if categorical_cols else pd.Series(dtype=int)
        low_card_cols: List[str] = []
        high_card_cols: List[str] = []
        for c in categorical_cols:
            uniq = int(card.get(c, 0))
            if uniq >= int(self.config_.high_cardinality_min_unique):
                high_card_cols.append(c)
            else:
                low_card_cols.append(c)

        transformers = []

        if numeric_cols:
            num_steps = []
            if numeric_missing_cols:
                num_steps.append(("imputer", SimpleImputer(strategy=self.config_.numeric_imputer)))

            if self.config_.numeric_scaler == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif self.config_.numeric_scaler == "minmax":
                num_steps.append(("scaler", MinMaxScaler()))
            elif self.config_.numeric_scaler == "robust":
                num_steps.append(("scaler", RobustScaler()))

            transformers.append(("num", Pipeline(num_steps), numeric_cols) if num_steps else ("num", "passthrough", numeric_cols))

        if low_card_cols:
            low_missing_cols = [c for c in low_card_cols if missing_frac_used.get(c, 0.0) > 0.0]
            low_steps = []
            if low_missing_cols:
                low_steps.append(("imputer", SimpleImputer(strategy=self.config_.categorical_imputer)))
            low_steps.append(("ohe", _make_ohe(self.config_)))
            transformers.append(("cat_low", Pipeline(low_steps), low_card_cols))

        if high_card_cols:
            high_missing_cols = [c for c in high_card_cols if missing_frac_used.get(c, 0.0) > 0.0]
            high_steps = []
            if high_missing_cols:
                high_steps.append(("imputer", SimpleImputer(strategy=self.config_.categorical_imputer)))
            high_steps.append(("freq", FrequencyEncoder(unseen_value=0.0)))
            transformers.append(("cat_high", Pipeline(high_steps), high_card_cols))

        if not transformers:
            empty = FunctionTransformer(lambda Z: np.zeros((len(Z), 0)), validate=False)
            self.pipeline_ = Pipeline([("empty", empty)])
            self.pipeline_.fit(X_used, y) if y is not None else self.pipeline_.fit(X_used)
            self.output_feature_names_ = []
            self.class_weights_ = None
            self._populate_applied(
                numeric_cols, categorical_cols, numeric_missing_cols,
                low_card_cols, high_card_cols, False, False, missing_frac_used
            )
            self._maybe_generate_report()
            return self

        ct = ColumnTransformer(transformers=transformers, remainder="drop")

        pre_pipe = Pipeline([("preprocess", ct)])
        pre_pipe.fit(X_used, y) if y is not None else pre_pipe.fit(X_used)

        fs_used = False
        self.fs_k_ = None
        self.pipeline_ = pre_pipe

        names_pre = self._safe_get_preprocess_feature_names(X_used) or []
        n_pre = len(names_pre) if names_pre else int(pre_pipe.transform(X_used.head(1)).shape[1])

        if (self.config_.feature_selection == "mutual_info") and (y is not None) and (n_pre > 0):
            k = max(int(self.config_.feature_fraction * max(1, n_pre)), 1)
            k = min(k, n_pre)
            fs = SelectKBest(score_func=mutual_info_classif, k=k)
            full_pipe = Pipeline([("preprocess", ct), ("fs", fs)])
            try:
                full_pipe.fit(X_used, y)
                self.pipeline_ = full_pipe
                fs_used = True
                self.fs_k_ = k
            except Exception as e:
                if self.verbose:
                    logger.debug("Feature selection disabled due to error: %s", e)
                self.pipeline_ = pre_pipe
                fs_used = False
                self.fs_k_ = None

        self.output_feature_names_ = self._compute_output_feature_names(X_used)

        self.class_weights_ = None
        balancing_used = False
        if y is not None and self.config_.balancing == "class_weight":
            ir = float(self.meta_.get("imbalance_ratio", 1.0))
            if ir > float(self.config_.imbalance_threshold):
                y_s = pd.Series(y)
                counts = y_s.value_counts().to_dict()
                total = int(len(y_s))
                n_classes = max(len(counts), 1)
                self.class_weights_ = {cls: total / (n_classes * cnt) for cls, cnt in counts.items()}
                balancing_used = True

        self._populate_applied(
            numeric_cols, categorical_cols, numeric_missing_cols,
            low_card_cols, high_card_cols, fs_used, balancing_used, missing_frac_used
        )

        self._maybe_generate_report()
        return self

    def transform(self, X):
        if self.pipeline_ is None:
            raise RuntimeError("Call fit before transform.")

        X = self._ensure_df(X)
        X = self._align_schema(X)

        X_used = X.drop(columns=self.drop_cols_, errors="ignore")
        X_used = self._handle_datetime_transform(X_used)

        return self.pipeline_.transform(X_used)

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        if self.output_feature_names_ is None:
            raise RuntimeError("Feature names not available; fit first.")
        return list(self.output_feature_names_)

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info("Saved AutomataPreprocessor to %s", path)

    @staticmethod
    def load(path: str):
        obj = joblib.load(path)
        if not isinstance(obj, AutomataPreprocessor):
            raise ValueError("Loaded object is not AutomataPreprocessor")
        logger.info("Loaded AutomataPreprocessor from %s", path)
        return obj

    @staticmethod
    def _ensure_df(X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _align_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_in_:
            return X
        X2 = X.copy()
        for c in self.feature_names_in_:
            if c not in X2.columns:
                X2[c] = np.nan
        X2 = X2[self.feature_names_in_]
        return X2

    def _handle_datetime_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config_ or PreprocessingConfig()
        col_types = self.analyzer_.infer_column_types(X)
        dt_cols = [c for c, t in col_types.items() if t == "datetime"]
        self.datetime_cols_ = dt_cols

        if not dt_cols:
            self.datetime_generated_cols_ = []
            return X

        if cfg.datetime_handling == "drop":
            self.datetime_generated_cols_ = []
            return X.drop(columns=dt_cols, errors="ignore")

        out = X.copy()
        gen_cols: List[str] = []
        for c in dt_cols:
            ser = pd.to_datetime(out[c], errors="coerce")
            if "year" in cfg.datetime_extract_parts:
                out[f"{c}__year"] = ser.dt.year; gen_cols.append(f"{c}__year")
            if "month" in cfg.datetime_extract_parts:
                out[f"{c}__month"] = ser.dt.month; gen_cols.append(f"{c}__month")
            if "day" in cfg.datetime_extract_parts:
                out[f"{c}__day"] = ser.dt.day; gen_cols.append(f"{c}__day")
            if "dayofweek" in cfg.datetime_extract_parts:
                out[f"{c}__dayofweek"] = ser.dt.dayofweek; gen_cols.append(f"{c}__dayofweek")

        out = out.drop(columns=dt_cols, errors="ignore")
        self.datetime_generated_cols_ = gen_cols
        return out

    def _handle_datetime_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config_ or PreprocessingConfig()
        if not self.datetime_cols_:
            return X

        if cfg.datetime_handling == "drop":
            return X.drop(columns=self.datetime_cols_, errors="ignore")

        out = X.copy()
        for c in self.datetime_cols_:
            if c not in out.columns:
                out[c] = pd.NaT
            ser = pd.to_datetime(out[c], errors="coerce")
            if f"{c}__year" in self.datetime_generated_cols_:
                out[f"{c}__year"] = ser.dt.year
            if f"{c}__month" in self.datetime_generated_cols_:
                out[f"{c}__month"] = ser.dt.month
            if f"{c}__day" in self.datetime_generated_cols_:
                out[f"{c}__day"] = ser.dt.day
            if f"{c}__dayofweek" in self.datetime_generated_cols_:
                out[f"{c}__dayofweek"] = ser.dt.dayofweek

        out = out.drop(columns=self.datetime_cols_, errors="ignore")
        return out

    def _populate_applied(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        numeric_missing_cols: List[str],
        low_card_cols: List[str],
        high_card_cols: List[str],
        feature_selection_used: bool,
        balancing_used: bool,
        missing_frac_used: Dict[str, float],
    ):
        cfg = self.config_ or PreprocessingConfig()
        low_missing_cols = [c for c in low_card_cols if missing_frac_used.get(c, 0.0) > 0.0]
        high_missing_cols = [c for c in high_card_cols if missing_frac_used.get(c, 0.0) > 0.0]

        self.applied_ = {
            "drop_cols": list(self.drop_cols_),
            "datetime_cols": list(self.datetime_cols_),
            "datetime_handling": cfg.datetime_handling,
            "datetime_generated_cols": list(self.datetime_generated_cols_),

            "numeric_cols": list(numeric_cols),
            "numeric_missing_cols": list(numeric_missing_cols),
            "numeric_imputer_used": bool(numeric_missing_cols),
            "numeric_scaler_used": bool(numeric_cols) and (cfg.numeric_scaler != "none"),

            "categorical_cols": list(categorical_cols),

            "low_card_cols": list(low_card_cols),
            "low_card_missing_cols": list(low_missing_cols),
            "low_card_imputer_used": bool(low_missing_cols),
            "low_card_encoder_used": bool(low_card_cols),

            "high_card_cols": list(high_card_cols),
            "high_card_missing_cols": list(high_missing_cols),
            "high_card_imputer_used": bool(high_missing_cols),
            "high_card_encoder_used": bool(high_card_cols),

            "feature_selection_used": bool(feature_selection_used),
            "feature_selection_method": (cfg.feature_selection if feature_selection_used else "none"),
            "feature_fraction": float(cfg.feature_fraction),
            "fs_k": self.fs_k_,

            "balancing_used": bool(balancing_used),
            "balancing_method": ("class_weight" if balancing_used else "none"),
            "imbalance_threshold": float(cfg.imbalance_threshold),
        }

    def _compute_output_feature_names(self, X_used: pd.DataFrame) -> List[str]:
        names_pre = self._safe_get_preprocess_feature_names(X_used) or []

        if self.pipeline_ is not None and "fs" in getattr(self.pipeline_, "named_steps", {}):
            fs = self.pipeline_.named_steps["fs"]
            try:
                mask = fs.get_support()
                if len(names_pre) == len(mask):
                    return list(np.array(names_pre, dtype=object)[mask])
            except Exception:
                pass

        return names_pre

    def _safe_get_preprocess_feature_names(self, X_used: pd.DataFrame) -> Optional[List[str]]:
        try:
            if self.pipeline_ is None or "preprocess" not in self.pipeline_.named_steps:
                return []
            ct = self.pipeline_.named_steps["preprocess"]

            if hasattr(ct, "get_feature_names_out"):
                try:
                    return list(ct.get_feature_names_out(input_features=list(X_used.columns)))
                except Exception:
                    try:
                        return list(ct.get_feature_names_out())
                    except Exception:
                        pass

            names: List[str] = []
            for name, trans, cols in getattr(ct, "transformers_", []):
                if name == "remainder":
                    continue
                if trans == "passthrough":
                    if isinstance(cols, (list, tuple)):
                        names.extend([f"{name}__{c}" for c in cols])
                    else:
                        names.append(f"{name}__{cols}")
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    try:
                        out = trans.get_feature_names_out(cols if isinstance(cols, (list, tuple)) else None)
                        names.extend(list(out))
                        continue
                    except Exception:
                        pass
                if isinstance(cols, (list, tuple)):
                    names.extend([f"{name}__{c}" for c in cols])
                else:
                    names.append(f"{name}__{cols}")

            return names
        except Exception:
            return None

    def _maybe_generate_report(self):
        if not self.report:
            return

        if self.report_generator_ref_:
            mod_name, fn_name = self.report_generator_ref_
            try:
                mod = importlib.import_module(mod_name)
                fn = getattr(mod, fn_name)
                fn(self, self.report_path)
                return
            except Exception as e:
                logger.warning("Report generator ref failed (%s.%s): %s", mod_name, fn_name, e)

        try:
            from __main__ import _generate_sensor_report
            _generate_sensor_report(self, self.report_path)
        except Exception as e:
            logger.warning("Report requested but generator is unavailable or failed: %s", e)


def preprocessing_logic(df: pd.DataFrame) -> pd.DataFrame:

    cols_to_drop = [
        "accuracy",
        "f1_macro",
        "precision_macro",
        "trained_model_size_kb",
        "inference_speed_ms",
        "model_name",
        "Task_id",
        "dataset_id",
        "model_id",
        "dataset_name",
        "static_usage_ram_kb",
        "dynamic_usage_ram_kb",
        "full_ram_usage_kb",
        "mean_feature_variance",
        "median_feature_variance",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    zero_variance_cols = [
        "n_leaves_mean",
        "dropout_rate_mean",
    ]
    df = df.drop(columns=[c for c in zero_variance_cols if c in df.columns], errors="ignore")

    if "regularization_supported" in df.columns:
        df = df.drop("regularization_supported", axis=1)

    return df


@dataclass
class PipelineState:
    task_id: str
    user_id: str
    task_type: str  # "sensor"
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


class SensorPipeline:

    def __init__(
        self,
        *,
        task_id: str,
        user_id: str,
        task_type: str,
        dataset_path: str,
        device_family_id: str,
        device_specs: Dict[str, Any],
        target_num_classes: Optional[int] = None,
        target_name: str,
        output_root: str = "runs",
        export_ext: str = ".h",
        min_accuracy: float = 0.50,
        seed: int = 42,

        batch_size: int = 128,
        num_workers: int = 0,
        final_epochs: Optional[int] = None,
        epochs: Optional[int] = None,

        title: str = None,
        description: str = None,
        visibility: str = None,

        quantization: str = None,
        optimization_strategy: str = None,
        training_speed: str = None,
        accuracy_tolerance: str = None,
        accuracy_drop_cap: float = 0.30,
        optimization_trigger_ratio: float = 0.70,
        robustness: str = None,
        outlier_removal: str = None,
    ):
        if task_type != "sensor":
            raise ValueError(f"SensorPipeline supports task_type='sensor' only, got {task_type}")

        self.task_id = task_id
        self.user_id = user_id
        self.task_type = task_type

        self.title = title
        self.description = description
        self.visibility = visibility

        self.quantization = quantization or "auto"
        self.optimization_strategy = optimization_strategy or "balanced"
        self.training_speed = training_speed or "auto"
        self.accuracy_tolerance = accuracy_tolerance or "auto"
        self.accuracy_drop_cap = float(accuracy_drop_cap)
        self.optimization_trigger_ratio = float(optimization_trigger_ratio)
        self.robustness = robustness
        self.outlier_removal = outlier_removal

        self.dataset_path = dataset_path
        self.device_family_id = device_family_id
        self.device_specs = normalize_device_specs(device_specs)
        self.target_num_classes = int(target_num_classes) if target_num_classes is not None else None
        self.target_name = str(target_name)

        self.export_ext = str(export_ext).strip().lower()
        if self.export_ext != "all" and not self.export_ext.startswith("."):
            self.export_ext = "." + self.export_ext

        self.output_dir = os.path.join(output_root, task_id)
        self.dataset_extract_dir = os.path.join(self.output_dir, "dataset")

        self.min_accuracy = float(min_accuracy)
        self.seed = int(seed)

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

        self._final_model = None
        self._final_model_path: Optional[str] = None
        self._report_path: Optional[str] = None
        self._preprocessing_report_path: Optional[str] = None
        self._prep_obj: Optional[Any] = None

        self._final_train_loader = None
        self._final_val_loader = None
        self._final_test_loader = None

        self.model_kind: Optional[str] = None
        self.best_model_name: Optional[str] = None
        self._xgb_sklearn_model = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self._label_encoder = None
        self._label_classes = None

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
            optimized = self._phase_optimizing(self._final_model, self.best_model_name)
            if optimized is not None:
                self._final_model = optimized

            self.update_status("packaging", "Saving final model + generating report...")
            self._phase_packaging()
            self._phase_report()

            self._phase_validate_final()

            self.update_status("completed", "Completed successfully.")
            return self._result()

        except (RuntimeError, ValueError) as e:
            return self._fail(str(e), e)
        except Exception as e:
            return self._fail(f"Unhandled pipeline error: {e}", e)

    def _prepare_dirs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dataset_extract_dir, exist_ok=True)

    def _unpack_and_validate_dataset(self) -> None:

        p = self.dataset_path
        if not isinstance(p, str) or len(p) == 0:
            raise ValueError("dataset_path must be a non-empty string.")
        if not os.path.exists(p):
            raise FileNotFoundError(f"dataset_path does not exist: {p}")
        if os.path.isdir(p):
            raise RuntimeError("dataset_path must be a file for sensor tasks (CSV/Excel/Parquet).")

        ext = os.path.splitext(p)[1].lower()
        supported = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}
        if ext not in supported:
            raise RuntimeError(f"Unsupported sensor dataset format '{ext}'. Supported: {sorted(supported)}")

        self.state.metrics.update({
            "dataset_mode": "file",
            "dataset_path": p,
            "dataset_ext": ext,
        })

    # ==========================================================
    # Phase 1: preprocessing 
    # ==========================================================

    def _phase_preprocessing(self) -> None:
        def _seed_everything(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        def _load_tabular(path: str) -> pd.DataFrame:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".csv":
                return pd.read_csv(path)
            if ext == ".tsv":
                return pd.read_csv(path, sep="\t")
            if ext in (".xlsx", ".xls"):
                return pd.read_excel(path)
            if ext == ".parquet":
                return pd.read_parquet(path)
            if ext == ".json":
                return pd.read_json(path)
            raise RuntimeError(f"Unsupported sensor dataset format '{ext}'.")

        _seed_everything(self.seed)

        df = _load_tabular(self.dataset_path)
        if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
            raise RuntimeError("Loaded dataset is empty or invalid.")

        if self.target_name not in df.columns:
            raise RuntimeError(f"target_name '{self.target_name}' not found in dataset columns.")

        X = df.drop(columns=[self.target_name])
        y = df[self.target_name]

        n_classes = int(pd.Series(y).nunique(dropna=True))
        if self.target_num_classes is None:
            self.target_num_classes = n_classes
        elif n_classes != self.target_num_classes:
            raise RuntimeError(
                f"Found {n_classes} classes in target '{self.target_name}', expected {self.target_num_classes}."
            )

        le = LabelEncoder()
        y_enc = le.fit_transform(pd.Series(y))
        self._label_encoder = le
        self._label_classes = le.classes_.tolist()

        prep = AutomataPreprocessor(
            verbose=True,
            report=False,
            report_path=os.path.join(self.output_dir, "sensor_preprocessing_report.pdf"),
            project_name="Sensor Pipeline - Preprocessing",
            logo_path=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
        )
        Xp = prep.fit_transform(X, y)
        yp = y_enc
        self._preprocessing_report_path = prep.report_path
        self._prep_obj = prep

        if isinstance(Xp, pd.DataFrame):
            Xp_df = Xp
        else:
            Xp_df = pd.DataFrame(Xp)

        X_train, X_test, y_train, y_test = train_test_split(
            Xp_df, yp, test_size=0.2, random_state=42
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_test, y_test
        self.X_test, self.y_test = X_test, y_test

        self.state.metrics.update({
            "dataset_shape": tuple(df.shape),
            "feature_shape": tuple(Xp_df.shape),
            "target_name": self.target_name,
            "target_num_classes": n_classes,
            "preprocessing_report_path": self._preprocessing_report_path,
            "class_labels": self._label_classes,
        })

    # ==========================================================
    # Phase 2: training 
    # ==========================================================

    def _phase_training(self) -> None:
        def build_meta_entries_for_all_models_from_preprocessed(
            X, y, task_id: int = 0, dataset_id: int = 0, dataset_name: str = "unknown"
        ) -> dict:
            import numpy as np
            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            from pandas.api.types import is_numeric_dtype
            from sklearn.decomposition import PCA

            def compute_dataset_features(X_df: pd.DataFrame, y_arr) -> dict:
                meta = {}
                try:
                    n_samples, n_features = X_df.shape
                except Exception:
                    n_samples, n_features = None, None
                meta["n_samples"] = int(n_samples) if n_samples is not None else None
                meta["n_features"] = int(n_features) if n_features is not None else None

                try:
                    dtypes = X_df.dtypes
                    numeric_mask = [is_numeric_dtype(dt) for dt in dtypes]
                    n_numeric = int(np.sum(numeric_mask))
                    n_categorical = int(len(dtypes) - n_numeric)

                    n_binary = 0
                    for col in X_df.columns:
                        vals = pd.Series(X_df[col]).dropna().unique()
                        if len(vals) == 2:
                            n_binary += 1

                    meta["n_numeric_features"] = n_numeric
                    meta["n_categorical_features"] = n_categorical
                    meta["n_binary_features"] = int(n_binary)
                except Exception:
                    numeric_mask = None
                    meta["n_numeric_features"] = None
                    meta["n_categorical_features"] = None
                    meta["n_binary_features"] = None

                try:
                    y_ser = pd.Series(y_arr).dropna()
                    counts = y_ser.value_counts().values
                    n_classes = int(len(counts))
                    probs = counts / counts.sum() if n_classes > 0 else np.array([])
                    class_balance_std = float(probs.std()) if n_classes > 0 else None
                    class_entropy = float(
                        -(probs * np.log2(probs + 1e-12)).sum()
                    ) if n_classes > 0 else None
                    meta["n_classes"] = int(n_classes)
                    meta["class_balance_std"] = class_balance_std
                    meta["class_entropy"] = class_entropy
                except Exception:
                    meta["n_classes"] = None
                    meta["class_balance_std"] = None
                    meta["class_entropy"] = None

                try:
                    num_cols = X_df.select_dtypes(include=[np.number])

                    mean_var = 0.0
                    med_var = 0.0
                    mean_corr = 0.0
                    max_corr = 0.0

                    if num_cols.shape[1] > 0 and num_cols.shape[0] > 1:
                        vars_ = num_cols.var(axis=0, ddof=1).values
                        if np.isfinite(vars_).sum() > 0:
                            mean_var = float(np.nanmean(vars_))
                            med_var = float(np.nanmedian(vars_))

                        max_corr_features = min(num_cols.shape[1], 50)
                        corr = num_cols.iloc[:, :max_corr_features].corr().abs().values
                        upper = corr[np.triu_indices_from(corr, k=1)]
                        finite_upper = upper[np.isfinite(upper)]
                        if finite_upper.size > 0:
                            mean_corr = float(finite_upper.mean())
                            max_corr = float(finite_upper.max())

                    meta["mean_feature_variance"] = mean_var
                    meta["median_feature_variance"] = med_var
                    meta["mean_corr_abs"] = mean_corr
                    meta["max_corr_abs"] = max_corr
                except Exception:
                    meta["mean_feature_variance"] = 0.0
                    meta["median_feature_variance"] = 0.0
                    meta["mean_corr_abs"] = 0.0
                    meta["max_corr_abs"] = 0.0

                try:
                    num_cols = X_df.select_dtypes(include=[np.number])

                    # 1) feature_skewness_mean
                    if num_cols.shape[1] > 0:
                        skews = num_cols.skew(axis=0, skipna=True)
                        skews = skews.replace([np.inf, -np.inf], np.nan)
                        feature_skewness_mean = float(
                            skews.mean(skipna=True)
                        ) if not skews.isna().all() else 0.0
                    else:
                        feature_skewness_mean = 0.0
                    meta["feature_skewness_mean"] = feature_skewness_mean

                    # 2) feature_kurtosis_mean
                    if num_cols.shape[1] > 0:
                        kurts = num_cols.kurt(axis=0, skipna=True)
                        kurts = kurts.replace([np.inf, -np.inf], np.nan)
                        feature_kurtosis_mean = float(
                            kurts.mean(skipna=True)
                        ) if not kurts.isna().all() else 0.0
                    else:
                        feature_kurtosis_mean = 0.0
                    meta["feature_kurtosis_mean"] = feature_kurtosis_mean

                    # 3) missing_percentage
                    if (
                        n_samples is not None
                        and n_features is not None
                        and n_samples > 0
                        and n_features > 0
                    ):
                        total_cells = float(n_samples * n_features)
                        missing_count = float(X_df.isna().sum().sum())
                        missing_percentage = missing_count / total_cells
                    else:
                        missing_percentage = 0.0
                    meta["missing_percentage"] = float(missing_percentage)

                    # 4) avg_cardinality_categorical
                    avg_card = 0.0
                    if "numeric_mask" in locals() and numeric_mask is not None:
                        cat_cols = [
                            col for col, isnum in zip(X_df.columns, numeric_mask) if not isnum
                        ]
                        if len(cat_cols) > 0:
                            cards = []
                            for col in cat_cols:
                                try:
                                    cards.append(X_df[col].nunique(dropna=True))
                                except Exception:
                                    continue
                            if len(cards) > 0:
                                avg_card = float(np.mean(cards))
                    meta["avg_cardinality_categorical"] = avg_card

                    # 5) complexity_ratio
                    if n_samples is not None and n_features is not None and n_samples > 0:
                        complexity_ratio = float(n_features) / float(n_samples)
                    else:
                        complexity_ratio = 0.0
                    meta["complexity_ratio"] = complexity_ratio

                    # 6) intrinsic_dim_estimate (PCA-based)
                    intrinsic_dim = 0.0
                    try:
                        if num_cols.shape[1] >= 2 and num_cols.shape[0] >= 5:
                            X_pca = num_cols.to_numpy(dtype=np.float32)
                            col_means = np.nanmean(X_pca, axis=0)
                            inds = np.where(np.isnan(X_pca))
                            if inds[0].size > 0:
                                X_pca[inds] = np.take(col_means, inds[1])

                            n_components = min(X_pca.shape[0], X_pca.shape[1])
                            if n_components >= 1:
                                pca = PCA(n_components=n_components)
                                pca.fit(X_pca)
                                cumsum = np.cumsum(pca.explained_variance_ratio_)
                                k = int(np.searchsorted(cumsum, 0.95) + 1)
                                intrinsic_dim = float(max(1, min(k, n_components)))
                    except Exception:
                        intrinsic_dim = 0.0

                    meta["intrinsic_dim_estimate"] = intrinsic_dim
                except Exception:
                    meta.setdefault("feature_skewness_mean", 0.0)
                    meta.setdefault("feature_kurtosis_mean", 0.0)
                    meta.setdefault("missing_percentage", 0.0)
                    meta.setdefault("avg_cardinality_categorical", 0.0)
                    meta.setdefault("complexity_ratio", 0.0)
                    meta.setdefault("intrinsic_dim_estimate", 0.0)

                return meta

            def _safe_stratify_or_none(y_enc: np.ndarray):
                values, counts = np.unique(y_enc, return_counts=True)
                if counts.min() < 2:
                    return None
                return y_enc

            # -------------------------------------------------------------------------
            # compute landmarks
            # -------------------------------------------------------------------------

            def compute_landmarks(X_train: pd.DataFrame, y_train) -> dict:
                landmarks = {
                    "landmark_lr_accuracy": 0.0,
                    "landmark_dt_depth3_accuracy": 0.0,
                    "landmark_knn3_accuracy": 0.0,
                    "landmark_random_noise_accuracy": 0.0,
                    "fisher_discriminant_ratio": 0.0,
                }

                y_arr = np.asarray(y_train)
                if y_arr.ndim > 1:
                    y_arr = y_arr.ravel()
                try:
                    y_arr = y_arr.astype(int)
                except Exception:
                    from sklearn.preprocessing import LabelEncoder
                    le_fallback = LabelEncoder()
                    y_arr = le_fallback.fit_transform(y_arr)

                if X_train.shape[0] < 5 or len(np.unique(y_arr)) < 2:
                    return landmarks

                LANDMARK_SUBSAMPLE_FRACTION = 0.15
                LANDMARK_MAX_ROWS = 1000
                LANDMARK_MIN_ROWS = 20

                if isinstance(X_train, pd.DataFrame):
                    X_num = X_train.select_dtypes(include=[np.number]).copy()

                    if X_num.shape[1] == 0:
                        X_num = pd.DataFrame(index=X_train.index)
                        for col in X_train.columns:
                            s = X_train[col]
                            if is_numeric_dtype(s):
                                X_num[col] = pd.to_numeric(s, errors="coerce").fillna(0)
                            else:
                                le_col = LabelEncoder()
                                X_num[col] = le_col.fit_transform(
                                    s.astype(str).fillna("__NA__")
                                )
                else:
                    X_num = pd.DataFrame(X_train)

                if X_num.shape[1] == 0:
                    return landmarks

                X_num = X_num.replace([np.inf, -np.inf], np.nan)
                vals = X_num.to_numpy(dtype=np.float32)

                if np.isnan(vals).any() or not np.isfinite(vals).all():
                    vals[~np.isfinite(vals)] = np.nan
                    col_means = np.nanmean(vals, axis=0)
                    col_means = np.where(np.isnan(col_means), 0.0, col_means)
                    inds = np.where(np.isnan(vals))
                    if inds[0].size > 0:
                        vals[inds] = np.take(col_means, inds[1])
                    X_num = pd.DataFrame(vals, columns=X_num.columns)

                n_rows = X_num.shape[0]
                RNG = np.random.RandomState(42)

                n_sub = min(LANDMARK_MAX_ROWS, int(LANDMARK_SUBSAMPLE_FRACTION * n_rows))
                if n_sub < LANDMARK_MIN_ROWS:
                    n_sub = LANDMARK_MIN_ROWS
                n_sub = min(n_sub, n_rows)

                idx = RNG.choice(n_rows, size=n_sub, replace=False)
                X_num_sub = X_num.iloc[idx].reset_index(drop=True).astype(np.float32)
                y_sub = y_arr[idx]

                if len(np.unique(y_sub)) < 2:
                    return landmarks

                strat_labels = _safe_stratify_or_none(y_sub)

                try:
                    Xtr, Xte, ytr, yte = train_test_split(
                        X_num_sub,
                        y_sub,
                        test_size=0.2,
                        random_state=42,
                        stratify=strat_labels,
                    )
                    clf = LogisticRegression(max_iter=50, C=0.1, solver="lbfgs")
                    clf.fit(Xtr, ytr)
                    acc = accuracy_score(yte, clf.predict(Xte))
                    landmarks["landmark_lr_accuracy"] = float(acc)
                except Exception:
                    pass

                try:
                    Xtr, Xte, ytr, yte = train_test_split(
                        X_num_sub,
                        y_sub,
                        test_size=0.2,
                        random_state=42,
                        stratify=strat_labels,
                    )
                    clf = __import__("sklearn.tree", fromlist=["DecisionTreeClassifier"]).DecisionTreeClassifier(
                        max_depth=3, min_samples_leaf=5, random_state=42
                    )
                    clf.fit(Xtr, ytr)
                    acc = accuracy_score(yte, clf.predict(Xte))
                    landmarks["landmark_dt_depth3_accuracy"] = float(acc)
                except Exception:
                    pass

                try:
                    X_knn = X_num_sub
                    if X_knn.shape[1] > 30:
                        cols = RNG.choice(X_knn.shape[1], size=30, replace=False)
                        X_knn = X_knn.iloc[:, cols]

                    Xtr, Xte, ytr, yte = train_test_split(
                        X_knn,
                        y_sub,
                        test_size=0.2,
                        random_state=42,
                        stratify=strat_labels,
                    )
                    clf = __import__("sklearn.neighbors", fromlist=["KNeighborsClassifier"]).KNeighborsClassifier(n_neighbors=3)
                    clf.fit(Xtr, ytr)
                    acc = accuracy_score(yte, clf.predict(Xte))
                    landmarks["landmark_knn3_accuracy"] = float(acc)
                except Exception:
                    pass

                try:
                    counts = np.bincount(y_sub)
                    probs = counts / counts.sum()
                    preds = RNG.choice(np.arange(len(probs)), size=len(y_sub), p=probs)
                    acc = accuracy_score(y_sub, preds)
                    landmarks["landmark_random_noise_accuracy"] = float(acc)
                except Exception:
                    pass

                try:
                    fdr_values = []
                    for j in range(X_num_sub.shape[1]):
                        xj = X_num_sub.iloc[:, j].values.astype(float)
                        mu = xj.mean()
                        num = 0.0
                        den = 0.0
                        for c in np.unique(y_sub):
                            mask_c = (y_sub == c)
                            xc = xj[mask_c]
                            if xc.size == 0:
                                continue
                            nc = xc.size
                            mu_c = xc.mean()
                            var_c = xc.var(ddof=1) if nc > 1 else 0.0
                            num += nc * (mu_c - mu) ** 2
                            den += nc * var_c
                        if den > 0:
                            fdr_values.append(num / (den + 1e-12))
                    if fdr_values:
                        landmarks["fisher_discriminant_ratio"] = float(np.mean(fdr_values))
                except Exception:
                    pass

                return landmarks

            # -------------------------------------------------------------------------
            # Compute dataset meta + landmarks
            # -------------------------------------------------------------------------
            try:
                ds_meta = compute_dataset_features(X, y)
            except Exception:
                ds_meta = {}

            try:
                lm = compute_landmarks(X, y)
            except Exception:
                lm = {k: 0.0 for k in [
                    "landmark_lr_accuracy",
                    "landmark_dt_depth3_accuracy",
                    "landmark_knn3_accuracy",
                    "landmark_random_noise_accuracy",
                    "fisher_discriminant_ratio",
                ]}

            for k in [
                "landmark_lr_accuracy",
                "landmark_dt_depth3_accuracy",
                "landmark_knn3_accuracy",
                "landmark_random_noise_accuracy",
                "fisher_discriminant_ratio",
            ]:
                if k not in lm or lm[k] is None:
                    lm[k] = 0.0

            meta_common = {}
            meta_common.update(ds_meta)
            meta_common.update(lm)

            base_ids = {
                "Task_id": int(task_id),
                "dataset_id": int(dataset_id),
                "dataset_name": str(dataset_name),
            }

            perf_placeholders = {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "precision_macro": 0.0,
                "trained_model_size_kb": 0.0,
                "inference_speed_ms": 0.0,
                "static_usage_ram_kb": 0.0,
                "dynamic_usage_ram_kb": 0.0,
                "full_ram_usage_kb": 0.0,
                "model_n_parameters": 0.0,
            }

            entries_by_model = {}

            for model_name, caps in MODEL_CAPABILITIES.items():
                combined = {}
                combined.update(base_ids)
                combined.update(meta_common)
                combined.update(caps)

                for k, v in perf_placeholders.items():
                    combined.setdefault(k, v)

                ordered_entry = {}
                for key in KEY_ORDER:
                    val = combined.get(key, None)
                    if val is None:
                        if key in STRING_KEYS:
                            val = "unknown"
                        else:
                            val = 0.0
                    ordered_entry[key] = val

                entries_by_model[model_name] = ordered_entry

            return entries_by_model

        class MLPNet(nn.Module):
            def __init__(self, input_dim, n_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_classes),
                )

            def forward(self, x):
                return self.net(x)

        class TinyConv1DNet(nn.Module):
            def __init__(self, input_dim, n_classes):
                super().__init__()
                self.conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveMaxPool1d(1)
                self.fc = nn.Linear(8, n_classes)  
                self.quant_pool = torch.ao.quantization.QuantStub()
                self.dequant_pool = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = x.unsqueeze(1)         
                x = self.conv(x)           
                x = self.relu(x)
                x = self.dequant_pool(x)
                x = self.pool(x).squeeze(-1) 
                x = self.quant_pool(x)
                return self.fc(x)


        class TinyConvNet(nn.Module):
            def __init__(self, input_dim, n_classes):
                super().__init__()
                self.conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveMaxPool1d(1)
                self.fc = nn.Linear(4, n_classes)  
                self.quant_pool = torch.ao.quantization.QuantStub()
                self.dequant_pool = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = x.unsqueeze(1)         
                x = self.conv(x)           
                x = self.relu(x)
                x = self.dequant_pool(x)
                x = self.pool(x).squeeze(-1) 
                x = self.quant_pool(x)
                return self.fc(x)


        class TinyRNNNet(nn.Module):
            def __init__(self, input_dim, n_classes):
                super().__init__()
                self.hidden_dim = 32
                self.rnn = nn.LSTM(input_size=1, hidden_size=self.hidden_dim, batch_first=True)
                self.fc = nn.Linear(self.hidden_dim, n_classes)

            def forward(self, x):
                x = x.unsqueeze(-1)
                out, (h, c) = self.rnn(x)
                return self.fc(out[:, -1, :])

        def make_logreg():
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline as SkPipeline
            return SkPipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("clf", LogisticRegression(max_iter=800, solver="lbfgs"))
            ])

        def make_rf():
            return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

        def make_xgboost():
            try:
                import xgboost as xgb
            except ImportError:
                return None
            return xgb.XGBClassifier(
                tree_method="hist",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
            )

        def make_cnn1d(input_dim, n_classes):
            return TinyConv1DNet(input_dim, n_classes)

        def make_tinyconv(input_dim, n_classes):
            return TinyConvNet(input_dim, n_classes)

        def make_tiny_rnn(input_dim, n_classes):
            return TinyRNNNet(input_dim, n_classes)

        def make_mlp(input_dim, n_classes):
            return MLPNet(input_dim, n_classes)

        MODELS = {
            "logreg": ("classic", make_logreg),
            "rf": ("classic", make_rf),
            "xgboost": ("classic", make_xgboost),
            "cnn1d": ("deep", make_cnn1d),
            "tiny_rnn": ("deep", make_tiny_rnn),
            "mlp": ("deep", make_mlp),
            "tinyconv": ("deep", make_tinyconv),
        }

        CLASSIC_PARAM_GRIDS = {
            "logreg": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs"],
                "clf__max_iter": [200, 500],
            },
            "rf": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        }

        DEEP_PARAM_CONFIGS = {
            "cnn1d": [
                {"lr": 1e-3, "batch_size": 32, "epochs": 15},
                {"lr": 1e-4, "batch_size": 64, "epochs": 20},
            ],
            "tiny_rnn": [
                {"lr": 1e-3, "batch_size": 32, "epochs": 20},
                {"lr": 5e-4, "batch_size": 64, "epochs": 25},
            ],
            "mlp": [
                {"lr": 1e-3, "batch_size": 32, "epochs": 20},
                {"lr": 1e-4, "batch_size": 64, "epochs": 30},
            ],
            "tinyconv": [
                {"lr": 1e-3, "batch_size": 32, "epochs": 20},
                {"lr": 5e-4, "batch_size": 64, "epochs": 25},
            ],
        }

        def get_model_size_mb_sklearn(model) -> float:
            data = json.dumps({"_": "placeholder"}).encode("utf-8")
            try:
                import pickle
                data = pickle.dumps(model)
            except Exception:
                pass
            return len(data) / (1024 ** 2)

        def get_model_size_mb_torch(model: nn.Module) -> float:
            fd, tmp_path = tempfile.mkstemp(suffix=".pth")
            os.close(fd)
            try:
                torch.save(model.state_dict(), tmp_path)
                size = os.path.getsize(tmp_path) / (1024 ** 2)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            return size

        def _size_kb_classic(model) -> float:
            try:
                import joblib
                tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
                tmp.close()
                joblib.dump(model, tmp.name)
                size_kb = os.path.getsize(tmp.name) / 1024.0
            finally:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass
            return float(size_kb)

        def _size_kb_xgb(booster) -> float:
            import xgboost as xgb
            tmp = tempfile.NamedTemporaryFile(suffix=".ubj", delete=False)
            tmp.close()
            try:
                booster.save_model(tmp.name)
                size_kb = os.path.getsize(tmp.name) / 1024.0
            finally:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass
            return float(size_kb)

        def train_and_tune_classic(model_name, make_fn, X_train, y_train, X_val, y_val):
            if model_name not in CLASSIC_PARAM_GRIDS:
                raise ValueError(f"No param grid defined for classic model: {model_name}")

            base_model = make_fn()
            if base_model is None:
                raise RuntimeError(f"Model factory returned None for {model_name}")

            param_grid = CLASSIC_PARAM_GRIDS[model_name]

            grid = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring="accuracy",
                cv=3,
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred)

            model_size_mb = get_model_size_mb_sklearn(best_model)

            return best_model, val_acc, model_size_mb

        def _train_one_epoch(model, loader, optimizer, device):
            model.train()
            total = 0
            correct = 0
            loss_sum = 0.0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item()) * xb.size(0)
                preds = out.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += int(xb.size(0))
            return loss_sum / max(1, total), correct / max(1, total)

        def _eval_model(model, loader, device):
            model.eval()
            total = 0
            correct = 0
            loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    loss_sum += float(loss.item()) * xb.size(0)
                    preds = out.argmax(dim=1)
                    correct += int((preds == yb).sum().item())
                    total += int(xb.size(0))
            return loss_sum / max(1, total), correct / max(1, total)

        def train_and_tune_deep(model_name, make_fn, X_train, y_train, X_val, y_val):
            if model_name not in DEEP_PARAM_CONFIGS:
                raise ValueError(f"No deep param configs defined for: {model_name}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            Xtr = torch.tensor(np.asarray(X_train), dtype=torch.float32)
            ytr = torch.tensor(np.asarray(y_train), dtype=torch.long)
            Xva = torch.tensor(np.asarray(X_val), dtype=torch.float32)
            yva = torch.tensor(np.asarray(y_val), dtype=torch.long)

            best_acc = -np.inf
            best_model = None
            best_cfg = None

            input_dim = Xtr.shape[1]
            n_classes = int(ytr.max().item() + 1) if ytr.numel() > 0 else int(self.target_num_classes)

            for cfg in DEEP_PARAM_CONFIGS[model_name]:
                model = make_fn(input_dim, n_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

                train_ds = TensorDataset(Xtr, ytr)
                val_ds = TensorDataset(Xva, yva)
                train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False)

                best_val_acc = -np.inf
                for _ in range(int(cfg["epochs"])):
                    _train_one_epoch(model, train_loader, optimizer, device)
                    _, val_acc = _eval_model(model, val_loader, device)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                if best_val_acc > best_acc:
                    best_acc = best_val_acc
                    best_model = model
                    best_cfg = cfg

            if best_model is None:
                raise RuntimeError(f"Deep training failed for {model_name}")

            model_size_mb = get_model_size_mb_torch(best_model)
            return best_model, best_acc, model_size_mb

        def run_best_candidate(best_model_name, X_train, y_train, X_val, y_val):
            if best_model_name not in MODELS:
                raise ValueError(f"Unknown model name: {best_model_name}")

            model_type, make_fn = MODELS[best_model_name]
            if model_type == "classic":
                return train_and_tune_classic(best_model_name, make_fn, X_train, y_train, X_val, y_val)
            if model_type == "deep":
                return train_and_tune_deep(best_model_name, make_fn, X_train, y_train, X_val, y_val)
            raise ValueError(f"Unknown model type for {best_model_name}: {model_type}")

        candidates = build_meta_entries_for_all_models_from_preprocessed(
            self.X_train, self.y_train,
            task_id=0,
            dataset_id=0,
            dataset_name=os.path.basename(self.dataset_path) if isinstance(self.dataset_path, str) else "unknown",
        )

        model_path = os.path.join("automl", "pipelines", "sensor", "sensor_pipeline_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing sensor pipeline model: {model_path}")

        import __main__
        __main__.preprocessing_logic = preprocessing_logic

        import joblib
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*serialized model.*",
                category=UserWarning,
            )
            
            loaded_model = joblib.load(model_path)

        best_name = None
        best_score = float("-inf")

        for name, features in candidates.items():
            df_entry = pd.DataFrame([features])

            if hasattr(loaded_model, "feature_names_in_"):
                missing = [c for c in loaded_model.feature_names_in_ if c not in df_entry.columns]
                for c in missing:
                    df_entry[c] = 0.0
                df_entry = df_entry[list(loaded_model.feature_names_in_)]
            else:
                df_entry = preprocessing_logic(df_entry)

            score = loaded_model.predict(df_entry)[0]

            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            raise RuntimeError("Failed to select best model candidate.")

        # Train 
        # best_name = "logreg"

        best_model, best_acc, best_size = run_best_candidate(
            best_name,
            self.X_train, self.y_train,
            self.X_val, self.y_val,
        )
        if best_name == "xgboost":
            try:
                import xgboost as xgb
                if isinstance(best_model, xgb.XGBClassifier):
                    self._xgb_sklearn_model = best_model
            except Exception:
                self._xgb_sklearn_model = None

        if MODELS[best_name][0] == "classic":
            test_pred = best_model.predict(self.X_test)
            test_acc = accuracy_score(self.y_test, test_pred)
        else:
            test_acc = float(best_acc)

        self.best_model_name = best_name
        self.model_kind = MODELS[best_name][0]
        self._final_model = best_model

        self.state.metrics.update({
            "best_model_name": best_name,
            "val_acc": float(best_acc),
            "test_acc": float(test_acc),
            "model_size_mb": float(best_size),
            "accuracy": float(test_acc),
            "model_name": best_name,
        })

        if self.model_kind == "classic":
            try:
                if best_name == "xgboost":
                    import xgboost as xgb
                    base_bst = best_model.get_booster() if hasattr(best_model, "get_booster") else best_model
                    self.state.metrics["sizeKBBefore"] = _size_kb_xgb(base_bst)
                else:
                    self.state.metrics["sizeKBBefore"] = _size_kb_classic(best_model)
            except Exception:
                self.state.metrics["sizeKBBefore"] = float(best_size) * 1024.0
        else:
            self.state.metrics["sizeKBBefore"] = float(best_size) * 1024.0

        if self.model_kind == "deep":
            Xtr = torch.tensor(np.asarray(self.X_train), dtype=torch.float32)
            ytr = torch.tensor(np.asarray(self.y_train), dtype=torch.long)
            Xva = torch.tensor(np.asarray(self.X_val), dtype=torch.float32)
            yva = torch.tensor(np.asarray(self.y_val), dtype=torch.long)
            Xte = torch.tensor(np.asarray(self.X_test), dtype=torch.float32)
            yte = torch.tensor(np.asarray(self.y_test), dtype=torch.long)

            self._final_train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=1, shuffle=False)
            self._final_val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=1, shuffle=False)
            self._final_test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=1, shuffle=False)

    # ==========================================================
    # Phase 3: optimizing 
    # ==========================================================

    def _phase_optimizing(self, best_model=None, best_model_name=None):

        self.state.metrics["optimization_failed"] = False
        self.state.metrics["optimization_fail_reason"] = None
        self.state.metrics["optimization_skipped"] = False
        self.state.metrics["optimization_skip_reason"] = None
        self.state.metrics["optimization_history"] = []
        self.state.metrics["optimization_attempted_levels"] = []
        self.state.metrics["optimization_attempt_accuracies"] = []
        
        import json
        import warnings
        import time
        import numpy as np
        import copy
        import os
        import tempfile
        try:
            import pandas as pd
        except ImportError:
            pass
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        try:
            import xgboost as xgb
        except ImportError:
            xgb = None
        try:
            from torch.ao.quantization import get_default_qconfig_mapping
            from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
            import torch.ao.quantization
            from torch import nn
            import torch
        except ImportError:
            pass

        def tabular_fp16(model: nn.Module) -> nn.Module:
            m = copy.deepcopy(model)
            m.eval()
            return m.half()

        def tabular_int8_dynamic(model: nn.Module) -> nn.Module:
            m = copy.deepcopy(model)
            m = m.eval().cpu()
            qmodel = torch.ao.quantization.quantize_dynamic(
                m,
                qconfig_spec={nn.Linear, nn.GRU},
                dtype=torch.qint8
            )
            return qmodel

        def tabular_int8_static(model: nn.Module, calibration_loader, num_calib_batches: int = 20, backend: str = "fbgemm") -> nn.Module:
            m = copy.deepcopy(model)
            m = m.eval().cpu()
            if hasattr(m, "conv") and hasattr(m, "relu"):
                try: torch.ao.quantization.fuse_modules(m, [["conv", "relu"]], inplace=True)
                except: pass
            if hasattr(m, "net") and isinstance(m.net, nn.Sequential):
                try: torch.ao.quantization.fuse_modules(m.net, [["0", "1"], ["2", "3"]], inplace=True)
                except: pass
            m = torch.ao.quantization.QuantWrapper(m)
            torch.backends.quantized.engine = backend
            m.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            prepared = torch.ao.quantization.prepare(m, inplace=False)
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= num_calib_batches: break
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch
                    prepared(x.cpu().float())
            quantized = torch.ao.quantization.convert(prepared, inplace=False)
            return quantized

        def build_lr_model(model, abs_weight_below=1e-4, dtype=np.float32):
            from sklearn.pipeline import Pipeline as SkPipeline
            optimized_model = copy.deepcopy(model)
            if isinstance(optimized_model, SkPipeline):
                clf = optimized_model.named_steps.get("clf", optimized_model.steps[-1][1])
            else:
                clf = optimized_model
            clf.coef_ = clf.coef_.astype(dtype, copy=False)
            clf.intercept_ = clf.intercept_.astype(dtype, copy=False)
            if abs_weight_below > 0:
                w = clf.coef_
                mask = np.abs(w) < abs_weight_below
                if np.any(mask):
                    w = w.copy()
                    w[mask] = 0.0
                    clf.coef_ = w
            return optimized_model

        def _to_xgb_data(X):
            if isinstance(X, pd.DataFrame):
                return X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            try:
                import scipy.sparse as _sp
                if _sp.issparse(X): return X.tocsr()
            except: pass
            return np.asarray(X, dtype=np.float32)

        def _xgb_feature_names(bst, X) -> Optional[list]:
            try: names = getattr(bst, "feature_names", None)
            except: names = None
            if names: return list(map(str, names))
            if isinstance(X, pd.DataFrame): return [str(c) for c in X.columns]
            try: return [str(i) for i in range(int(X.shape[1]))]
            except: return None

        def _get_size_mb(model) -> float:
            try:
                if hasattr(model, "state_dict"):
                    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        torch.save(model.state_dict(), tmp_path)
                        size = os.path.getsize(tmp_path) / (1024 * 1024)
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                    return size
                
                suffix = ".json" if (xgb and isinstance(model, (xgb.Booster, xgb.XGBClassifier))) else ".joblib"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    if xgb and isinstance(model, (xgb.XGBClassifier, xgb.Booster)):
                        bst = model.get_booster() if hasattr(model, "get_booster") else model
                        bst.save_model(tmp_path)
                    else:
                        import joblib
                        joblib.dump(model, tmp_path)
                    size = os.path.getsize(tmp_path) / (1024 * 1024)
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                return size
            except Exception:
                return 0.0

        def _estimate_peak_ram(model, sample_input=None, device="cpu"):

            static_kb = _get_size_mb(model) * 1024.0
            dynamic_kb = 0.0

            if hasattr(model, "forward"):
                try:
                    activations = []
                    def hook_fn(module, input, output):
                        if isinstance(output, torch.Tensor):
                            activations.append(output.element_size() * output.numel())
                        elif isinstance(output, (list, tuple)):
                            for o in output:
                                if isinstance(o, torch.Tensor):
                                    activations.append(o.element_size() * o.numel())
                    
                    hooks = []
                    for name, module in model.named_modules():
                        if len(list(module.children())) == 0:
                             hooks.append(module.register_forward_hook(hook_fn))
                    
                    if sample_input is not None:
                         try:
                             p = next(model.parameters())
                             target_dev = p.device
                             target_dtype = p.dtype
                         except:
                             target_dev = device
                             target_dtype = torch.float32

                         if isinstance(sample_input, (list, tuple)):
                             inp = sample_input[0].to(target_dev)
                         else:
                             inp = sample_input.to(target_dev)
                         
                         if target_dtype == torch.float16:
                             inp = inp.half()
                         else:
                             inp = inp.float()

                         with torch.no_grad():
                             model(inp)
                    
                    if activations:
                        dynamic_kb = max(activations) / 1024.0
                    
                    for h in hooks: h.remove()
                    
                except Exception as e:
                    pass

            else:
                 overhead_kb = 100.0 
                 input_kb = 0.0
                 if sample_input is not None:
                     try:
                         input_kb = sample_input.nbytes / 1024.0
                     except: 
                        try: input_kb = (sample_input.element_size() * sample_input.numel()) / 1024.0
                        except: pass

                 if xgb and isinstance(model, (xgb.Booster, xgb.XGBClassifier)):
                     dynamic_kb = (input_kb * 2.5) + overhead_kb
                 else:
                     dynamic_kb = (input_kb * 2.0) + overhead_kb

            return static_kb, dynamic_kb

        def _meets_constraints(size_mb, ram_static_kb=None, ram_dynamic_kb=None) -> bool:
            specs = self.device_specs or {}
            flash_limit = specs.get("flash_mb", 0)
            
            if flash_limit and flash_limit > 0:
                if size_mb is None or size_mb > flash_limit: 
                    return False

            ram_limit = specs.get("ram_kb", 0)
            if ram_limit and ram_limit > 0:
                 if ram_static_kb is not None or ram_dynamic_kb is not None:
                    #  usage = (ram_static_kb or 0) + (ram_dynamic_kb or 0)
                     usage = (ram_dynamic_kb or 0)
                     if usage > ram_limit: 
                         return False
            
            return True

        def _probe_forward(model, xb, device) -> Tuple[bool, str]:
            try:
                if isinstance(model, nn.Module):
                    try:
                        p = next(model.parameters())
                        target_dtype = p.dtype
                        target_device = p.device
                    except StopIteration:
                        # Quantized models have no parameters
                        target_dtype = torch.float32
                        target_device = device
                    except Exception:
                        target_dtype = torch.float32
                        target_device = device

                    inp = xb.to(target_device)
                    if target_dtype == torch.float16: inp = inp.half()
                    else: inp = inp.float()
                    model(inp)
                return True, "ok"
            except Exception as e:
                return False, str(e)

        def _measure_latency_deep(model, loader, device, runs=50) -> float:
            xb, _ = next(iter(loader))
            _probe_forward(model, xb, device)
            latencies = []
            with torch.no_grad():
                for _ in range(runs):
                    t0 = time.perf_counter()
                    _probe_forward(model, xb, device)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
            return (sum(latencies) / len(latencies)) / max(1, xb.size(0))

        def _evaluate_acc_deep(model, loader, device) -> float:
            correct = 0; total = 0
            model.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    yb = yb.to(device)
                    valid, _ = _probe_forward(model, xb, device)
                    if not valid: continue
                    try:
                        try:
                            p = next(model.parameters())
                            target_device = p.device
                            target_dtype = p.dtype
                        except StopIteration:
                            target_device = device
                            target_dtype = torch.float32

                        inp = xb.to(target_device)
                        if target_dtype == torch.float16: inp = inp.half()
                        else: inp = inp.float()

                        out = model(inp)
                        preds = out.argmax(dim=1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                    except Exception: 
                        pass
            return float(correct) / max(1, total)

        def _measure_latency_xgb_dmatrix(bst, X, n_warmup=10, n_runs=100) -> float:
            X_xgb = _to_xgb_data(X)
            fn = _xgb_feature_names(bst, X)
            dm = xgb.DMatrix(X_xgb, feature_names=fn)
            for _ in range(n_warmup): bst.predict(dm)
            t0 = time.perf_counter()
            for _ in range(n_runs): bst.predict(dm)
            total_ms = (time.perf_counter() - t0) * 1000.0
            return (total_ms / n_runs) / max(1, dm.num_row())

        def _measure_latency_sklearn_generic(model, X, runs=50) -> float:
            X_batch = X[:1]
            try: model.predict(X_batch)
            except: pass
            t0 = time.perf_counter()
            for _ in range(runs): model.predict(X_batch)
            return ((time.perf_counter() - t0) * 1000.0 / runs) / max(1, len(X_batch))

        if best_model is None: return None
        is_deep = best_model_name in ["cnn1d", "tiny_rnn", "mlp", "tinyconv"]
        
        strat = (self.optimization_strategy or "Balanced").lower()
        q_user = (self.quantization or "Automatic").lower()

        if q_user == "auto": q_user = "automatic"
        acc_tol = (self.accuracy_tolerance or "Automatic").lower()
        
        self.state.metrics["optimization_strategy"] = strat.capitalize()
        self.state.metrics["quantization_requested"] = q_user.capitalize()
        self.state.metrics["accuracy_tolerance"] = acc_tol.capitalize()
        
        user_cap = 1.0 
        if self.accuracy_drop_cap is not None:
            try: user_cap = float(self.accuracy_drop_cap)
            except: pass
        self.state.metrics["accuracy_drop_cap"] = f"{user_cap*100:.1f}%"
        self.state.metrics["accuracy_drop_allowed"] = user_cap 

        base_acc = 0.0
        base_size = 0.0
        base_lat = 0.0
        base_model = best_model
        
        candidates = [] 

        try:
            if is_deep:
                try:
                    p = next(base_model.parameters())
                    device = p.device
                except: device = torch.device("cpu")
                base_acc = _evaluate_acc_deep(base_model, self._final_test_loader, device)
                base_size = _get_size_mb(base_model)
                base_lat = _measure_latency_deep(base_model, self._final_test_loader, device)
                
                xb_sample, _ = next(iter(self._final_test_loader))
                base_st, base_dy = _estimate_peak_ram(base_model, xb_sample, device)

            else:
                if best_model_name == "xgboost":
                    bst = best_model.get_booster() if hasattr(best_model, "get_booster") else best_model
                    dtest = xgb.DMatrix(_to_xgb_data(self.X_test), feature_names=_xgb_feature_names(bst, self.X_test))
                    preds = bst.predict(dtest)
                    if preds.ndim > 1: y_hat = np.argmax(preds, axis=1)
                    else: y_hat = (preds > 0.5).astype(int)
                    base_acc = accuracy_score(self.y_test, y_hat)
                    base_size = _get_size_mb(bst)
                    base_lat = _measure_latency_xgb_dmatrix(bst, self.X_test)
                    base_model = bst 
                    
                    base_st, base_dy = _estimate_peak_ram(base_model, self.X_test[:1])

                else:
                    preds = best_model.predict(self.X_test)
                    base_acc = accuracy_score(self.y_test, preds)
                    base_size = _get_size_mb(best_model)
                    base_lat = _measure_latency_sklearn_generic(best_model, self.X_test)

                    base_st, base_dy = _estimate_peak_ram(base_model, self.X_test[:1])

        except Exception as e:
            self.state.metrics["optimization_failed"] = True
            self.state.metrics["optimization_fail_reason"] = f"Baseline eval failed: {e}"
            return best_model

        self.state.metrics["accuracyBefore"] = base_acc
        self.state.metrics["sizeKBBefore"] = base_size * 1024.0
        self.state.metrics["latencyMsBefore"] = base_lat

        met_constraints = _meets_constraints(base_size, base_st, base_dy)
        cand_hist = {
            "name": "Baseline (Original)",
            "status": "success" if met_constraints else "rejected",
            "acc": base_acc, "size_mb": base_size, "latency_ms": base_lat,
            "size_kb": base_size * 1024, "latency_us": base_lat * 1000,
            "ram_static_kb": base_st, "ram_dynamic_kb": base_dy,
            "reason": "" if met_constraints else "Baseline exceeds constraints"
        }
        self.state.metrics["optimization_history"].append(cand_hist)
        
        cand_full = cand_hist.copy()
        cand_full["model_obj"] = base_model
        candidates.append(cand_full)

        if q_user == "automatic" and strat in ["balanced", "auto"] and met_constraints:
            self.state.metrics["optimization_skipped"] = True
            self.state.metrics["optimization_skip_reason"] = "Baseline already meets strict constraints (Flash & RAM)"
            self.state.metrics["optimization_level"] = "Baseline"
            self._final_model = base_model
            self.state.metrics["accuracyAfter"] = base_acc
            self.state.metrics["sizeKBAfter"] = base_size * 1024.0
            self.state.metrics["latencyMsAfter"] = base_lat
            return base_model

        # ---------------------------------------------------------
        # 4. Deep Optimization 
        # ---------------------------------------------------------
        if is_deep:
            jobs = []
            if q_user in ["automatic", "float16"]:
                if torch.cuda.is_available():
                    jobs.append(("FP16", tabular_fp16, "cuda"))
                else:
                    self.state.metrics["optimization_history"].append({
                        "name": "FP16", "status": "skipped", 
                        "reason": "CUDA not available", "acc": None, "size_kb": None, "latency_us": None
                    })
            if q_user in ["automatic", "dynamic int8"]:
                jobs.append(("INT8 Dynamic", tabular_int8_dynamic, "cpu"))
            if q_user in ["automatic", "static int8"]:
                jobs.append(("INT8 Static", lambda m: tabular_int8_static(m, self._final_train_loader), "cpu"))

            for name, fn, dev_str in jobs:
                try:
                    m = copy.deepcopy(base_model)
                    m_opt = fn(m)
                    dev = torch.device(dev_str)
                    
                    # Probe
                    xb_probe, _ = next(iter(self._final_test_loader))
                    ok, r = _probe_forward(m_opt, xb_probe, dev)
                    if not ok:
                        self.state.metrics["optimization_history"].append({"name": name, "status": "failed", "reason": str(r)})
                        continue
                        
                    acc = _evaluate_acc_deep(m_opt, self._final_test_loader, dev)
                    sz = _get_size_mb(m_opt)
                    lat = _measure_latency_deep(m_opt, self._final_test_loader, dev)

                    st, dy = _estimate_peak_ram(m_opt, xb_probe, dev)
                    
                    drop_pct = (base_acc - acc) / base_acc * 100.0 if base_acc > 0 else 0
                    cap_pct = user_cap * 100.0
                    meets_constraints = _meets_constraints(sz, st, dy)
                    
                    status = "rejected"
                    reason = ""
                    if drop_pct > cap_pct:
                        reason = f"Drop {drop_pct:.1f}% > Cap {cap_pct:.1f}%"
                    elif not meets_constraints:
                        reason = "Size/RAM limit exceeded"
                    else:
                        status = "success"
                    
                    rec_history = {
                        "name": name, "status": status, "acc": acc, "size_mb": sz, "latency_ms": lat,
                        "size_kb": sz * 1024, "latency_us": lat * 1000,
                        "ram_static_kb": st, "ram_dynamic_kb": dy,
                        "drop_pct": drop_pct, "reason": reason
                    }
                    self.state.metrics["optimization_history"].append(rec_history)
                    
                    if status == "success": 
                        rec_cand = rec_history.copy()
                        rec_cand["model_obj"] = m_opt
                        candidates.append(rec_cand)
                except Exception as e:
                    self.state.metrics["optimization_history"].append({"name": name, "status": "failed", "reason": str(e)})

        # ---------------------------------------------------------
        # 5. Classic Optimization 
        # ---------------------------------------------------------
        else:
            if best_model_name == "xgboost":
                import xgboost as xgb
                base_bst = None
                if hasattr(best_model, "get_booster"):
                    base_bst = best_model.get_booster()
                elif isinstance(best_model, xgb.Booster):
                    base_bst = best_model
                
                if base_bst is None:
                    self.state.metrics["optimization_history"].append({"name": "XGB Init", "status": "failed", "reason": "Baseline booster not found"})
                else:
                    fn_train = _xgb_feature_names(base_bst, self.X_train)
                    dtrain = xgb.DMatrix(_to_xgb_data(self.X_train), label=self.y_train, feature_names=fn_train)
                    
                    fn_test = _xgb_feature_names(base_bst, self.X_test)
                    dtest = xgb.DMatrix(_to_xgb_data(self.X_test), feature_names=fn_test)

                    def _xgb_acc(bst, dm):
                        preds = bst.predict(dm)
                        if hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[1] > 1:
                            y_hat = np.argmax(preds, axis=1)
                        else:
                            y_hat = (preds > 0.5).astype(int)
                        return float(accuracy_score(self.y_test, y_hat))

                    base_acc_xgb = base_acc 
                    base_params = {"tree_method": "hist", "seed": 42, "objective": "binary:logistic", "eval_metric": "logloss"}
                    try:
                        cfg = json.loads(base_bst.save_config())
                        obj = cfg.get("learner", {}).get("objective", {}).get("name")
                        if obj: base_params["objective"] = obj
                        nc = cfg.get("learner", {}).get("learner_model_param", {}).get("num_class")
                        if nc: base_params["num_class"] = int(nc)
                        
                        if (obj and obj.startswith("multi")) or (base_params.get("num_class", 0) > 1):
                             base_params["eval_metric"] = "mlogloss"
                    except: pass

                    X_tr, X_val, y_tr, y_val = train_test_split(_to_xgb_data(self.X_train), self.y_train, test_size=0.2, random_state=42, stratify=self.y_train)
                    dm_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=fn_train)
                    dm_val = xgb.DMatrix(X_val, label=y_val, feature_names=fn_train)

                    def _size_kb(bst):
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
                                tmp_path = tmp.name
                            bst.save_model(tmp_path)
                            sz = os.path.getsize(tmp_path) / 1024.0
                            try: os.remove(tmp_path)
                            except: pass
                            return sz
                        except: return 0.0
                    
                    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 350.0, 500.0, 750.0, 1000.0]
                    found_best = False
                    
                    for g in gammas:
                        if found_best: break
                        
                        try:
                            p = base_params.copy()
                            p.update({"eta": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "max_depth": 6, "gamma": g})
                            bst_cand = xgb.train(p, dm_tr, num_boost_round=200, evals=[(dm_val, "val")], verbose_eval=False, early_stopping_rounds=25)
                            
                            acc = _xgb_acc(bst_cand, dtest)
                            sz_kb = _size_kb(bst_cand)
                            lat = _measure_latency_xgb_dmatrix(bst_cand, self.X_test)
                            
                            st, dy = _estimate_peak_ram(bst_cand, self.X_test[:1])

                            drop_pct = (base_acc_xgb - acc) / base_acc_xgb * 100.0 if base_acc_xgb > 0 else 0.0
                            
                            cap_pct = user_cap * 100.0
                            meets_constraints = _meets_constraints(sz_kb / 1024.0, st, dy)

                            status = "rejected"
                            reason = ""
                            if drop_pct > cap_pct: reason = f"Drop {drop_pct:.1f}% > Cap {cap_pct:.1f}%"
                            elif not meets_constraints: reason = "Exceeds Flash/RAM"
                            else:
                                status = "success"
                                reason = "Selected (First Valid)"
                            
                            rec_hist = {
                                "name": f"XGB Gamma={g}", "status": status,
                                "acc": acc, "size_kb": sz_kb, "latency_us": lat * 1000.0,
                                "size_mb": sz_kb/1024.0, "latency_ms": lat,
                                "ram_static_kb": st, "ram_dynamic_kb": dy,
                                "gamma": g, "method": "retrain", "drop_pct": drop_pct, "reason": reason
                            }
                            self.state.metrics["optimization_history"].append(rec_hist)
                            
                            rec_cand = rec_hist.copy()
                            rec_cand["model_obj"] = bst_cand
                            candidates.append(rec_cand)
                            
                            if status == "success":
                                self.state.metrics["optimization_level"] = f"Gamma={g}"
                                self.state.metrics["accuracy"] = acc
                                self.state.metrics["accuracyAfter"] = acc
                                self.state.metrics["sizeKBAfter"] = sz_kb
                                self.state.metrics["latencyMsAfter"] = lat 
                                best_model = bst_cand
                                found_best = True
                                break 
                        except Exception as e:
                            self.state.metrics["optimization_history"].append({"name": f"XGB Gamma={g}", "status": "failed", "reason": str(e)})

                    if not found_best:
                        for pg in gammas:
                            if found_best: break
                            try:
                                r = xgb.train({"process_type": "update", "updater": "refresh", "refresh_leaf": True}, dtrain=dm_tr, num_boost_round=1, xgb_model=base_bst)
                                bst_p = xgb.train({"process_type": "update", "updater": "prune", "gamma": pg, "max_depth": 6}, dtrain=dm_tr, num_boost_round=1, xgb_model=r)
                                
                                acc = _xgb_acc(bst_p, dtest)
                                sz_kb = _size_kb(bst_p)
                                lat = _measure_latency_xgb_dmatrix(bst_p, self.X_test)
                                
                                st, dy = _estimate_peak_ram(bst_p, self.X_test[:1])

                                drop = (base_acc_xgb - acc) / base_acc_xgb * 100.0 if base_acc_xgb > 0 else 0.0
                                
                                cap_pct = user_cap * 100.0
                                meets_constraints = _meets_constraints(sz_kb / 1024.0, st, dy)
                                
                                status = "rejected"
                                reason = ""
                                if drop > cap_pct: reason = f"Drop {drop:.1f}% > Cap {cap_pct:.1f}%"
                                elif not meets_constraints: reason = "Exceeds Flash/RAM"
                                else:
                                    status = "success"
                                    reason = "Selected (Pruned Valid)"
                                    
                                rec_hist = {
                                    "name": f"XGB Pruned Gamma={pg}", "status": status,
                                    "acc": acc, "size_kb": sz_kb, "latency_us": lat * 1000.0,
                                    "size_mb": sz_kb/1024.0, "latency_ms": lat,
                                    "ram_static_kb": st, "ram_dynamic_kb": dy,
                                    "gamma": pg, "method": "prune", "drop_pct": drop, "reason": reason
                                }
                                self.state.metrics["optimization_history"].append(rec_hist)
                                
                                rec_cand = rec_hist.copy()
                                rec_cand["model_obj"] = bst_p
                                candidates.append(rec_cand)
                                
                                if status == "success":
                                    found_best = True
                                    break 
                            except Exception as e:
                                self.state.metrics["optimization_history"].append({"name": f"XGB Pruned Gamma={pg}", "status": "failed", "reason": str(e)})

                    if not found_best:
                        self.state.metrics["optimization_level"] = "baseline"

            elif best_model_name == "rf":
                try:
                    current_est = getattr(base_model, "n_estimators", 100)
                    for n in [200, 150, 100, 50, 25, 15, 10, 5, 1]:
                        if n > current_est: continue 
                        
                        name = f"RF n_est={n}"
                        m = copy.deepcopy(base_model)
                        m.estimators_ = m.estimators_[:n]
                        m.n_estimators = n
                        
                        acc = accuracy_score(self.y_test, m.predict(self.X_test))
                        sz = _get_size_mb(m)
                        lat = _measure_latency_sklearn_generic(m, self.X_test)
                        
                        st, dy = _estimate_peak_ram(m, self.X_test[:1])
                        
                        drop_pct = (base_acc - acc) / base_acc * 100.0 if base_acc > 0 else 0
                        cap_pct = user_cap * 100.0
                        meets_constraints = _meets_constraints(sz, st, dy)
                        
                        status = "rejected"
                        reason = ""
                        if drop_pct > cap_pct:
                            reason = f"Drop {drop_pct:.1f}% > Cap {cap_pct:.1f}%"
                        elif not meets_constraints:
                            reason = "Exceeds Flash/RAM"
                        else:
                            status = "success"
                        
                        rec_history = {
                            "name": name, "status": status,
                            "acc": acc, "size_mb": sz, "latency_ms": lat,
                            "size_kb": sz * 1024, "latency_us": lat * 1000,
                            "ram_static_kb": st, "ram_dynamic_kb": dy,
                            "drop_pct": drop_pct, "reason": reason
                        }
                        self.state.metrics["optimization_history"].append(rec_history)
                        if status == "success":
                            rec_cand = rec_history.copy()
                            rec_cand["model_obj"] = m
                            candidates.append(rec_cand)
                            break  
                except Exception as e:
                     self.state.metrics["optimization_history"].append({"name": "RF Pruning", "status": "failed", "reason": str(e)})

            elif best_model_name == "logreg":
                try:
                    thresholds = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
                    for thresh in thresholds:
                        name = f"LogReg Thresh={thresh:.0e}"
                        m = build_lr_model(base_model, abs_weight_below=thresh)
                        
                        acc = accuracy_score(self.y_test, m.predict(self.X_test))
                        sz = _get_size_mb(m)
                        lat = _measure_latency_sklearn_generic(m, self.X_test)

                        st, dy = _estimate_peak_ram(m, self.X_test[:1])
                        
                        drop_pct = (base_acc - acc) / base_acc * 100.0 if base_acc > 0 else 0
                        cap_pct = user_cap * 100.0
                        meets_constraints = _meets_constraints(sz, st, dy)
                        
                        status = "rejected"
                        reason = ""
                        if drop_pct > cap_pct:
                            reason = f"Drop {drop_pct:.1f}% > Cap {cap_pct:.1f}%"
                        elif not meets_constraints:
                            reason = "Exceeds Flash/RAM"
                        else:
                            status = "success"
                        
                        rec_history = {
                            "name": name, "status": status,
                            "acc": acc, "size_mb": sz, "latency_ms": lat,
                            "size_kb": sz * 1024, "latency_us": lat * 1000,
                            "ram_static_kb": st, "ram_dynamic_kb": dy,
                            "drop_pct": drop_pct, "reason": reason
                        }
                        self.state.metrics["optimization_history"].append(rec_history)
                        
                        if status == "success":
                            rec_cand = rec_history.copy()
                            rec_cand["model_obj"] = m
                            candidates.append(rec_cand)
                            break  
                except Exception as e:
                    self.state.metrics["optimization_history"].append({"name": "LogReg Sweep", "status": "failed", "reason": str(e)})

        # ---------------------------------------------------------
        # 6. Selection Strategy
        # ---------------------------------------------------------
        valid_cands = [c for c in candidates if c["status"] == "success"]
        if not valid_cands:
            all_cands = [c for c in candidates if c.get("model_obj") is not None]
            if all_cands:
                all_cands.sort(key=lambda c: (c.get("size_mb", float('inf')), -c.get("acc", 0)))
                best_failed = all_cands[0]
                
                self._final_model = best_failed["model_obj"]
                self.state.metrics["optimization_failed"] = True
                self.state.metrics["optimization_fail_reason"] = "No candidates met constraints. Using best failed."
                self.state.metrics["accuracyAfter"] = best_failed["acc"]
                self.state.metrics["sizeKBAfter"] = (best_failed.get("size_mb") or 0) * 1024.0
                self.state.metrics["latencyMsAfter"] = best_failed.get("latency_ms")
                self.state.metrics["optimization_level"] = f"{best_failed['name']} (best failed)"
                
                if is_deep:
                    self.state.metrics["quantization_applied"] = best_failed["name"]
                    self.best_model_name = f"{best_model_name}_{best_failed['name'].lower().replace(' ', '_')}"
                    self.state.metrics["best_model_name"] = self.best_model_name
                else:
                    self.state.metrics["quantization_applied"] = "none"

                return self._final_model
            
            self.state.metrics["optimization_failed"] = True
            self.state.metrics["optimization_fail_reason"] = "No candidates available."
            self.state.metrics["accuracyAfter"] = base_acc
            self.state.metrics["sizeKBAfter"] = base_size * 1024.0
            self.state.metrics["optimization_level"] = "baseline"
            return best_model

        def _score(c):
            acc = c["acc"]; lat = c["latency_ms"] or 1e-9; sz = c["size_mb"] or 1e-9

            if "balanced" in strat: return (acc * 1000) - (lat * 0.01) - (sz * 0.01)
            if "accuracy" in strat and "latency" in strat: return (acc * 1000) - (lat * 0.1)  
            if "ram" in strat and "size" in strat: return -(sz + sz) 
            if "accuracy" in strat: return acc  
            if "latency" in strat: return -lat  
            if "size" in strat: return -sz  
            if "ram" in strat: return -sz  
            return acc  # Default fallback

        valid_cands.sort(key=_score, reverse=True)
        selected = valid_cands[0]

        self._final_model = selected["model_obj"]
        self.state.metrics["accuracyAfter"] = selected["acc"]
        self.state.metrics["sizeKBAfter"] = (selected["size_mb"] or 0) * 1024.0
        self.state.metrics["latencyMsAfter"] = selected["latency_ms"]
        self.state.metrics["optimization_level"] = selected["name"]
        self.state.metrics["quantization_applied"] = selected["name"] if is_deep else "none"
        
        if is_deep:
            self.best_model_name = f"{best_model_name}_{selected['name'].lower().replace(' ', '_')}"
            self.state.metrics["best_model_name"] = self.best_model_name

        # Compat
        self.state.metrics["optimization_attempted_levels"] = [c["name"] for c in candidates]
        self.state.metrics["optimization_attempt_accuracies"] = [
            (c["name"], c.get("acc", 0.0), (c.get("size_mb") or 0.0)*1024.0) for c in self.state.metrics["optimization_history"] if c.get("acc") is not None
        ]

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
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
        warnings.filterwarnings("ignore", message=r".*deprecated.*", module="tensorflow")
        warnings.filterwarnings("ignore", message=r".*tf\.executing_eagerly_outside_functions.*deprecated.*")
        warnings.filterwarnings("ignore", message=r".*tf\.logging\..*deprecated.*")
        warnings.filterwarnings("ignore", message=r".*tf\.control_flow_v2_enabled.*deprecated.*")
        warnings.filterwarnings("ignore", message=r".*tf\.losses\..*deprecated.*", category=UserWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_probability")
        warnings.filterwarnings("ignore", category=UserWarning, module="keras")
        warnings.filterwarnings("ignore", message=r".*Exporting a model to ONNX.*GRU.*batch_size.*", category=UserWarning)

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
                m = getattr(self, "state", None)
                if m is not None and hasattr(m, "metrics"):
                    for k in ("model_name", "selected_model", "final_model_name", "best_model_name"):
                        v = m.metrics.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
            except Exception:
                pass
            return "unknown_model"

        MODEL_NAME = _infer_model_name()

        def _is_torch_model(model) -> bool:
            try:
                import torch
            except Exception:
                return False
            return isinstance(model, torch.nn.Module)

        def _is_tf_model(model) -> bool:
            try:
                import tensorflow as tf
            except Exception:
                return False
            return isinstance(model, tf.keras.Model)

        def _is_classic_model(model) -> bool:
            return (not _is_torch_model(model)) and (not _is_tf_model(model))

        def _is_xgboost_model(model) -> bool:
            try:
                import xgboost as xgb
                return isinstance(model, (xgb.XGBClassifier, xgb.Booster))
            except Exception:
                return False

        def _is_sklearn_pipeline(obj) -> bool:
            try:
                from sklearn.pipeline import Pipeline
                return isinstance(obj, Pipeline)
            except Exception:
                return False

        def _unwrap_pipeline_scaler_and_clf(model):
            scaler = None
            clf = model
            if _is_sklearn_pipeline(model):
                steps = list(model.steps)
                clf = steps[-1][1]
                for _, step in steps:
                    if step.__class__.__name__ in ("StandardScaler", "MinMaxScaler", "RobustScaler"):
                        scaler = step
                        break
            return scaler, clf

        def _is_logreg_or_pipeline_logreg(model) -> bool:
            try:
                from sklearn.linear_model import LogisticRegression
            except Exception:
                return False
            _, clf = _unwrap_pipeline_scaler_and_clf(model)
            return isinstance(clf, LogisticRegression)

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
            if self._final_train_loader is None:
                raise RuntimeError("train_loader missing; cannot create export input.")
            try:
                xb, _ = next(iter(self._final_train_loader))
            except Exception as e:
                raise RuntimeError(f"Could not fetch batch for export: {e}")
            if not isinstance(xb, torch.Tensor):
                raise RuntimeError("Export sample is not a torch.Tensor")
            try:
                model_device = next(self._final_model.parameters()).device
            except Exception:
                model_device = getattr(self, "device", torch.device("cpu"))
            x = xb[:1].to(model_device)
            try:
                if next(self._final_model.parameters()).dtype == torch.float16:
                    x = x.half()
            except Exception:
                pass
            return x

        def export_onnx_torch(model, sample_input, out_base, opset: int = 11):
            import torch
            from pathlib import Path

            out_path = Path(out_base).with_suffix(".onnx")
            
            model = model.cpu().float()
            sample_input = sample_input.cpu().float()
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

        def export_onnx_sklearn(model, out_base):
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
            except Exception as e:
                raise RuntimeError(f"skl2onnx missing: {e}")

            if getattr(self, "X_train", None) is None:
                raise RuntimeError("X_train is None; cannot infer n_features for skl2onnx export")

            Xtr = self.X_train
            try:
                if hasattr(Xtr, "shape"):
                    n_features = int(Xtr.shape[1])
                else:
                    n_features = len(Xtr.columns) if hasattr(Xtr, "columns") else len(Xtr[0])
            except Exception as e:
                raise RuntimeError(f"Could not infer n_features: {e}")

            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onx = convert_sklearn(model, initial_types=initial_type)
            out_path = Path(out_base).with_suffix(".onnx")
            out_path.write_bytes(onx.SerializeToString())
            return out_path

        def export_onnx_xgboost(model, out_base):

            try:
                from onnxmltools.convert import convert_xgboost
                from onnxmltools.convert.common.data_types import FloatTensorType
            except Exception as e:
                raise RuntimeError(f"onnxmltools missing for xgboost->onnx export: {e}")

            if getattr(self, "X_train", None) is None:
                raise RuntimeError("X_train is None; cannot infer n_features for xgboost->onnx export")

            Xtr = self.X_train
            try:
                if hasattr(Xtr, "shape"):
                    n_features = int(Xtr.shape[1])
                else:
                    n_features = len(Xtr.columns) if hasattr(Xtr, "columns") else len(Xtr[0])
            except Exception as e:
                raise RuntimeError(f"Could not infer n_features: {e}")

            initial_type = [("float_input", FloatTensorType([None, n_features]))]

            try:
                onx = convert_xgboost(model, initial_types=initial_type)
            except TypeError:
                onx = convert_xgboost(model, initial_type=initial_type)

            out_path = Path(out_base).with_suffix(".onnx")
            out_path.write_bytes(onx.SerializeToString())
            return out_path

        def _classic_linear_to_keras_model(classic_model):
            import numpy as np
            import tensorflow as tf

            scaler, clf = _unwrap_pipeline_scaler_and_clf(classic_model)

            if not (hasattr(clf, "coef_") and hasattr(clf, "intercept_")):
                raise TypeError(f"Not a LogisticRegression-like model: {type(clf)}")

            coef = np.asarray(clf.coef_, dtype=np.float32)
            intercept = np.asarray(clf.intercept_, dtype=np.float32)

            if coef.ndim != 2:
                raise RuntimeError(f"Unexpected coef_ shape: {coef.shape}")

            n_classes = int(coef.shape[0])
            n_features = int(coef.shape[1])

            if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                mean = np.asarray(scaler.mean_, dtype=np.float32)
                scale = np.asarray(scaler.scale_, dtype=np.float32)
                coef = coef / scale.reshape(1, -1)
                intercept = intercept - (coef @ mean.reshape(-1, 1)).reshape(-1)

            inp = tf.keras.Input(shape=(n_features,), name="input")

            if n_classes == 1:
                out = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True, name="dense")(inp)
                km = tf.keras.Model(inp, out)
                km.get_layer("dense").set_weights([coef.T.astype(np.float32), intercept.reshape(1).astype(np.float32)])
                return km

            out = tf.keras.layers.Dense(n_classes, activation="softmax", use_bias=True, name="dense")(inp)
            km = tf.keras.Model(inp, out)
            km.get_layer("dense").set_weights([coef.T.astype(np.float32), intercept.reshape(n_classes).astype(np.float32)])
            return km

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


        def _convert_tiny_rnn_to_keras(torch_model, seq_len):
            import numpy as np
            import tensorflow as tf
            
            sd = torch_model.state_dict()
            
            w_ih = sd["rnn.weight_ih_l0"].cpu().numpy().T
            w_hh = sd["rnn.weight_hh_l0"].cpu().numpy().T
            b_ih = sd["rnn.bias_ih_l0"].cpu().numpy()
            b_hh = sd["rnn.bias_hh_l0"].cpu().numpy()
            b = b_ih + b_hh
            
            w_fc = sd["fc.weight"].cpu().numpy().T
            b_fc = sd["fc.bias"].cpu().numpy()
            
            hidden_dim = w_hh.shape[0]
            n_classes = w_fc.shape[1]
            
            inp = tf.keras.Input(shape=(seq_len,), name="input")
            x = tf.keras.layers.Reshape((seq_len, 1))(inp)
            
            x = tf.keras.layers.LSTM(
                units=hidden_dim,
                return_sequences=False,
                use_bias=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                unroll=False 
            )(x)
            
            out = tf.keras.layers.Dense(n_classes, activation="linear")(x)
            
            k_model = tf.keras.Model(inp, out)
            
            lstm_layer = k_model.layers[2]
            dense_layer = k_model.layers[3]
            
            if not isinstance(lstm_layer, tf.keras.layers.LSTM):
                raise RuntimeError(f"Expected LSTM layer at index 2, got {type(lstm_layer)}")
            
            lstm_layer.set_weights([w_ih, w_hh, b])
            dense_layer.set_weights([w_fc, b_fc])
            
            return k_model

        def _torch_to_tflite_via_onnx(model, out_base: str, quantize: bool) -> Path:

            if type(model).__name__ == "TinyRNNNet":
                try:
                    sample_input = _get_sample_batch_for_export()
                    seq_len = int(sample_input.shape[1])
                    keras_model = _convert_tiny_rnn_to_keras(model, seq_len)
                    return export_tflite_from_keras(keras_model, out_base, quantize)
                except Exception as e:
                    print(f"TinyRNN Keras fallback failed: {e}. Falling back to ONNX.")

            import copy
            model = copy.deepcopy(model).cpu().float()

            sample_input = _get_sample_batch_for_export()
            if hasattr(sample_input, "float"): 
                 sample_input = sample_input.cpu().float()
            else:
                 sample_input = sample_input.cpu()
                 
            onnx_path = export_onnx_torch(model, sample_input, out_base, opset=11)

            try:
                import onnx
                _ensure_onnx_mapping_shim(onnx)
                from onnx_tf.backend import prepare
            except Exception as e:
                raise RuntimeError(f"onnx / onnx-tf not available for torch->tflite conversion: {e}")

            _patch_tf_compat_modules()

            tmp_dir = tempfile.mkdtemp(prefix="onnx_tf_savedmodel_")
            try:
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

        def export_tflite_from_keras(model, out_base, quantize: bool = True):
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

        def export_bin_from_tflite(tflite_path, out_base):
            data = Path(tflite_path).read_bytes()
            return _write_bytes(out_base, data, ".bin")

        def export_h_from_bin(bin_path, out_base):
            data = Path(bin_path).read_bytes()
            return _bytes_to_c_header(data, out_base, var_name="model_data")

        def _should_quantize() -> bool:
            q = self.state.metrics.get("quantization_applied")
            if q is None and hasattr(self, "quantization"):
                q = self.quantization
            if q is None:
                return True
            qs = str(q).strip().lower()
            return qs not in ("none", "false", "0", "fp32", "float32", "no")

        def export_bin_classic(model, out_base):
            if _is_xgboost_model(model):
                tmp = Path(out_base).with_suffix(".xgb.json")
                model.save_model(str(tmp))
                raw = tmp.read_bytes()
                try:
                    tmp.unlink()
                except Exception:
                    pass
                return _write_bytes(out_base, raw, ".bin")

            try:
                import joblib
            except Exception as e:
                raise RuntimeError(f"joblib missing for classic .bin export: {e}")

            buf = io.BytesIO()
            joblib.dump(model, buf)
            raw = buf.getvalue()
            return _write_bytes(out_base, raw, ".bin")

        def export_h_classic(model, out_base):
            try:
                import m2cgen as m2c
                c_code = m2c.export_to_c(model)
                p = Path(out_base).with_suffix(".h")
                header = (
                    "#ifndef MODEL_C_CODE_H\n"
                    "#define MODEL_C_CODE_H\n\n"
                    "// Generated by m2cgen (C inference code)\n\n"
                    f"{c_code}\n\n"
                    "#endif\n"
                )
                p.write_text(header)
                return p
            except Exception:
                bin_path = export_bin_classic(model, out_base)
                data = Path(bin_path).read_bytes()
                return _bytes_to_c_header(data, out_base, var_name="model_bytes")

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
                    if _is_torch_model(model):
                        sample_input = _get_sample_batch_for_export()
                        p = export_onnx_torch(model, sample_input, out_base)
                        return True, "", str(p)
                    if _is_xgboost_model(model):
                        p = export_onnx_xgboost(model, out_base)
                        return True, "", str(p)
                    if _is_classic_model(model):
                        p = export_onnx_sklearn(model, out_base)
                        return True, "", str(p)
                    return _reason("ONNX export requires torch or classic model")

                if ext == ".tflite":
                    q = _should_quantize()

                    if _is_tf_model(model):
                        p = export_tflite_from_keras(model, out_base, quantize=q)
                        return True, "", str(p)

                    if _is_classic_model(model) and _is_logreg_or_pipeline_logreg(model):
                        km = _classic_linear_to_keras_model(model)
                        p = export_tflite_from_keras(km, out_base, quantize=q)
                        return True, "", str(p)

                    if _is_torch_model(model):
                        if self._final_model is not None:
                            try:
                                self._final_model_fp32 = copy.deepcopy(self._final_model)
                            except Exception:
                                self._final_model_fp32 = self._final_model
                        else:
                            self._final_model_fp32 = None
                        model_to_export = model
                        if hasattr(self, "_final_model_fp32") and self._final_model_fp32 is not None:
                             model_to_export = self._final_model_fp32

                        try:
                            p = _torch_to_tflite_via_onnx(model_to_export, out_base, quantize=q)
                            return True, "", str(p)
                        except Exception as e:
                            reason = f"torch->tflite conversion failed: {type(e).__name__}: {e}"
                            if "tiny_rnn" in MODEL_NAME.lower():
                                explain = (
                                    "tiny_rnn torch->tflite uses ONNX->TF conversion (onnx-tf) then TFLite conversion. "
                                    "RNN operators (e.g., LSTM/GRU/RNN blocks) are commonly represented differently "
                                    "between PyTorch->ONNX and TensorFlow graphs. The ONNX->TF bridge may not implement "
                                    "some RNN-related ops or may require specific opset/version combinations. "
                                    "If this happens, the export will fail even though ONNX export succeeds. "
                                    "Mitigations: (1) replace RNN with 1D conv/TCN, (2) implement the model in tf.keras, "
                                    "(3) pin compatible versions of onnx/onnx-tf/tensorflow/opset and re-export."
                                )
                                self.state.metrics["tflite_failure_tiny_rnn_reason"] = reason
                                self.state.metrics["tflite_failure_tiny_rnn_explain"] = explain
                            return _reason(reason)

                    return _reason(f".tflite export not supported for model type: {type(model)}")

                if ext == ".bin":
                    q = _should_quantize()

                    if _is_tf_model(model) or (_is_classic_model(model) and _is_logreg_or_pipeline_logreg(model)) or _is_torch_model(model):
                        ok, reason, tflite_path = _export_ext(model, out_base, ".tflite")
                        if not ok:
                            return _reason(f".bin requires .tflite first; reason: {reason}")
                        p = export_bin_from_tflite(Path(tflite_path), out_base)
                        return True, "", str(p)

                    if _is_classic_model(model):
                        p = export_bin_classic(model, out_base)
                        return True, "", str(p)

                    return _reason(".bin export not supported for this model type")

                if ext == ".h":
                    if _is_tf_model(model) or (_is_classic_model(model) and _is_logreg_or_pipeline_logreg(model)) or _is_torch_model(model):
                        ok, reason, bin_path = _export_ext(model, out_base, ".bin")
                        if not ok:
                            return _reason(f".h requires .bin first; reason: {reason}")
                        p = export_h_from_bin(Path(bin_path), out_base)
                        return True, "", str(p)

                    if _is_classic_model(model):
                        p = export_h_classic(model, out_base)
                        return True, "", str(p)

                    return _reason(".h export not supported for this model type")

                if ext == ".kmodel":
                    return _reason("not supported right now")
                if ext == ".engine":
                    return _reason("not supported right now")

                return _reason("extension not supported")

            except Exception as e:
                return False, f"{type(e).__name__}: {e}", ""

        requested = _normalize_ext(self.export_ext)
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
        self.state.metrics["export_ext_attempted"] = list(exts_to_try)
        self.state.metrics["packaging_model_name"] = MODEL_NAME

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

        if ".tflite" in export_errors:
            self.state.metrics["tflite_export_failed"] = True
            self.state.metrics["tflite_export_failure_reason"] = export_errors[".tflite"]

        if not exported_paths:
            raise RuntimeError(
                f"No exports succeeded. "
                f"family={self.device_family_id}, "
                f"requested={requested}, "
                f"errors={export_errors}"
            )

        self._final_model_path = exported_paths[0]

        self.state.metrics["exported_exts"] = sorted({Path(p).suffix.lower() for p in exported_paths})

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
                self.state.metrics["export_zip_path"] = zip_path
                self._final_model_path = zip_path
            except Exception as e:
                self.state.metrics["export_zip_error"] = str(e)

            if self._final_model_path.endswith(".zip") and os.path.isfile(self._final_model_path):
                try:
                    zsize = float(os.path.getsize(self._final_model_path) / (1024 * 1024))
                    sizes_mb[os.path.basename(self._final_model_path)] = zsize
                except Exception:
                    pass

        # Final metrics
        try:
            final_name = os.path.basename(self._final_model_path)
            final_size = sizes_mb.get(final_name)
        except Exception:
            final_size = None

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

    # ==========================================================
    # Phase 5: report
    # ==========================================================

    def _phase_report(self) -> None:
        if self._prep_obj is None:
            raise RuntimeError("Preprocessing state missing; cannot generate report.")
        
        import os
        import tempfile
        import datetime
        from collections import Counter
        
        from PIL import Image as PILImage
        
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
            Image as RLImage, LongTable
        )
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        
        def _wrap_token(s: str) -> str:
            return str(s)
        
        styles = getSampleStyleSheet()
        H1 = ParagraphStyle("H1", parent=styles["Heading1"], alignment=1, spaceAfter=10)
        H2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=6, spaceAfter=8)
        body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=6, leading=13)
        small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=6)
        caption = ParagraphStyle("cap", parent=styles["BodyText"], fontSize=9, leading=11, alignment=1, spaceAfter=10)
        sub = ParagraphStyle("sub", parent=body, leftIndent=18, spaceAfter=6)
        
        cover_title = ParagraphStyle("cover_title", parent=styles["Title"], alignment=1, fontSize=24, spaceAfter=18, leading=30)
        cover_subtitle = ParagraphStyle("cover_subtitle", parent=styles["h2"], alignment=1, fontSize=16, spaceAfter=12, leading=20)
        cover_meta = ParagraphStyle("cover_meta", parent=styles["Normal"], alignment=1, fontSize=10, textColor=colors.gray, spaceAfter=24)
        cover_desc = ParagraphStyle("cover_desc", parent=styles["Normal"], alignment=1, fontSize=12, leading=16, spaceAfter=0)
        
        mono_wrap = ParagraphStyle(
            "mono_wrap",
            parent=small,
            fontName="Courier",
            fontSize=7.5,
            leading=9,
            wordWrap="CJK",
            splitLongWords=1,
        )

        def _fmt_num(v, places=3):
            try:
                return f"{float(v):.{places}f}"
            except Exception:
                return str(v) if v not in (None, "") else ""

        def _fmt_latency_ms(v):
            try:
                f = float(v)
            except Exception:
                return str(v) if v not in (None, "") else ""
            if f == 0.0:
                return "0.000"
            if f < 0.001:
                return f"{f * 1000.0:.2f} µs"
            return f"{f:.3f}"


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

        def _convert_image_to_png(img_path: str) -> str:
            img = PILImage.open(img_path).convert("RGBA")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.close()
            img.save(tmp.name, format="PNG")
            return tmp.name
        
        def _save_fig_to_png(fig) -> str:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.close()
            fig.savefig(
                tmp.name,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.06,
                facecolor="white",
            )
            plt.close(fig)
            return tmp.name
        
        def _fit_rl_image(img_path: str, max_w: float, max_h: float) -> RLImage:
            im = PILImage.open(img_path)
            w, h = im.size
            im.close()
            scale = min(max_w / float(w), max_h / float(h))
            return RLImage(img_path, width=w * scale, height=h * scale)
        
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
        CHART_MAX_H = 6.8 * cm
        
        def _style_axes(ax, grid_axis: str):
            ax.set_axisbelow(True)
            ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_linewidth(0.8)
                ax.spines[s].set_alpha(0.7)
            ax.tick_params(axis="both", which="both", length=3, width=0.8)
            return ax
        
        def _add_bar_labels(ax, bars, labels, orientation: str):
            try:
                if orientation == "v":
                    ax.bar_label(bars, labels=labels, padding=6, fontsize=8)
                else:
                    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
            except Exception:
                pass
        
        def _chart_missingness(missing_frac: dict, top_n: int = 12, threshold: float | None = None) -> str | None:
            if not missing_frac:
                return None
            items = sorted(missing_frac.items(), key=lambda x: x[1], reverse=True)[:top_n]
            if not items or max(v for _, v in items) <= 0:
                return None
        
            cols_, vals_ = zip(*items)
            vals_pct = [v * 100 for v in vals_]
        
            def _short(s: str, n: int = 22) -> str:
                s = str(s)
                return s if len(s) <= n else (s[: n - 1] + "…")
        
            cols_disp = [_short(c) for c in cols_]
        
            fig, ax = plt.subplots(figsize=(7.1, 3.9))
            bars = ax.barh(range(len(cols_disp)), vals_pct)
            ax.set_yticks(range(len(cols_disp)))
            ax.set_yticklabels(cols_disp)
            ax.invert_yaxis()
            ax.set_xlim(0, max(100, max(vals_pct) * 1.15))
            ax.set_xlabel("Missing values (%)")
            ax.set_title("Missingness (top columns)")
            _style_axes(ax, grid_axis="x")
        
            if threshold is not None:
                try:
                    t = float(threshold) * 100.0
                    ax.axvline(t, linestyle=":", linewidth=1.2)
                    ax.text(t, ax.get_ylim()[0], f"  drop @ {t:.0f}%", va="bottom", fontsize=8)
                except Exception:
                    pass
        
            _add_bar_labels(ax, bars, [f"{v:.1f}%" for v in vals_pct], orientation="h")
            fig.tight_layout()
            return _save_fig_to_png(fig)
        
        def _chart_coltype_counts(col_types: dict) -> str | None:
            if not col_types:
                return None
            c = Counter(col_types.values())
        
            preferred = ["numeric", "categorical", "datetime"]
            labels = [k for k in preferred if k in c] + [k for k in sorted(c.keys()) if k not in preferred]
            values = [c[k] for k in labels]
            if sum(values) == 0:
                return None
        
            fig, ax = plt.subplots(figsize=(6.6, 3.4))
            bars = ax.bar(labels, values, width=0.58)
            ax.set_ylabel("Number of columns")
            ax.set_title("Feature type composition (before encoding)")
            _style_axes(ax, grid_axis="y")
            _add_bar_labels(ax, bars, [str(v) for v in values], orientation="v")
            fig.tight_layout()
            return _save_fig_to_png(fig)
        
        def _chart_class_counts(class_counts: dict) -> str | None:
            if not class_counts:
                return None
            items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            labels = [str(k) for k, _ in items]
            values = [v for _, v in items]
            total = sum(values)
            if total == 0:
                return None
        
            fig, ax = plt.subplots(figsize=(6.9, 3.8))
            bars = ax.bar(labels, values, width=0.58)
            ax.set_ylabel("Samples")
            ax.set_title("Target class distribution")
            _style_axes(ax, grid_axis="y")

            bar_labels = [f"{v}\n({(v/total)*100:.1f}%)" for v in values]
            _add_bar_labels(ax, bars, bar_labels, orientation="v")

            ax.tick_params(axis="x", rotation=0)
            ax.margins(y=0.18)
            fig.tight_layout()
            return _save_fig_to_png(fig)
        
        def _chart_feature_source_breakdown(feature_names: list[str]) -> str | None:
            if not feature_names:
                return None
        
            def src(n: str) -> str:
                return n.split("__", 1)[0] if "__" in n else "features"
        
            c = Counter(src(n) for n in feature_names)
            items = sorted(c.items(), key=lambda x: x[1], reverse=True)
            labels = [k for k, _ in items]
            values = [v for _, v in items]
            if sum(values) == 0:
                return None
        
            fig, ax = plt.subplots(figsize=(6.8, 3.4))
            bars = ax.barh(labels, values, height=0.58)
            ax.invert_yaxis()
            ax.set_xlabel("Number of output features")
            ax.set_title("Final feature contribution by transformer")
            _style_axes(ax, grid_axis="x")
            _add_bar_labels(ax, bars, [str(v) for v in values], orientation="h")
            fig.tight_layout()
            return _save_fig_to_png(fig)
        
        def _generate_sensor_report(
            prep,
            path: str = "prep_report.pdf",
            report_data: Optional[Dict[str, Any]] = None,
            deployment_data: Optional[Dict[str, Any]] = None,
        ):
            meta = getattr(prep, "meta_", None) or {}
            meta_used = getattr(prep, "meta_used_", None) or {}
            applied = getattr(prep, "applied_", None) or {}
            cfg = getattr(prep, "config_", None)
            report_data = report_data or {}
            deployment_data = deployment_data or {}
        
            doc = SimpleDocTemplate(
                path,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=3.1*cm,
                bottomMargin=2.0*cm
            )
        
            story = []
            tmp_files = []
        
            logo_path = getattr(prep, "logo_path", None)
            logo_png = None
            if logo_path and os.path.exists(logo_path):
                try:
                    logo_png = _convert_image_to_png(logo_path)
                    tmp_files.append(logo_png)
                except Exception:
                    logo_png = None
        
            def draw_header_footer(canvas, doc_):
                canvas.saveState()
        
                header_top = doc_.pagesize[1] - 1.0*cm
                header_bottom = doc_.pagesize[1] - doc_.topMargin + 0.25*cm
        
                if logo_png:
                    lw, lh = (1.25*cm, 1.25*cm)
                    x = doc_.leftMargin
                    y = header_top - lh
                    canvas.drawImage(
                        logo_png, x, y,
                        width=lw, height=lh,
                        preserveAspectRatio=True, mask="auto"
                    )
        
                title_y = doc_.pagesize[1] - 1.35*cm
                if logo_png:
                    title_y = header_top - (lh / 2.0) - (0.35 * cm)
                canvas.setFont("Helvetica-Bold", 12)
                canvas.drawCentredString(
                    doc_.pagesize[0] / 2.0,
                    title_y,
                    "Automata-AI Sensor Report"
                )
        
                canvas.setLineWidth(0.4)
                canvas.setStrokeColor(colors.grey)
                canvas.line(doc_.leftMargin, header_bottom, doc_.pagesize[0] - doc_.rightMargin, header_bottom)
        
                canvas.setFont("Helvetica", 8)
                canvas.setFillColor(colors.black)
                canvas.drawString(doc_.leftMargin, 1.15*cm, f"Page {doc_.page}")
                canvas.drawRightString(
                    doc_.pagesize[0] - doc_.rightMargin,
                    1.15*cm,
                    f"© {datetime.datetime.now().year} Automata AI — All rights reserved"
                )
        
                canvas.restoreState()
        
            story.append(Spacer(1, 2.6 * cm))
        
            if logo_png:
                cover_logo = _fit_rl_image(logo_png, max_w=6.5 * cm, max_h=6.5 * cm)
                cover_logo.hAlign = "CENTER"
                story.append(cover_logo)
        
            story.append(Spacer(1, 1.2 * cm))
        
            story.append(Paragraph("Automata-AI Sensor Report", cover_title))
            story.append(Paragraph("Automated Pipeline Report", cover_subtitle))
            story.append(Paragraph(
                f"Generated on {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                cover_meta
            ))
        
            story.append(Spacer(1, 1.2 * cm))
        
            story.append(Paragraph(
                "This report summarizes dataset characteristics, preprocessing steps, "
                "training results, optimization decisions, and export artifacts for sensor models.",
                cover_desc
            ))

            story.append(PageBreak())
            story.append(Paragraph("1. Dataset Overview", H2))
        
            n_rows = meta_used.get("n_rows", meta.get("n_rows", ""))
            n_cols_raw = meta.get("n_cols", "")
            n_cols_used = meta_used.get("n_cols", "")
        
            col_types_used = meta_used.get("col_types", meta.get("col_types", {})) or {}
            missing_used = meta_used.get("missing_frac", meta.get("missing_frac", {})) or {}
            card_used = meta_used.get("cardinality", meta.get("cardinality", {})) or {}
        
            type_counts = Counter(col_types_used.values()) if col_types_used else Counter()
            dropped_cols = applied.get("drop_cols", []) or []
            dt_cols = applied.get("datetime_cols", []) or []
            low_cols = applied.get("low_card_cols", []) or []
            high_cols = applied.get("high_card_cols", []) or []
        
            avg_missing = (sum(missing_used.values()) / max(len(missing_used), 1)) if missing_used else 0.0
            max_missing = max(missing_used.values()) if missing_used else 0.0
        
            overview_rows = [
                ["Samples", str(n_rows)],
                ["Raw features (before drops)", str(n_cols_raw)],
                ["Features used (after drops/datetime)", str(n_cols_used)],
                ["Numeric columns", str(type_counts.get("numeric", 0))],
                ["Categorical columns", str(type_counts.get("categorical", 0))],
                ["Datetime columns (detected)", str(type_counts.get("datetime", 0))],
                ["Dropped columns", str(len(dropped_cols))],
                ["Avg missing rate (across columns)", f"{avg_missing*100:.2f}%"],
                ["Max missing rate (single column)", f"{max_missing*100:.2f}%"],
                ["Low-card categorical columns", str(len(low_cols))],
                ["High-card categorical columns", str(len(high_cols))],
            ]
        
            if "class_counts" in meta:
                counts = meta.get("class_counts", {}) or {}
                ir = meta.get("imbalance_ratio", None)
                overview_rows += [
                    ["# Classes", str(len(counts))],
                    ["Imbalance ratio", f"{float(ir):.3f}" if ir is not None else ""],
                ]
        
            t = Table(overview_rows, colWidths=[7.5*cm, 8.5*cm])
            t.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.Color(0.97,0.97,0.97)]),
                ("LEFTPADDING", (0,0), (-1,-1), 6),
                ("RIGHTPADDING", (0,0), (-1,-1), 6),
                ("TOPPADDING", (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4*cm))
        
            chart_paths = []
            try:
                p1 = _chart_coltype_counts(col_types_used)
                if p1: chart_paths.append(("Column types", p1))
                p2 = _chart_missingness(missing_used, top_n=12, threshold=getattr(cfg, "drop_missing_threshold", None))
                if p2: chart_paths.append(("Missingness", p2))
                p3 = _chart_class_counts(meta.get("class_counts", {}) if "class_counts" in meta else {})
                if p3: chart_paths.append(("Target classes", p3))
                for _, p in chart_paths:
                    tmp_files.append(p)
        
                for title, p in chart_paths:
                    img = _fit_rl_image(p, max_w=doc.width, max_h=CHART_MAX_H)
                    story.append(img)
                    story.append(Paragraph(title, caption))
                    story.append(Spacer(1, 0.45*cm))
            except Exception:
                pass
        
            if dropped_cols:
                story.append(Paragraph(f"<b>Dropped columns ({len(dropped_cols)}):</b> {', '.join(map(str, dropped_cols[:80]))}"
                                       + (" ..." if len(dropped_cols) > 80 else "") + ".",
                    small))
            if dt_cols and applied.get("datetime_handling", "") == "drop":
                story.append(Paragraph(f"<b>Datetime columns dropped ({len(dt_cols)}):</b> {', '.join(map(str, dt_cols[:80]))}"
                                       + (" ..." if len(dt_cols) > 80 else "") + ".",
                    small))
        
            if cfg is not None:
                cfg_rows = [
                    ["drop_missing_threshold", str(getattr(cfg, "drop_missing_threshold", ""))],
                    ["high_cardinality_min_unique", str(getattr(cfg, "high_cardinality_min_unique", ""))],
                    ["high_cardinality_threshold", str(getattr(cfg, "high_cardinality_threshold", ""))],
                    ["numeric_imputer", str(getattr(cfg, "numeric_imputer", ""))],
                    ["numeric_scaler", str(getattr(cfg, "numeric_scaler", ""))],
                    ["categorical_imputer", str(getattr(cfg, "categorical_imputer", ""))],
                    ["datetime_handling", str(getattr(cfg, "datetime_handling", ""))],
                    ["feature_selection", str(getattr(cfg, "feature_selection", ""))],
                    ["feature_fraction", str(getattr(cfg, "feature_fraction", ""))],
                    ["balancing", str(getattr(cfg, "balancing", ""))],
                    ["imbalance_threshold", str(getattr(cfg, "imbalance_threshold", ""))],
                ]
                story.append(Spacer(1, 0.2*cm))
                story.append(Paragraph("Configuration Snapshot", ParagraphStyle("h3", parent=styles["Heading3"], spaceAfter=6)))
                tc = Table(cfg_rows, colWidths=[7.5*cm, 8.5*cm])
                tc.setStyle(TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                    ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.Color(0.97,0.97,0.97)]),
                    ("LEFTPADDING", (0,0), (-1,-1), 6),
                    ("RIGHTPADDING", (0,0), (-1,-1), 6),
                    ("TOPPADDING", (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ]))
                story.append(tc)
        
            story.append(PageBreak())
            story.append(Paragraph("2. Preprocessing Steps Applied", H2))
        
            if applied.get("drop_cols"):
                story.append(Paragraph(
                    f"<b>• Column removal:</b> Columns with excessive missing values or constant values were removed. "
                    f"Dropped: <b>{', '.join(applied['drop_cols'][:120])}</b>"
                    + (" ..." if len(applied["drop_cols"]) > 120 else "") + ".",
                    body
                ))
        
            if applied.get("datetime_cols"):
                story.append(Paragraph(
                    f"<b>• Datetime handling:</b> Detected datetime columns: <b>{', '.join(applied['datetime_cols'])}</b>. "
                    f"Handling mode: <b>{applied.get('datetime_handling','')}</b>.",
                    body
                ))
                if applied.get("datetime_generated_cols"):
                    story.append(Paragraph(
                        f"– Extracted datetime parts into: <b>{', '.join(applied['datetime_generated_cols'])}</b>.",
                        sub
                    ))
        
            if applied.get("numeric_cols"):
                story.append(Paragraph(
                    f"<b>• Numeric processing:</b> Numeric features: <b>{', '.join(applied['numeric_cols'][:80])}</b>"
                    + (" ..." if len(applied["numeric_cols"]) > 80 else "") + ".",
                    body
                ))
                if applied.get("numeric_imputer_used"):
                    story.append(Paragraph(
                        f"– Missing numeric values imputed using <b>{getattr(cfg,'numeric_imputer','')}</b> for: "
                        f"<b>{', '.join(applied.get('numeric_missing_cols', []))}</b>.",
                        sub
                    ))
                if applied.get("numeric_scaler_used"):
                    story.append(Paragraph(
                        f"– Scaling applied using <b>{getattr(cfg,'numeric_scaler','')}</b>.",
                        sub
                    ))
        
            if applied.get("low_card_cols"):
                story.append(Paragraph(
                    f"<b>• Low-card categorical:</b> <b>{', '.join(applied['low_card_cols'][:80])}</b>"
                    + (" ..." if len(applied["low_card_cols"]) > 80 else "") + ".",
                    body
                ))
                if applied.get("low_card_imputer_used"):
                    story.append(Paragraph(
                        f"– Missing values imputed using <b>{getattr(cfg,'categorical_imputer','')}</b>.",
                        sub
                    ))
                story.append(Paragraph("– One-hot encoding applied.", sub))
        
            if applied.get("high_card_cols"):
                story.append(Paragraph(
                    f"<b>• High-card categorical:</b> <b>{', '.join(applied['high_card_cols'][:80])}</b>"
                    + (" ..." if len(applied["high_card_cols"]) > 80 else "") + ".",
                    body
                ))
                story.append(Paragraph("– Frequency encoding applied to avoid feature explosion.", sub))
        
            if applied.get("feature_selection_used"):
                story.append(Paragraph(
                    f"<b>• Feature selection:</b> Method <b>{applied.get('feature_selection_method','')}</b>, "
                    f"kept fraction <b>{applied.get('feature_fraction','')}</b>, k = <b>{applied.get('fs_k','')}</b>.",
                    body
                ))
        
            if applied.get("balancing_used"):
                story.append(Paragraph(
                    f"<b>• Imbalance handling:</b> Class weights computed (threshold {applied.get('imbalance_threshold','')}).",
                    body
                ))
        
            story.append(PageBreak())

            story.append(Paragraph("3. Output Feature Summary", H2))
        
            try:
                out_names = list(prep.get_feature_names_out())
            except Exception:
                out_names = list(getattr(prep, "output_feature_names_", []) or [])
        
            story.append(Paragraph(f"The preprocessing pipeline produced <b>{len(out_names)}</b> final features.", body))
        
            try:
                p_feat = _chart_feature_source_breakdown(out_names)
                if p_feat:
                    tmp_files.append(p_feat)
                    img = _fit_rl_image(p_feat, max_w=doc.width, max_h=CHART_MAX_H)
                    story.append(img)
                    story.append(Paragraph("Final feature contribution by transformer (count)", caption))
            except Exception:
                pass
        
            if out_names:
                story.append(Spacer(1, 0.15*cm))
        
                def _grp(n: str) -> str:
                    n = str(n)
                    if n.startswith("num__"):
                        return "Numeric"
                    if n.startswith("cat_low__"):
                        return "Low-card categorical (one-hot)"
                    if n.startswith("cat_high__"):
                        return "High-card categorical (frequency)"
                    if "__" in n:
                        return n.split("__", 1)[0]
                    return "Other"
        
                groups: dict[str, list[str]] = {}
                for n in out_names:
                    groups.setdefault(_grp(n), []).append(str(n))
        
                preferred_order = ["Numeric", "Low-card categorical (one-hot)", "High-card categorical (frequency)"]
                ordered_groups = [g for g in preferred_order if g in groups] + [g for g in sorted(groups.keys()) if g not in preferred_order]
        
                summary_rows = [["Group", "Count", "Examples (first 3)"]]
                for g in ordered_groups:
                    feats = groups[g]
                    ex = ", ".join(_wrap_token(x) for x in feats[:3]) + (" ..." if len(feats) > 3 else "")
                    summary_rows.append([Paragraph(g, small), str(len(feats)), Paragraph(ex, mono_wrap)])
        
                summary_tbl = Table(summary_rows, colWidths=[6.0*cm, 1.6*cm, doc.width - 7.6*cm])
                summary_tbl.setStyle(TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                    ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING", (0,0), (-1,-1), 6),
                    ("RIGHTPADDING", (0,0), (-1,-1), 6),
                    ("TOPPADDING", (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.Color(0.97,0.97,0.97)]),
                ]))
                story.append(summary_tbl)
                story.append(Spacer(1, 0.35*cm))
        
                max_items_per_group = 240
                ncols = 3
                colw = doc.width / float(ncols)
        
                for gi, g in enumerate(ordered_groups):
                    feats = groups[g]
                    shown = feats[:max_items_per_group]
                    truncated = len(feats) > max_items_per_group
        
                    header = Paragraph(f"<b>{g} ({len(feats)})</b>" + (f" — showing first {max_items_per_group}" if truncated else ""), small)
                    data = [[header, "", ""]]
                    for i in range(0, len(shown), ncols):
                        row = shown[i:i+ncols]
                        if len(row) < ncols:
                            row += [""] * (ncols - len(row))
                        data.append([Paragraph(_wrap_token(x), mono_wrap) if x else "" for x in row])
        
                    tf = LongTable(data, colWidths=[colw]*ncols, repeatRows=1)
                    tf.setStyle(TableStyle([
                        ("SPAN", (0,0), (-1,0)),
                        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
                        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                        ("VALIGN", (0,0), (-1,-1), "TOP"),
                        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.Color(0.97,0.97,0.97)]),
                        ("LEFTPADDING", (0,0), (-1,-1), 5),
                        ("RIGHTPADDING", (0,0), (-1,-1), 5),
                        ("TOPPADDING", (0,0), (-1,-1), 3),
                        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                    ]))
                    story.append(tf)
                    if gi < len(ordered_groups) - 1:
                        story.append(Spacer(1, 0.30*cm))

            # Model Report 
            if report_data:
                story.append(PageBreak())
                story.append(Paragraph("4. Model Report", H2))

                ds_name = str(report_data.get("dataset_name", ""))
                fam_name = str(report_data.get("device_family_name", ""))
                fam_id = str(report_data.get("device_family_id", ""))
                saved_as = str(report_data.get("saved_as", ""))
                attempted = report_data.get("attempted_exts", []) or []
                attempted_str = ", ".join(attempted) if attempted else ""
                exported_paths = report_data.get("exported_paths", []) or []

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
                    dlat_txt, dlat_col = "+0.0%", colors.red if failed_flag else colors.green
                    dsz_txt, dsz_col = "+0.0%", colors.red if failed_flag else colors.green
                else:
                    dlat_txt, dlat_col = _delta(lat_b, lat_a)
                    dsz_txt, dsz_col = _delta(sz_b, sz_a)

                cards = [
                    ["Accuracy", _fmt_acc(acc), "", ""],
                    ["Latency (ms / sample)", _fmt_latency_ms(lat_a), "After optimizing", dlat_txt],
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
                ba_rows = [
                    ["Metric", "Before", "After", "Delta"],
                    ["Accuracy", _fmt_acc(acc_b), _fmt_acc(acc_a), acc_delta],
                    ["Latency per sample", _fmt_latency_ms(lat_b), _fmt_latency_ms(lat_a), dlat_txt],
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
                    [_p("Model name"), _p(report_data.get("model_name", "")), _p("Device family"), _p(f"{fam_name} ({fam_id})")],
                    [_p("Dataset"), _p(ds_name), _p("Export requested"), _p(report_data.get("export_ext_requested", ""))],
                    [_p("Export allowed"), _p(", ".join(report_data.get("export_ext_allowed", []) or [])), _p("Saved as"), _p(saved_as)],
                    [_p("Target name"), _p(report_data.get("target_name", "")), _p("Classes"), _p(report_data.get("target_num_classes", ""))],
                    [_p("Optimization strategy"), _p(report_data.get("optimization_strategy", "")),_p("Optimize if size >"), _p(report_data.get("optimization_trigger_ratio_pct", ""))],
                    [_p("Accuracy tolerance"), _p(report_data.get("accuracy_tolerance", "")), _p("Allowed accuracy drop"), _p(report_data.get("accuracy_drop_allowed_pct", ""))],
                    [_p("Quantization requested"), _p(report_data.get("quantization_requested", "")), _p("Quantization applied"), _p(report_data.get("quantization_applied_display", ""))],
                    [_p("Optimization skipped"), _p(report_data.get("optimization_skipped_display", "")), _p("Skip reason"), _p(report_data.get("optimization_skip_reason_display", ""))],
                    [_p("Optimization failed"), _p(report_data.get("optimization_failed_display", "")), _p("Failure reason"), _p(report_data.get("optimization_fail_reason_display", ""))],
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

                fb_parts = []
                min_acc = report_data.get("min_accuracy")
                if acc is not None and min_acc is not None:
                    try:
                        if float(acc) >= float(min_acc):
                            fb_parts.append(f"Accuracy meets the minimum requirement ({_fmt_acc(acc)}).")
                        else:
                            fb_parts.append(f"Accuracy is below the minimum requirement ({_fmt_acc(acc)}).")
                    except Exception:
                        pass
                if dsz_txt:
                    fb_parts.append(f"Model size change: {dsz_txt}.")
                if dlat_txt:
                    fb_parts.append(f"Latency change: {dlat_txt}.")
                attempts_sentence = ""
                attempts_disp = report_data.get("optimization_attempted_levels_display", "")
                attempts_acc = report_data.get("optimization_attempt_accuracies_display", "")
                if attempts_disp:
                    if attempts_acc:
                        attempts_sentence = f"Optimization attempts: {attempts_disp} (accuracy: {attempts_acc})."
                    else:
                        attempts_sentence = f"Optimization attempts: {attempts_disp}."
                feedback_text = " ".join([p for p in fb_parts if p] + ([attempts_sentence] if attempts_sentence else [])) or "Model summary based on measured metrics."

                opt_history = report_data.get("optimization_history", [])
                if opt_history:
                    story.append(Spacer(1, 0.35 * cm))
                    story.append(Paragraph("Optimization Details", H2))
                    
                    hist_rows = [["Candidate", "Status", "Accuracy", "Size (KB)", "Latency (µs)", "Reason"]]
                    
                    for item in opt_history:
                        name = str(item.get("name", "unknown"))
                        if len(name) > 40: name = name[:40] + "..."
                        status = str(item.get("status", "unknown"))
                        if len(status) > 20: status = status[:20] + "..."
                        
                        acc_val = item.get("acc")
                        sz_val = item.get("size_kb")
                        if sz_val is None and item.get("size_mb") is not None:
                            sz_val = float(item.get("size_mb")) * 1024
                        lat_val = item.get("latency_us")
                        if lat_val is None and item.get("latency_ms") is not None:
                            lat_val = float(item.get("latency_ms")) * 1000
                        reason_val = str(item.get("reason", "") or "")
                        if len(reason_val) > 150:
                            reason_val = reason_val[:150] + "..."
                        
                        acc_str = f"{float(acc_val)*100:.1f}%" if acc_val is not None else "-"
                        sz_str = f"{float(sz_val):.2f}" if sz_val is not None else "-"
                        lat_str = f"{float(lat_val):.2f}" if lat_val is not None else "-"
                        
                        status_style = small
                        if status == "success":
                            status_style = ParagraphStyle("s_ok", parent=small, textColor=colors.green)
                        elif status == "failed":
                            status_style = ParagraphStyle("s_fail", parent=small, textColor=colors.red)
                        elif status == "rejected":
                            status_style = ParagraphStyle("s_rej", parent=small, textColor=colors.orange)
                        elif status == "skipped":
                            status_style = ParagraphStyle("s_skip", parent=small, textColor=colors.gray)
                            
                        hist_rows.append([
                            Paragraph(name, small),
                            Paragraph(status, status_style),
                            Paragraph(acc_str, small),
                            Paragraph(sz_str, small),
                            Paragraph(lat_str, small),
                            Paragraph(reason_val, ParagraphStyle("reason", parent=small, fontSize=7, leading=8)),
                        ])
                    
                    hist_tbl = Table(hist_rows, colWidths=[
                        doc.width * 0.20, 
                        doc.width * 0.12, 
                        doc.width * 0.12, 
                        doc.width * 0.12, 
                        doc.width * 0.15, 
                        doc.width * 0.29  
                    ])
                    hist_tbl.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]))
                    story.append(hist_tbl)
                    story.append(Spacer(1, 0.35 * cm))

                story.append(Paragraph("Model Feedback", H2))
                story.append(Paragraph(feedback_text, body))

            if deployment_data:
                story.append(PageBreak())
                story.append(Paragraph("5. Deployment Guidance", H2))

                fam_name = deployment_data.get("family_name", "")
                fam_id = deployment_data.get("family_id", "")
                frameworks = deployment_data.get("frameworks", []) or []
                allowed_exts = deployment_data.get("allowed_exts", []) or []
                exported_paths = deployment_data.get("exported_paths", []) or []

                exported_exts = []
                for p in exported_paths:
                    try:
                        ext = os.path.splitext(str(p))[1].lower()
                        if ext:
                            exported_exts.append(ext)
                    except Exception:
                        pass
                exported_exts = sorted(set(exported_exts))

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
                        f"<b>Exported format(s):</b> {', '.join(exported_exts)}",
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

                story.append(Spacer(1, 0.25 * cm))
                story.append(Paragraph("Minimal Example Usage", H2))
                code_lines = [
                    "# adapt to your runtime/framework",
                    "model = load_model(\"exported_model\")",
                    "raw_input = read_sensor_sample()",
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

                mapping = report_data.get("class_label_mapping", []) or []
                if mapping:
                    mapping_preview = ", ".join([f"{i} -> {lbl}" for i, lbl in mapping[:30]])
                    if len(mapping) > 30:
                        mapping_preview += " ..."
                    story.append(Spacer(1, 0.2 * cm))
                    story.append(Paragraph(f"<b>Label mapping (index -> label):</b> {mapping_preview}", body))

                story.append(Spacer(1, 0.2 * cm))
            doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
        
            for f in tmp_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
        
            print(f"[INFO] Preprocessing report saved to {path}")
        
        report_path = os.path.join(self.output_dir, "report.pdf")
        def _clean_dataset_name(name: str) -> str:
            s = str(name or "")
            if not s:
                return s
            m = re.match(r"^[0-9a-f]{16,}_(.+)$", s)
            return m.group(1) if m else s

        report_data = {
            "model_name": self.state.metrics.get("model_name") or self.best_model_name,
            "model_kind": self.model_kind,
            "target_name": self.target_name,
            "target_num_classes": self.target_num_classes,
            "class_labels": (self._label_classes or []),
            "class_label_mapping": list(enumerate(self._label_classes or [])),
            "dataset_name": _clean_dataset_name(os.path.basename(str(self.dataset_path))) if self.dataset_path else "",
            "device_family_name": get_family_name(self.device_family_id),
            "device_family_id": self.device_family_id,
            "accuracy": self.state.metrics.get("accuracy"),
            "accuracyBefore": self.state.metrics.get("accuracyBefore"),
            "accuracyAfter": self.state.metrics.get("accuracyAfter"),
            "latencyMsBefore": self.state.metrics.get("latencyMsBefore"),
            "latencyMsAfter": self.state.metrics.get("latencyMsAfter"),
            "sizeKBBefore": self.state.metrics.get("sizeKBBefore"),
            "sizeKBAfter": self.state.metrics.get("sizeKBAfter"),
            "export_ext_requested": self.state.metrics.get("export_ext_requested", self.export_ext),
            "export_ext_allowed": self.state.metrics.get("export_ext_allowed", family_allowed_formats(self.device_family_id)),
            "exported_model_paths": self.state.metrics.get("exported_model_paths", []),
            "saved_as": (os.path.splitext(self._final_model_path)[1].lower() if self._final_model_path else ""),
            "attempted_exts": self.state.metrics.get("export_ext_attempted", self.state.metrics.get("export_ext_allowed", family_allowed_formats(self.device_family_id))),
            "min_accuracy": self.min_accuracy,
            "optimization_strategy": self.state.metrics.get("optimization_strategy"),
            "optimization_level": self.state.metrics.get("optimization_level"),
            "accuracy_tolerance": self.state.metrics.get("accuracy_tolerance"),
            "accuracy_drop_allowed": self.state.metrics.get("accuracy_drop_allowed"),
            "accuracy_drop_cap": self.state.metrics.get("accuracy_drop_cap"),
            "quantization_requested": self.state.metrics.get("quantization_requested"),
            "quantization_applied": self.state.metrics.get("quantization_applied"),
            "optimization_skipped": self.state.metrics.get("optimization_skipped"),
            "optimization_skip_reason": self.state.metrics.get("optimization_skip_reason"),
            "optimization_attempted_levels": self.state.metrics.get("optimization_attempted_levels"),
            "optimization_failed": self.state.metrics.get("optimization_failed"),
            "optimization_fail_reason": self.state.metrics.get("optimization_fail_reason"),
            "optimization_attempt_accuracies": self.state.metrics.get("optimization_attempt_accuracies"),
            "optimization_history": self.state.metrics.get("optimization_history", []),
        }
        try:
            report_data["accuracy_drop_allowed_pct"] = f"{float(self.state.metrics.get('accuracy_drop_allowed') or 0.0) * 100.0:.1f}%"
        except Exception:
            report_data["accuracy_drop_allowed_pct"] = ""
        try:
            report_data["accuracy_drop_cap_pct"] = f"{float(self.state.metrics.get('accuracy_drop_cap') or 0.0) * 100.0:.1f}%"
        except Exception:
            report_data["accuracy_drop_cap_pct"] = ""
        try:
            report_data["optimization_trigger_ratio_pct"] = f"{float(self.optimization_trigger_ratio) * 100.0:.0f}%"
        except Exception:
            report_data["optimization_trigger_ratio_pct"] = ""
        report_data["optimization_skipped_display"] = (
            self.state.metrics.get("optimization_skipped_display")
            or ("Yes" if report_data.get("optimization_skipped") else "No")
        )
        report_data["optimization_failed_display"] = (
            "Yes" if report_data.get("optimization_failed") else "No"
        )
        attempted = report_data.get("optimization_attempted_levels") or []
        report_data["optimization_attempted_levels_display"] = " -> ".join(attempted) if attempted else ""
        report_data["quantization_applied_display"] = (
            report_data.get("quantization_applied") or "None"
        )
        attempt_acc = report_data.get("optimization_attempt_accuracies") or []
        attempt_acc_parts = []
        acc_map = {}
        size_map = {}
        for row in attempt_acc:
            try:
                lvl = str(row[0])
            except Exception:
                continue
            accv = None
            sizev = None
            if len(row) > 1:
                accv = row[1]
            if len(row) > 2:
                sizev = row[2]
            if accv is not None:
                try:
                    acc_map[lvl] = f"{float(accv) * 100.0:.1f}%"
                except Exception:
                    acc_map[lvl] = str(accv)
            if sizev is not None:
                try:
                    size_map[lvl] = f"{float(sizev):.2f} KB"
                except Exception:
                    size_map[lvl] = str(sizev)
        for lvl in attempted:
            acc_txt = acc_map.get(str(lvl), "n/a")
            size_txt = size_map.get(str(lvl), "n/a")
            attempt_acc_parts.append(f"{lvl}: {acc_txt}, {size_txt}")
        report_data["optimization_attempt_accuracies_display"] = "; ".join(attempt_acc_parts)
        reason_map = {
            "within_specs_balanced": "Within specs",
            "strategy_accuracy": "Accuracy prioritized",
            "exceeds_flash_limit_before_optimization": "Over flash limit before optimization",
            "optimization_no_viable_candidate": "No viable optimization candidate",
            "no_candidate_within_accuracy_tolerance": "No candidate within accuracy tolerance",
            "no_candidate_meets_specs": "No candidate met device specs",
            "int8_not_supported_for_model": "Int8 not supported for this model",
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
        _generate_sensor_report(
            self._prep_obj,
            path=report_path,
            report_data=report_data,
            deployment_data=deployment_data,
        )
        
        self._report_path = report_path
        self.state.metrics["report_path"] = report_path
                    
    # ==========================================================
    # Final validation
    # ==========================================================

    def _phase_validate_final(self) -> None:
        if self._final_model is None:
            raise RuntimeError("No final model available for validation.")

        test_acc = self.state.metrics.get("test_acc", None)
        if test_acc is None:
            raise RuntimeError("Missing test_acc metric; cannot validate minimum accuracy.")

        size_mb = float(self.state.metrics.get("final_model_size_mb") or self.state.metrics.get("model_size_mb") or 1e9)
        
        if self.state.metrics.get("optimization_failed"):
            self.state.metrics["validation_note"] = "Using best available model despite constraint violation."
            return

        if self.model_kind == "classic":
            if float(test_acc) < float(self.min_accuracy):
                raise RuntimeError(f"Accuracy {test_acc:.4f} < min_accuracy {self.min_accuracy:.2f}")
            flash_mb_limit = float(self.device_specs.get("flash_mb", 0.0))
            # if flash_mb_limit > 0 and float(size_mb) > flash_mb_limit:
            #     raise RuntimeError(
            #         f"Model size ({size_mb:.3f} MB) is too large for the selected device's flash memory ({flash_mb_limit:.3f} MB)."
            #     )
            return

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
            raise RuntimeError(reason)
