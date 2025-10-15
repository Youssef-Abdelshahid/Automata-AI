import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# Helper to create lightweight landmark models
def _make_light_clf(name: str, _JOBS=-1):
    nm = name.lower()
    if "logreg" in nm: return LogisticRegression(max_iter=200, solver="lbfgs")
    if "linearsvc" in nm: return LinearSVC(tol=1e-2, max_iter=1000, dual='auto')
    if "knn" in nm: return KNeighborsClassifier(n_neighbors=3, n_jobs=_JOBS)
    if "decisiontree" in nm: return DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
    if "randomforest" in nm: return RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42, n_jobs=_JOBS)
    if "extratrees" in nm: return ExtraTreesClassifier(n_estimators=20, max_depth=4, random_state=42, n_jobs=_JOBS)
    if "gradientboosting" in nm: return GradientBoostingClassifier(n_estimators=25, max_depth=2, random_state=42)
    return None

LANDMARK_MODELS = {
    "landmark_LogReg": _make_light_clf("logreg"),
    "landmark_LinearSVC": _make_light_clf("linearsvc"),
    "landmark_KNN_5": _make_light_clf("knn"),
    "landmark_DecisionTree_d5": _make_light_clf("decisiontree"),
    "landmark_RandomForest_m": _make_light_clf("randomforest"),
    "landmark_ExtraTrees_m": _make_light_clf("extratrees"),
    "landmark_GradientBoosting": _make_light_clf("gradientboosting"),
}

def extract_meta_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    meta = {}
    meta["n_features"] = X.shape[1]
    meta["n_samples"] = X.shape[0]
    meta["n_classes"] = len(np.unique(y))
    counts = pd.Series(y).value_counts(normalize=True)
    meta["class_balance"] = counts.min() / counts.max() if len(counts) > 1 else 1.0
    
    # Landmark models
    for name, model in LANDMARK_MODELS.items():
        try:
            score = np.mean(cross_val_score(model, X, y, cv=3, scoring="accuracy"))
            meta[name] = score
        except Exception:
            meta[name] = np.nan
    
    return pd.DataFrame([meta])

def recommend_top_models(meta_features_df: pd.DataFrame, meta_learner_path: str, top_n: int = 3):
    """
    Uses a pre-trained meta-learner to predict the best models.
    """
    try:
        bundle = joblib.load(meta_learner_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Meta-learner not found at {meta_learner_path}. Please ensure it's in the project root.")

    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    
    # Ensure columns match what the meta-learner was trained on
    required_cols = pipeline.feature_names_in_
    meta_features_df_aligned = pd.DataFrame(columns=required_cols)
    for col in required_cols:
        if col in meta_features_df.columns:
            meta_features_df_aligned[col] = meta_features_df[col]
        else:
            meta_features_df_aligned[col] = 0 # Or np.nan, depending on how the model was trained
            
    scores = pipeline.predict_proba(meta_features_df_aligned)[0]
    
    decoded_classes = label_encoder.classes_
    model_scores = dict(zip(decoded_classes, scores))
    
    top_models = sorted(model_scores, key=model_scores.get, reverse=True)[:top_n]
    
    return top_models