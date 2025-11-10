from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import time
import joblib
import os
from sklearn.metrics import accuracy_score


def data_splitting(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    # Split 70% train, 15% validation, 15% test
    # First split off 30% (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # Then split that 30% evenly into 15% + 15%
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_best_model(results):
    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df.sort_values("accuracy", ascending=False))

    results_df["score"] = (
    results_df["accuracy"]
    - 0.001 * results_df["size_mb"]
    - 0.0001 * results_df["inference_ms"]
    ) 
    best_tradeoff_index = results_df["score"].idxmax()
    return best_tradeoff_index


def model_optimization(pipeline, df, target):
    from sklearn.svm import SVC
    X_train, y_train, X_val, y_val, X_test, y_test = data_splitting(df, target)
    model = pipeline.named_steps["model"]

    if model == "SVC":
        return "wrong model"

    preprocessor = pipeline.named_steps["preprocessing"]

    param_grid = [
    {"kernel": "linear", "C": 0.1, "gamma": "scale"},
    {"kernel": "linear", "C": 1, "gamma": "scale"},
    {"kernel": "rbf", "C": 1, "gamma": 0.01},
    {"kernel": "rbf", "C": 10, "gamma": 0.01},
    {"kernel": "rbf", "C": 1, "gamma": 0.001},
]

    results = []
    models = []

    for params in param_grid:
        print(f"Testing {params}")

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", SVC(**params, probability=True, random_state=42))
        ])

        # --- Train ---
        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train

        # --- Evaluate accuracy ---
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        # --- Measure inference time ---
        start_inf = time.perf_counter()
        _ = model.predict(X_val)
        inf_time = (time.perf_counter() - start_inf) / len(X_val)  # per-sample latency

        # --- Measure size ---
        joblib.dump(model, "temp_model.joblib")
        size_mb = os.path.getsize("temp_model.joblib") / (1024 * 1024)

        results.append({
            "kernel": params["kernel"],
            "C": params["C"],
            "gamma": params["gamma"],
            "accuracy": acc,
            "train_time_s": train_time,
            "inference_ms": inf_time * 1000,
            "size_mb": size_mb
        })

        models.append(model)
        
    best_tradeoff_index = get_best_model(results)

    return models[best_tradeoff_index]


