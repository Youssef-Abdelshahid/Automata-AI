from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# Candidate models with their full tuning grids
CANDIDATE_MODELS = {
    "LogReg": (
        LogisticRegression(max_iter=800),
        {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    ),
    "LinearSVC": (
        LinearSVC(max_iter=2000, dual='auto'),
        {"C": [0.1, 1, 10]}
    ),
    "KNN-5": (
        KNeighborsClassifier(n_jobs=-1),
        {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
    ),
    "DecisionTree-d5": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [5, 10], "min_samples_leaf": [1, 5, 10]}
    ),
    "RandomForest-m": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    ),
    "ExtraTrees-m": (
        ExtraTreesClassifier(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100], "max_depth": [5, 10, None]}
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    )
}

def train_best_model(X, y, recommended_models: list):
    """
    Tunes the recommended models and returns the best one.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model_so_far = None
    best_score = -1.0
    best_model_name = ""

    print(f"Starting tuning for top models: {recommended_models}")

    for mname in recommended_models:
        if mname not in CANDIDATE_MODELS:
            print(f"Warning: Recommended model '{mname}' not in candidate list. Skipping.")
            continue
        
        base_model, param_grid = CANDIDATE_MODELS[mname]
        
        grid_search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        
        y_pred = grid_search.best_estimator_.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
        print(f"-> Tuned {mname}: Accuracy = {score:.4f} (Params: {grid_search.best_params_})")
        
        if score > best_score:
            best_score = score
            best_model_so_far = grid_search.best_estimator_
            best_model_name = mname

    print(f"\n✅ Final best model selected: {best_model_name} with accuracy: {best_score:.4f}")
    
    return best_model_so_far, best_score