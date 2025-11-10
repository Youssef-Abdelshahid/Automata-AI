from optimized_models import model_optimization

def optimizing_model(pipeline, df, target):
    model_name = type(pipeline.named_steps["model"]).__name__
    print(f"🔍 Detected model: {model_name}")

    match model_name:
        case "SVC":
            print("⚙️ Running SVM optimization...")

            optimized_model = model_optimization(pipeline, df, target)
 
            pass

        case "RandomForestClassifier":
            print("🌲 Running Random Forest optimization...")
            # Add RandomForest tuning (n_estimators, max_depth, etc.)
            # tune_random_forest(pipeline)
            pass

        case "XGBClassifier":
            print("⚡ Running XGBoost optimization...")
            # Add XGBoost tuning
            # tune_xgboost(pipeline)
            pass

        case "LGBMClassifier":
            print("💡 Running LightGBM optimization...")
            # Add LightGBM tuning logic
            pass

        case "CatBoostClassifier":
            print("🐈 Running CatBoost optimization...")
            # Add CatBoost tuning logic
            pass

        case "LogisticRegression":
            print("📈 Running Logistic Regression optimization...")
            # Add LogisticRegression tuning logic
            pass

        case "KNeighborsClassifier":
            print("👥 Running KNN optimization...")
            # Add KNN tuning logic
            pass

        case "DecisionTreeClassifier":
            print("🌳 Running Decision Tree optimization...")
            # Add DecisionTree tuning logic
            pass

        case "GaussianNB":
            print("🧮 Running Naive Bayes optimization...")
            # Add GaussianNB tuning logic
            pass

        case "GradientBoostingClassifier":
            print("🚀 Running Gradient Boosting optimization...")
            # Add GradientBoosting tuning logic
            pass

        case _:
            print(f"⚠️ No optimization routine found for {model_name}")

    return optimized_model



    

