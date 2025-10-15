# /api_project/main.py

import pandas as pd
import joblib
import io
import uuid
import time # Import time to measure duration
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Import custom modules
from processing import DataProcessor
from meta_learning import extract_meta_features, recommend_top_models
from trainer import train_best_model

app = FastAPI(
    title="Automated Model Training API",
    description="Upload a dataset to automatically train the best model.",
    version="1.0.0"
)

# --- CORS middleware section ---
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:5500", # Default port for VS Code Live Server
    "null" # Allow requests from local files
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_registry = {}

# Pydantic model for prediction input
class PredictionInput(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/train/")
async def train_model(
    target_column: str,
    meta_learner_path: str = "best_model_RandomForest_1756377093.joblib",
    file: UploadFile = File(...)
):
    print("\n\n--- [NEW JOB RECEIVED] ---")
    start_time = time.time()

    # Phase 1: Load Data
    try:
        print("[PHASE 1/6] Reading uploaded file into memory...")
        contents = await file.read()
        print("...File read successfully. Converting to DataFrame...")
        df = pd.read_csv(io.BytesIO(contents))
        print(f"✅ PHASE 1 COMPLETE: DataFrame created with shape {df.shape}. (Took {time.time() - start_time:.2f}s)")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the dataset.")

    phase_start_time = time.time()
    # Phase 2: Preprocess Data
    try:
        print("\n[PHASE 2/6] Starting data preprocessing...")
        processor = DataProcessor()
        X, y = processor.fit_transform(df, target_column)
        print(f"✅ PHASE 2 COMPLETE: Data preprocessed. (Took {time.time() - phase_start_time:.2f}s)")
    except Exception as e:
        print(f"❌ ERROR IN PHASE 2: {e}")
        raise HTTPException(status_code=500, detail=f"Error during data preprocessing: {e}")

    phase_start_time = time.time()
    # Phase 3: Meta-Feature Extraction
    try:
        print("\n[PHASE 3/6] Starting meta-feature extraction...")
        meta_features = extract_meta_features(X, y)
        print(f"✅ PHASE 3 COMPLETE: Meta-features extracted. (Took {time.time() - phase_start_time:.2f}s)")
    except Exception as e:
        print(f"❌ ERROR IN PHASE 3: {e}")
        raise HTTPException(status_code=500, detail=f"Error during meta-feature extraction: {e}")

    phase_start_time = time.time()
    # Phase 4: Recommend Models
    try:
        print("\n[PHASE 4/6] Recommending top models using meta-learner...")
        recommended_models = recommend_top_models(meta_features, meta_learner_path)
        print(f"✅ PHASE 4 COMPLETE: Top models recommended: {recommended_models}. (Took {time.time() - phase_start_time:.2f}s)")
    except Exception as e:
        print(f"❌ ERROR IN PHASE 4: {e}")
        raise HTTPException(status_code=500, detail=f"Error during model recommendation: {e}")

    phase_start_time = time.time()
    # Phase 5: Train Best Model
    try:
        print("\n[PHASE 5/6] Starting final model training...")
        best_model, best_score = train_best_model(X, y, recommended_models)
        print(f"✅ PHASE 5 COMPLETE: Best model trained. (Took {time.time() - phase_start_time:.2f}s)")
    except Exception as e:
        print(f"❌ ERROR IN PHASE 5: {e}")
        raise HTTPException(status_code=500, detail=f"Error during final model training: {e}")

    phase_start_time = time.time()
    # Phase 6: Save Artifact
    print("\n[PHASE 6/6] Saving model artifact...")
    model_id = str(uuid.uuid4())
    artifact = {
        "model": best_model,
        "processor": processor,
        "accuracy": best_score,
        "model_name": type(best_model).__name__
    }
    joblib.dump(artifact, f"saved_models/{model_id}.joblib")

    model_registry[model_id] = {
        "accuracy": best_score,
        "model_name": type(best_model).__name__
    }
    print(f"✅ PHASE 6 COMPLETE: Artifact saved with ID {model_id}. (Took {time.time() - phase_start_time:.2f}s)")

    print(f"\n--- [JOB SUCCEEDED] --- Total time: {time.time() - start_time:.2f}s ---\n")
    return {
        "message": "Training successful!",
        "model_id": model_id,
        "model_name": artifact["model_name"],
        "test_set_accuracy": best_score
    }

@app.post("/predict/{model_id}")
async def predict(model_id: str, input_data: PredictionInput):
    try:
        artifact = joblib.load(f"saved_models/{model_id}.joblib")
        model = artifact["model"]
        processor = artifact["processor"]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found.")

    input_df = pd.DataFrame(input_data.data)
    processed_input = processor.transform(input_df)
    predictions_encoded = model.predict(processed_input)
    predictions = processor.target_encoder.inverse_transform(predictions_encoded)

    return {"predictions": predictions.tolist()}

@app.get("/models")
async def get_models():
    return model_registry