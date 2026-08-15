import os
import joblib
import numpy as np
import pandas as pd
from cancer_config import MODELS_DIR, CANCER_TYPES

def load_artifact(cancer_type: str) -> dict:
    """Load the saved model artifact for a given cancer type."""
    if cancer_type not in CANCER_TYPES:
        raise ValueError(f"Unknown cancer type: {cancer_type}")

    cfg = CANCER_TYPES[cancer_type]
    model_path = os.path.join(MODELS_DIR, cfg["model_file"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not trained yet for '{cancer_type}'. "
            f"Please run the corresponding training script first. "
            f"Expected file: {model_path}"
        )

    return joblib.load(model_path)


def predict_user_csv(cancer_type: str, csv_file_path: str) -> dict:
    """Run inference on a user-uploaded CSV file using a trained model."""
    artifact = load_artifact(cancer_type)
    cfg = CANCER_TYPES[cancer_type]

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = list(artifact["feature_names"])
    label_map = cfg["label_map"]

    # Read user CSV
    df = pd.read_csv(csv_file_path)

    # Identify sample IDs (if present)
    sample_ids = (
        df["sample_id"].astype(str).tolist()
        if "sample_id" in df.columns
        else [f"sample_{i+1}" for i in range(len(df))]
    )

    # --- Do not fail on missing genes. Fill missing with 0.0 ---
    X = pd.DataFrame(0.0, index=df.index, columns=feature_names)
    present_genes = [g for g in feature_names if g in df.columns]
    X[present_genes] = df[present_genes].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0)

    # Scale
    X_scaled = scaler.transform(X.values)

    # Predict
    preds = model.predict(X_scaled)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)

    # --- EXTRACT IMPORTANCES (Multi-Model Support) ---
    # 1. Try standard tree-based feature_importances_ (Random Forest, XGBoost)
    global_importances = getattr(model, "feature_importances_", None)
    
    # 2. Fallback for Linear models (Logistic Regression, Linear SVM)
    if global_importances is None:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            # Take absolute values of coefficients for ranking
            global_importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

    # Build result rows
    results = []
    for i, sid in enumerate(sample_ids):
        row = {
            "sample_id": sid,
            "predicted_label": int(preds[i]),
            "predicted_class": label_map.get(int(preds[i]), str(preds[i])),
        }
        if proba is not None:
            row["confidence"] = round(float(proba[i, preds[i]]), 4)
            for cls_idx in range(proba.shape[1]):
                cls_name = label_map.get(cls_idx, f"class_{cls_idx}")
                row[f"P({cls_name})"] = round(float(proba[i, cls_idx]), 4)
        
        # --- PATIENT-SPECIFIC CONTRIBUTING GENES ---
        if int(preds[i]) == 1:
            if global_importances is not None:
                # Model has importances (Tree or Linear)
                top_global_idx = np.argsort(global_importances)[-15:][::-1]
                patient_values = X_scaled[i, top_global_idx]
                sorted_local_idx = top_global_idx[np.argsort(patient_values)[::-1]]
            else:
                # Fallback for RBF SVM (Black box model): Rank purely by patient expression
                patient_values = X_scaled[i]
                sorted_local_idx = np.argsort(patient_values)[-15:][::-1]
            
            # Take the top 7 highest expressed important genes for this patient
            contributing_genes = [feature_names[idx] for idx in sorted_local_idx[:7]]
            row["contributing_genes"] = contributing_genes
        else:
            # Empty list if Normal
            row["contributing_genes"] = []

        results.append(row)

    return {
        "cancer_type": cancer_type,
        "cancer_label": cfg["label"],
        "model_algorithm": artifact.get("metadata", {}).get("algorithm", "Unknown"),
        "n_biomarkers": len(feature_names),
        "n_samples": len(results),
        "predictions": results,
    }


def list_available_models() -> list:
    """Return list of cancer types that have a trained model ready."""
    available = []
    for cancer_type, cfg in CANCER_TYPES.items():
        path = os.path.join(MODELS_DIR, cfg["model_file"])
        available.append({
            "cancer_type": cancer_type,
            "label": cfg["label"],
            "trained": os.path.exists(path),
            "label_map": cfg["label_map"],
        })
    return available