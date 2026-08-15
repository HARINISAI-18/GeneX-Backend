import os
import joblib
import pandas as pd
from cancer_config import MODELS_DIR, CANCER_TYPES

def route_to_model(csv_path: str) -> str:
    """
    Mock AI Agent: Determines which model to use.
    1. First, it checks the filename for keywords (e.g., 'breast', 'pancreatic').
    2. If no keyword is found, it falls back to gene column overlap.
    """
    filename = os.path.basename(csv_path).lower()
    print(f"[Agent] Analyzing uploaded file: {filename}...")
    
    # --- STEP 1: Filename Keyword Routing (Foolproof for demos) ---
    keyword_map = {
        "breast": "breast_cancer",
        "pancreatic": "pancreatic",
        "leukemia": "leukemia",
        "aml": "leukemia",
        "colorectal": "colorectal",
        "ovarian": "ovarian",
        "lung": "lung_cancer",
        "skin": "skin_melanoma",
        "esophageal": "esophageal",
        "cervical": "cervical"
    }
    
    for keyword, cancer_type in keyword_map.items():
        if keyword in filename:
            # Verify this model is actually trained
            model_path = os.path.join(MODELS_DIR, CANCER_TYPES[cancer_type]["model_file"])
            if os.path.exists(model_path):
                print(f"[Agent] Filename keyword '{keyword}' detected. Routing to {cancer_type} model.")
                return cancer_type
                
    # --- STEP 2: Fallback - Gene Column Overlap Routing ---
    print("[Agent] No filename keyword found. Falling back to gene column overlap analysis...")
    df = pd.read_csv(csv_path)
    csv_cols = set(df.columns)
    
    best_match = None
    max_overlap = 0
    
    for cancer_type, cfg in CANCER_TYPES.items():
        model_path = os.path.join(MODELS_DIR, cfg["model_file"])
        if os.path.exists(model_path):
            artifact = joblib.load(model_path)
            model_genes = set(artifact["feature_names"])
            
            overlap = len(csv_cols.intersection(model_genes))
            print(f"[Agent] Overlap with {cancer_type}: {overlap} genes")
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = cancer_type
                
    if best_match is None or max_overlap == 0:
        raise ValueError("Agent could not match the uploaded CSV to any trained model.")
        
    print(f"[Agent] Decision: Routing to {best_match} model.")
    return best_match