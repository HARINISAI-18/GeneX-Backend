"""Configuration for all supported cancer types."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Each cancer type config: training files, model file name, and metadata columns
CANCER_TYPES = {
    "skin_melanoma": {
        "label": "Skin Melanoma",
        "model_file": "skin_melanoma.pkl",
        "train_file": "SKIN_MELANOMA_TRAIN.csv",
        "test_file": "SKIN_MELANOMA_TEST.csv",
        "meta_cols": ["sample_id", "study", "sample_type", "common_class", "label"],
        "label_col": "label",
        "label_map": {0: "Non-Malignant / Normal / Benign", 1: "Melanoma (Cancerous)"},
    },
    "breast_cancer": {
        "label": "Breast Cancer",
        "model_file": "breast_cancer.pkl",
        "train_file": "breast_cancer_train_80.csv",
        "test_file": "breast_cancer_test_20.csv",
        "meta_cols": ["sample_id", "disease"],
        "label_col": "label",
        "label_map": {0: "Normal", 1: "Breast Cancer"},
    },
    "esophageal": {
        "label": "Esophageal Cancer",
        "model_file": "esophageal.pkl",
        "train_file": "ESOPHAGEAL_GSE20347_TRAIN.csv",
        "test_file": "ESOPHAGEAL_GSE20347_TEST.csv",
        "meta_cols": ["sample_id", "patient_id", "disease_status", "label"],
        "label_col": "label",
        "label_map": {0: "No Cancer", 1: "Esophageal Cancer"},
    },
    "lung_cancer": {
        "label": "Lung Cancer",
        "model_file": "lung_cancer.pkl",
        "train_file": "LUNG_train_80.csv",
        "test_file": "LUNG_test_20.csv",
        "meta_cols": ["sample_id", "patient_id", "disease_status", "label"],
        "label_col": "label",
        "label_map": {0: "Normal", 1: "Lung Cancer"},
    },
    "ovarian": {
        "label": "Ovarian Cancer",
        "model_file": "ovarian.pkl",
        "train_file": "OVARIAN_train.csv",
        "test_file": "OVARIAN_test.csv",
        "meta_cols": ["sample_id"],
        "label_col": "label",
        "label_map": {0: "Normal", 1: "Tumor / Ovarian Cancer"},
    },
    "cervical": {
        "label": "Cervical Cancer",
        "model_file": "cervical.pkl",
        "train_file": "CESC_train.csv",
        "test_file": "CESC_test.csv",
        "meta_cols": ["sample_id", "disease_stage", "stage_label"],
        "label_col": "stage_label",
        "label_map": {0: "Normal", 1: "CIN1", 2: "CIN3", 3: "Cervical Cancer"},
    },
    "pancreatic": {
        "label": "Pancreatic Cancer",
        "model_file": "pancreatic.pkl",
        "train_file": "PANCREATIC_TRAIN.csv",
        "test_file": "PANCREATIC_TEST.csv",
        "meta_cols": ["sample_id", "sample_type", "disease_status", "label"],
        "label_col": "label",
        "label_map": {0: "Non_Cancer", 1: "Pancreatic Cancer"},
    },
    "leukemia": {
        "label": "Leukemia (AML)",
        "model_file": "leukemia.pkl",
        "train_file": "AML_train_80_percent.csv",
        "test_file": "AML_test_20_percent.csv",
        "meta_cols": ["Sample_ID", "Class", "Label"],
        "label_col": "Label",
        "label_map": {0: "Healthy_Negative", 1: "AML_Positive"},
    },
    "colorectal": {
        "label": "Colorectal Cancer",
        "model_file": "colorectal.pkl",
        "train_file": "train_samples_colorectal.csv",
        "test_file": "",  # No test file provided in notebook
        "meta_cols": ["sample_id", "disease_status", "label"],
        "label_col": "label",
        "label_map": {0: "Normal", 1: "Colorectal Cancer"},
    },
}