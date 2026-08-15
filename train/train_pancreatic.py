# -*- coding: utf-8 -*-
"""Pancreatic Cancer QEA Pipeline - VS Code training script."""

import os, sys, warnings, datetime, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_config import DATA_DIR, MODELS_DIR, CANCER_TYPES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["pancreatic"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["pancreatic"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["pancreatic"]["model_file"])

TARGET_COL = "label"
METADATA_COLS = ["sample_id", "sample_type", "disease_status", "label"]
K_ANOVA = 100
POP_SIZE = 20
N_GENERATIONS = 30
CV_FOLDS = 5
FEATURE_PENALTY = 0.0015
THETA_STEP = 0.05 * np.pi

def split_X_y(df, target_col=TARGET_COL, meta_cols=METADATA_COLS):
    y = df[target_col].astype(int).values
    X = df.drop(columns=[c for c in meta_cols if c in df.columns])
    return X, y

def qbits_to_binary(alpha, beta, rng_local):
    prob_1 = beta ** 2
    r = rng_local.random(len(prob_1))
    return (r < prob_1).astype(int)

def fitness_function(binary_vector, X, y, cv):
    n_selected = int(binary_vector.sum())
    if n_selected == 0:
        return 0.0, 0.0
    idx = np.where(binary_vector == 1)[0]
    X_sub = X[:, idx]
    clf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    try:
        acc = cross_val_score(clf, X_sub, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    except Exception:
        acc = 0.0
    return acc - (FEATURE_PENALTY * n_selected), acc

def rotation_update(alpha, beta, individual_bit, best_bit, individual_fit, best_fit, theta_step):
    new_alpha, new_beta = alpha.copy(), beta.copy()
    for i in range(len(alpha)):
        a, b = alpha[i], beta[i]
        x_bit, b_bit = individual_bit[i], best_bit[i]
        delta_theta = 0.0
        if individual_fit < best_fit:
            if x_bit == 0 and b_bit == 1:
                delta_theta = theta_step if a * b >= 0 else -theta_step
            elif x_bit == 1 and b_bit == 0:
                delta_theta = -theta_step if a * b >= 0 else theta_step
        if delta_theta != 0.0:
            cos_t, sin_t = np.cos(delta_theta), np.sin(delta_theta)
            new_alpha[i] = a * cos_t - b * sin_t
            new_beta[i] = a * sin_t + b * cos_t
    return new_alpha, new_beta

def main():
    print("=" * 80)
    print("TRAINING: PANCREATIC CANCER QEA PIPELINE")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    X_train_raw, y_train = split_X_y(train_df)
    X_test_raw, y_test = split_X_y(test_df)
    gene_names = X_train_raw.columns.to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    k = min(K_ANOVA, X_train_scaled.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=k)
    X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
    X_test_anova = anova_selector.transform(X_test_scaled)
    anova_mask = anova_selector.get_support()
    anova_genes = gene_names[anova_mask]
    print(f"ANOVA reduced: {X_train_scaled.shape[1]} -> {k} genes")

    print("\nRunning QEA optimization...")
    N_GENES_POOL = X_train_anova.shape[1]
    rng = np.random.default_rng(RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    alpha_pop = np.full((POP_SIZE, N_GENES_POOL), 1 / np.sqrt(2))
    beta_pop = np.full((POP_SIZE, N_GENES_POOL), 1 / np.sqrt(2))
    best_binary_global, best_fitness_global, best_acc_global = None, -np.inf, 0.0

    for gen in range(N_GENERATIONS):
        binaries = np.zeros((POP_SIZE, N_GENES_POOL), dtype=int)
        fitnesses = np.zeros(POP_SIZE)
        for p in range(POP_SIZE):
            b_vec = qbits_to_binary(alpha_pop[p], beta_pop[p], rng)
            if b_vec.sum() == 0: b_vec[rng.integers(0, N_GENES_POOL)] = 1
            binaries[p] = b_vec
            fit, acc = fitness_function(b_vec, X_train_anova, y_train, cv)
            fitnesses[p] = fit
        
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness_global:
            best_fitness_global = fitnesses[gen_best_idx]
            best_binary_global = binaries[gen_best_idx].copy()
        
        for p in range(POP_SIZE):
            alpha_pop[p], beta_pop[p] = rotation_update(
                alpha_pop[p], beta_pop[p], binaries[p], best_binary_global,
                fitnesses[p], best_fitness_global, THETA_STEP)
            norm = np.sqrt(alpha_pop[p]**2 + beta_pop[p]**2)
            norm[norm == 0] = 1.0
            alpha_pop[p] /= norm
            beta_pop[p] /= norm

    qea_gene_idx = np.where(best_binary_global == 1)[0]
    qea_selected_genes = anova_genes[qea_gene_idx]
    print(f"QEA selected {len(qea_selected_genes)} biomarkers")

    X_train_final = X_train_anova[:, qea_gene_idx]
    X_test_final = X_test_anova[:, qea_gene_idx]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
    }
    
    best_f1, best_model_name, best_model = -1, None, None
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1, best_model_name, best_model = f1, name, model

    print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")

    selected_indices = [list(gene_names).index(g) for g in qea_selected_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[selected_indices]
    final_scaler.scale_ = scaler.scale_[selected_indices]
    final_scaler.var_ = scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "pancreatic",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(qea_selected_genes),
        "metadata": {
            "algorithm": best_model_name,
            "best_params": best_model.get_params(),
            "test_f1": float(best_f1),
            "training_date": datetime.datetime.utcnow().isoformat(),
        }
    }
    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")

if __name__ == "__main__":
    main()