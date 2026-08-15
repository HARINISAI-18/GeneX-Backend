# -*- coding: utf-8 -*-
"""
Skin Melanoma QEA Biomarker Prediction - VS Code training script.
Saves a trained model artifact to backend/models/skin_melanoma.pkl
"""

import os
import sys
import warnings
import datetime
import joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Make backend dir importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_config import DATA_DIR, MODELS_DIR, CANCER_TYPES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["skin_melanoma"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["skin_melanoma"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["skin_melanoma"]["model_file"])

META_COLS = ["sample_id", "study", "sample_type", "common_class", "label"]
LABEL_COL = "label"


def split_X_y(df, gene_cols, label_col=LABEL_COL):
    X = df[gene_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    y = df[label_col].values if label_col in df.columns else None
    return X, y


class QuantumInspiredEvolutionaryAlgorithm:
    """Same QEA logic as the original notebook."""

    def __init__(self, n_features, fitness_fn, pop_size=20, generations=30,
                 min_features=5, max_features=40, rotation_angle=0.05 * np.pi,
                 random_state=42, verbose=True):
        self.n_features = n_features
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.generations = generations
        self.min_features = min_features
        self.max_features = max_features
        self.rotation_angle = rotation_angle
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose
        self.alpha = np.full((pop_size, n_features), 1 / np.sqrt(2))
        self.beta = np.full((pop_size, n_features), 1 / np.sqrt(2))
        self.best_solution = None
        self.best_fitness = -np.inf
        self.history = []

    def _observe(self):
        probs = self.beta ** 2
        rand = self.rng.random(probs.shape)
        return (rand < probs).astype(int)

    def _repair(self, binary_solution):
        n_selected = binary_solution.sum()
        idx_on = np.where(binary_solution == 1)[0]
        idx_off = np.where(binary_solution == 0)[0]
        if n_selected < self.min_features and len(idx_off) > 0:
            need = self.min_features - n_selected
            turn_on = self.rng.choice(idx_off, size=min(need, len(idx_off)), replace=False)
            binary_solution[turn_on] = 1
        elif n_selected > self.max_features and len(idx_on) > 0:
            excess = n_selected - self.max_features
            turn_off = self.rng.choice(idx_on, size=min(excess, len(idx_on)), replace=False)
            binary_solution[turn_off] = 0
        return binary_solution

    def _rotate(self, binary_pop, best_binary):
        for i in range(self.pop_size):
            for j in range(self.n_features):
                x_ij = binary_pop[i, j]
                b_j = best_binary[j]
                a, b = self.alpha[i, j], self.beta[i, j]
                if x_ij == b_j:
                    continue
                delta_theta = self.rotation_angle
                if b_j == 1:
                    s = 1 if (a * b) >= 0 else -1
                else:
                    s = -1 if (a * b) >= 0 else 1
                theta = s * delta_theta
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                self.alpha[i, j], self.beta[i, j] = cos_t * a - sin_t * b, sin_t * a + cos_t * b

    def run(self):
        for gen in range(self.generations):
            binary_pop = self._observe()
            binary_pop = np.array([self._repair(ind.copy()) for ind in binary_pop])
            fitnesses = np.array([self.fitness_fn(ind) for ind in binary_pop])
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[gen_best_idx]
                self.best_solution = binary_pop[gen_best_idx].copy()
            self._rotate(binary_pop, self.best_solution)
            self.history.append(self.best_fitness)
            if self.verbose and (gen % 5 == 0 or gen == self.generations - 1):
                n_sel = int(self.best_solution.sum())
                print(f"Gen {gen+1:>3}/{self.generations} | best fitness = {self.best_fitness:.4f} | "
                      f"genes selected = {n_sel}")
        return self.best_solution, self.best_fitness


def main():
    print("=" * 80)
    print("TRAINING: SKIN MELANOMA QEA BIOMARKER PIPELINE")
    print("=" * 80)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    META_COLS_PRESENT = [c for c in META_COLS if c in train_df.columns]
    gene_cols = [c for c in train_df.columns if c not in META_COLS_PRESENT]
    print(f"Detected {len(gene_cols)} gene-expression features.")

    X_train_raw, y_train = split_X_y(train_df, gene_cols)
    X_test_raw, y_test = split_X_y(test_df, gene_cols)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=gene_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=gene_cols)

    # ANOVA
    K_ANOVA = min(150, X_train_scaled.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=K_ANOVA)
    anova_selector.fit(X_train_scaled, y_train)
    anova_mask = anova_selector.get_support()
    anova_genes = np.array(gene_cols)[anova_mask]
    print(f"ANOVA selected top {K_ANOVA} genes.")

    X_train_anova = X_train_scaled[anova_genes].reset_index(drop=True)
    X_test_anova = X_test_scaled[anova_genes].reset_index(drop=True)
    X_train_anova_arr = X_train_anova.values

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def qea_fitness(binary_solution):
        idx = np.where(binary_solution == 1)[0]
        if len(idx) == 0:
            return -1.0
        X_subset = X_train_anova_arr[:, idx]
        clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                     random_state=RANDOM_STATE, class_weight="balanced")
        scores = cross_val_score(clf, X_subset, y_train, cv=cv_splitter, scoring="f1")
        parsimony_penalty = 0.0005 * len(idx)
        return scores.mean() - parsimony_penalty

    print("\nRunning QEA optimization...")
    qea = QuantumInspiredEvolutionaryAlgorithm(
        n_features=X_train_anova_arr.shape[1],
        fitness_fn=qea_fitness,
        pop_size=20, generations=30,
        min_features=8, max_features=40,
        rotation_angle=0.03 * np.pi,
        random_state=RANDOM_STATE,
    )
    best_binary, best_fitness = qea.run()
    qea_selected_idx = np.where(best_binary == 1)[0]
    qea_selected_genes = anova_genes[qea_selected_idx]
    print(f"\nQEA selected {len(qea_selected_genes)} biomarkers (best CV F1 = {best_fitness:.4f})")

    X_train_final = X_train_anova[qea_selected_genes].values
    X_test_final = X_test_anova[qea_selected_genes].values

    # Train models
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models_and_grids = {
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                          scale_pos_weight=scale_pos_weight),
            {"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"n_estimators": [200, 300], "max_depth": [3, 5, None]},
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        ),
    }

    fitted_models = {}
    results = {}
    for name, (estimator, grid) in models_and_grids.items():
        search = GridSearchCV(estimator, grid, scoring="f1", cv=cv_splitter, n_jobs=-1)
        search.fit(X_train_final, y_train)
        fitted_models[name] = search.best_estimator_
        cv_scores = cross_val_score(search.best_estimator_, X_train_final, y_train,
                                    cv=cv_splitter, scoring="f1")
        results[name] = {"cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std()}
        print(f"{name:15s} | CV F1 = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Pick best by CV F1
    BEST_MODEL_NAME = max(results, key=lambda n: results[n]["cv_f1_mean"])
    best_model = fitted_models[BEST_MODEL_NAME]
    print(f"\nBest model: {BEST_MODEL_NAME}")

    # Evaluate on test
    y_test_pred = best_model.predict(X_test_final)
    y_test_proba = best_model.predict_proba(X_test_final)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    try:
        test_auc = roc_auc_score(y_test, y_test_proba)
    except Exception:
        test_auc = float("nan")
    print(f"\nTest Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | ROC-AUC: {test_auc:.4f}")

    # Build mini-scaler for selected genes only
    selected_indices = [gene_cols.index(g) for g in qea_selected_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[selected_indices]
    final_scaler.scale_ = scaler.scale_[selected_indices]
    final_scaler.var_ = scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "skin_melanoma",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(qea_selected_genes),
        "metadata": {
            "algorithm": BEST_MODEL_NAME,
            "best_params": best_model.get_params(),
            "cv_f1": float(results[BEST_MODEL_NAME]["cv_f1_mean"]),
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_roc_auc": float(test_auc),
            "training_date": datetime.datetime.utcnow().isoformat(),
        }
    }

    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")
    print(f"   Algorithm: {BEST_MODEL_NAME}")
    print(f"   Biomarkers: {len(qea_selected_genes)}")


if __name__ == "__main__":
    main()