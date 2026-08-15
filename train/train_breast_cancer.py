# -*- coding: utf-8 -*-
"""Breast Cancer QEA Pipeline - VS Code training script."""

import os, sys, warnings, datetime, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
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

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["breast_cancer"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["breast_cancer"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["breast_cancer"]["model_file"])

TARGET_COL = "label"
META_COLS = ["sample_id", "disease"]
K_ANOVA = 100
QEA_MIN_GENES = 15
QEA_MAX_GENES = 20
QEA_POP_SIZE = 20
QEA_GENERATIONS = 60


class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, X, y, n_features, min_genes=15, max_genes=20,
                 pop_size=20, generations=60, random_state=42):
        self.X, self.y = X, y
        self.n_features = n_features
        self.min_genes, self.max_genes = min_genes, max_genes
        self.pop_size, self.generations = pop_size, generations
        self.rng = np.random.default_rng(random_state)
        self.alpha = np.full((pop_size, n_features), 1 / np.sqrt(2))
        self.beta = np.full((pop_size, n_features), 1 / np.sqrt(2))
        self.best_solution, self.best_fitness = None, -np.inf
        self.history = []
        self.surrogate = RandomForestClassifier(n_estimators=60, max_depth=6,
            class_weight="balanced", random_state=random_state, n_jobs=-1)
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def _observe(self, alpha_row):
        probs = alpha_row ** 2
        r = self.rng.random(self.n_features)
        return (r < probs).astype(int)

    def _fitness(self, bits):
        idx = np.where(bits == 1)[0]
        n_sel = len(idx)
        if n_sel < 2:
            return -1.0
        X_sub = self.X[:, idx]
        try:
            scores = cross_val_score(self.surrogate, X_sub, self.y, cv=self.cv, scoring="f1")
            perf = scores.mean()
        except Exception:
            return -1.0
        if n_sel < self.min_genes:
            penalty = 0.01 * (self.min_genes - n_sel)
        elif n_sel > self.max_genes:
            penalty = 0.01 * (n_sel - self.max_genes)
        else:
            penalty = 0.0
        return perf - penalty

    def _rotate(self, alpha_row, beta_row, bits, best_bits, generation):
        theta0 = 0.05 * (1 - generation / self.generations) + 0.01
        new_alpha, new_beta = alpha_row.copy(), beta_row.copy()
        for i in range(self.n_features):
            a, b = alpha_row[i], beta_row[i]
            x_i, b_i = bits[i], best_bits[i]
            if x_i == b_i:
                delta = 0.0
            else:
                s = 1.0 if (a * b) >= 0 else -1.0
                if x_i == 0 and b_i == 1:
                    delta = s * theta0
                elif x_i == 1 and b_i == 0:
                    delta = -s * theta0
                else:
                    delta = 0.0
            cos_d, sin_d = np.cos(delta), np.sin(delta)
            new_a = a * cos_d - b * sin_d
            new_b = a * sin_d + b * cos_d
            norm = np.sqrt(new_a ** 2 + new_b ** 2) + 1e-12
            new_alpha[i], new_beta[i] = new_a / norm, new_b / norm
        return new_alpha, new_beta

    def run(self, verbose=True):
        for gen in range(self.generations):
            gen_best_fit, gen_best_bits = -np.inf, None
            observed_bits_all = []
            for p in range(self.pop_size):
                bits = self._observe(self.alpha[p])
                observed_bits_all.append(bits)
                fit = self._fitness(bits)
                if fit > gen_best_fit:
                    gen_best_fit, gen_best_bits = fit, bits
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = bits.copy()
            if self.best_solution is None:
                self.best_solution = gen_best_bits
            for p in range(self.pop_size):
                self.alpha[p], self.beta[p] = self._rotate(
                    self.alpha[p], self.beta[p], observed_bits_all[p], self.best_solution, gen)
            self.history.append(self.best_fitness)
            if verbose and (gen % 10 == 0 or gen == self.generations - 1):
                n_sel = int(self.best_solution.sum())
                print(f"  Gen {gen+1:3d}/{self.generations} | best F1: {self.best_fitness:.4f} | genes: {n_sel}")
        return np.where(self.best_solution == 1)[0], self.best_fitness


def split_features_target(df, target_col=TARGET_COL, meta_cols=META_COLS):
    y = df[target_col].values if target_col in df.columns else None
    drop_cols = [c for c in meta_cols + [target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def main():
    print("=" * 80)
    print("TRAINING: BREAST CANCER QEA PIPELINE")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    X_train_raw, y_train = split_features_target(train_df)
    X_test_raw, y_test = split_features_target(test_df)
    gene_columns = X_train_raw.columns.tolist()
    X_test_raw = X_test_raw.reindex(columns=gene_columns)

    train_medians = X_train_raw.median(numeric_only=True)
    X_train_raw = X_train_raw.fillna(train_medians)
    X_test_raw = X_test_raw.fillna(train_medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # ANOVA
    k = min(K_ANOVA, X_train_scaled.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=k)
    X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
    X_test_anova = anova_selector.transform(X_test_scaled)
    anova_mask = anova_selector.get_support()
    anova_genes = np.array(gene_columns)[anova_mask]
    print(f"ANOVA reduced: {X_train_scaled.shape[1]} -> {X_train_anova.shape[1]} genes")

    # QEA
    print("\nRunning QEA optimization...")
    qea = QuantumInspiredEvolutionaryAlgorithm(
        X_train_anova, y_train, n_features=X_train_anova.shape[1],
        min_genes=QEA_MIN_GENES, max_genes=QEA_MAX_GENES,
        pop_size=QEA_POP_SIZE, generations=QEA_GENERATIONS, random_state=42)
    qea_idx, qea_fitness = qea.run(verbose=True)
    biomarker_genes = anova_genes[qea_idx]
    print(f"\nQEA selected {len(biomarker_genes)} biomarkers")

    X_train_final = X_train_anova[:, qea_idx]
    X_test_final = X_test_anova[:, qea_idx]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42),
    }
    for m in models.values():
        m.fit(X_train_final, y_train)

    best_model_name = None
    best_f1 = -1
    test_metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_final)
        y_proba = model.predict_proba(X_test_final)[:, 1]
        f1 = f1_score(y_test, y_pred)
        test_metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1,
            "auc": roc_auc_score(y_test, y_proba),
        }
        if f1 > best_f1:
            best_f1, best_model_name = f1, name
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")

    # Build mini-scaler for selected genes
    biomarker_indices_in_raw = [gene_columns.index(g) for g in biomarker_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[biomarker_indices_in_raw]
    final_scaler.scale_ = scaler.scale_[biomarker_indices_in_raw]
    final_scaler.var_ = scaler.var_[biomarker_indices_in_raw]
    final_scaler.n_features_in_ = len(biomarker_indices_in_raw)

    artifact = {
        "cancer_type": "breast_cancer",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(biomarker_genes),
        "metadata": {
            "algorithm": best_model_name,
            "best_params": best_model.get_params(),
            "test_accuracy": float(test_metrics[best_model_name]["accuracy"]),
            "test_f1": float(test_metrics[best_model_name]["f1"]),
            "test_auc": float(test_metrics[best_model_name]["auc"]),
            "training_date": datetime.datetime.utcnow().isoformat(),
        }
    }
    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()