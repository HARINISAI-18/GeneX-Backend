# -*- coding: utf-8 -*-
"""Lung Cancer QEA Pipeline - VS Code training script."""

import os, sys, warnings, datetime, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_config import DATA_DIR, MODELS_DIR, CANCER_TYPES

SEED = 42
np.random.seed(SEED)

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["lung_cancer"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["lung_cancer"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["lung_cancer"]["model_file"])

META_COLS = ["sample_id", "patient_id", "disease_status", "label"]
TARGET_COL = "label"
ANOVA_TOP_K = 200
QEA_POP_SIZE = 20
QEA_GENERATIONS = 40
QEA_ROTATION_ANGLE = 0.05 * np.pi
QEA_ALPHA = 0.9
QEA_BETA = 0.1
QEA_CV_FOLDS = 5


class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, n_features, X, y, pop_size=20, generations=40,
                 rotation_angle=0.05 * np.pi, alpha_w=0.9, beta_w=0.1,
                 cv_folds=5, min_features=5, seed=42):
        self.n_features = n_features
        self.X, self.y = X, y
        self.pop_size, self.generations = pop_size, generations
        self.rotation_angle = rotation_angle
        self.alpha_w, self.beta_w = alpha_w, beta_w
        self.cv_folds, self.min_features = cv_folds, min_features
        self.rng = np.random.RandomState(seed)
        self.qubits = np.full((n_features, 2), 1.0 / np.sqrt(2))
        self.surrogate = LogisticRegression(max_iter=500, random_state=seed, solver="liblinear")
        self.best_solution, self.best_fitness = None, -np.inf
        self.fitness_history = []

    def _observe(self):
        population = np.zeros((self.pop_size, self.n_features), dtype=int)
        probs_select = self.qubits[:, 1] ** 2
        for p in range(self.pop_size):
            rand_vals = self.rng.rand(self.n_features)
            population[p] = (rand_vals < probs_select).astype(int)
            if population[p].sum() < self.min_features:
                extra_needed = self.min_features - population[p].sum()
                off_idx = np.where(population[p] == 0)[0]
                if len(off_idx) > 0:
                    turn_on = self.rng.choice(off_idx, size=min(extra_needed, len(off_idx)), replace=False)
                    population[p, turn_on] = 1
        return population

    def _fitness(self, mask):
        selected_idx = np.where(mask == 1)[0]
        if len(selected_idx) == 0:
            return -1.0
        X_sub = self.X[:, selected_idx]
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=SEED)
        try:
            scores = cross_val_score(self.surrogate, X_sub, self.y, cv=skf, scoring="accuracy", n_jobs=-1)
            acc = scores.mean()
        except Exception:
            acc = 0.0
        feature_ratio = len(selected_idx) / self.n_features
        return self.alpha_w * acc - self.beta_w * feature_ratio

    def _rotate_qubits(self, best_mask, population, fitnesses):
        for p in range(self.pop_size):
            for i in range(self.n_features):
                b_i = population[p, i]
                best_i = best_mask[i]
                alpha, beta = self.qubits[i]
                if b_i == best_i:
                    continue
                if fitnesses[p] < self.best_fitness:
                    if b_i == 0 and best_i == 1:
                        delta = self.rotation_angle
                    elif b_i == 1 and best_i == 0:
                        delta = -self.rotation_angle
                    else:
                        delta = 0.0
                else:
                    delta = 0.0
                new_alpha = alpha * np.cos(delta) - beta * np.sin(delta)
                new_beta = alpha * np.sin(delta) + beta * np.cos(delta)
                norm = np.sqrt(new_alpha ** 2 + new_beta ** 2)
                if norm > 0:
                    new_alpha, new_beta = new_alpha / norm, new_beta / norm
                new_alpha = np.clip(new_alpha, 0.02, 0.999)
                new_beta = np.sqrt(max(0.0, 1 - new_alpha ** 2))
                self.qubits[i] = [new_alpha, new_beta]

    def _mutate(self, rate=0.02):
        for i in range(self.n_features):
            if self.rng.rand() < rate:
                self.qubits[i] = self.qubits[i][::-1]

    def run(self, verbose=True):
        for gen in range(1, self.generations + 1):
            population = self._observe()
            fitnesses = np.array([self._fitness(ind) for ind in population])
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[gen_best_idx]
                self.best_solution = population[gen_best_idx].copy()
            self._rotate_qubits(self.best_solution, population, fitnesses)
            self._mutate()
            self.fitness_history.append(self.best_fitness)
            if verbose and (gen % 5 == 0 or gen == 1):
                n_sel = int(self.best_solution.sum())
                print(f"  Gen {gen:3d}/{self.generations} | Best fitness: {self.best_fitness:.4f} | Genes: {n_sel}")
        return self.best_solution, self.best_fitness, self.fitness_history


def main():
    print("=" * 80)
    print("TRAINING: LUNG CANCER QEA PIPELINE")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    gene_cols = [c for c in train_df.columns if c not in META_COLS]
    X_train_raw = train_df[gene_cols].copy()
    y_train = train_df[TARGET_COL].values
    X_test_raw = test_df[gene_cols].copy()
    y_test = test_df[TARGET_COL].values

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    anova_k = min(ANOVA_TOP_K, X_train_scaled.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=anova_k)
    X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
    X_test_anova = anova_selector.transform(X_test_scaled)
    anova_mask = anova_selector.get_support()
    anova_gene_names = np.array(gene_cols)[anova_mask]
    print(f"ANOVA reduced: {X_train_scaled.shape[1]} -> {X_train_anova.shape[1]} genes")

    print("\nRunning QEA optimization...")
    qea = QuantumInspiredEvolutionaryAlgorithm(
        n_features=X_train_anova.shape[1], X=X_train_anova, y=y_train,
        pop_size=QEA_POP_SIZE, generations=QEA_GENERATIONS,
        rotation_angle=QEA_ROTATION_ANGLE, alpha_w=QEA_ALPHA, beta_w=QEA_BETA,
        cv_folds=QEA_CV_FOLDS, seed=SEED)
    best_mask, best_fitness, _ = qea.run(verbose=True)
    qea_selected_idx = np.where(best_mask == 1)[0]
    qea_gene_names = anova_gene_names[qea_selected_idx]
    print(f"\nQEA selected {len(qea_gene_names)} biomarkers")

    X_train_final = X_train_anova[:, qea_selected_idx]
    X_test_final = X_test_anova[:, qea_selected_idx]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=SEED, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear"),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }
    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        trained_models[name] = model
        test_pred = model.predict(X_test_final)
        test_proba = model.predict_proba(X_test_final)[:, 1]
        results.append({
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, test_pred),
            "F1": f1_score(y_test, test_pred),
            "ROC-AUC": roc_auc_score(y_test, test_proba),
        })
    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
    print("\n" + results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # Build mini-scaler for selected genes
    selected_indices = [gene_cols.index(g) for g in qea_gene_names]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[selected_indices]
    final_scaler.scale_ = scaler.scale_[selected_indices]
    final_scaler.var_ = scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "lung_cancer",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(qea_gene_names),
        "metadata": {
            "algorithm": best_model_name,
            "best_params": best_model.get_params(),
            "test_accuracy": float(results_df.iloc[0]["Test Accuracy"]),
            "test_f1": float(results_df.iloc[0]["F1"]),
            "test_auc": float(results_df.iloc[0]["ROC-AUC"]),
            "training_date": datetime.datetime.utcnow().isoformat(),
        }
    }
    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()