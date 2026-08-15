# -*- coding: utf-8 -*-
"""Ovarian Cancer QEA Pipeline - VS Code training script."""

import os, sys, warnings, datetime, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_config import DATA_DIR, MODELS_DIR, CANCER_TYPES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["ovarian"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["ovarian"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["ovarian"]["model_file"])

CONFIG = {
    "anova_k": 100, "qea_pop_size": 20, "qea_generations": 30, "qea_cv_folds": 3,
    "qea_rotation_angle": 0.03 * np.pi, "qea_mutation_rate": 0.05,
    "qea_penalty_lambda": 0.03, "qea_min_features": 3,
}


def split_features_labels(df, dataset_name):
    label_col = None
    for cand in ["label", "Label", "LABEL", "class", "Class", "target", "Target"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]
    y = df[label_col]
    X = df.drop(columns=[label_col])
    non_numeric_cols = []
    for col in X.columns:
        coerced = pd.to_numeric(X[col], errors="coerce")
        if coerced.isna().mean() > 0.5:
            non_numeric_cols.append(col)
        else:
            X[col] = coerced
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)
    try:
        y = y.astype(int)
    except Exception:
        y = pd.to_numeric(y, errors="coerce").astype(int)
    return X, y


class QEAFeatureSelector:
    def __init__(self, n_features, base_classifier, pop_size=20, generations=30,
                 cv_folds=3, rotation_angle=0.03 * np.pi, mutation_rate=0.05,
                 penalty_lambda=0.03, min_features=3, random_state=42):
        self.n_features = n_features
        self.base_classifier = base_classifier
        self.pop_size, self.generations = pop_size, generations
        self.cv_folds, self.rotation_angle = cv_folds, rotation_angle
        self.mutation_rate, self.penalty_lambda = mutation_rate, penalty_lambda
        self.min_features = min_features
        self.rng = np.random.RandomState(random_state)
        inv_sqrt2 = 1.0 / np.sqrt(2)
        self.alpha = np.full((pop_size, n_features), inv_sqrt2)
        self.beta = np.full((pop_size, n_features), inv_sqrt2)
        self.best_solution, self.best_fitness = None, -np.inf
        self.fitness_history = []

    def _observe(self):
        probs_one = self.alpha ** 2
        rand_matrix = self.rng.rand(self.pop_size, self.n_features)
        population = (rand_matrix < probs_one).astype(int)
        for i in range(self.pop_size):
            if population[i].sum() < self.min_features:
                extra_idx = self.rng.choice(self.n_features, size=self.min_features, replace=False)
                population[i, extra_idx] = 1
        return population

    def _fitness(self, mask, X, y):
        if mask.sum() == 0:
            return -1.0
        X_sub = X[:, mask.astype(bool)]
        try:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(self.base_classifier, X_sub, y, cv=skf, scoring="accuracy")
            acc = scores.mean()
        except Exception:
            acc = 0.0
        return acc - self.penalty_lambda * (mask.sum() / self.n_features)

    def _rotate(self, population):
        best = self.best_solution
        for i in range(self.pop_size):
            for j in range(self.n_features):
                bit = population[i, j]
                if bit != best[j]:
                    theta = self.rotation_angle if best[j] == 1 else -self.rotation_angle
                    a, b = self.alpha[i, j], self.beta[i, j]
                    new_a = a * np.cos(theta) - b * np.sin(theta)
                    new_b = a * np.sin(theta) + b * np.cos(theta)
                    norm = np.sqrt(new_a ** 2 + new_b ** 2)
                    if norm > 1e-12:
                        self.alpha[i, j], self.beta[i, j] = new_a / norm, new_b / norm
            mutate_mask = self.rng.rand(self.n_features) < self.mutation_rate
            self.alpha[i, mutate_mask] = 1 / np.sqrt(2)
            self.beta[i, mutate_mask] = 1 / np.sqrt(2)

    def fit(self, X, y, verbose=True):
        for gen in range(1, self.generations + 1):
            population = self._observe()
            fitnesses = np.array([self._fitness(ind, X, y) for ind in population])
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[gen_best_idx]
                self.best_solution = population[gen_best_idx].copy()
            self.fitness_history.append(self.best_fitness)
            if self.best_solution is not None:
                self._rotate(population)
            if verbose and (gen % 5 == 0 or gen == 1 or gen == self.generations):
                n_sel = int(self.best_solution.sum()) if self.best_solution is not None else 0
                print(f"  [QEA] Gen {gen:3d}/{self.generations} | best fitness={self.best_fitness:.4f} | features={n_sel}")
        return self.best_solution, self.best_fitness, self.fitness_history


def main():
    print("=" * 80)
    print("TRAINING: OVARIAN CANCER QEA PIPELINE")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    X_train_raw, y_train = split_features_labels(train_df, "Train")
    X_test_raw, y_test = split_features_labels(test_df, "Test")

    common_cols = X_train_raw.columns.intersection(X_test_raw.columns)
    common_cols = list(common_cols)
    X_train_raw = X_train_raw[common_cols]
    X_test_raw = X_test_raw[common_cols]
    gene_names = np.array(common_cols)

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    k = min(CONFIG["anova_k"], X_train_scaled.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=k)
    X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
    anova_mask = anova_selector.get_support()
    X_test_anova = X_test_scaled[:, anova_mask]
    anova_genes = gene_names[anova_mask]
    print(f"ANOVA reduced: {X_train_scaled.shape[1]} -> {k} genes")

    print("\nRunning QEA optimization...")
    base_clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    qea = QEAFeatureSelector(
        n_features=X_train_anova.shape[1], base_classifier=base_clf,
        pop_size=CONFIG["qea_pop_size"], generations=CONFIG["qea_generations"],
        cv_folds=CONFIG["qea_cv_folds"], rotation_angle=CONFIG["qea_rotation_angle"],
        mutation_rate=CONFIG["qea_mutation_rate"], penalty_lambda=CONFIG["qea_penalty_lambda"],
        min_features=min(CONFIG["qea_min_features"], X_train_anova.shape[1]),
        random_state=RANDOM_STATE)
    best_mask, best_fitness, _ = qea.fit(X_train_anova, y_train.values)
    best_mask = best_mask.astype(bool)
    selected_genes = anova_genes[best_mask]
    print(f"\nQEA selected {len(selected_genes)} biomarkers")

    X_train_final = X_train_anova[:, best_mask]
    X_test_final = X_test_anova[:, best_mask]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=RANDOM_STATE),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        y_proba = model.predict_proba(X_test_final)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = float("nan")
        f1 = f1_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": acc, "auc": auc, "f1": f1}
        print(f"  {name}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    best_model_name = max(results, key=lambda n: (results[n]["auc"] if not np.isnan(results[n]["auc"]) else results[n]["accuracy"]))
    best_model = results[best_model_name]["model"]
    print(f"\nBest model: {best_model_name}")

    # Build mini-scaler for selected genes
    selected_indices = [list(gene_names).index(g) for g in selected_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[selected_indices]
    final_scaler.scale_ = scaler.scale_[selected_indices]
    final_scaler.var_ = scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "ovarian",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(selected_genes),
        "metadata": {
            "algorithm": best_model_name,
            "best_params": best_model.get_params(),
            "test_accuracy": float(results[best_model_name]["accuracy"]),
            "test_f1": float(results[best_model_name]["f1"]),
            "test_auc": float(results[best_model_name]["auc"]),
            "training_date": datetime.datetime.utcnow().isoformat(),
        }
    }
    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()