# -*- coding: utf-8 -*-
"""Esophageal Cancer QEA Pipeline - VS Code training script."""

import os, sys, warnings, datetime, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_config import DATA_DIR, MODELS_DIR, CANCER_TYPES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["esophageal"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["esophageal"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["esophageal"]["model_file"])

NON_FEATURE_COLS = ["sample_id", "patient_id", "disease_status", "label"]
K_ANOVA = 500
QEA_POP_SIZE = 12
QEA_GENERATIONS = 20
QEA_ROTATION_ANGLE = 0.05 * np.pi
QEA_MUTATION_RATE = 0.05
QEA_FEATURE_PENALTY = 0.004
QEA_MIN_FEATURES = 5
CV_FOLDS = 3


class DataPreprocessor:
    def __init__(self, train_path, test_path):
        self.train_path, self.test_path = train_path, test_path
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_and_split(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        self.feature_names = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
        X_train = train_df[self.feature_names].values.astype(float)
        y_train = train_df["label"].values.astype(int)
        X_test = test_df[self.feature_names].values.astype(float)
        y_test = test_df["label"].values.astype(int)
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            col_medians = np.nanmedian(X_train, axis=0)
            for X in (X_train, X_test):
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_medians, inds[1])
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return (X_train, y_train), (X_test, y_test)


class AnovaFeatureSelector:
    def __init__(self, k=K_ANOVA):
        self.k = k
        self.selector, self.selected_idx, self.selected_names = None, None, None

    def fit_select(self, X_train, y_train, feature_names):
        k = min(self.k, X_train.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_train_sel = self.selector.fit_transform(X_train, y_train)
        self.selected_idx = self.selector.get_support(indices=True)
        self.selected_names = np.array(feature_names)[self.selected_idx]
        return X_train_sel

    def transform(self, X):
        return X[:, self.selected_idx]


class QuantumInspiredFeatureSelector:
    def __init__(self, n_features, pop_size=QEA_POP_SIZE, generations=QEA_GENERATIONS,
                 rotation_angle=QEA_ROTATION_ANGLE, mutation_rate=QEA_MUTATION_RATE,
                 feature_penalty=QEA_FEATURE_PENALTY, min_features=QEA_MIN_FEATURES,
                 cv_folds=CV_FOLDS, random_state=RANDOM_STATE):
        self.n_features = n_features
        self.pop_size, self.generations = pop_size, generations
        self.rotation_angle, self.mutation_rate = rotation_angle, mutation_rate
        self.feature_penalty, self.min_features, self.cv_folds = feature_penalty, min_features, cv_folds
        self.rng = np.random.RandomState(random_state)
        init_include_prob = 0.06
        theta0 = np.arcsin(np.sqrt(init_include_prob))
        self.theta = np.full((pop_size, n_features), theta0)
        self.best_bits, self.best_fitness = None, -np.inf
        self.history = []

    def _observe(self):
        beta_sq = np.sin(self.theta) ** 2
        r = self.rng.rand(self.pop_size, self.n_features)
        return (r < beta_sq).astype(int)

    def _fitness(self, bits, X, y):
        n_selected = bits.sum()
        if n_selected < self.min_features:
            return -1.0
        idx = np.where(bits == 1)[0]
        X_sub = X[:, idx]
        clf = RandomForestClassifier(n_estimators=30, max_depth=5,
                                     random_state=RANDOM_STATE, n_jobs=1)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
        try:
            scores = cross_val_score(clf, X_sub, y, cv=cv, scoring="accuracy")
            cv_acc = scores.mean()
        except ValueError:
            cv_acc = cross_val_score(clf, X_sub, y, cv=min(3, self.cv_folds), scoring="accuracy").mean()
        return cv_acc - self.feature_penalty * n_selected

    def _rotate(self, population, fitness_vals):
        for i in range(self.pop_size):
            for j in range(self.n_features):
                xi, bi = population[i, j], self.best_bits[j]
                if xi == bi:
                    continue
                alpha = np.cos(self.theta[i, j])
                beta = np.sin(self.theta[i, j])
                if fitness_vals[i] < self.best_fitness:
                    if bi == 1:
                        delta = self.rotation_angle if alpha * beta >= 0 else -self.rotation_angle
                    else:
                        delta = -self.rotation_angle if alpha * beta >= 0 else self.rotation_angle
                    self.theta[i, j] += delta
        self.theta = np.clip(self.theta, 1e-3, np.pi / 2 - 1e-3)

    def _mutate(self):
        mask = self.rng.rand(self.pop_size, self.n_features) < self.mutation_rate
        self.theta[mask] = (np.pi / 2) - self.theta[mask]

    def optimize(self, X, y, verbose=True):
        for gen in range(1, self.generations + 1):
            population = self._observe()
            fitness_vals = np.array([self._fitness(ind, X, y) for ind in population])
            gen_best_idx = np.argmax(fitness_vals)
            if fitness_vals[gen_best_idx] > self.best_fitness:
                self.best_fitness = fitness_vals[gen_best_idx]
                self.best_bits = population[gen_best_idx].copy()
            self.history.append(self.best_fitness)
            if self.best_bits is not None:
                self._rotate(population, fitness_vals)
            self._mutate()
            if verbose and (gen % 5 == 0 or gen == 1):
                n_sel = int(self.best_bits.sum()) if self.best_bits is not None else 0
                print(f"    Gen {gen:3d}/{self.generations} | best fitness={self.best_fitness:.4f} | panel size={n_sel}")
        return self.best_bits, self.best_fitness


def main():
    print("=" * 80)
    print("TRAINING: ESOPHAGEAL CANCER QEA PIPELINE")
    print("=" * 80)

    prep = DataPreprocessor(TRAIN_PATH, TEST_PATH)
    (X_train, y_train), (X_test, y_test) = prep.load_and_split()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train class balance -> No Cancer: {(y_train==0).sum()}, Cancer: {(y_train==1).sum()}")

    anova = AnovaFeatureSelector(k=K_ANOVA)
    X_train_anova = anova.fit_select(X_train, y_train, prep.feature_names)
    X_test_anova = anova.transform(X_test)
    print(f"ANOVA reduced: {X_train.shape[1]} -> {X_train_anova.shape[1]} genes")

    print("\nRunning QEA optimization...")
    qea = QuantumInspiredFeatureSelector(n_features=X_train_anova.shape[1])
    best_bits, best_fitness = qea.optimize(X_train_anova, y_train, verbose=True)
    selected_idx = np.where(best_bits == 1)[0]
    selected_genes = anova.selected_names[selected_idx]
    print(f"\nQEA selected {len(selected_genes)} biomarkers")

    X_train_qea = X_train_anova[:, selected_idx]
    X_test_qea = X_test_anova[:, selected_idx]

    final_clf = RandomForestClassifier(n_estimators=300, max_depth=5,
                                       random_state=RANDOM_STATE, n_jobs=-1)
    final_clf.fit(X_train_qea, y_train)
    y_test_pred = final_clf.predict(X_test_qea)
    y_test_proba = final_clf.predict_proba(X_test_qea)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    try:
        test_auc = roc_auc_score(y_test, y_test_proba)
    except Exception:
        test_auc = float("nan")
    print(f"\nTest Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

    # Build mini-scaler for selected genes
    selected_indices = [prep.feature_names.index(g) for g in selected_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = prep.scaler.mean_[selected_indices]
    final_scaler.scale_ = prep.scaler.scale_[selected_indices]
    final_scaler.var_ = prep.scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "esophageal",
        "model": final_clf,
        "scaler": final_scaler,
        "feature_names": list(selected_genes),
        "metadata": {
            "algorithm": "RandomForest",
            "best_params": final_clf.get_params(),
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_auc": float(test_auc),
            "training_date": datetime.datetime.utcnow().isoformat(),
        }
    }
    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()