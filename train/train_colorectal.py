# -*- coding: utf-8 -*-
"""Colorectal Cancer QEA Pipeline - VS Code training script."""

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
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_config import DATA_DIR, MODELS_DIR, CANCER_TYPES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["colorectal"]["train_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["colorectal"]["model_file"])

NON_FEATURE_COLS = ["sample_id", "disease_status", "label"]
K_ANOVA = 100

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, n_features, X, y, pop_size=20, n_generations=30, gamma=0.15,
                 rotation_angle=0.05 * np.pi, min_features=5, cv_folds=3, random_state=42):
        self.n_features = n_features
        self.X, self.y = X, y
        self.pop_size, self.n_generations = pop_size, n_generations
        self.gamma, self.rotation_angle = gamma, rotation_angle
        self.min_features, self.cv_folds = min_features, cv_folds
        self.rng = np.random.RandomState(random_state)
        self.alpha = np.full((pop_size, n_features), 1 / np.sqrt(2))
        self.beta = np.full((pop_size, n_features), 1 / np.sqrt(2))
        self.best_chromosome, self.best_fitness = None, -np.inf

    def _measure(self, alpha, beta):
        probs = beta ** 2
        rand_vals = self.rng.random(self.n_features)
        binary = (rand_vals < probs).astype(int)
        if binary.sum() < self.min_features:
            zero_idx = np.where(binary == 0)[0]
            chosen = self.rng.choice(zero_idx, size=self.min_features - binary.sum(), replace=False)
            binary[chosen] = 1
        return binary

    def _fitness(self, binary_chromosome):
        selected = np.where(binary_chromosome == 1)[0]
        if len(selected) == 0: return -np.inf
        X_sub = self.X[:, selected]
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        n_classes_min = np.min(np.bincount(self.y))
        n_splits = min(self.cv_folds, n_classes_min) if n_classes_min >= 2 else 2
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            acc = cross_val_score(clf, X_sub, self.y, cv=skf, scoring="accuracy").mean()
        except ValueError:
            clf.fit(X_sub, self.y)
            acc = clf.score(X_sub, self.y)
        return acc - self.gamma * (len(selected) / self.n_features)

    def _rotate_qbits(self, alpha, beta, binary, best_binary):
        new_alpha, new_beta = alpha.copy(), beta.copy()
        for i in range(self.n_features):
            if binary[i] == best_binary[i]: continue
            delta = self.rotation_angle if best_binary[i] == 1 else -self.rotation_angle
            a, b = alpha[i], beta[i]
            new_a, new_b = a * np.cos(delta) - b * np.sin(delta), a * np.sin(delta) + b * np.cos(delta)
            norm = np.sqrt(new_a ** 2 + new_b ** 2)
            new_alpha[i], new_beta[i] = new_a / norm, new_b / norm
        return new_alpha, new_beta

    def run(self, verbose=True):
        for gen in range(self.n_generations):
            binaries = [self._measure(self.alpha[p], self.beta[p]) for p in range(self.pop_size)]
            fitness_vals = [self._fitness(b) for b in binaries]
            gen_best_idx = int(np.argmax(fitness_vals))
            if fitness_vals[gen_best_idx] > self.best_fitness:
                self.best_fitness = fitness_vals[gen_best_idx]
                self.best_chromosome = binaries[gen_best_idx].copy()
            for p in range(self.pop_size):
                self.alpha[p], self.beta[p] = self._rotate_qbits(self.alpha[p], self.beta[p], binaries[p], self.best_chromosome)
            if verbose and (gen % 5 == 0 or gen == self.n_generations - 1):
                print(f"Gen {gen+1:>2}/{self.n_generations} | Best Fitness: {self.best_fitness:.4f} | Genes: {int(self.best_chromosome.sum())}")
        return self.best_chromosome, self.best_fitness

def main():
    print("=" * 80)
    print("TRAINING: COLORECTAL CANCER QEA PIPELINE")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Train shape: {train_df.shape}")

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    X_train_raw = train_df[feature_cols].values
    y_train = train_df["label"].values

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    anova_selector = SelectKBest(score_func=f_classif, k=min(K_ANOVA, X_train_scaled.shape[1]))
    X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
    anova_genes = np.array(feature_cols)[anova_selector.get_support()]
    print(f"ANOVA reduced: {len(feature_cols)} -> {len(anova_genes)} genes")

    print("\nRunning QEA optimization...")
    qea = QuantumInspiredEvolutionaryAlgorithm(n_features=X_train_anova.shape[1], X=X_train_anova, y=y_train)
    best_bits, best_fit = qea.run(verbose=True)
    qea_idx = np.where(best_bits == 1)[0]
    selected_genes = anova_genes[qea_idx]
    print(f"QEA selected {len(selected_genes)} biomarkers")

    X_train_final = X_train_anova[:, qea_idx]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE, eval_metric="logloss"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    }
    
    # No test set provided in notebook, train on all data
    best_acc, best_model_name, best_model = -1, None, None
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        acc = cross_val_score(model, X_train_final, y_train, cv=skf, scoring="accuracy").mean()
        if acc > best_acc:
            best_acc, best_model_name = acc, name
    best_model = models[best_model_name].fit(X_train_final, y_train)

    print(f"\nBest model: {best_model_name} (CV Accuracy = {best_acc:.4f})")

    selected_indices = [feature_cols.index(g) for g in selected_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[selected_indices]
    final_scaler.scale_ = scaler.scale_[selected_indices]
    final_scaler.var_ = scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "colorectal",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(selected_genes),
        "metadata": {
            "algorithm": best_model_name,
            "best_params": best_model.get_params(),
            "cv_accuracy": float(best_acc),
            "training_date": datetime.datetime.utcnow().isoformat(),
        }
    }
    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")

if __name__ == "__main__":
    main()