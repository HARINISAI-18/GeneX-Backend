# -*- coding: utf-8 -*-
"""Leukemia (AML) QEA Pipeline - VS Code training script."""

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

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["leukemia"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["leukemia"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["leukemia"]["model_file"])

TARGET_COL = "Label"
METADATA_COLS = ["Sample_ID", "Class", "Label"]
K_ANOVA = 100

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, n_qubits, X, y, pop_size=20, generations=25,
                 delta_theta=0.05 * np.pi, feature_penalty=0.05,
                 cv_folds=3, min_features=5, random_state=42):
        self.n_qubits = n_qubits
        self.X, self.y = X, y
        self.pop_size, self.generations = pop_size, generations
        self.delta_theta = delta_theta
        self.feature_penalty = feature_penalty
        self.cv_folds, self.min_features = cv_folds, min_features
        self.rng = np.random.RandomState(random_state)
        self.alpha = np.full((pop_size, n_qubits), 1 / np.sqrt(2))
        self.beta = np.full((pop_size, n_qubits), 1 / np.sqrt(2))
        self.best_binary, self.best_fitness = None, -np.inf

    def _observe(self):
        r = self.rng.rand(self.pop_size, self.n_qubits)
        return (r < self.beta ** 2).astype(int)

    def _fitness(self, binary_individual):
        idx = np.where(binary_individual == 1)[0]
        if len(idx) < self.min_features:
            return -1.0
        X_sub = self.X[:, idx]
        clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced", n_jobs=-1)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_sub, self.y, cv=skf, scoring="f1_weighted")
        return scores.mean() - (self.feature_penalty * (len(idx) / self.n_qubits))

    def _rotation_update(self, pop_binary):
        b = self.best_binary
        for i in range(self.pop_size):
            for j in range(self.n_qubits):
                x_ij = pop_binary[i, j]
                if x_ij == b[j]: continue
                a, be = self.alpha[i, j], self.beta[i, j]
                sign = np.sign(a * be) if np.sign(a * be) != 0 else 1.0
                theta = self.delta_theta * sign if b[j] == 1 else -self.delta_theta * sign
                self.alpha[i, j], self.beta[i, j] = a * np.cos(theta) - be * np.sin(theta), a * np.sin(theta) + be * np.cos(theta)

    def run(self, verbose=True):
        for gen in range(1, self.generations + 1):
            pop_binary = self._observe()
            fitness_vals = np.array([self._fitness(ind) for ind in pop_binary])
            gen_best_idx = np.argmax(fitness_vals)
            if fitness_vals[gen_best_idx] > self.best_fitness:
                self.best_fitness = fitness_vals[gen_best_idx]
                self.best_binary = pop_binary[gen_best_idx].copy()
            if self.best_binary is not None:
                self._rotation_update(pop_binary)
            if verbose and (gen % 5 == 0 or gen == 1 or gen == self.generations):
                print(f"  Gen {gen:3d}/{self.generations} | Best fitness = {self.best_fitness:.4f} | Biomarkers = {int(self.best_binary.sum())}")
        return self.best_binary, self.best_fitness

def split_X_y(df, target_col=TARGET_COL, meta_cols=METADATA_COLS):
    y = df[target_col].astype(int).values
    X = df.drop(columns=[c for c in meta_cols if c in df.columns])
    return X, y

def main():
    print("=" * 80)
    print("TRAINING: LEUKEMIA (AML) QEA PIPELINE")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    X_train_raw, y_train = split_X_y(train_df)
    X_test_raw, y_test = split_X_y(test_df)
    gene_columns = X_train_raw.columns.to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    anova_selector = SelectKBest(score_func=f_classif, k=min(K_ANOVA, X_train_scaled.shape[1]))
    X_train_anova = anova_selector.fit_transform(X_train_scaled, y_train)
    X_test_anova = anova_selector.transform(X_test_scaled)
    anova_genes = gene_columns[anova_selector.get_support()]
    print(f"ANOVA reduced: {X_train_scaled.shape[1]} -> {len(anova_genes)} genes")

    print("\nRunning QEA optimization...")
    qea = QuantumInspiredEvolutionaryAlgorithm(n_qubits=X_train_anova.shape[1], X=X_train_anova, y=y_train)
    best_mask, best_fitness = qea.run(verbose=True)
    biomarker_idx = np.where(best_mask == 1)[0]
    biomarker_genes = anova_genes[biomarker_idx]
    print(f"QEA selected {len(biomarker_genes)} biomarkers")

    X_train_final = X_train_anova[:, biomarker_idx]
    X_test_final = X_test_anova[:, biomarker_idx]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss", scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    }
    
    best_f1, best_model_name, best_model = -1, None, None
    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1, best_model_name, best_model = f1, name, model

    print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")

    selected_indices = [list(gene_columns).index(g) for g in biomarker_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[selected_indices]
    final_scaler.scale_ = scaler.scale_[selected_indices]
    final_scaler.var_ = scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "leukemia",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(biomarker_genes),
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