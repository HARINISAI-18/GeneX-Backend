# -*- coding: utf-8 -*-
"""Cervical Cancer QEA Pipeline - VS Code training script."""

import os, sys, warnings, datetime, joblib
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cancer_config import DATA_DIR, MODELS_DIR, CANCER_TYPES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_PATH = os.path.join(DATA_DIR, CANCER_TYPES["cervical"]["train_file"])
TEST_PATH = os.path.join(DATA_DIR, CANCER_TYPES["cervical"]["test_file"])
MODEL_OUT = os.path.join(MODELS_DIR, CANCER_TYPES["cervical"]["model_file"])

STAGE_MAP = {0: "Normal", 1: "CIN1", 2: "CIN3", 3: "Cervical_Cancer"}
META_COLS = ["sample_id", "disease_stage", "stage_label"]
EXCLUDE_STAGES = ["CIN1", "CIN3"]

N_QBITS = None
POP_SIZE = 20
GENERATIONS = 25
THETA_STEP = 0.05 * np.pi
INIT_PROB = 0.30
SIZE_PENALTY = 0.02
MIN_GENES = 2


def adaptive_cv_folds(y, max_folds=5, min_folds=2):
    counts = pd.Series(y).value_counts()
    return int(np.clip(counts.min(), min_folds, max_folds))


def main():
    print("=" * 80)
    print("TRAINING: CERVICAL CANCER QEA PIPELINE")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    # Remove CIN1 and CIN3
    train_df = train_df[~train_df["disease_stage"].isin(EXCLUDE_STAGES)].reset_index(drop=True)
    test_df = test_df[~test_df["disease_stage"].isin(EXCLUDE_STAGES)].reset_index(drop=True)

    remaining_labels = sorted(train_df["stage_label"].unique())
    label_remap = {old: new for new, old in enumerate(remaining_labels)}
    for _df in (train_df, test_df):
        _df["stage_label"] = _df["stage_label"].map(label_remap).astype(int)
    global STAGE_MAP
    STAGE_MAP = {label_remap[k]: v for k, v in STAGE_MAP.items() if k in label_remap}

    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    X_train_raw = train_df[feature_cols]
    y_train = train_df["stage_label"].astype(int).values
    X_test_raw = test_df[feature_cols]
    y_test = test_df["stage_label"].astype(int).values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s = scaler.transform(X_test_imp)

    CV_FOLDS = adaptive_cv_folds(y_train)
    print(f"Using {CV_FOLDS}-fold stratified CV")

    K = min(200, X_train_s.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=K)
    X_train_anova = anova_selector.fit_transform(X_train_s, y_train)
    anova_mask = anova_selector.get_support()
    anova_features = np.array(feature_cols)[anova_mask]
    X_test_anova = anova_selector.transform(X_test_s)
    print(f"ANOVA reduced: {len(feature_cols)} -> {K} genes")

    global N_QBITS
    N_QBITS = X_train_anova.shape[1]

    skf_qea = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    qea_fitness_clf = RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE,
                                             class_weight="balanced", n_jobs=-1)

    def collapse_qbits(theta_row, rng):
        prob_one = np.sin(theta_row) ** 2
        r = rng.random(len(theta_row))
        return (r < prob_one).astype(int)

    def evaluate_fitness(bits):
        idx = np.where(bits == 1)[0]
        if len(idx) < MIN_GENES:
            return 0.0, idx
        X_sub = X_train_anova[:, idx]
        try:
            scores = cross_val_score(qea_fitness_clf, X_sub, y_train, cv=skf_qea, scoring="accuracy")
            acc = scores.mean()
        except Exception:
            return 0.0, idx
        penalty = SIZE_PENALTY * (len(idx) / N_QBITS)
        return acc - penalty, idx

    def rotation_update(theta, pop_bits, best_bits):
        for i in range(pop_bits.shape[0]):
            for j in range(theta.shape[1]):
                b, bb = pop_bits[i, j], best_bits[j]
                if b == bb:
                    continue
                alpha, beta = np.cos(theta[i, j]), np.sin(theta[i, j])
                direction = 1 if (alpha * beta) >= 0 else -1
                if bb == 1 and b == 0:
                    theta[i, j] += direction * THETA_STEP
                elif bb == 0 and b == 1:
                    theta[i, j] -= direction * THETA_STEP
        return theta

    print("\nRunning QEA optimization...")
    rng = np.random.default_rng(RANDOM_STATE)
    theta = np.full((POP_SIZE, N_QBITS), np.arcsin(np.sqrt(INIT_PROB)))
    best_bits, best_fitness = None, -np.inf
    for gen in range(GENERATIONS):
        pop_bits = np.array([collapse_qbits(theta[i], rng) for i in range(POP_SIZE)])
        fitnesses = np.array([evaluate_fitness(b)[0] for b in pop_bits])
        gen_best = np.argmax(fitnesses)
        if fitnesses[gen_best] > best_fitness:
            best_fitness = fitnesses[gen_best]
            best_bits = pop_bits[gen_best].copy()
        theta = rotation_update(theta, pop_bits, best_bits)
        if gen % 5 == 0 and gen > 0:
            mut_mask = rng.random(theta.shape) < 0.01
            theta[mut_mask] = np.pi / 2 - theta[mut_mask]
        if gen % 5 == 0 or gen == GENERATIONS - 1:
            print(f"  Gen {gen:2d}/{GENERATIONS} | best fitness = {best_fitness:.4f} | genes = {int(best_bits.sum())}")

    qea_selected_idx = np.where(best_bits == 1)[0]
    qea_selected_genes = anova_features[qea_selected_idx]
    print(f"\nQEA selected {len(qea_selected_genes)} biomarkers")

    X_train_final = X_train_anova[:, qea_selected_idx]
    X_test_final = X_test_anova[:, qea_selected_idx]

    skf_final = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    param_grids = {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {"n_estimators": [150, 300], "max_depth": [None, 6]}),
        "XGBoost": (
            xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", verbosity=0),
            {"n_estimators": [150, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
        "SVM": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}),
    }

    trained_models = {}
    for name, (estimator, grid) in param_grids.items():
        try:
            search = GridSearchCV(estimator, grid, cv=skf_final, scoring="f1_macro", n_jobs=-1)
            search.fit(X_train_final, y_train)
            trained_models[name] = search.best_estimator_
            print(f"{name}: best params = {search.best_params_}")
        except Exception as e:
            print(f"{name}: grid search failed ({e}); using defaults.")
            estimator.fit(X_train_final, y_train)
            trained_models[name] = estimator

    eval_rows = []
    test_preds = {}
    for name, model in trained_models.items():
        pred = model.predict(X_test_final)
        test_preds[name] = pred
        eval_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
            "Macro Precision": precision_score(y_test, pred, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        })
    results_df = pd.DataFrame(eval_rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    print("\n" + results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # Build mini-scaler for selected genes
    selected_indices = [feature_cols.index(g) for g in qea_selected_genes]
    final_scaler = StandardScaler()
    final_scaler.mean_ = scaler.mean_[selected_indices]
    final_scaler.scale_ = scaler.scale_[selected_indices]
    final_scaler.var_ = scaler.var_[selected_indices]
    final_scaler.n_features_in_ = len(selected_indices)

    artifact = {
        "cancer_type": "cervical",
        "model": best_model,
        "scaler": final_scaler,
        "feature_names": list(qea_selected_genes),
        "metadata": {
            "algorithm": best_model_name,
            "best_params": best_model.get_params(),
            "test_accuracy": float(results_df.iloc[0]["Accuracy"]),
            "test_f1": float(results_df.iloc[0]["Macro F1"]),
            "training_date": datetime.datetime.utcnow().isoformat(),
            "stage_map": STAGE_MAP,
        }
    }
    joblib.dump(artifact, MODEL_OUT, compress=3)
    print(f"\n✅ Model artifact saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()