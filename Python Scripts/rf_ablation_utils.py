import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# GLOBAL COLUMN NAMES
# =========================================================

COL_COMPOUND = "Compound"
COL_DENSITY = "density (g/cm^3)"
COL_CP = "Cp (J/kg K)"
COL_E = "Elastic Modulus (GPA)"
COL_CTE = "CTE (C^-1) * 10e6"
COL_LOGK = "log( (k) (W m-1 K-1))"

CLASS_COLS = [
    "oxide",
    "carbide",
    "borides",
    "nitride",
    "silicates/aluminosilicates"
]

ALL_NUMERIC_COLS = [COL_DENSITY, COL_CP, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in ALL_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # CTE column is stored as values in "times 10^6"
    df[COL_CTE] = df[COL_CTE] * 1e-6

    return df


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_rf(X: pd.DataFrame, y: pd.Series, rf_params: dict, cv) -> dict:
    rf = RandomForestRegressor(**rf_params)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", rf)
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=("r2", "neg_mean_squared_error"),
        n_jobs=-1,
        return_train_score=False
    )

    rmse_folds = np.sqrt(-scores["test_neg_mean_squared_error"])
    r2_folds = scores["test_r2"]

    model.fit(X, y)
    pred_train = model.predict(X)

    train_r2 = 1 - np.sum((y - pred_train) ** 2) / np.sum((y - y.mean()) ** 2)

    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.named_steps["rf"].feature_importances_
    }).sort_values("importance", ascending=False)

    return {
        "rmse_folds": rmse_folds,
        "r2_folds": r2_folds,
        "cv_rmse_mean": float(rmse_folds.mean()),
        "cv_r2_mean": float(r2_folds.mean()),
        "train_r2": float(train_r2),
        "feature_importance": fi
    }


# =========================================================
# ABLATION RUNNER
# =========================================================

def run_ablation_suite(
    df: pd.DataFrame,
    target_col: str,
    full_feature_cols: list,
    out_dir: str,
    short_name: str
):
    os.makedirs(out_dir, exist_ok=True)

    mask = ~df[target_col].isna()
    base_df = df.loc[mask].copy()

    y = base_df[target_col].copy()

    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # -----------------------------------------------------
    # 1) MODEL COMPLEXITY ABLATION
    # -----------------------------------------------------
    complexity_configs = {
    "simple": {
        "n_estimators": 200,
        "max_depth": 4,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "max_features": 0.6,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    },
    "medium": {
        "n_estimators": 400,
        "max_depth": 8,
        "min_samples_split": 6,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    },
    "flexible": {
        "n_estimators": 800,
        "max_depth": 16,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    },
    "very_flexible": {
        "n_estimators": 1200,
        "max_depth": 24,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    },
    "extreme": {
        "n_estimators": 1600,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    }
}

    complexity_rows = []
    for config_name, rf_params in complexity_configs.items():
        X = base_df[full_feature_cols].copy()

        res = evaluate_rf(X, y, rf_params, kf)

        complexity_rows.append({
            "target": short_name,
            "ablation_type": "model_complexity",
            "setting": config_name,
            "rows_used": len(X),
            "n_features": X.shape[1],
            "cv_rmse_mean": res["cv_rmse_mean"],
            "cv_r2_mean": res["cv_r2_mean"],
            "train_r2": res["train_r2"]
        })

    complexity_df = pd.DataFrame(complexity_rows)
    complexity_df.to_csv(
        os.path.join(out_dir, f"ablation_model_complexity_{short_name}.csv"),
        index=False
    )

    # -----------------------------------------------------
    # 2) FEATURE REMOVAL ABLATION
    # -----------------------------------------------------
    feature_removal_rows = []
    default_rf = complexity_configs["medium"]

    # full baseline
    X_full = base_df[full_feature_cols].copy()
    res_full = evaluate_rf(X_full, y, default_rf, kf)

    feature_removal_rows.append({
        "target": short_name,
        "ablation_type": "feature_removal",
        "setting": "full_model",
        "rows_used": len(X_full),
        "n_features": X_full.shape[1],
        "cv_rmse_mean": res_full["cv_rmse_mean"],
        "cv_r2_mean": res_full["cv_r2_mean"],
        "train_r2": res_full["train_r2"]
    })

    # remove one feature at a time
    for removed_feature in full_feature_cols:
        reduced_features = [c for c in full_feature_cols if c != removed_feature]
        X_reduced = base_df[reduced_features].copy()

        res = evaluate_rf(X_reduced, y, default_rf, kf)

        feature_removal_rows.append({
            "target": short_name,
            "ablation_type": "feature_removal",
            "setting": f"remove__{removed_feature}",
            "rows_used": len(X_reduced),
            "n_features": X_reduced.shape[1],
            "cv_rmse_mean": res["cv_rmse_mean"],
            "cv_r2_mean": res["cv_r2_mean"],
            "train_r2": res["train_r2"]
        })

    feature_removal_df = pd.DataFrame(feature_removal_rows)
    feature_removal_df.to_csv(
        os.path.join(out_dir, f"ablation_feature_removal_{short_name}.csv"),
        index=False
    )

    # -----------------------------------------------------
    # 3) FEATURE GROUP ABLATION
    # -----------------------------------------------------
    continuous_candidates = [c for c in full_feature_cols if c not in CLASS_COLS]
    class_candidates = [c for c in full_feature_cols if c in CLASS_COLS]

    group_sets = {
        "continuous_only": continuous_candidates,
        "class_only": class_candidates,
        "full_model": full_feature_cols
    }

    feature_group_rows = []
    best_fi = None

    for group_name, feat_list in group_sets.items():
        X_group = base_df[feat_list].copy()

        res = evaluate_rf(X_group, y, default_rf, kf)

        feature_group_rows.append({
            "target": short_name,
            "ablation_type": "feature_group",
            "setting": group_name,
            "rows_used": len(X_group),
            "n_features": X_group.shape[1],
            "cv_rmse_mean": res["cv_rmse_mean"],
            "cv_r2_mean": res["cv_r2_mean"],
            "train_r2": res["train_r2"]
        })

        if group_name == "full_model":
            best_fi = res["feature_importance"]

    feature_group_df = pd.DataFrame(feature_group_rows)
    feature_group_df.to_csv(
        os.path.join(out_dir, f"ablation_feature_group_{short_name}.csv"),
        index=False
    )

    if best_fi is not None:
        best_fi.to_csv(
            os.path.join(out_dir, f"feature_importance_{short_name}.csv"),
            index=False
        )

    # -----------------------------------------------------
    # COMBINED SUMMARY
    # -----------------------------------------------------
    summary_df = pd.concat(
        [complexity_df, feature_removal_df, feature_group_df],
        axis=0,
        ignore_index=True
    )

    summary_df.to_csv(
        os.path.join(out_dir, f"ablation_summary_{short_name}.csv"),
        index=False
    )

    print(f"\nFinished ablation suite for {short_name}")
    print(summary_df)