import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# =========================================================
# PATHS
# =========================================================

PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\Training_Data_Sheet_Thermal_Conductivity.csv"
OUT_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\ablation_ridge_lasso_k.csv"


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(PATH)
df.columns = df.columns.str.strip()


# =========================================================
# TARGET + FEATURES
# =========================================================

target_col = "thermal conductivity (k) (W m-1 K-1)"

feature_cols = [
    "density (g/cm^3)",
    "Cp (J/kg K)",
    "Elastic Modulus (GPA)",
    "CTE (C^-1) * 10e-6",
    "oxide",
    "carbide",
    "borides",
    "nitride",
    "silicates/aluminosilicates"
]

if target_col not in df.columns:
    raise ValueError(f"Missing target column: {target_col}")

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")


X = df[feature_cols].copy()
y = df[target_col].copy()


# =========================================================
# CLEAN TYPES
# =========================================================

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

y = pd.to_numeric(y, errors="coerce")

# Convert CTE into true 1/C
X["CTE (C^-1) * 10e-6"] = X["CTE (C^-1) * 10e-6"] * 1e-6

mask = ~y.isna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print(f"Rows used: {len(X)}")
print(f"Train rows: {len(X_train)} | Validation rows: {len(X_val)}")


# =========================================================
# HYPERPARAMETER GRIDS
# =========================================================

RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0]


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(model_name, alpha, model):
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, pred_val))

    train_r2 = r2_score(y_train, pred_train)
    val_r2 = r2_score(y_val, pred_val)

    coef = model.named_steps["model"].coef_
    n_nonzero = int(np.sum(np.abs(coef) > 1e-12))

    return {
        "model": model_name,
        "alpha": alpha,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "n_nonzero_coeffs": n_nonzero
    }


# =========================================================
# RUN ABLATION
# =========================================================

results = []

for alpha in RIDGE_ALPHAS:
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha))
    ])

    res = evaluate_model("Ridge", alpha, ridge)
    results.append(res)

for alpha in LASSO_ALPHAS:
    lasso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=alpha, max_iter=50000))
    ])

    res = evaluate_model("Lasso", alpha, lasso)
    results.append(res)


# =========================================================
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["model", "alpha"]).reset_index(drop=True)

results_df.to_csv(OUT_PATH, index=False)

print("\nAblation results:")
print(results_df)
print(f"\nSaved to:\n{OUT_PATH}")