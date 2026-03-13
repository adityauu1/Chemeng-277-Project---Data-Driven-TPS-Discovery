import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\Updated Training Data Sheet.csv"
OUT_DIR = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(PATH)
df.columns = df.columns.str.strip()

COL_DENSITY = "density (g/cm^3)"
COL_CP = "Cp (J/kg K)"
COL_E = "Elastic Modulus (GPA)"
COL_CTE = "CTE (C^-1) * 10e6"
COL_LOGK = "log( (k) (W m-1 K-1))"
CLASS_COLS = ["oxide", "carbide", "borides", "nitride", "silicates/aluminosilicates"]

for col in [COL_DENSITY, COL_CP, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[COL_CTE] = df[COL_CTE] * 1e-6

feature_cols = [COL_DENSITY, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS
target_col = COL_CP

X = df[feature_cols].copy()
y = df[target_col].copy()

mask = ~y.isna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()

rf = RandomForestRegressor(
    n_estimators=800,
    max_depth=15,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", rf)
])

kf = KFold(n_splits=4, shuffle=True, random_state=42)

rmse_folds = np.sqrt(-cross_val_score(
    model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
))
r2_folds = cross_val_score(
    model, X, y, cv=kf, scoring="r2", n_jobs=-1
)

model.fit(X, y)
pred_train = model.predict(X)
train_r2 = 1 - np.sum((y - pred_train) ** 2) / np.sum((y - y.mean()) ** 2)

print("\nRandom Forest — Heat Capacity")
print("-----------------------------")
print("4-fold CV RMSE folds:", rmse_folds)
print("4-fold CV R2 folds:", r2_folds)
print("4-fold CV Mean RMSE:", rmse_folds.mean())
print("4-fold CV Mean R2:", r2_folds.mean())
print("Train R2:", train_r2)

metrics = pd.DataFrame([{
    "target": "heat_capacity",
    "rows_used": len(X),
    "cv_rmse_mean": rmse_folds.mean(),
    "cv_r2_mean": r2_folds.mean(),
    "train_r2": train_r2
}])
metrics.to_csv(os.path.join(OUT_DIR, "metrics_rf_heat_capacity_4fold.csv"), index=False)

fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.named_steps["rf"].feature_importances_
}).sort_values("importance", ascending=False)
fi.to_csv(os.path.join(OUT_DIR, "feature_importance_rf_heat_capacity_4fold.csv"), index=False)