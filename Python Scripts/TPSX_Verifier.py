import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =============================
# PATHS
# =============================

TRAIN_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\Updated Training Data Sheet.csv"
TPSX_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\tpsx_dataset_full.xlsx"
OUTPUT_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\TPSX_verification\TPSX_sanity_check_results_flexible_feature_groups_logk.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# =============================
# LOAD TRAIN DATA
# =============================

train = pd.read_csv(TRAIN_PATH)
train.columns = train.columns.str.strip()

COL_DENSITY = "density (g/cm^3)"
COL_CP = "Cp (J/kg K)"
COL_E = "Elastic Modulus (GPA)"
COL_CTE = "CTE (C^-1) * 10e6"
COL_K = "log( (k) (W m-1 K-1))"

CLASS_COLS = ["oxide", "carbide", "borides", "nitride", "silicates/aluminosilicates"]

ALL_NUMERIC_COLS = [COL_DENSITY, COL_CP, COL_E, COL_CTE, COL_K] + CLASS_COLS

for col in ALL_NUMERIC_COLS:
    train[col] = pd.to_numeric(train[col], errors="coerce")

# training sheet CTE is stored as x10^-6 numbers
train[COL_CTE] = train[COL_CTE] * 1e-6

# create log(k)
train["log_k"] = np.where(train[COL_K] > 0, np.log(train[COL_K]), np.nan)
COL_LOGK = "log_k"


# =============================
# RANDOM FOREST SETTINGS
# FLEXIBLE MODEL
# =============================

RF_PARAMS = dict(
    n_estimators=800,
    max_depth=16,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)


# =============================
# TRAIN MODEL FUNCTION
# =============================

def train_model(features, target):
    X = train[features].copy()
    y = pd.to_numeric(train[target], errors="coerce")

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    mask = ~y.isna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    rf = RandomForestRegressor(**RF_PARAMS)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", rf)
    ])

    model.fit(X, y)
    return model


# =============================
# FEATURE SETS
# =============================

FULL_logk = [COL_DENSITY, COL_CP, COL_E, COL_CTE] + CLASS_COLS
CONT_logk = [COL_DENSITY, COL_CP, COL_E, COL_CTE]
CLASS_logk = CLASS_COLS

FULL_E = [COL_DENSITY, COL_CP, COL_CTE, COL_LOGK] + CLASS_COLS
CONT_E = [COL_DENSITY, COL_CP, COL_CTE, COL_LOGK]
CLASS_E = CLASS_COLS

FULL_density = [COL_CP, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS
CONT_density = [COL_CP, COL_E, COL_CTE, COL_LOGK]
CLASS_density = CLASS_COLS

FULL_cp = [COL_DENSITY, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS
CONT_cp = [COL_DENSITY, COL_E, COL_CTE, COL_LOGK]
CLASS_cp = CLASS_COLS

FULL_cte = [COL_DENSITY, COL_CP, COL_E, COL_LOGK] + CLASS_COLS
CONT_cte = [COL_DENSITY, COL_CP, COL_E, COL_LOGK]
CLASS_cte = CLASS_COLS


# =============================
# TRAIN ALL MODELS
# =============================

model_logk_full = train_model(FULL_logk, COL_LOGK)
model_logk_cont = train_model(CONT_logk, COL_LOGK)
model_logk_class = train_model(CLASS_logk, COL_LOGK)

model_E_full = train_model(FULL_E, COL_E)
model_E_cont = train_model(CONT_E, COL_E)
model_E_class = train_model(CLASS_E, COL_E)

model_density_full = train_model(FULL_density, COL_DENSITY)
model_density_cont = train_model(CONT_density, COL_DENSITY)
model_density_class = train_model(CLASS_density, COL_DENSITY)

model_cp_full = train_model(FULL_cp, COL_CP)
model_cp_cont = train_model(CONT_cp, COL_CP)
model_cp_class = train_model(CLASS_cp, COL_CP)

model_cte_full = train_model(FULL_cte, COL_CTE)
model_cte_cont = train_model(CONT_cte, COL_CTE)
model_cte_class = train_model(CLASS_cte, COL_CTE)


# =============================
# LOAD TPSX EXCEL DATA
# =============================

tps = pd.read_excel(TPSX_PATH)
tps.columns = tps.columns.str.strip()

print("TPSX columns:")
print(tps.columns.tolist())


# =============================
# HELPER: ENCODE MATERIAL CLASS
# =============================

def encode_class(material_class):
    material_class = str(material_class).strip().lower()
    class_map = {c: 0 for c in CLASS_COLS}

    if "oxide" in material_class:
        class_map["oxide"] = 1
    elif "carbide" in material_class:
        class_map["carbide"] = 1
    elif "boride" in material_class:
        class_map["borides"] = 1
    elif "nitride" in material_class:
        class_map["nitride"] = 1
    elif "silicate" in material_class or "aluminosilicate" in material_class:
        class_map["silicates/aluminosilicates"] = 1

    return [class_map[c] for c in CLASS_COLS]


def make_df(values, columns):
    return pd.DataFrame([values], columns=columns)


# =============================
# SANITY CHECK PREDICTIONS
# =============================

print("\nTPSX SANITY CHECK PREDICTIONS — FLEXIBLE MODEL + FEATURE GROUPS + LOG(K)\n")

results = []

for _, row in tps.iterrows():
    material = row["Material"]
    material_class = row["Material Class"]

    density_kgm3 = pd.to_numeric(row["Density (kg/m^3)"], errors="coerce")
    k = pd.to_numeric(row["Thermal_Conductivity_W_mK"], errors="coerce")
    cp = pd.to_numeric(row["Heat_Capacity_J_kgK"], errors="coerce")
    E_pa = pd.to_numeric(row["Elastic_Modulus_Pa"], errors="coerce")
    cte = pd.to_numeric(row["CTE (1/K)"], errors="coerce")

    density = density_kgm3 / 1000 if pd.notna(density_kgm3) else np.nan
    E = E_pa / 1e9 if pd.notna(E_pa) else np.nan
    logk_actual = np.log(k) if pd.notna(k) and k > 0 else np.nan

    class_vec = encode_class(material_class)

    # log(k)
    X_logk_full = make_df([density, cp, E, cte] + class_vec, FULL_logk)
    X_logk_cont = make_df([density, cp, E, cte], CONT_logk)
    X_logk_class = make_df(class_vec, CLASS_logk)

    pred_logk_full = model_logk_full.predict(X_logk_full)[0]
    pred_logk_cont = model_logk_cont.predict(X_logk_cont)[0]
    pred_logk_class = model_logk_class.predict(X_logk_class)[0]

    pred_k_full = np.exp(pred_logk_full)
    pred_k_cont = np.exp(pred_logk_cont)
    pred_k_class = np.exp(pred_logk_class)

    # E
    X_E_full = make_df([density, cp, cte, pred_logk_full] + class_vec, FULL_E)
    X_E_cont = make_df([density, cp, cte, pred_logk_cont], CONT_E)
    X_E_class = make_df(class_vec, CLASS_E)

    pred_E_full = model_E_full.predict(X_E_full)[0]
    pred_E_cont = model_E_cont.predict(X_E_cont)[0]
    pred_E_class = model_E_class.predict(X_E_class)[0]

    # density
    X_density_full = make_df([cp, E, cte, pred_logk_full] + class_vec, FULL_density)
    X_density_cont = make_df([cp, E, cte, pred_logk_cont], CONT_density)
    X_density_class = make_df(class_vec, CLASS_density)

    pred_density_full = model_density_full.predict(X_density_full)[0]
    pred_density_cont = model_density_cont.predict(X_density_cont)[0]
    pred_density_class = model_density_class.predict(X_density_class)[0]

    # Cp
    X_cp_full = make_df([density, E, cte, pred_logk_full] + class_vec, FULL_cp)
    X_cp_cont = make_df([density, E, cte, pred_logk_cont], CONT_cp)
    X_cp_class = make_df(class_vec, CLASS_cp)

    pred_cp_full = model_cp_full.predict(X_cp_full)[0]
    pred_cp_cont = model_cp_cont.predict(X_cp_cont)[0]
    pred_cp_class = model_cp_class.predict(X_cp_class)[0]

    # CTE
    X_cte_full = make_df([density, cp, E, pred_logk_full] + class_vec, FULL_cte)
    X_cte_cont = make_df([density, cp, E, pred_logk_cont], CONT_cte)
    X_cte_class = make_df(class_vec, CLASS_cte)

    pred_cte_full = model_cte_full.predict(X_cte_full)[0]
    pred_cte_cont = model_cte_cont.predict(X_cte_cont)[0]
    pred_cte_class = model_cte_class.predict(X_cte_class)[0]

    results.append([
        material,
        material_class,

        k,
        logk_actual,
        pred_logk_full,
        pred_logk_cont,
        pred_logk_class,
        pred_k_full,
        pred_k_cont,
        pred_k_class,
        abs(k - pred_k_full) if pd.notna(k) else np.nan,

        E,
        pred_E_full,
        pred_E_cont,
        pred_E_class,
        abs(E - pred_E_full) if pd.notna(E) else np.nan,

        density,
        pred_density_full,
        pred_density_cont,
        pred_density_class,
        abs(density - pred_density_full) if pd.notna(density) else np.nan,

        cp,
        pred_cp_full,
        pred_cp_cont,
        pred_cp_class,
        abs(cp - pred_cp_full) if pd.notna(cp) else np.nan,

        cte,
        pred_cte_full,
        pred_cte_cont,
        pred_cte_class,
        abs(cte - pred_cte_full) if pd.notna(cte) else np.nan
    ])


# =============================
# SAVE RESULTS
# =============================

results = pd.DataFrame(results, columns=[
    "Material",
    "Material Class",

    "k_actual_W_mK",
    "logk_actual",
    "logk_pred_full",
    "logk_pred_continuous_only",
    "logk_pred_class_only",
    "k_pred_full_W_mK",
    "k_pred_continuous_only_W_mK",
    "k_pred_class_only_W_mK",
    "k_abs_error_full",

    "E_actual_GPa",
    "E_pred_full",
    "E_pred_continuous_only",
    "E_pred_class_only",
    "E_abs_error_full",

    "density_actual_gcm3",
    "density_pred_full",
    "density_pred_continuous_only",
    "density_pred_class_only",
    "density_abs_error_full",

    "Cp_actual_J_kgK",
    "Cp_pred_full",
    "Cp_pred_continuous_only",
    "Cp_pred_class_only",
    "Cp_abs_error_full",

    "CTE_actual_1_per_K",
    "CTE_pred_full",
    "CTE_pred_continuous_only",
    "CTE_pred_class_only",
    "CTE_abs_error_full"
])

print(results)
results.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved results to:\n{OUTPUT_PATH}")