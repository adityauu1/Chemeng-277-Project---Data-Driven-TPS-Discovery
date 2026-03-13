import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PATHS
# =========================================================

TRAIN_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\Updated Training Data Sheet.csv"
TPSX_INPUT_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\tpsx_materials_missing_values.csv"
OUTPUT_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\TPSX_verification\tpsx_missing_values_filled.xlsx"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# =========================================================
# TRAINING DATA COLUMN NAMES
# =========================================================

TR_COL_DENSITY = "density (g/cm^3)"
TR_COL_CP = "Cp (J/kg K)"
TR_COL_E = "Elastic Modulus (GPA)"
TR_COL_CTE = "CTE (C^-1) * 10e6"
TR_COL_K = "log( (k) (W m-1 K-1))"

CLASS_COLS = ["oxide", "carbide", "borides", "nitride", "silicates/aluminosilicates"]


# =========================================================
# TPSX COLUMN NAMES
# =========================================================

TP_COL_MATERIAL = "Material"
TP_COL_CLASS = "Material Class"
TP_COL_DENSITY = "Density (kg/m^3)"
TP_COL_K = "Thermal_Conductivity_W_mK"
TP_COL_CP = "Heat_Capacity_J_kgK"
TP_COL_E = "Elastic_Modulus_Pa"
TP_COL_CTE = "CTE (1/K)"


# =========================================================
# LOAD TRAINING DATA
# =========================================================

train = pd.read_csv(TRAIN_PATH)
train.columns = train.columns.str.strip()

for col in [TR_COL_DENSITY, TR_COL_CP, TR_COL_E, TR_COL_CTE, TR_COL_K] + CLASS_COLS:
    train[col] = pd.to_numeric(train[col], errors="coerce")

# convert CTE to true 1/K
train[TR_COL_CTE] = train[TR_COL_CTE] * 1e-6

# TRAINING DATA ALREADY STORES log(k)
train["log_k"] = pd.to_numeric(train[TR_COL_K], errors="coerce")
TR_COL_LOGK = "log_k"


# =========================================================
# RANDOM FOREST SETTINGS
# =========================================================

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


# =========================================================
# TRAIN MODEL FUNCTION
# =========================================================

def train_model(features, target):

    X = train[features].copy()
    y = pd.to_numeric(train[target], errors="coerce")

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    mask = ~y.isna()

    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(**RF_PARAMS))
    ])

    model.fit(X, y)

    return model


# =========================================================
# TRAIN MODELS
# =========================================================

model_logk = train_model(
    [TR_COL_DENSITY, TR_COL_CP, TR_COL_E, TR_COL_CTE] + CLASS_COLS,
    TR_COL_LOGK
)

model_E = train_model(
    [TR_COL_DENSITY, TR_COL_CP, TR_COL_CTE, TR_COL_LOGK] + CLASS_COLS,
    TR_COL_E
)

model_density = train_model(
    [TR_COL_CP, TR_COL_E, TR_COL_CTE, TR_COL_LOGK] + CLASS_COLS,
    TR_COL_DENSITY
)

model_cp = train_model(
    [TR_COL_DENSITY, TR_COL_E, TR_COL_CTE, TR_COL_LOGK] + CLASS_COLS,
    TR_COL_CP
)

model_cte = train_model(
    [TR_COL_DENSITY, TR_COL_CP, TR_COL_E, TR_COL_LOGK] + CLASS_COLS,
    TR_COL_CTE
)


# =========================================================
# LOAD TPSX DATA
# =========================================================

tps = pd.read_csv(TPSX_INPUT_PATH)
tps.columns = tps.columns.str.strip()

print("Target sheet columns:")
print(tps.columns.tolist())


# =========================================================
# MATERIAL CLASS ENCODER
# =========================================================

def encode_class(material_class):

    s = str(material_class).strip().lower()

    class_map = {c: 0 for c in CLASS_COLS}

    if "oxide" in s or "ceramic" in s or "zirconia" in s:
        class_map["oxide"] = 1

    elif "carbide" in s or "sic" in s:
        class_map["carbide"] = 1

    elif "boride" in s:
        class_map["borides"] = 1

    elif "nitride" in s:
        class_map["nitride"] = 1

    elif "silicate" in s:
        class_map["silicates/aluminosilicates"] = 1

    return [class_map[c] for c in CLASS_COLS]


# =========================================================
# UNIT CONVERSION
# =========================================================

def density_kgm3_to_gcm3(x):
    return x / 1000 if pd.notna(x) else np.nan

def density_gcm3_to_kgm3(x):
    return x * 1000 if pd.notna(x) else np.nan

def E_pa_to_gpa(x):
    return x / 1e9 if pd.notna(x) else np.nan

def E_gpa_to_pa(x):
    return x * 1e9 if pd.notna(x) else np.nan


# =========================================================
# INTERNAL COLUMN PREP
# =========================================================

tps["_density_gcm3"] = pd.to_numeric(tps[TP_COL_DENSITY], errors="coerce").apply(density_kgm3_to_gcm3)

tps["_k"] = pd.to_numeric(tps[TP_COL_K], errors="coerce")

tps["_cp"] = pd.to_numeric(tps[TP_COL_CP], errors="coerce")

tps["_E_gpa"] = pd.to_numeric(tps[TP_COL_E], errors="coerce").apply(E_pa_to_gpa)

tps["_cte"] = pd.to_numeric(tps[TP_COL_CTE], errors="coerce")


# convert k → log(k)
tps["_logk"] = np.nan
mask_k = tps["_k"] > 0
tps.loc[mask_k, "_logk"] = np.log(tps.loc[mask_k, "_k"])


# =========================================================
# ITERATIVE MISSING VALUE PREDICTION
# =========================================================

def make_df(vals, cols):
    return pd.DataFrame([vals], columns=cols)

N_PASSES = 5

for _ in range(N_PASSES):

    for idx, row in tps.iterrows():

        class_vec = encode_class(row.get(TP_COL_CLASS, ""))

        density = row["_density_gcm3"]
        logk = row["_logk"]
        cp = row["_cp"]
        E = row["_E_gpa"]
        cte = row["_cte"]


        if pd.isna(logk):

            X = make_df(
                [density, cp, E, cte] + class_vec,
                [TR_COL_DENSITY, TR_COL_CP, TR_COL_E, TR_COL_CTE] + CLASS_COLS
            )

            pred_logk = model_logk.predict(X)[0]

            tps.at[idx, "_logk"] = pred_logk
            tps.at[idx, "_k"] = float(np.exp(pred_logk))


        density = tps.at[idx, "_density_gcm3"]
        logk = tps.at[idx, "_logk"]
        cp = tps.at[idx, "_cp"]
        E = tps.at[idx, "_E_gpa"]
        cte = tps.at[idx, "_cte"]


        if pd.isna(E):

            X = make_df(
                [density, cp, cte, logk] + class_vec,
                [TR_COL_DENSITY, TR_COL_CP, TR_COL_CTE, TR_COL_LOGK] + CLASS_COLS
            )

            tps.at[idx, "_E_gpa"] = model_E.predict(X)[0]


        density = tps.at[idx, "_density_gcm3"]
        logk = tps.at[idx, "_logk"]
        cp = tps.at[idx, "_cp"]
        E = tps.at[idx, "_E_gpa"]
        cte = tps.at[idx, "_cte"]


        if pd.isna(density):

            X = make_df(
                [cp, E, cte, logk] + class_vec,
                [TR_COL_CP, TR_COL_E, TR_COL_CTE, TR_COL_LOGK] + CLASS_COLS
            )

            tps.at[idx, "_density_gcm3"] = model_density.predict(X)[0]


        density = tps.at[idx, "_density_gcm3"]
        logk = tps.at[idx, "_logk"]
        cp = tps.at[idx, "_cp"]
        E = tps.at[idx, "_E_gpa"]
        cte = tps.at[idx, "_cte"]


        if pd.isna(cp):

            X = make_df(
                [density, E, cte, logk] + class_vec,
                [TR_COL_DENSITY, TR_COL_E, TR_COL_CTE, TR_COL_LOGK] + CLASS_COLS
            )

            tps.at[idx, "_cp"] = model_cp.predict(X)[0]


        density = tps.at[idx, "_density_gcm3"]
        logk = tps.at[idx, "_logk"]
        cp = tps.at[idx, "_cp"]
        E = tps.at[idx, "_E_gpa"]
        cte = tps.at[idx, "_cte"]


        if pd.isna(cte):

            X = make_df(
                [density, cp, E, logk] + class_vec,
                [TR_COL_DENSITY, TR_COL_CP, TR_COL_E, TR_COL_LOGK] + CLASS_COLS
            )

            tps.at[idx, "_cte"] = model_cte.predict(X)[0]


# =========================================================
# WRITE BACK ORIGINAL UNITS
# =========================================================

missing_density = pd.to_numeric(tps[TP_COL_DENSITY], errors="coerce").isna()
missing_k = pd.to_numeric(tps[TP_COL_K], errors="coerce").isna()
missing_cp = pd.to_numeric(tps[TP_COL_CP], errors="coerce").isna()
missing_E = pd.to_numeric(tps[TP_COL_E], errors="coerce").isna()
missing_cte = pd.to_numeric(tps[TP_COL_CTE], errors="coerce").isna()

tps.loc[missing_density, TP_COL_DENSITY] = tps.loc[missing_density, "_density_gcm3"].apply(density_gcm3_to_kgm3)
tps.loc[missing_k, TP_COL_K] = tps.loc[missing_k, "_k"]
tps.loc[missing_cp, TP_COL_CP] = tps.loc[missing_cp, "_cp"]
tps.loc[missing_E, TP_COL_E] = tps.loc[missing_E, "_E_gpa"].apply(E_gpa_to_pa)
tps.loc[missing_cte, TP_COL_CTE] = tps.loc[missing_cte, "_cte"]


# =========================================================
# FLAGS
# =========================================================

tps["density_filled"] = missing_density
tps["k_filled"] = missing_k
tps["Cp_filled"] = missing_cp
tps["E_filled"] = missing_E
tps["CTE_filled"] = missing_cte


# =========================================================
# SAVE
# =========================================================

internal_cols = ["_density_gcm3","_k","_cp","_E_gpa","_cte","_logk"]

tps_out = tps.drop(columns=internal_cols, errors="ignore")

tps_out.to_excel(OUTPUT_PATH, index=False)

print("\nSaved completed file to:")
print(OUTPUT_PATH)