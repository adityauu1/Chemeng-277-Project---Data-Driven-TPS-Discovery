import pandas as pd
import numpy as np

# =========================================================
# PATHS
# =========================================================

INPUT_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\TPSX_verification\tpsx_missing_values_filled.xlsx"
OUTPUT_PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\TPSX_verification\tpsx_ranked_materials.xlsx"

# =========================================================
# COLUMN NAMES
# =========================================================

COL_MATERIAL = "Material"
COL_CLASS = "Material Class"
COL_DENSITY = "Density (kg/m^3)"
COL_K = "Thermal_Conductivity_W_mK"
COL_CP = "Heat_Capacity_J_kgK"
COL_E = "Elastic_Modulus_Pa"
COL_CTE = "CTE (1/K)"

# =========================================================
# WEIGHTS (SUM = 1)
# =========================================================

WEIGHTS = {
    COL_DENSITY: 0.25,
    COL_K: 0.35,
    COL_CP: 0.10,
    COL_E: 0.10,
    COL_CTE: 0.20
}

# =========================================================
# NORMALIZATION FUNCTIONS
# =========================================================

def minmax_high_better(series):

    s = pd.to_numeric(series, errors="coerce")

    smin = s.min()
    smax = s.max()

    if np.isclose(smax, smin):
        return pd.Series(1.0, index=s.index)

    return (s - smin) / (smax - smin)


def minmax_low_better(series):

    s = pd.to_numeric(series, errors="coerce")

    smin = s.min()
    smax = s.max()

    if np.isclose(smax, smin):
        return pd.Series(1.0, index=s.index)

    return (smax - s) / (smax - smin)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_excel(INPUT_PATH)
df.columns = df.columns.str.strip()

for col in [COL_DENSITY, COL_K, COL_CP, COL_E, COL_CTE]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================================================
# PROPERTY SCORES (0–1)
# =========================================================

df["score_density"] = minmax_low_better(df[COL_DENSITY])
df["score_k"] = minmax_low_better(df[COL_K])
df["score_cp"] = minmax_high_better(df[COL_CP])
df["score_E"] = minmax_low_better(df[COL_E])
df["score_cte"] = minmax_low_better(df[COL_CTE])

# =========================================================
# TPS SCORE
# =========================================================

df["TPS_score"] = (
    WEIGHTS[COL_DENSITY] * df["score_density"] +
    WEIGHTS[COL_K] * df["score_k"] +
    WEIGHTS[COL_CP] * df["score_cp"] +
    WEIGHTS[COL_E] * df["score_E"] +
    WEIGHTS[COL_CTE] * df["score_cte"]
)

# Optional 0–100 score
df["TPS_score_100"] = df["TPS_score"] * 100

# =========================================================
# RANK MATERIALS
# =========================================================

df = df.sort_values("TPS_score", ascending=False).reset_index(drop=True)

df["TPS_rank"] = np.arange(1, len(df) + 1)

# =========================================================
# SAVE
# =========================================================

df.to_excel(OUTPUT_PATH, index=False)

print(df[[COL_MATERIAL, "TPS_rank", "TPS_score", "TPS_score_100"]])
print("\nSaved ranking file to:")
print(OUTPUT_PATH)