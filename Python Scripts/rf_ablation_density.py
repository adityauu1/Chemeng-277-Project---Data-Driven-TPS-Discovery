import os
from rf_ablation_utils import (
    load_dataset,
    run_ablation_suite,
    COL_DENSITY,
    COL_CP,
    COL_E,
    COL_CTE,
    COL_LOGK,
    CLASS_COLS
)

PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\Updated Training Data Sheet.csv"
OUT_DIR = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\ablation_v2\ablation_density"

df = load_dataset(PATH)

feature_cols = [COL_CP, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS
target_col = COL_DENSITY

run_ablation_suite(
    df=df,
    target_col=target_col,
    full_feature_cols=feature_cols,
    out_dir=OUT_DIR,
    short_name="density"
)