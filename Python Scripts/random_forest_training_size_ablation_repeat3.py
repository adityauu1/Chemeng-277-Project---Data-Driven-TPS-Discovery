# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.model_selection import KFold, cross_validate
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor


# # =========================================================
# # PATHS
# # =========================================================

# PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\Updated Training Data Sheet.csv"
# OUT_DIR = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\ablation_v2\training_size_ablation"
# os.makedirs(OUT_DIR, exist_ok=True)


# # =========================================================
# # LOAD DATA
# # =========================================================

# df = pd.read_csv(PATH)
# df.columns = df.columns.str.strip()

# print("Columns in dataset:")
# print(df.columns.tolist())


# # =========================================================
# # COLUMN DEFINITIONS
# # =========================================================

# COL_DENSITY = "density (g/cm^3)"
# COL_CP = "Cp (J/kg K)"
# COL_E = "Elastic Modulus (GPA)"
# COL_CTE = "CTE (C^-1) * 10e6"
# COL_LOGK = "log( (k) (W m-1 K-1))"

# CLASS_COLS = [
#     "oxide",
#     "carbide",
#     "borides",
#     "nitride",
#     "silicates/aluminosilicates"
# ]

# ALL_NUMERIC_COLS = [COL_DENSITY, COL_CP, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS

# for col in ALL_NUMERIC_COLS:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# # Convert CTE from micro-scale notation to 1/K
# df[COL_CTE] = df[COL_CTE] * 1e-6


# # =========================================================
# # SETTINGS
# # =========================================================

# TARGETS = {
#     "logk": {
#         "target_col": COL_LOGK,
#         "feature_cols": [COL_DENSITY, COL_CP, COL_E, COL_CTE] + CLASS_COLS,
#         "y_label_rmse": "Cross-Validated RMSE (log k)",
#         "y_label_r2": r"Cross-Validated $R^2$ (log k)"
#     },
#     "elastic_modulus": {
#         "target_col": COL_E,
#         "feature_cols": [COL_DENSITY, COL_CP, COL_CTE, COL_LOGK] + CLASS_COLS,
#         "y_label_rmse": "Cross-Validated RMSE (Elastic Modulus, GPa)",
#         "y_label_r2": r"Cross-Validated $R^2$ (Elastic Modulus)"
#     },
#     "density": {
#         "target_col": COL_DENSITY,
#         "feature_cols": [COL_CP, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS,
#         "y_label_rmse": "Cross-Validated RMSE (Density, g/cm^3)",
#         "y_label_r2": r"Cross-Validated $R^2$ (Density)"
#     },
#     "heat_capacity": {
#         "target_col": COL_CP,
#         "feature_cols": [COL_DENSITY, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS,
#         "y_label_rmse": "Cross-Validated RMSE (Cp, J/kg·K)",
#         "y_label_r2": r"Cross-Validated $R^2$ (Cp)"
#     },
#     "CTE": {
#         "target_col": COL_CTE,
#         "feature_cols": [COL_DENSITY, COL_CP, COL_E, COL_LOGK] + CLASS_COLS,
#         "y_label_rmse": r"Cross-Validated RMSE (CTE, 1/K)",
#         "y_label_r2": r"Cross-Validated $R^2$ (CTE)"
#     }
# }

# REPEATS = 3

# RF_PARAMS = dict(
#     n_estimators=800,
#     max_depth=16,
#     min_samples_split=3,
#     min_samples_leaf=1,
#     max_features="sqrt",
#     bootstrap=True,
#     random_state=42,
#     n_jobs=-1
# )

# kf = KFold(n_splits=4, shuffle=True, random_state=42)


# # =========================================================
# # EVALUATION FUNCTION
# # =========================================================

# def evaluate_subset(X, y):
#     rf = RandomForestRegressor(**RF_PARAMS)

#     model = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("rf", rf)
#     ])

#     scores = cross_validate(
#         model,
#         X,
#         y,
#         cv=kf,
#         scoring=("r2", "neg_mean_squared_error"),
#         n_jobs=-1
#     )

#     rmse = np.sqrt(-scores["test_neg_mean_squared_error"])
#     r2 = scores["test_r2"]

#     return rmse.mean(), r2.mean()


# # =========================================================
# # RUN ABLATION
# # =========================================================

# all_runs = []

# for target_name, config in TARGETS.items():
#     target_col = config["target_col"]
#     feature_cols = config["feature_cols"]

#     base = df[feature_cols + [target_col]].dropna()
#     max_n = len(base)

#     # Every increment of 10, starting from 10, up to max_n
#     subset_sizes = list(range(10, max_n + 1, 10))

#     # Add max explicitly if it is not already a multiple of 10
#     if max_n not in subset_sizes:
#         subset_sizes.append(max_n)

#     print("\n==========================")
#     print("Target:", target_name)
#     print("Rows available:", max_n)
#     print("Subset sizes:", subset_sizes)
#     print("==========================")

#     for size in subset_sizes:
#         for rep in range(REPEATS):
#             subset = base.sample(
#                 n=size,
#                 random_state=rep
#             )

#             X = subset[feature_cols]
#             y = subset[target_col]

#             rmse, r2 = evaluate_subset(X, y)

#             print(
#                 f"{target_name:>15} | "
#                 f"n={size:>3} | "
#                 f"rep={rep} | "
#                 f"RMSE={rmse:.6g} | "
#                 f"R2={r2:.4f}"
#             )

#             all_runs.append({
#                 "target": target_name,
#                 "subset_size": str(size),
#                 "subset_numeric": size,
#                 "repeat": rep,
#                 "cv_rmse": rmse,
#                 "cv_r2": r2
#             })

# runs_df = pd.DataFrame(all_runs)


# # =========================================================
# # AGGREGATE RESULTS
# # =========================================================

# summary_df = (
#     runs_df
#     .groupby(["target", "subset_size", "subset_numeric"])
#     .agg(
#         cv_rmse_mean=("cv_rmse", "mean"),
#         cv_rmse_std=("cv_rmse", "std"),
#         cv_r2_mean=("cv_r2", "mean"),
#         cv_r2_std=("cv_r2", "std")
#     )
#     .reset_index()
# )

# summary_df = summary_df.sort_values(["target", "subset_numeric"])


# # =========================================================
# # SAVE RESULTS
# # =========================================================

# runs_path = os.path.join(OUT_DIR, "training_size_ablation_all_runs.csv")
# summary_path = os.path.join(OUT_DIR, "training_size_ablation_summary.csv")

# runs_df.to_csv(runs_path, index=False)
# summary_df.to_csv(summary_path, index=False)

# print("\nSaved files:")
# print(runs_path)
# print(summary_path)

# print("\nSummary:")
# print(summary_df)


# # =========================================================
# # PLOT INDIVIDUAL TARGET RESULTS
# # =========================================================

# for target_name, config in TARGETS.items():
#     temp = summary_df[summary_df["target"] == target_name].copy()
#     temp = temp.sort_values("subset_numeric")

#     x = temp["subset_numeric"].values
#     r2_mean = temp["cv_r2_mean"].values
#     r2_std = temp["cv_r2_std"].fillna(0).values
#     rmse_mean = temp["cv_rmse_mean"].values
#     rmse_std = temp["cv_rmse_std"].fillna(0).values

#     # R^2 plot
#     plt.figure(figsize=(8, 5))
#     plt.errorbar(
#         x,
#         r2_mean,
#         yerr=r2_std,
#         marker="o",
#         linewidth=2,
#         capsize=4
#     )
#     plt.xlabel("Training Set Size")
#     plt.ylabel(config["y_label_r2"])
#     plt.title(f"Training Size Ablation: R² vs Training Set Size ({target_name})")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     r2_plot_path = os.path.join(OUT_DIR, f"{target_name}_r2_vs_training_size.png")
#     plt.savefig(r2_plot_path, dpi=300, bbox_inches="tight")
#     plt.show()
#     plt.close()

#     # RMSE plot
#     plt.figure(figsize=(8, 5))
#     plt.errorbar(
#         x,
#         rmse_mean,
#         yerr=rmse_std,
#         marker="o",
#         linewidth=2,
#         capsize=4
#     )
#     plt.xlabel("Training Set Size")
#     plt.ylabel(config["y_label_rmse"])
#     plt.title(f"Training Size Ablation: RMSE vs Training Set Size ({target_name})")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     rmse_plot_path = os.path.join(OUT_DIR, f"{target_name}_rmse_vs_training_size.png")
#     plt.savefig(rmse_plot_path, dpi=300, bbox_inches="tight")
#     plt.show()
#     plt.close()


# # =========================================================
# # PLOT COMBINED R^2 ACROSS ALL TARGETS
# # =========================================================

# plt.figure(figsize=(10, 6))
# for target_name in TARGETS.keys():
#     temp = summary_df[summary_df["target"] == target_name].copy()
#     temp = temp.sort_values("subset_numeric")

#     plt.errorbar(
#         temp["subset_numeric"].values,
#         temp["cv_r2_mean"].values,
#         yerr=temp["cv_r2_std"].fillna(0).values,
#         marker="o",
#         linewidth=2,
#         capsize=3,
#         label=target_name
#     )

# plt.xlabel("Training Set Size")
# plt.ylabel(r"Cross-Validated $R^2$")
# plt.title("Training Size Ablation: R² vs Training Set Size (All Targets)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()

# combined_r2_path = os.path.join(OUT_DIR, "all_targets_r2_vs_training_size.png")
# plt.savefig(combined_r2_path, dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()


# # =========================================================
# # PLOT COMBINED RMSE ACROSS ALL TARGETS
# # =========================================================

# plt.figure(figsize=(10, 6))
# for target_name in TARGETS.keys():
#     temp = summary_df[summary_df["target"] == target_name].copy()
#     temp = temp.sort_values("subset_numeric")

#     plt.errorbar(
#         temp["subset_numeric"].values,
#         temp["cv_rmse_mean"].values,
#         yerr=temp["cv_rmse_std"].fillna(0).values,
#         marker="o",
#         linewidth=2,
#         capsize=3,
#         label=target_name
#     )

# plt.xlabel("Training Set Size")
# plt.ylabel("Cross-Validated RMSE")
# plt.title("Training Size Ablation: RMSE vs Training Set Size (All Targets)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()

# combined_rmse_path = os.path.join(OUT_DIR, "all_targets_rmse_vs_training_size.png")
# plt.savefig(combined_rmse_path, dpi=300, bbox_inches="tight")
# plt.show()
# plt.close()

# print("\nSaved plot files to:")
# print(OUT_DIR)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PATHS
# =========================================================

PATH = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\Updated Training Data Sheet.csv"
OUT_DIR = r"C:\Users\adity\OneDrive\Desktop\Chemeng 277\Chemeng 277 Project\ablation_v2\training_size_ablation"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(PATH)
df.columns = df.columns.str.strip()

print("Columns in dataset:")
print(df.columns.tolist())


# =========================================================
# COLUMN DEFINITIONS
# =========================================================

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

for col in ALL_NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert CTE from micro-scale notation to 1/K
df[COL_CTE] = df[COL_CTE] * 1e-6


# =========================================================
# SETTINGS
# =========================================================

TARGETS = {
    "logk": {
        "target_col": COL_LOGK,
        "feature_cols": [COL_DENSITY, COL_CP, COL_E, COL_CTE] + CLASS_COLS,
        "y_label_rmse": "Cross-Validated RMSE (log k)",
        "y_label_r2": r"Cross-Validated $R^2$ (log k)"
    },
    "elastic_modulus": {
        "target_col": COL_E,
        "feature_cols": [COL_DENSITY, COL_CP, COL_CTE, COL_LOGK] + CLASS_COLS,
        "y_label_rmse": "Cross-Validated RMSE (Elastic Modulus, GPa)",
        "y_label_r2": r"Cross-Validated $R^2$ (Elastic Modulus)"
    },
    "density": {
        "target_col": COL_DENSITY,
        "feature_cols": [COL_CP, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS,
        "y_label_rmse": "Cross-Validated RMSE (Density, g/cm^3)",
        "y_label_r2": r"Cross-Validated $R^2$ (Density)"
    },
    "heat_capacity": {
        "target_col": COL_CP,
        "feature_cols": [COL_DENSITY, COL_E, COL_CTE, COL_LOGK] + CLASS_COLS,
        "y_label_rmse": "Cross-Validated RMSE (Cp, J/kg·K)",
        "y_label_r2": r"Cross-Validated $R^2$ (Cp)"
    },
    "CTE": {
        "target_col": COL_CTE,
        "feature_cols": [COL_DENSITY, COL_CP, COL_E, COL_LOGK] + CLASS_COLS,
        "y_label_rmse": r"Cross-Validated RMSE (CTE, 1/K)",
        "y_label_r2": r"Cross-Validated $R^2$ (CTE)"
    }
}

REPEATS = 3

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

kf = KFold(n_splits=4, shuffle=True, random_state=42)


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_subset(X, y):
    rf = RandomForestRegressor(**RF_PARAMS)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", rf)
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=("r2", "neg_mean_squared_error"),
        n_jobs=-1
    )

    rmse = np.sqrt(-scores["test_neg_mean_squared_error"])
    r2 = scores["test_r2"]

    return rmse.mean(), r2.mean()


# =========================================================
# RUN ABLATION
# =========================================================

all_runs = []

for target_name, config in TARGETS.items():
    target_col = config["target_col"]
    feature_cols = config["feature_cols"]

    base = df[feature_cols + [target_col]].dropna()
    max_n = len(base)

    subset_sizes = list(range(10, max_n + 1, 10))
    if max_n not in subset_sizes:
        subset_sizes.append(max_n)

    print("\n==========================")
    print("Target:", target_name)
    print("Rows available:", max_n)
    print("Subset sizes:", subset_sizes)
    print("==========================")

    for size in subset_sizes:
        for rep in range(REPEATS):
            subset = base.sample(
                n=size,
                random_state=rep
            )

            X = subset[feature_cols]
            y = subset[target_col]

            rmse, r2 = evaluate_subset(X, y)

            print(
                f"{target_name:>15} | "
                f"n={size:>3} | "
                f"rep={rep} | "
                f"RMSE={rmse:.6g} | "
                f"R2={r2:.4f}"
            )

            all_runs.append({
                "target": target_name,
                "subset_size": str(size),
                "subset_numeric": size,
                "repeat": rep,
                "cv_rmse": rmse,
                "cv_r2": r2
            })

runs_df = pd.DataFrame(all_runs)


# =========================================================
# AGGREGATE RESULTS
# =========================================================

summary_df = (
    runs_df
    .groupby(["target", "subset_size", "subset_numeric"])
    .agg(
        cv_rmse_mean=("cv_rmse", "mean"),
        cv_rmse_std=("cv_rmse", "std"),
        cv_r2_mean=("cv_r2", "mean"),
        cv_r2_std=("cv_r2", "std")
    )
    .reset_index()
)

summary_df = summary_df.sort_values(["target", "subset_numeric"])


# =========================================================
# SAVE RESULTS
# =========================================================

runs_path = os.path.join(OUT_DIR, "training_size_ablation_all_runs.csv")
summary_path = os.path.join(OUT_DIR, "training_size_ablation_summary.csv")

runs_df.to_csv(runs_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\nSaved files:")
print(runs_path)
print(summary_path)

print("\nSummary:")
print(summary_df)


# =========================================================
# PLOT INDIVIDUAL TARGET RESULTS (NO ERROR BARS)
# =========================================================

for target_name, config in TARGETS.items():
    temp = summary_df[summary_df["target"] == target_name].copy()
    temp = temp.sort_values("subset_numeric")

    x = temp["subset_numeric"].values
    r2_mean = temp["cv_r2_mean"].values
    rmse_mean = temp["cv_rmse_mean"].values

    # R2 plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        r2_mean,
        marker="o",
        linewidth=2
    )
    plt.xlabel("Training Set Size")
    plt.ylabel(config["y_label_r2"])
    plt.title(f"Training Size Ablation: R² vs Training Set Size ({target_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    r2_plot_path = os.path.join(OUT_DIR, f"{target_name}_r2_vs_training_size.png")
    plt.savefig(r2_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # RMSE plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        rmse_mean,
        marker="o",
        linewidth=2
    )
    plt.xlabel("Training Set Size")
    plt.ylabel(config["y_label_rmse"])
    plt.title(f"Training Size Ablation: RMSE vs Training Set Size ({target_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    rmse_plot_path = os.path.join(OUT_DIR, f"{target_name}_rmse_vs_training_size.png")
    plt.savefig(rmse_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================================================
# PLOT COMBINED R2 ACROSS ALL TARGETS (NO ERROR BARS)
# =========================================================

plt.figure(figsize=(10, 6))
for target_name in TARGETS.keys():
    temp = summary_df[summary_df["target"] == target_name].copy()
    temp = temp.sort_values("subset_numeric")

    plt.plot(
        temp["subset_numeric"].values,
        temp["cv_r2_mean"].values,
        marker="o",
        linewidth=2,
        label=target_name
    )

plt.xlabel("Training Set Size")
plt.ylabel(r"Cross-Validated $R^2$")
plt.title("Training Size Ablation: R² vs Training Set Size (All Targets)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

combined_r2_path = os.path.join(OUT_DIR, "all_targets_r2_vs_training_size.png")
plt.savefig(combined_r2_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# =========================================================
# PLOT COMBINED RMSE ACROSS ALL TARGETS (NO ERROR BARS)
# =========================================================

plt.figure(figsize=(10, 6))
for target_name in TARGETS.keys():
    temp = summary_df[summary_df["target"] == target_name].copy()
    temp = temp.sort_values("subset_numeric")

    plt.plot(
        temp["subset_numeric"].values,
        temp["cv_rmse_mean"].values,
        marker="o",
        linewidth=2,
        label=target_name
    )

plt.xlabel("Training Set Size")
plt.ylabel("Cross-Validated RMSE")
plt.title("Training Size Ablation: RMSE vs Training Set Size (All Targets)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

combined_rmse_path = os.path.join(OUT_DIR, "all_targets_rmse_vs_training_size.png")
plt.savefig(combined_rmse_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nSaved plot files to:")
print(OUT_DIR)