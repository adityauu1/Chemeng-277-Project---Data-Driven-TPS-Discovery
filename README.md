# Chemeng-277-Project---Data-Driven-TPS-Discovery

Machine learning framework for predicting material properties and screening Thermal Protection System (TPS) candidate materials using linear regression and random forest models. The project includes ablation studies, property prediction, and ranking of high-temperature materials for aerospace applications.

## Project Overview

This project develops a data-driven workflow to:

- Predict missing material properties from known properties
- Evaluate model performance using ablation studies
- Validate predictions against known TPS materials
- Rank materials for thermal protection system (TPS) applications


Material data was compiled from:

- Materials Project
- TPSX Materials Property Database

Properties used in training:

- Density
- Thermal conductivity
- Heat capacity at constant pressure (Cp)
- Elastic modulus
- Coefficient of thermal expansion (CTE)
- Material class (oxide, carbide, boride, nitride, silicate) via one-hot encoding

Models used:

- Linear Regression (L1 / L2 regularization)
- Random Forest Regressor
- Log-transform regression for thermal conductivity
- Random forest ablation studies
- Training size ablation
- Property ranking model

## Repository Structure

README.md

Training_Data_Sheet_Thermal_Conductivity.csv
tpsx_dataset_full.xlsx
tpsx_materials_missing_values.csv

linear_regression.py
linear_regression_log_transform.py

random_forest_thermal_conductivity.py
random_forest_density.py
random_forest_elastic_modulus.py
random_forest_heat_capacity.py
random_forest_CTE.py

rf_ablation_logk.py
rf_ablation_density.py
rf_ablation_elastic.py
rf_ablation_heat_capacity.py
rf_ablation_CTE.py
rf_ablation_utils.py

random_forest_training_size_ablation_repeat3.py

final_property_predictor.py
material_ranker.py

## Workflow

### 1. Linear Regression Baseline

Files:
- linear_regression.py
- linear_regression_log_transform.py

Purpose:
- Establish baseline model performance
- Test L1 / L2 regularization
- Evaluate the effect of log(k) transformation
- Compare RMSE and R²

### 2. Random Forest Property Prediction

Files:
- random_forest_thermal_conductivity.py
- random_forest_density.py
- random_forest_elastic_modulus.py
- random_forest_heat_capacity.py
- random_forest_CTE.py

Purpose:
- Predict each property separately
- Apply median imputation for missing data
- Standardize features
- Use K-fold cross validation
- Evaluate RMSE and R²

### 3. Random Forest Ablation Studies

Files:
- rf_ablation_logk.py
- rf_ablation_density.py
- rf_ablation_elastic.py
- rf_ablation_heat_capacity.py
- rf_ablation_CTE.py
- rf_ablation_utils.py

Purpose:
- Test different random forest model complexities
- Vary number of trees
- Vary maximum depth
- Vary minimum samples to split
- Vary minimum samples per leaf
- Vary number of features considered per split
- Examine effect of removing individual features
- Compare continuous-only vs discrete-only features

Goal:
- Identify the best-performing model configuration

### 4. Training Size Ablation

File:
- random_forest_training_size_ablation_repeat3.py

Purpose:
- Evaluate the effect of dataset size on model performance
- Determine whether the model is data-limited

### 5. Final Property Prediction

File:
- final_property_predictor.py

Purpose:
- Predict missing properties for TPS candidates
- Use the best model identified from ablation studies

Input:
- tpsx_materials_missing_values.csv

Output:
- Dataset with predicted missing properties

### 6. Material Ranking

File:
- material_ranker.py

Purpose:
- Rank TPS materials using a weighted property score

Ranking is based on:
- Density
- Thermal conductivity
- Cp
- Elastic modulus
- CTE

Standardization:
- Min-max normalization is used before ranking

Weights:
- w_density = 0.25
- w_k = 0.35
- w_Cp = 0.10
- w_E = 0.10
- w_CTE = 0.20

Goal:
- Identify the most promising TPS candidates

## Method Summary

The modelling framework predicts unknown material properties from known material properties and then uses those predicted values to rank candidate TPS materials. A total of 319 materials representative of TPS-relevant classes, including oxides, ceramics, borides, nitrides, and silicates, were used.

Linear regression was first used as a baseline model. Missing values were handled using median imputation, and features were standardized. Ridge and Lasso regression were tested across a range of regularization strengths. Because thermal conductivity spans multiple orders of magnitude, a logarithmic transformation of thermal conductivity was also evaluated to improve performance.

Random forest models were then used because they can capture nonlinear relationships and feature interactions more effectively than linear regression. Median imputation, standardization, and K-fold cross validation were used. Ablation studies were performed to determine the effect of model complexity, feature removal, feature type, and training dataset size.

The best-performing random forest configuration was then used to predict missing properties in TPS candidate datasets. These predicted properties were finally used in a weighted ranking framework grounded in physical TPS requirements, emphasizing low density and low thermal conductivity.

## Requirements

Python 3.9+

Install dependencies with:

pip install numpy pandas scikit-learn matplotlib openpyxl

## Example Usage

Run linear regression:
python linear_regression.py

Run log-transform linear regression:
python linear_regression_log_transform.py

Run random forest thermal conductivity model:
python random_forest_thermal_conductivity.py

Run ablation study:
python rf_ablation_logk.py

Run training size ablation:
python random_forest_training_size_ablation_repeat3.py

Predict missing properties:
python final_property_predictor.py

Rank materials:
python material_ranker.py

## Expected Outputs

Depending on the script, outputs may include:
- RMSE and R² metrics
- Feature importance values
- Ablation study comparisons
- Predicted missing material properties
- Ranked TPS candidate materials
- Figures showing training size effects

## Notes

- Thermal conductivity is especially challenging to predict due to its wide spread across materials.
- Log-transforming thermal conductivity improves model performance.
- Random forest models outperform linear regression for this dataset.
- Increasing dataset size improves performance, indicating the framework is somewhat data-limited.
- The workflow is intended as a screening tool, not as a replacement for experimental validation.

## Author

Aditya Udgaonkar
Keshav Dhir
ChemE 277 Project
Data-Driven TPS Discovery
Stanford University
