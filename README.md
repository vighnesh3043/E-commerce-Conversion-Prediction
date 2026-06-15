# E-Commerce Conversion Prediction — Summer Analytics 2026 (Mini-Hackathon Week 2)

Predicts whether a user converts (`Converted` = 1/0) on an e-commerce platform using 12 anonymized user-level features covering demographics, browsing behavior, traffic source, and campaign data.

## Approach

- **EDA**: Identified a moderate class imbalance (~31% converted / 69% not converted), missing values in `Age`, `Income`, and `Time_On_Site` (10–18% of rows), and conversion-rate variation across `Device_Type` and `Traffic_Source`.
- **Preprocessing**: Median imputation for numeric features; most-frequent imputation and one-hot encoding for categorical features (`Device_Type`, `Traffic_Source`).
- **Feature engineering**: Added three derived ratios to capture engagement intensity and purchasing power:
  - `Pages_per_min` = Pages_Viewed / (Time_On_Site + 1)
  - `Products_per_page` = Products_Viewed / (Pages_Viewed + 1)
  - `Income_per_age` = Income / (Age + 1)
- **Modeling**: Compared Logistic Regression, Random Forest, and LightGBM with class-balanced weighting on an 80/20 stratified split. Random Forest (300 trees, max_depth=8, `class_weight='balanced'`) gave the best validation F1 (~0.567) and was selected as the final model.
- **Final prediction**: Retrained Random Forest on combined train + public test data (13,000 records) and generated predictions for the private test set.

## Repository Contents

- `notebook.ipynb` — full reproducible pipeline (EDA → preprocessing → feature engineering → modeling → predictions)
- `submission.csv` — final predictions (`User_ID`, `Converted`) for `private_test.csv`
- `report.pdf` — one-page methodology summary

## Results

| Model | Validation F1 |
|---|---|
| Logistic Regression | ~0.55 |
| Random Forest (final) | ~0.567 |
| LightGBM | lower |

Public test F1 (train-only model): ~0.52
