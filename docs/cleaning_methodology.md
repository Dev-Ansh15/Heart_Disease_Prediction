# Data Cleaning Methodology (Day 3)

## Overview
This document describes the steps followed to clean the `heart_disease_uci.csv` dataset.

## Steps
1. **Missing Values**
   - Numerical columns imputed using **KNN Imputer (k=5)**
   - Categorical columns imputed using **Mode**

2. **Data Normalization**
   - TRUE/FALSE values converted to **1/0**

3. **Outlier Handling**
   - IQR method used to detect outliers
   - Outliers capped to lower/upper IQR limits

4. **Invalid/Consistent Data**
   - Corrected inconsistent labels in categorical features
   - Ensured ranges are valid (e.g., age > 0)

## Validation
- Compared distributions before and after cleaning
- Verified missing values reduced to **zero**
- Ensured dataset shape consistent (no dropped rows unnecessarily)

## Deliverables
- Cleaned dataset in `data/processed/`
- Intermediate dataset in `data/interim/`
- Visuals in `visuals/`
- Report: `reports/cleaning_report.pdf`
