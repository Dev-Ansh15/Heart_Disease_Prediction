# Feature Documentation

## 1. Numerical Transformations
- **Normalization:** MinMaxScaler applied to `chol`, `trestbps`.
- **Standardization:** StandardScaler applied to continuous variables.

## 2. Categorical Encoding
- **One-hot encoding:** `sex`, `cp`, `thal`, `slope`
- **Label encoding:** For categorical variables with many categories.

## 3. Derived Features
- **BMI:** Calculated from weight and height.
- **Age Groups:** Binned into Young (<30), Middle-aged (30–50), Senior (50–70), Elderly (>70).
- **Risk Score:** Linear combination of `chol` (60%) and `trestbps` (40%).

## 4. Feature Selection
- **RandomForest importance** computed.
- **Chi-Square test** applied to categorical predictors.

## 5. Output Dataset
- Final engineered dataset stored as `data/processed/heart_disease_features.csv`.
