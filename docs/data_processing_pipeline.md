# Heart Disease Prediction – Data Processing Pipeline

**Project:** Heart Disease Prediction  
**Author:** Intern 6  

---

## Objective

To clean, process, enrich, and split the Heart Disease dataset for machine learning modeling while maintaining data integrity, balanced target distribution, and reproducibility.

---

## 1. Data Acquisition

- **Raw data sources:** `heart_disease_uci.csv` and supplementary files in `other_sources/`.
- **Image data:** Patient images (`patient2.png`) for OCR extraction of medical values (e.g., BP, cholesterol).

---

## 2. Data Cleaning

- **Missing values handled:**
  - Numerical columns → KNN Imputer (`k=5`)
  - Categorical columns → Mode imputation
- **Boolean and string standardization:**
  - `"TRUE"` → 1
  - `"FALSE"` → 0
- **Outlier detection:**
  - IQR method
  - Outlier capping to nearest bound
- **Cleaned dataset saved to:**
  - `data/interim/heart_disease_cleaned_partial.csv`
  - `data/processed/heart_disease_cleaned.csv`

---

## 3. OCR Integration

- **Image preprocessing:** Grayscale, noise reduction, thresholding
- **Text extraction:** Tesseract OCR
- **Medical value parsing:**
  - Blood Pressure
  - Cholesterol
- **Values appended** to dataset for feature enrichment

---

## 4. Feature Engineering

- **Derived features:**
  - `BMI = weight / (height/100)^2`
  - `age_group` categorized as Young, Middle-aged, Senior, Elderly
  - `risk_score = 0.6 * cholesterol + 0.4 * resting blood pressure`
- **Encoding:**
  - Categorical variables → One-hot encoding (drop first category)
- **Scaling:**
  - Normalize (`MinMaxScaler`) → `chol`, `trestbps`
  - Standardize (`StandardScaler`) → `thalach`, `oldpeak`
- **Feature selection (optional analysis):**
  - RandomForest feature importance
  - SelectKBest (Chi2)

---

## 5. Data Splitting

- Stratified train-validation-test split (70%-15%-15%) to maintain target balance
- **Split validation:**
  - Distribution check for class imbalance
  - NaN checks for all splits
- **Datasets saved to:**
  - `data/final/train_dataset.csv`
  - `data/final/validation_dataset.csv`
  - `data/final/test_dataset.csv`

---

## 6. Summary Statistics

- Dataset summary includes:
  - Column types
  - Missing values
  - Unique values
  - Mean, standard deviation, min, max for numerical columns
- **Visualizations:**
  - Class distributions
  - Correlation heatmap
  - Boxplots for outlier analysis
  - Histograms for feature distribution
