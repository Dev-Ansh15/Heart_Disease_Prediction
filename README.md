#  Heart Disease Prediction Pipeline

Pipeline for early detection of heart disease. It integrates data cleaning, OCR-based medical value extraction, feature engineering, and stratified dataset splitting to prepare data for predictive modeling.

---

## 🔹 Features

- Cleaned and standardized datasets  
- OCR extraction from patient images (BP, cholesterol, etc.)  
- Engineered features (BMI, age groups, risk score)  
- Stratified train/validation/test splits (70/15/15)  
- Ready-to-use datasets for ML modeling  

---

## 📂 Project Structure

HEART_DISEASE_PROJECT/
├── data/ # raw, interim, processed, final splits
├── notebooks/ # Exploratory Analysis, Cleaning, Feature Engineering, Data Splitting
├── scripts/ # Cleaning, Feature Engineering, OCR, Splitting, Pipeline
├── reports/ # Quality, Feature, OCR, and Dataset reports
├── visuals/ # Histograms, Boxplots, Correlation, Class Distribution
├── docs/ # Methodology & documentation
├── tests/ # Unit tests
├── requirements.txt
├── README.md
└── LICENSE


---

## ⚙️ Tech Stack

- **Python 3.8+**  
- **Pandas, NumPy** – Data handling  
- **Matplotlib, Seaborn** – Visualization  
- **Scikit-learn** – Machine learning  
- **Tesseract OCR** – Image-based text extraction  

---

## 🚀 Getting Started

1. Place raw CSV and images in `data/raw/`  
2. Run notebooks/scripts for:
   - EDA → `02_dataset_exploration.ipynb`  
   - Cleaning → `03_data_cleaning.ipynb`  
   - Feature Engineering → `05_feature_engineering.ipynb`  
   - Data Splitting → `06_data_splitting_integrating.ipynb`  
   - OCR → `04_OCR_pipeline.ipynb`  
3. Outputs saved in `data/final/`  

---

## 🤝 Contributing

Fork the repo  → Create branch  → Commit changes  → Open PR   

---

