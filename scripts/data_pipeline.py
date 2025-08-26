import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import sys
sys.path.append("../scripts")  # path to your scripts folder

from feature_engineering import create_derived_features, encode_all_categoricals, normalize_features, standardize_features


# --- Integrate full pipeline ---
def integrate_pipeline(df):
    df_new = create_derived_features(df)
    df_new = encode_all_categoricals(df_new, ['sex' 'cp', 'thal', 'slope'])
    df_new = normalize_features(df_new, ['chol', 'trestbps'])
    df_new = standardize_features(df_new, ['thalch', 'oldpeak'])
    return df_new

# --- Stratified Split ---
def stratified_split(df, target_col, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train / temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    
    # Second split: val / test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- Validate Data Splits ---
def validate_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    # Check balanced distribution
    print("Train target distribution:\n", y_train.value_counts(normalize=True))
    print("Validation target distribution:\n", y_val.value_counts(normalize=True))
    print("Test target distribution:\n", y_test.value_counts(normalize=True))
    
    # Check for NaNs
    for split, X in zip(["Train", "Validation", "Test"], [X_train, X_val, X_test]):
        if X.isnull().sum().sum() > 0:
            print(f"⚠️ NaNs found in {split} set")
        else:
            print(f"{split} set: No missing values")

# --- Dataset Summary ---
def dataset_summary(df):
    summary = pd.DataFrame({
        "Feature": df.columns,
        "Type": df.dtypes,
        "Missing": df.isnull().sum(),
        "Unique": df.nunique(),
        "Mean": df.mean(numeric_only=True),
        "Std": df.std(numeric_only=True),
        "Min": df.min(numeric_only=True),
        "Max": df.max(numeric_only=True)
    })
    return summary
