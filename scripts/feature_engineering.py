# scripts/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2

# --- Normalization ---
def normalize_features(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

# --- Standardization ---
def standardize_features(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

# --- Encode all categoricals automatically  ---
def encode_all_categoricals(df):
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype in ['object', 'category', 'bool']:  # Simplified condition
            try:
                ohe = OneHotEncoder(sparse_output=False, drop="first")
            except TypeError:
                ohe = OneHotEncoder(sparse=False, drop="first")
            encoded = ohe.fit_transform(df_encoded[[col]])
            new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]  # Fixed variable name
            encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
    return df_encoded

# --- Derived Features ---
def create_derived_features(df):
    df_new = df.copy()
    if {'weight', 'height'}.issubset(df_new.columns):
        df_new['BMI'] = df_new['weight'] / ((df_new['height'] / 100) ** 2)
    if 'age' in df_new.columns:
        df_new['age_group'] = pd.cut(
            df_new['age'],
            bins=[0, 30, 50, 70, 100],
            labels=['Young', 'Middle-aged', 'Senior', 'Elderly']
        )
    if {'chol', 'trestbps'}.issubset(df_new.columns):
        df_new['risk_score'] = df_new['chol'] * 0.6 + df_new['trestbps'] * 0.4
    return df_new

# --- Feature Selection (COMPREHENSIVE FIX) ---
def feature_selection_analysis(df, target_col):
    df_copy = df.copy()

    # Separate features/target
    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]

    # Convert ALL columns to numeric, handling different data types
    X_numeric = pd.DataFrame(index=X.index)
    for col in X.columns:
        if X[col].dtype == 'bool':
            X_numeric[col] = X[col].astype(int).astype(float)
        elif X[col].dtype == 'object':
            # For object columns, use LabelEncoder
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X_numeric[col] = le.fit_transform(X[col].astype(str)).astype(float)
        elif X[col].dtype == 'category':
            # For category columns, use codes
            X_numeric[col] = X[col].cat.codes.astype(float)
        else:
            # For numeric columns
            X_numeric[col] = X[col].astype(float)

    # --- RandomForest for importance ---
    model = RandomForestClassifier(random_state=42)
    model.fit(X_numeric, y)
    importance_scores = pd.DataFrame({
        "feature": X_numeric.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # --- SelectKBest (chi2) ---
    # Ensure all values are non-negative for chi2
    X_pos = X_numeric.copy()
    for col in X_pos.columns:
        min_val = X_pos[col].min()
        if min_val < 0:
            X_pos[col] = X_pos[col] - min_val + 1e-6
        else:
            X_pos[col] = X_pos[col] + 1e-6
    
    selector = SelectKBest(chi2, k=10)
    X_new = selector.fit_transform(X_pos, y)
    selected_features = X_numeric.columns[selector.get_support()]

    return selected_features, importance_scores




