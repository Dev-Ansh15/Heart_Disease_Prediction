import sys
sys.path.append("../scripts")

from data_cleaning import clean_dataset
from feature_engineering import create_derived_features, encode_all_categoricals, normalize_features, standardize_features

# --- Integrate full pipeline ---
def integrate_pipeline(df):
    # Step 1: Clean dataset
    df_cleaned = clean_dataset(df)

    # Step 2: Derived features
    df_features = create_derived_features(df_cleaned)

    # Step 3: Encode categorical columns automatically
    df_encoded = encode_all_categoricals(df_features)

    # Step 4: Normalize & standardize selected columns
    norm_cols = ['chol', 'trestbps'] if all(x in df_encoded.columns for x in ['chol', 'trestbps']) else []
    std_cols = ['thalch', 'oldpeak'] if all(x in df_encoded.columns for x in ['thalch', 'oldpeak']) else []

    if norm_cols:
        df_encoded = normalize_features(df_encoded, norm_cols)
    if std_cols:
        df_encoded = standardize_features(df_encoded, std_cols)

    return df_encoded
