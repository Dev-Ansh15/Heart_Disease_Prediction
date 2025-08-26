import pandas as pd
import numpy as np
import pytest
from scripts.data_cleaning import clean_dataset, detect_outliers

@pytest.fixture
def sample_raw_data():
    """Create a dummy dataset with missing values and outliers."""
    data = {
        "age": [45, 60, None, 30, 200],  # 200 is an outlier
        "sex": ["M", "F", "M", None, "F"],
        "cholesterol": [250, None, 180, 190, 500],  # 500 is an outlier
        "blood_pressure": [140, 160, None, 120, 300],  # 300 is an outlier
        "target": [1, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)

def test_no_missing_values_after_cleaning(sample_raw_data):
    cleaned = clean_dataset(sample_raw_data)
    assert cleaned.isnull().sum().sum() == 0, "There are still missing values after cleaning."

def test_outliers_are_capped(sample_raw_data):
    cleaned = clean_dataset(sample_raw_data)

    assert cleaned["cholesterol"].max() <= 500  # upper capped value
    assert cleaned["cholesterol"].min() >= 180  # lower capped value

    assert cleaned["blood_pressure"].max() <= 300
    assert cleaned["blood_pressure"].min() >= 120

def test_categorical_imputation(sample_raw_data):
    cleaned = clean_dataset(sample_raw_data)
    assert cleaned["sex"].isnull().sum() == 0, "Categorical missing values not filled."

def test_numeric_columns_type(sample_raw_data):
    cleaned = clean_dataset(sample_raw_data)
    num_cols = ["age", "cholesterol", "blood_pressure"]
    for col in num_cols:
        assert np.issubdtype(cleaned[col].dtype, np.number), f"{col} is not numeric."

def test_target_column_binary(sample_raw_data):
    cleaned = clean_dataset(sample_raw_data)
    assert set(cleaned["target"].unique()).issubset({0, 1}), "Target column contains invalid values."

def test_detect_outliers_function(sample_raw_data):
    outliers = detect_outliers(sample_raw_data, "cholesterol")
    assert not outliers.empty, "detect_outliers failed to find expected outliers."
    assert 500 in outliers["cholesterol"].values, "Known outlier not detected."
