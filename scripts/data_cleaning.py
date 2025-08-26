# scripts/data_cleaning.py
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def detect_outliers(df, column):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans dataset: missing values, outliers, invalid entries."""
    
    df = df.copy()

    # Replace categorical TRUE/FALSE with 1/0
    df.replace({"TRUE": 1, "FALSE": 0}, inplace=True)

    # Handle missing values - KNN Imputation for numerical
    imputer = KNNImputer(n_neighbors=5)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Handle outliers - capping with IQR limits
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower,
                           np.where(df[col] > upper, upper, df[col]))

    return df
