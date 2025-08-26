# --- Encoding ---
def encode_categorical(df, cols):
    df_encoded = df.copy()
    for col in cols:
        if df_encoded[col].dtype == 'object' or df_encoded[col].nunique() < 10:
            # Handle scikit-learn version differences
            try:
                ohe = OneHotEncoder(sparse_output=False, drop="first")  # new versions
            except TypeError:
                ohe = OneHotEncoder(sparse=False, drop="first")  # old versions

            encoded = ohe.fit_transform(df_encoded[[col]])
            new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]  # skip first due to drop="first"
            encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), encoded_df], axis=1)
        else:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded