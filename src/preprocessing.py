import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    df = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    #columns with placeholder tokens in them
    for col in diagnostics_config["numeric_text_columns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    for col in df.columns:
        mask = df[col].astype(str).str.strip().isin(placeholder_tokens)
        df.loc[mask, col] = np.nan

    #values that break domain rules
    for col, rule in diagnostics_config["validity_rules"].items():
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            lo, hi = rule.get("min", -np.inf), rule.get("max", np.inf)
            df.loc[values.notna() & ~values.between(lo, hi), col] = np.nan
 
    # one canonical label per category
    for col, mapping in diagnostics_config["canonical_categories"].items():
        if col in df.columns:
            not_null = df[col].notna()
            cleaned = df.loc[not_null, col].astype(str).str.strip().str.lower()
            df.loc[not_null, col] = cleaned.map(mapping).fillna(cleaned)

    df =df.drop_duplicates()
    id_col = diagnostics_config.get("id_column")
    if id_col in df.columns:
        df = df.drop_duplicates(subset=id_col,keep="first")

    return df.reset_index(drop=True)

def add_missingness_indicators(df:pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    df = df.copy()
    for col in mnar_indicator_sources:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df

def impute_missing(X_train, X_test, numeric_features, categorical_features, numeric_strategy, categorical_strategy):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in numeric_features:
        if col in X_train.columns:
            fill_value = X_train[col].median() if numeric_strategy == "median" else X_train[col].mean()
            X_train[col]= X_train[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)

    for col in categorical_features:
        if col in X_train.columns:
            fill_value = X_train[col].mode(dropna=True).iloc[0]
            X_train[col] = X_train[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)

    return X_train, X_test

def preprocess(df: pd.DataFrame, config:dict):
    data_config = config["data"]
    diag_config = config["diagnostics"]
    preprocessing_config = config["preprocessing"]

    df = clean_dataset(df, diag_config)
    df = add_missingness_indicators(df, preprocessing_config.get("mnar_indicator_sources", []))

    y = df[data_config["target"]]
    extras = df[[data_config["sensitive_attr"], "score_text"]].copy()

    columns_to_exclude = [data_config["target"], data_config["sensitive_attr"]] + [
        c for c in data_config.get("drop_columns", []) if c in df.columns
    ]

    X = df.drop(columns=columns_to_exclude)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, 
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    X_train, X_test = impute_missing(
        X_train, X_test,
        numeric_features=preprocessing_config.get("numeric_features"),
        categorical_features=preprocessing_config.get("categorical_features"),  
        numeric_strategy=preprocessing_config["imputation"]["numeric_strategy"],
        categorical_strategy=preprocessing_config["imputation"]["categorical_strategy"]
        )
        


    # one-hot encode all non-numeric columns, no further thought
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train.columns, fill_value=0)


    return X_train, X_test, y_train, y_test, extras_test
