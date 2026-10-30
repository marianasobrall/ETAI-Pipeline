import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)


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

    return df.reset_index(drop=True)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """Exact duplicate rows first, then repeated ids (first one kept)."""
    df = df.drop_duplicates()
    if id_column in df.columns:
        df = df.drop_duplicates(subset=id_column, keep="first")
    return df.reset_index(drop=True)


def add_missingness_indicators(df:pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    df = df.copy()
    for col in mnar_indicator_sources:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """X (no target, no sensitive attribute, no drop_columns), y, and the extras the
    fairness report needs (sensitive attribute + COMPAS's own score_text)."""
    df = add_missingness_indicators(df, mnar_indicator_sources)
    target, sensitive_attr = data_config["target"], data_config["sensitive_attr"]

    y = df[target].astype(int)
    extras = df[[c for c in [sensitive_attr, "score_text"] if c in df.columns]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in data_config.get("drop_columns", []) if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)
    return X, y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """Stratified split into the development set and the locked test set."""
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)


class CountEncoder(BaseEstimator, TransformerMixin):
    """Replaces each category by how often it appears in the training data (0 if unseen)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.counts_ = [X[col].value_counts() for col in X.columns]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([
            X[col].map(counts).fillna(0).to_numpy(dtype=float)
            for col, counts in zip(X.columns, self.counts_)
        ])


_SCALERS = {
    "none": None,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _build_encoder(name: str, random_state=None):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", drop="first")
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if name == "count":
        return CountEncoder()
    if name == "target":
        # cross-fitted on internal folds, so a row's own label never leaks into its encoding
        return TargetEncoder(target_type="binary", random_state=random_state)
    raise ValueError(f"Unknown encoder: {name}. Options: onehot, ordinal, count, target")


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    numeric_strategy = preprocessing_config["imputation"]["numeric_strategy"]
    categorical_strategy = preprocessing_config["imputation"]["categorical_strategy"]
    scaler_name = preprocessing_config.get("scaler", "standard")
    if scaler_name not in _SCALERS:
        raise ValueError(f"Unknown scaler: {scaler_name}. Options: {list(_SCALERS)}")
    scaler = _SCALERS[scaler_name]

    numeric_steps = [("impute", SimpleImputer(strategy=numeric_strategy))]
    if scaler is not None:
        numeric_steps.append(("scale", scaler()))
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=categorical_strategy)),
        ("encode", _build_encoder(preprocessing_config.get("encoder", "onehot"),
                                  preprocessing_config.get("random_state"))),
    ])

    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Week 6 feature engineering experiment. Not used by the pipeline: none beat the fold std."""
    out = df.copy()
    adult_years = (out["age"] - 17).clip(lower=1)
    known_priors = out["priors_count"].notna()

    out["priors_per_adult_year"] = out["priors_count"] / adult_years
    out["log_priors"] = np.log1p(out["priors_count"])
    out["has_priors"] = (out["priors_count"] > 0).astype(float).where(known_priors)
    out["has_juv_felony"] = (out["juv_fel_count"] > 0).astype(float).where(out["juv_fel_count"].notna())
    out["young_felony"] = ((out["age"] < 25) & (out["c_charge_degree"] == "F")).astype(float) \
        .where(out["age"].notna() & out["c_charge_degree"].notna())

    # share of juvenile offenses that were felonies (0 if no juvenile record): severity, not volume
    juv_total = out[["juv_fel_count", "juv_misd_count", "juv_other_count"]].sum(axis=1, min_count=3)
    out["juv_felony_share"] = (out["juv_fel_count"] / juv_total).where(juv_total != 0, 0.0)

    # many priors while young: an interaction a linear model or a stump can't build on its own
    young = out["age"] < 25
    out["priors_x_young"] = (out["priors_count"] * young.astype(float)) \
        .where(out["age"].notna() & out["priors_count"].notna())
    return out
