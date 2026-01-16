import pandas as pd
import numpy as np
from pathlib import Path

def get_indicators(df):
    """
    Returns list of indicators (columns after metadata) from the long-format dataframe.
    """
    metadata_cols = [
        "geo", "year", "geo_label", "nuts_level", "country_code",
        "country_name", "is_el_regional_unit",
    ]
    return [c for c in df.columns if c not in metadata_cols]

def pivot_to_wide(df):
    """
    Transforms long-form data to wide format in memory.
    """
    indicators = get_indicators(df)
    wide_df = df.pivot(index="geo", columns="year", values=indicators)
    wide_df.columns = [f"{ind}_{yr}" for ind, yr in wide_df.columns]
    return wide_df.reset_index()

def get_base_indicators_from_wide(wide_df):
    """
    Infers base indicator names from a wide-format dataframe.
    """
    indicators = set()
    for col in wide_df.columns:
        if col == "geo":
            continue
        if "_" in col:
            indicators.add(col.rsplit("_", 1)[0])
    return sorted(indicators)

def prepare_indicator_dataset(wide_df, target_indicator, target_year=None):
    """
    Prepares train/eval and forecast splits for a specific indicator in memory.
    """
    if target_year is None:
        years = []
        prefix = f"{target_indicator}_"
        for c in wide_df.columns:
            if c.startswith(prefix):
                 remainder = c[len(prefix):]
                 if remainder.isdigit():
                     years.append(int(remainder))
        if not years:
            return None, None, None, None
        target_year = max(years)

    target_col = f"{target_indicator}_{target_year}"
    if target_col not in wide_df.columns:
        return None, None, None, target_year

    train_eval_df = wide_df[wide_df[target_col].notna()].copy()
    forecast_df = wide_df[wide_df[target_col].isna()].copy()

    feature_cols = [c for c in wide_df.columns if not c.endswith(str(target_year)) and c != "geo"]
    # Filter only target indicator history
    single_feature_cols = []
    prefix = target_indicator + "_"
    for c in feature_cols:
        if c.startswith(prefix):
            remainder = c[len(prefix):]
            if remainder.isdigit():
                single_feature_cols.append(c)
    
    single_feature_cols.sort(key=lambda x: int(x.split('_')[-1]))

    return train_eval_df, forecast_df, {
        "single": single_feature_cols,
        "multi": feature_cols,
        "target": target_col,
    }, target_year

def impute_missing_values(X):
    """
    Fills NaNs and 0s with row-wise interpolation/means.
    """
    X = X.copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace(0, np.nan)

    col_prefixes = {}
    for col in X.columns:
        if "_" in col:
            prefix = "_".join(col.split("_")[:-1])
            col_prefixes.setdefault(prefix, []).append(col)

    for prefix, cols in col_prefixes.items():
        try:
            sorted_cols = sorted(cols, key=lambda c: int(c.split('_')[-1]))
        except ValueError:
            sorted_cols = cols
            
        subset = X[sorted_cols]
        subset_interp = subset.interpolate(method='linear', axis=1, limit_direction='both')
        row_means = subset_interp.mean(axis=1)
        subset_final = subset_interp.apply(lambda s: s.fillna(row_means))
        X[sorted_cols] = subset_final

    return X.fillna(0)

def preprocess_df_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full in-memory preprocessing pipeline.
    """
    if df.empty:
        return pd.DataFrame()
        
    df["year"] = df["year"].astype(int)
    wide_df = pivot_to_wide(df)

    if 'geo' in wide_df.columns:
        geo_col = wide_df['geo']
        numeric_df = wide_df.drop(columns=['geo'])
    else:
        geo_col = None
        numeric_df = wide_df

    imputed_df = impute_missing_values(numeric_df)
    
    if geo_col is not None:
        imputed_df.insert(0, 'geo', geo_col)
    
    return imputed_df