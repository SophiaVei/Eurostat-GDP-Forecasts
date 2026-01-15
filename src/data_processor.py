import pandas as pd
import numpy as np
from pathlib import Path



def load_raw_data(filepath):
    """
    Loads a NUTS 2 dataset (long format: one row per geo-year).
    Ensures the year column is integer.
    Works for economy, labour, tourism, etc.
    """
    df = pd.read_csv(filepath)
    df["year"] = df["year"].astype(int)
    return df


def get_indicators(df):
    """
    Returns list of economic indicators (columns after metadata) from the
    long-format dataframe.
    """
    metadata_cols = [
        "geo",
        "year",
        "geo_label",
        "nuts_level",
        "country_code",
        "country_name",
        "is_el_regional_unit",
    ]
    return [c for c in df.columns if c not in metadata_cols]


def pivot_to_wide(df):
    """
    Transforms long-form data to wide format:
    - One row per region (geo)
    - Columns are indicator_year (e.g. gdp_mio_eur_2008)
    """
    indicators = get_indicators(df)

    wide_df = df.pivot(index="geo", columns="year", values=indicators)

    # Flatten multi-index columns: ('gdp_mio_eur', 2008) -> 'gdp_mio_eur_2008'
    wide_df.columns = [f"{ind}_{yr}" for ind, yr in wide_df.columns]

    return wide_df.reset_index()


def load_wide_data(filepath):
    """
    Loads a precomputed wide-format dataframe from disk.
    """
    return pd.read_csv(filepath)


def get_base_indicators_from_wide(wide_df):
    """
    Infers base indicator names from a wide-format dataframe whose columns
    are of the form <indicator>_<year> plus the 'geo' column.
    """
    indicators = set()
    for col in wide_df.columns:
        if col == "geo":
            continue
        # Split on the last underscore to separate indicator from year
        if "_" in col:
            indicators.add(col.rsplit("_", 1)[0])
    return sorted(indicators)


def prepare_indicator_dataset(wide_df, target_indicator, target_year=None):
    """
    Prepares train/eval and forecast splits for a specific indicator.

    Args:
        wide_df: The wide-format DataFrame (one row per geo).
        target_indicator: Name of the indicator to predict (e.g., 'gdp_mio_eur').
        target_year: Target year to forecast/evaluate. If None, detects the last available year.
    """
    # Auto-detect target year if not provided
    if target_year is None:
        years = []
        prefix = f"{target_indicator}_"
        for c in wide_df.columns:
            if c.startswith(prefix):
                 # Extract year part
                 remainder = c[len(prefix):]
                 if remainder.isdigit():
                     years.append(int(remainder))
        
        if not years:
            return None, None, None, None  # Updated to return target_year too
        
        target_year = max(years)

    target_col = f"{target_indicator}_{target_year}"

    if target_col not in wide_df.columns:
        # If the target year doesn't exist for this indicator, we can't train/eval
        return None, None, None, target_year

    # Split into regions where target year is present vs missing
    train_eval_df = wide_df[wide_df[target_col].notna()].copy()
    forecast_df = wide_df[wide_df[target_col].isna()].copy()

    # Define features (X): all indicator columns for years < target_year
    feature_cols = [
        c for c in wide_df.columns if not c.endswith(str(target_year)) and c != "geo"
    ]
    # Filter out any future columns if present (improbable but good for safety)
    feature_cols = [c for c in feature_cols if not any(c.endswith(f"_{y}") for y in range(target_year + 1, target_year + 20))]

    # X_single: just target_indicator history (exact match, not sub-indicators)
    # Match columns like "gva_2008" but NOT "gva_sector_a_2008"
    # The pattern should be: {target_indicator}_{year} where year is numeric
    single_feature_cols = []
    prefix = target_indicator + "_"
    for c in feature_cols:
        if c.startswith(prefix):
            # Remove the prefix and check if what remains is just a year
            remainder = c[len(prefix):]
            if remainder.isdigit():  # If remainder is just digits, it's a year - this is the exact indicator
                single_feature_cols.append(c)
    
    # Sort history columns by year to ensure temporal order
    single_feature_cols.sort(key=lambda x: int(x.split('_')[-1]))

    return train_eval_df, forecast_df, {
        "single": single_feature_cols,
        "multi": feature_cols,
        "target": target_col,
    }, target_year


def impute_missing_values(X):
    """
    Fills NaNs and 0s with the mean of the available values for each indicator
    in that specific row. 0s are treated as missing values.

    This is shared across all models so that they receive identically
    preprocessed feature matrices.
    """
    X = X.copy()
    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace(0, np.nan)

    # Group columns by indicator prefix (everything except the trailing year)
    col_prefixes = {}
    for col in X.columns:
        if "_" in col:
            prefix = "_".join(col.split("_")[:-1])
            col_prefixes.setdefault(prefix, []).append(col)

    # Interpolate time series for each indicator group
    for prefix, cols in col_prefixes.items():
        # Sort cols by year to ensure correct interpolation order
        try:
            sorted_cols = sorted(cols, key=lambda c: int(c.split('_')[-1]))
        except ValueError:
            # Fallback if column names don't end in year
            sorted_cols = cols
            
        subset = X[sorted_cols]
        # Linear interpolation across columns (axis=1)
        # limit_direction='both' fills leading/trailing NaNs if possible
        subset_interp = subset.interpolate(method='linear', axis=1, limit_direction='both')
        
        # Determine if we still have NaNs (e.g. all empty rows)
        # Apply row-mean fallback for any stubbornly missing values
        row_means = subset_interp.mean(axis=1)
        subset_final = subset_interp.apply(lambda s: s.fillna(row_means))
        
        X[sorted_cols] = subset_final

    # Final fallback for cases where entire rows are NaN for an indicator
    return X.fillna(0)


def preprocess_to_wide_file(raw_filepath, wide_out_filepath):
    """
    End-to-end preprocessing step:
    - Load long-format CSV
    - Pivot to wide format
    - Save the resulting wide dataframe to disk

    All models should load this same wide file for comparability.
    """
    df = load_raw_data(raw_filepath)
    wide_df = pivot_to_wide(df)

    # Separate metadata (geo) from data to preserve it during imputation
    if 'geo' in wide_df.columns:
        geo_col = wide_df['geo']
        numeric_df = wide_df.drop(columns=['geo'])
    else:
        geo_col = None
        numeric_df = wide_df

    # Apply robust interpolation/imputation to numeric data
    # This ensures the stored wide file is clean and gap-free
    imputed_df = impute_missing_values(numeric_df)
    
    # Restore metadata
    if geo_col is not None:
        imputed_df.insert(0, 'geo', geo_col)
    
    wide_df = imputed_df
    
    Path(wide_out_filepath).parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(wide_out_filepath, index=False)
    return wide_df


if __name__ == "__main__":
    # Preprocess ALL datasets when running this script directly
    import os
    
    # Resolve paths relative to this script
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_DIR = SCRIPT_DIR.parent
    DATA_DIR = REPO_DIR / "data"

    datasets = [
        ("economy_nuts2_all_columns.csv", "economy_nuts2_wide.csv"),
        ("labour_nuts2_all_columns.csv", "labour_nuts2_wide.csv"),
        ("tourism_nuts2_all_columns.csv", "tourism_nuts2_wide.csv"),
        ("greek_tourism_nuts2_all_columns.csv", "greek_tourism_nuts2_wide.csv"),
    ]

    for raw_name, wide_name in datasets:
        raw_path = DATA_DIR / raw_name
        wide_path = DATA_DIR / wide_name
        
        if raw_path.exists():
            print(f"\n--- Processing {raw_name} ---")
            wide = preprocess_to_wide_file(str(raw_path), str(wide_path))
            print(f"Saved wide file: {wide_name}")
            print(f"Shape: {wide.shape}")
            
            # Brief check on one indicator to verify
            test_indicator = wide.columns[1].rsplit('_', 1)[0] # Grab first real indicator
            te, f, cols, t_year = prepare_indicator_dataset(wide, test_indicator)
            if te is not None:
                print(f"Test indicator: {test_indicator} (Target Year: {t_year})")
        else:
            print(f"\nSkipping {raw_name} (File not found)")