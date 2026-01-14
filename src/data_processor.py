import pandas as pd
import numpy as np
from pathlib import Path


def load_economy_data(filepath):
    """
    Loads the multi-indicator NUTS 2 dataset (long format: one row per geo-year).
    Ensures the year column is integer.
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


def prepare_indicator_dataset(wide_df, target_indicator, target_year=2023):
    """
    Prepares train/eval and forecast splits for a specific indicator.

    Args:
        wide_df: The wide-format DataFrame (one row per geo).
        target_indicator: Name of the indicator to predict (e.g., 'gdp_mio_eur').
        target_year: Target year to forecast/evaluate (default: 2023).
    """
    target_col = f"{target_indicator}_{target_year}"

    if target_col not in wide_df.columns:
        # If the target year doesn't exist for this indicator, we can't train/eval
        return None, None, None

    # Split into regions where target year is present vs missing
    train_eval_df = wide_df[wide_df[target_col].notna()].copy()
    forecast_df = wide_df[wide_df[target_col].isna()].copy()

    # Define features (X): all indicator columns for years < target_year
    feature_cols = [
        c for c in wide_df.columns if not c.endswith(str(target_year)) and c != "geo"
    ]

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

    return train_eval_df, forecast_df, {
        "single": single_feature_cols,
        "multi": feature_cols,
        "target": target_col,
    }


def impute_missing_values(X):
    """
    Fills NaNs and 0s with the mean of the available values for each indicator
    in that specific row. 0s are treated as missing values.

    This is shared across all models so that they receive identically
    preprocessed feature matrices.
    """
    X = X.copy()
    X = X.replace(0, np.nan)

    # Group columns by indicator prefix (everything except the trailing year)
    col_prefixes = {}
    for col in X.columns:
        if "_" in col:
            prefix = "_".join(col.split("_")[:-1])
            col_prefixes.setdefault(prefix, []).append(col)

    # Fill row-wise mean for each indicator group
    for prefix, cols in col_prefixes.items():
        row_means = X[cols].mean(axis=1)
        X[cols] = X[cols].apply(lambda s: s.fillna(row_means))

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
    df = load_economy_data(raw_filepath)
    wide_df = pivot_to_wide(df)
    Path(wide_out_filepath).parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(wide_out_filepath, index=False)
    return wide_df


if __name__ == "__main__":
    # Simple manual test / convenience entrypoint for preprocessing.
    import os

    # Adjust these paths as needed for your environment
    RAW_PATH = r"c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv"
    WIDE_OUT = r"c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_wide.csv"

    if os.path.exists(RAW_PATH):
        print("Preprocessing raw economy data to wide format...")
        wide = preprocess_to_wide_file(RAW_PATH, WIDE_OUT)
        print(f"Wide DF Shape: {wide.shape}")
        print(f"Sample columns: {wide.columns[:5].tolist()}")

        te, f, cols = prepare_indicator_dataset(wide, "gdp_mio_eur")
        print(f"Train/Eval rows: {len(te)}")
        print(f"Forecast rows: {len(f)}")
        print(f"Single features count: {len(cols['single'])}")
        print(f"Multi features count: {len(cols['multi'])}")
        print(f"Saved preprocessed wide file to: {WIDE_OUT}")