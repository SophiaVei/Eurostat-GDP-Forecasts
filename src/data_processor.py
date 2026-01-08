import pandas as pd
import numpy as np

def load_economy_data(filepath):
    """
    Loads the multi-indicator NUTS 2 dataset.
    """
    df = pd.read_csv(filepath)
    # Ensure year is integer
    df['year'] = df['year'].astype(int)
    return df

def get_indicators(df):
    """
    Returns list of economic indicators (columns after metadata).
    """
    metadata_cols = ['geo', 'year', 'geo_label', 'nuts_level', 'country_code', 'country_name', 'is_el_regional_unit']
    return [c for c in df.columns if c not in metadata_cols]

def pivot_to_wide(df):
    """
    Transforms long-form data to wide format: One row per region, columns are indicator_year.
    """
    indicators = get_indicators(df)
    
    # Pivot
    wide_df = df.pivot(index='geo', columns='year', values=indicators)
    
    # Flatten multi-index columns: ('gdp_mio_eur', 2008) -> 'gdp_mio_eur_2008'
    wide_df.columns = [f"{ind}_{yr}" for ind, yr in wide_df.columns]
    
    return wide_df.reset_index()

def prepare_indicator_dataset(wide_df, target_indicator, target_year=2023):
    """
    Prepares X and y for training/eval for a specific indicator.
    
    Args:
        wide_df: The wide-format DataFrame.
        target_indicator: Name of the indicator to predict (e.g., 'gdp_mio_eur').
    """
    # Identify target column
    target_col = f"{target_indicator}_{target_year}"
    
    if target_col not in wide_df.columns:
        # If the target year doesn't exist for this indicator, we can't train/eval
        return None, None, None
        
    # Split into regions where 2023 is present vs missing
    train_eval_df = wide_df[wide_df[target_col].notna()].copy()
    forecast_df = wide_df[wide_df[target_col].isna()].copy()
    
    # Define features (X): all indicator columns for years < 2023
    # Use indicators list to ensure we filter correctly
    feature_cols = [c for c in wide_df.columns if not c.endswith(str(target_year)) and c != 'geo']
    
    # X_single: just target_indicator history
    single_feature_cols = [c for c in feature_cols if c.startswith(target_indicator + "_")]
    
    return train_eval_df, forecast_df, {
        'single': single_feature_cols,
        'multi': feature_cols,
        'target': target_col
    }

if __name__ == "__main__":
    # Test
    import os
    path = r'c:\Users\Sofia\Eurostat-GDP-Forecasts\data\economy_nuts2_all_columns.csv'
    if os.path.exists(path):
        df = load_economy_data(path)
        wide = pivot_to_wide(df)
        print(f"Wide DF Shape: {wide.shape}")
        print(f"Sample columns: {wide.columns[:5].tolist()}")
        
        te, f, cols = prepare_indicator_dataset(wide, 'gdp_mio_eur')
        print(f"Train/Eval rows: {len(te)}")
        print(f"Forecast rows: {len(f)}")
        print(f"Single features count: {len(cols['single'])}")
        print(f"Multi features count: {len(cols['multi'])}")
