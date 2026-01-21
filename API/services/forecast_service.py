import joblib
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import sys

# Import src_shared logic
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src_shared"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from download_data import fetch_live_data_as_df
from data_processor import preprocess_df_to_wide
from forecast_utils import TimesFMRegionalWrapper

try:
    import timesfm
    TIMESFM_AVAILABLE = True
except (ImportError, OSError):
    TIMESFM_AVAILABLE = False

MODELS_DIR = CURRENT_DIR.parent / "models"

def get_forecasts(domain: str, indicator: str, nuts_code: str = None):
    """
    On-Demand Inference:
    1. Loads model
    2. Fetches live data from Skillscapes API
    3. Transforms to wide in memory
    4. Predicts
    5. Returns JSON
    """
    model_path = MODELS_DIR / f"{domain}_{indicator}.pkl"
    if not model_path.exists():
        return {"error": f"Model for {indicator} in {domain} not found."}

    # Load Model Metadata
    try:
        data = joblib.load(model_path)
        model = data['model']
        features = data['features']
        trained_target_year = data.get('trained_target_year', 2023)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    # Identify columns needed for this model
    # Features look like: indicator_year or base_indicator_year
    # We need to extract the base column names to fetch from the API
    cols_to_fetch = set()
    for f in features:
        if "_" in f:
            # Check if it ends in a year
            parts = f.rsplit("_", 1)
            if parts[-1].isdigit():
                cols_to_fetch.add(parts[0])
            else:
                cols_to_fetch.add(f)
        else:
            cols_to_fetch.add(f)
    
    # Always include the target indicator itself to detect the latest year
    cols_to_fetch.add(indicator)

    # Fetch Live Data
    print(f"Fetching live data for {indicator} dependencies: {cols_to_fetch}")
    raw_df = fetch_live_data_as_df(domain, list(cols_to_fetch))
    if raw_df.empty:
        return {"error": "Failed to fetch data from Skillscapes API."}

    # Transform to Wide in Memory
    wide_df = preprocess_df_to_wide(raw_df)
    if wide_df.empty:
        return {"error": "Data preprocessing failed (empty result)."}

    # Calculate last available year per region from raw_df (pre-imputation)
    raw_df_copy = raw_df.copy()
    raw_df_copy["year"] = pd.to_numeric(raw_df_copy["year"], errors='coerce')
    # Filter rows where target indicator is not null to find the real last year
    indicator_data = raw_df_copy[raw_df_copy[indicator].notna()]
    if indicator_data.empty:
        # Fallback to general years if target indicator is missing everywhere
        last_years_map = {geo: int(raw_df_copy["year"].max()) for geo in wide_df["geo"]}
    else:
        last_years_map = indicator_data.groupby('geo')['year'].max().astype(int).to_dict()

    # Handle TimesFM special case
    is_timesfm = "TimesFM" in str(type(model)) or (isinstance(model, str) and model == "TimesFM_Pretrained")
    if is_timesfm and isinstance(model, str):
        if TIMESFM_AVAILABLE:
            model = TimesFMRegionalWrapper(target_prefix=indicator)
        else:
            return {"error": "TimesFM model requested but library not installed in container."}

    # Prepare Inference Input X and determine Forecast Years per region
    prefix = f"{indicator}_"
    
    # Global fallback for max data year and train input
    data_years_cols = [int(c.split('_')[-1]) for c in wide_df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
    global_max_data_year = max(data_years_cols) if data_years_cols else 2023
    
    train_years = []
    for f in features:
        if "_" in f and f.split('_')[-1].isdigit():
            train_years.append(int(f.split('_')[-1]))
    max_train_input = max(train_years) if train_years else global_max_data_year

    X_rows = []
    region_forecast_years = []
    
    for _, row in wide_df.iterrows():
        geo = row['geo']
        region_last_year = last_years_map.get(geo, global_max_data_year)
        region_forecast_years.append(int(region_last_year + 1))
        
        if is_timesfm:
            # TimesFM logic: extract indicators in order
            hist_cols = [c for c in wide_df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
            hist_cols.sort(key=lambda c: int(c.split('_')[-1]))
            # Note: We currently pass the full history (up to global_max_data_year).
            # The model wrapper should handle any trailing 0s or we could trim here.
            X_row = row[hist_cols].to_dict()
        else:
            # Time Shift Lag Logic
            shift = region_last_year - max_train_input
            X_row = {}
            for feat in features:
                if "_" in feat and feat.split('_')[-1].isdigit():
                    parts = feat.rsplit('_', 1)
                    base, yr = parts[0], int(parts[1])
                    target_yr = yr + shift
                    source_col = f"{base}_{target_yr}"
                    X_row[feat] = row[source_col] if source_col in wide_df.columns else np.nan
                else:
                    X_row[feat] = row[feat] if feat in wide_df.columns else np.nan
        X_rows.append(X_row)

    X = pd.DataFrame(X_rows)
    X = X.fillna(0)
    
    # Filter by NUTS if requested
    if nuts_code:
        # Create a boolean mask on the original wide_df to keep indices aligned
        mask = (wide_df['geo'] == nuts_code)
        if not mask.any():
            return {"error": f"NUTS code {nuts_code} not found in the fetched data."}
        
        # Filter everything using the mask
        X = X[mask.values] # X indices are 0..N, matching wide_df rows
        wide_df_filtered = wide_df[mask]
        region_forecast_years = [y for i, y in enumerate(region_forecast_years) if mask.iloc[i]]
    else:
        wide_df_filtered = wide_df

    # Predict
    try:
        preds = model.predict(X)
        preds = np.maximum(0, preds)
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    # Format Results
    results = []
    for i, val in enumerate(preds):
        results.append({
            "geo": wide_df_filtered.iloc[i]['geo'],
            "indicator": indicator,
            "year": int(region_forecast_years[i]),
            "value": round(float(val), 2),
            "model": data.get('model_name', 'Unknown'),
            "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
    
    return {
        "domain": domain,
        "indicator": indicator,
        "count": len(results),
        "data": results
    }

def get_api_metadata():
    models_dir = MODELS_DIR
    model_files = list(models_dir.glob("*.pkl"))
    metadata = {}
    for model_path in model_files:
        name = model_path.stem
        found_domain = None
        indicator = None
        if name.startswith("greek_tourism_"):
            found_domain = "greek_tourism"
            indicator = name[len("greek_tourism_"):]
        else:
            for d in ["economy", "labour", "tourism"]:
                if name.startswith(f"{d}_"):
                    found_domain = d
                    indicator = name[len(f"{d}_"):]
                    break
        if found_domain and indicator:
            if found_domain not in metadata: metadata[found_domain] = []
            if indicator not in metadata[found_domain]: metadata[found_domain].append(indicator)
    for d in metadata: metadata[d].sort()
    return metadata
