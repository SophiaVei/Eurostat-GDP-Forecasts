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

    # Handle TimesFM special case
    is_timesfm = "TimesFM" in str(type(model)) or (isinstance(model, str) and model == "TimesFM_Pretrained")
    
    if is_timesfm and isinstance(model, str):
        if TIMESFM_AVAILABLE:
            model = TimesFMRegionalWrapper(target_prefix=indicator)
        else:
            return {"error": "TimesFM model requested but library not installed in container."}

    # Prepare Inference Input X
    if is_timesfm:
        prefix = f"{indicator}_"
        hist_cols = [c for c in wide_df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
        hist_cols.sort(key=lambda c: int(c.split('_')[-1]))
        X = wide_df[hist_cols]
        last_year = int(hist_cols[-1].split('_')[-1])
        forecast_year = last_year + 1
    else:
        # Time Shift Lag Logic
        train_years = []
        for f in features:
             if "_" in f and f.split('_')[-1].isdigit():
                 train_years.append(int(f.split('_')[-1]))
        
        if not train_years:
            X = pd.DataFrame(0, index=wide_df.index, columns=features)
            for f in features:
                if f in wide_df.columns:
                    X[f] = wide_df[f]
            forecast_year = datetime.datetime.now().year + 1
        else:
            max_train_input = max(train_years)
            prefix = f"{indicator}_"
            data_years = [int(c.split('_')[-1]) for c in wide_df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
            if not data_years:
                return {"error": f"No historical data years found for {indicator}."}
            max_data_year = max(data_years)
            shift = max_data_year - max_train_input
            
            X = pd.DataFrame(index=wide_df.index)
            for feat in features:
                if "_" in feat and feat.split('_')[-1].isdigit():
                    parts = feat.rsplit('_', 1)
                    base = parts[0]
                    yr = int(parts[1])
                    target_yr = yr + shift
                    source_col = f"{base}_{target_yr}"
                    X[feat] = wide_df[source_col] if source_col in wide_df.columns else np.nan
                else:
                    X[feat] = wide_df[feat] if feat in wide_df.columns else np.nan
            forecast_year = trained_target_year + shift

    X = X.fillna(0)
    
    # Filter by NUTS if requested
    if nuts_code:
        mask = (wide_df['geo'] == nuts_code)
        if not mask.any():
            return {"error": f"NUTS code {nuts_code} not found in the fetched data."}
        X = X[mask]
        wide_df_filtered = wide_df[mask]
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
            "year": int(forecast_year),
            "value": float(val),
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
