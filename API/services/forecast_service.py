import joblib
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import sys
from database import get_database

try:
    import timesfm
    TIMESFM_AVAILABLE = True
except (ImportError, OSError):
    TIMESFM_AVAILABLE = False

# Import internal Wrapper
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent / "src_shared"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forecast_utils import TimesFMRegionalWrapper

def run_forecasts():
    """
    Loads models from API/models, fetches data from Mongo, predicts next year, saves to Mongo.
    """
    db = get_database()
    datasets_collection = db["datasets"]
    forecasts_collection = db["forecasts"]
    models_dir = CURRENT_DIR.parent / "models"
    
    # Iterate pickles
    model_files = list(models_dir.glob("*.pkl"))
    results = []
    skipped = []
    
    print(f"Starting forecast run for {len(model_files)} indicators...")
    
    for model_path in model_files:
        try:
            data = joblib.load(model_path)
            model = data['model']
            features = data['features']
            indicator = data['indicator']
            domain = data['domain']
            
            # TimesFM Re-init
            if isinstance(model, str) and model == "TimesFM_Pretrained":
                 if TIMESFM_AVAILABLE:
                     try:
                        import timesfm as tfm_lib
                        model = TimesFMRegionalWrapper(target_prefix=indicator)
                     except Exception as e:
                        print(f"Failed to init TimesFM for {indicator}: {e}")
                        continue
                 else:
                     continue
            
            # Fetch Data from MongoDB (Domain Collection)
            collection = db[domain]
            
            # Get History Docs
            cursor = collection.find({"type": "history"})
            start_docs = list(cursor)
            if not start_docs:
                print(f"Skipping {indicator}: No historical data found in collection '{domain}'.")
                skipped.append(f"{indicator} (Missing {domain} data)")
                continue
                
            rows = []
            for d in start_docs:
                row = d['data']
                row['geo'] = d['geo']
                rows.append(row)
            
            wide_df = pd.DataFrame(rows)
            
            is_timesfm = "TimesFM" in str(type(model))
            
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
                     # Static features only?
                    X = pd.DataFrame(0, index=wide_df.index, columns=features)
                    # Try to map static
                    for f in features:
                        if f in wide_df.columns:
                            X[f] = wide_df[f]
                    forecast_year = pd.Timestamp.now().year + 1
                else:
                    max_train_input = max(train_years)
                    
                    prefix = f"{indicator}_"
                    data_years = [int(c.split('_')[-1]) for c in wide_df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()]
                    if not data_years:
                        continue
                    max_data_year = max(data_years)
                    
                    shift = max_data_year - max_train_input
                    
                    X = pd.DataFrame(index=wide_df.index)
                    for feat in features:
                        if "_" in feat and feat.split('_')[-1].isdigit():
                            base = "_".join(feat.split('_')[:-1])
                            yr = int(feat.split('_')[-1])
                            target_yr = yr + shift
                            source_col = f"{base}_{target_yr}"
                            
                            if source_col in wide_df.columns:
                                X[feat] = wide_df[source_col]
                            else:
                                X[feat] = np.nan
                        else:
                            if feat in wide_df.columns:
                                X[feat] = wide_df[feat]
                            else:
                                X[feat] = np.nan
                                
                    forecast_year = data['trained_target_year'] + shift

            X = X.fillna(0)
            
            try:
                preds = model.predict(X)
                preds = np.maximum(0, preds)
            except Exception as e:
                print(f"Prediction failed for {indicator}: {e}")
                continue

            # Save to Mongo (Same collection, type='forecast')
            collection.delete_many({
                "type": "forecast",
                "indicator": indicator, 
                "year": int(forecast_year)
            })
            
            new_docs = []
            for i, val in enumerate(preds):
                geo = wide_df.iloc[i]['geo']
                new_docs.append({
                    "geo": geo,
                    "type": "forecast",
                    "indicator": indicator,
                    "year": int(forecast_year),
                    "value": float(val),
                    "model": data['model_name'],
                    "run_at": datetime.datetime.now(datetime.timezone.utc)
                })
            
            if new_docs:
                collection.insert_many(new_docs)
                results.append(f"{indicator} -> {forecast_year}")
                
        except Exception as e:
            print(f"Error processing {model_path.name}: {e}")
            continue

    return {
        "status": "success", 
        "summary": results,
        "skipped": skipped,
        "total_processed": len(results),
        "total_skipped": len(skipped)
    }

def get_forecasts(domain: str, indicator: str = None):
    """
    Retrieves forecasts from the domain collection.
    """
    db = get_database()
    collection = db[domain]
    
    query = {"type": "forecast"}
    if indicator:
        query["indicator"] = indicator
        
    cursor = collection.find(query, {"_id": 0}) # Exclude ObjectID
    results = list(cursor)
    
    return {
        "domain": domain,
        "indicator": indicator if indicator else "all",
        "count": len(results),
        "data": results
    }

def get_api_metadata():
    """
    Scans the API/models directory to identify all available domains and indicators.
    """
    models_dir = CURRENT_DIR.parent / "models"
    model_files = list(models_dir.glob("*.pkl"))
    
    metadata = {}
    
    # Filename pattern: {domain}_{indicator}.pkl
    for model_path in model_files:
        name = model_path.stem
        # Try to find the domain by splitting
        # Possible domains: economy, labour, tourism, greek_tourism
        parts = name.split("_")
        
        found_domain = None
        indicator = None
        
        # Priority check for multi-word domains like greek_tourism
        if name.startswith("greek_tourism_"):
            found_domain = "greek_tourism"
            indicator = name[len("greek_tourism_"):]
        else:
            # economy_, labour_, tourism_
            for d in ["economy", "labour", "tourism"]:
                if name.startswith(f"{d}_"):
                    found_domain = d
                    indicator = name[len(f"{d}_"):]
                    break
        
        if found_domain and indicator:
            if found_domain not in metadata:
                metadata[found_domain] = []
            if indicator not in metadata[found_domain]:
                metadata[found_domain].append(indicator)
                
    # Sort for predictability
    for d in metadata:
        metadata[d].sort()
        
    return metadata
