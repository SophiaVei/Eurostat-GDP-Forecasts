import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
import sys

# Move torch/timesfm import to the top
try:
    import timesfm
    TIMESFM_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: TimesFM not available ({type(e).__name__}: {e})")
    TIMESFM_AVAILABLE = False

# Ensure src is on path for imports
CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_processor import (
    load_wide_data,
    prepare_indicator_dataset,
    impute_missing_values
)
from forecast_utils import (
    TimesFMRegionalWrapper,
    EnsembleRegionalWrapper
)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

def get_model_class(model_name):
    name = model_name.lower()
    if 'ensemble' in name:
        return EnsembleRegionalWrapper
    elif 'timesfm' in name:
        if TIMESFM_AVAILABLE:
             return TimesFMRegionalWrapper
        else:
             print("Warning: TimesFM requested but not available. Falling back to Ensemble.")
             return EnsembleRegionalWrapper
    elif 'xgboost' in name:
        return XGBRegressor
    elif 'linear' in name:
        return Ridge
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def export_models():
    # Paths
    print(f"Repo Dir: {REPO_DIR}")
    data_dir = REPO_DIR / "data"
    results_dir = REPO_DIR / "results"
    
    # Target Directory for API Models
    api_models_dir = REPO_DIR / "API" / "models"
    os.makedirs(api_models_dir, exist_ok=True)
    
    domains = ['economy', 'labour', 'tourism', 'greek_tourism']
    
    for domain in domains:
        print(f"\n=== Processing Domain: {domain} ===")
        domain_results_dir = results_dir / domain
        master_comp_path = domain_results_dir / "master_comparison.csv"
        detailed_path = domain_results_dir / "detailed_metrics.csv"
        
        if not master_comp_path.exists():
            print(f"Skipping {domain} (No master_comparison.csv found)")
            continue
            
        try:
            master_df = pd.read_csv(master_comp_path)
            # Load detailed metrics to find Feature Set (Single vs Multi)
            if detailed_path.exists():
                det_df = pd.read_csv(detailed_path)
            else:
                det_df = pd.DataFrame()
        except Exception as e:
            print(f"Error reading metadata for {domain}: {e}")
            continue

        # Load data
        wide_file = data_dir / f"{domain}_nuts2_wide.csv"
        if not wide_file.exists():
            print(f"Wide data not found for {domain}")
            continue
        wide_df = load_wide_data(str(wide_file))
        
        for _, row in master_df.iterrows():
            indicator = row['indicator']
            winner_model = row['Winner']
            
            # Determine Feature Set
            feature_set = "Multi" # Default
            if not det_df.empty:
                spec_row = det_df[(det_df['indicator'] == indicator) & (det_df['Model'] == winner_model)]
                if not spec_row.empty:
                    feature_set = spec_row.iloc[0]['Better']
            
            print(f"  Exporting {indicator}: {winner_model} ({feature_set})")
            
            # Prepare Data (Full History)
            training_matrix, _, col_info, target_year = prepare_indicator_dataset(wide_df, indicator)
            
            if training_matrix is None:
                print(f"    Skipping (Data issue)")
                continue
                
            target_col = col_info['target']
            feature_cols = col_info['multi'] if feature_set == "Multi" else col_info['single']
            
            X = training_matrix[feature_cols]
            y = training_matrix[target_col]
            X_imp = impute_missing_values(X)
            
            # Init Model
            try:
                ModelClass = get_model_class(winner_model)
            except ValueError as e:
                print(f"    {e}")
                continue
                
            if winner_model == 'xgboost':
                model = ModelClass(n_estimators=200, max_depth=4, learning_rate=0.1, objective='reg:absoluteerror')
            elif winner_model == 'linear':
                model = ModelClass(alpha=1.0)
            else:
                model = ModelClass(target_prefix=indicator)
            
            # Fit
            try:
                model.fit(X_imp, y)
            except Exception as e:
                print(f"    Failed to train: {e}")
                continue
            
            # Save object
            # Handle TimesFM serialization issue
            if "TimesFM" in str(type(model)):
                 model_to_save = "TimesFM_Pretrained"
            else:
                 model_to_save = model

            model_data = {
                'model': model_to_save,
                'features': feature_cols,
                'feature_set': feature_set,
                'trained_target_year': target_year,
                'indicator': indicator,
                'domain': domain,
                'model_name': winner_model
            }
            
            # Filename: {domain}_{indicator}.pkl
            save_path = api_models_dir / f"{domain}_{indicator}.pkl"
            joblib.dump(model_data, save_path)

    print(f"\nModels exported to {api_models_dir}")

if __name__ == "__main__":
    export_models()
