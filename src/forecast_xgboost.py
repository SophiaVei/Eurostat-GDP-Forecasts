import os
from pathlib import Path
import xgboost as xgb
from forecast_utils import run_model_comparison
from data_processor import preprocess_to_wide_file

def main():
    # Use relative paths from script location
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_DIR = SCRIPT_DIR.parent
    
    RAW_FILE = REPO_DIR / "data" / "economy_nuts2_all_columns.csv"
    WIDE_FILE = REPO_DIR / "data" / "economy_nuts2_wide.csv"
    OUTPUT_DIR = REPO_DIR / "results" / "xgboost"
    
    # Ensure preprocessed wide file exists
    if not WIDE_FILE.exists():
        print(f"Preprocessing raw data to wide format: {WIDE_FILE}")
        preprocess_to_wide_file(str(RAW_FILE), str(WIDE_FILE))
    
    DATA_FILE = str(WIDE_FILE)
    
    # Advanced XGBoost parameters for Phase 2 optimization
    model_params = {
        'n_estimators': 500,        # Increased trees for better depth
        'max_depth': 4,
        'learning_rate': 0.05,     # Smaller learning rate for stability
        'gamma': 0.5,              # Minimum loss reduction for a split
        'reg_lambda': 1.5,         # L2 regularization
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:absoluteerror'
    }
    
    run_model_comparison(
        model_class=xgb.XGBRegressor,
        model_name='xgboost',
        data_file=DATA_FILE,
        output_dir=str(OUTPUT_DIR),
        **model_params
    )

if __name__ == "__main__":
    main()
