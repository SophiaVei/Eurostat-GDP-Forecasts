import os
from pathlib import Path
from sklearn.linear_model import Ridge
from forecast_utils import run_model_comparison
from data_processor import preprocess_to_wide_file

def main():
    # Use relative paths from script location
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_DIR = SCRIPT_DIR.parent
    
    RAW_FILE = REPO_DIR / "data" / "economy_nuts2_all_columns.csv"
    WIDE_FILE = REPO_DIR / "data" / "economy_nuts2_wide.csv"
    OUTPUT_DIR = REPO_DIR / "results" / "linear"
    
    # Ensure preprocessed wide file exists
    if not WIDE_FILE.exists():
        print(f"Preprocessing raw data to wide format: {WIDE_FILE}")
        preprocess_to_wide_file(str(RAW_FILE), str(WIDE_FILE))
    
    DATA_FILE = str(WIDE_FILE)
    
    # Using Ridge regression to handle many features (Multi-Feature) and prevent overfitting
    # alpha=1.0 provides moderate regularization
    run_model_comparison(
        model_class=Ridge,
        model_name='linear',
        data_file=DATA_FILE,
        output_dir=str(OUTPUT_DIR),
        alpha=1.0
    )

if __name__ == "__main__":
    main()
