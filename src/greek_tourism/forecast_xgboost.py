from pathlib import Path
import sys

import xgboost as xgb

# Ensure parent src directory is on sys.path so we can import shared modules
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_SRC = CURRENT_DIR.parent
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from data_processor import preprocess_to_wide_file  # type: ignore  # noqa: E402
from forecast_utils import run_model_comparison  # type: ignore  # noqa: E402
from download_data import download_greek_tourism_default  # type: ignore  # noqa: E402


def main():
    # Resolve paths relative to repo root
    script_dir = Path(__file__).resolve().parent          # .../src/greek_tourism
    repo_dir = script_dir.parent.parent                   # repo root

    raw_file = repo_dir / "data" / "greek_tourism_nuts2_all_columns.csv"
    wide_file = repo_dir / "data" / "greek_tourism_nuts2_wide.csv"
    output_dir = repo_dir / "results" / "greek_tourism" / "xgboost"

    # Ensure raw data exists
    if not raw_file.exists():
        print("Raw greek_tourism data not found, downloading from /greek-tourism endpoint...")
        download_greek_tourism_default()

    # Ensure preprocessed wide file exists
    if not wide_file.exists():
        print(f"Preprocessing raw data to wide format: {wide_file}")
        preprocess_to_wide_file(str(raw_file), str(wide_file))

    data_file = str(wide_file)

    model_params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "gamma": 0.5,
        "reg_lambda": 1.5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "objective": "reg:absoluteerror",
    }

    run_model_comparison(
        model_class=xgb.XGBRegressor,
        model_name="xgboost",
        data_file=data_file,
        output_dir=str(output_dir),
        **model_params,
    )


if __name__ == "__main__":
    main()
