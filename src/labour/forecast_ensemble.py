from pathlib import Path
import sys

# Ensure parent src directory is on sys.path so we can import shared modules
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_SRC = CURRENT_DIR.parent
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from data_processor import preprocess_to_wide_file  # type: ignore  # noqa: E402
from forecast_utils import (  # type: ignore  # noqa: E402
    run_model_comparison,
    EnsembleRegionalWrapper,
)
from download_data import download_labour_default  # type: ignore  # noqa: E402


def main():
    # Resolve paths relative to repo root
    script_dir = Path(__file__).resolve().parent          # .../src/labour
    repo_dir = script_dir.parent.parent                   # repo root

    raw_file = repo_dir / "data" / "labour_nuts2_all_columns.csv"
    wide_file = repo_dir / "data" / "labour_nuts2_wide.csv"
    output_dir = repo_dir / "results" / "labour" / "ensemble"

    # Ensure raw data exists
    if not raw_file.exists():
        print("Raw labour data not found, downloading from /labour endpoint...")
        download_labour_default()

    # Ensure preprocessed wide file exists
    if not wide_file.exists():
        print(f"Preprocessing raw data to wide format: {wide_file}")
        preprocess_to_wide_file(str(raw_file), str(wide_file))

    data_file = str(wide_file)

    run_model_comparison(
        model_class=EnsembleRegionalWrapper,
        model_name="ensemble",
        data_file=data_file,
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()
