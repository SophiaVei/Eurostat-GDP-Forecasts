from pathlib import Path
import sys

from sklearn.linear_model import Ridge

# Ensure parent src directory is on sys.path so we can import shared modules
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_SRC = CURRENT_DIR.parent
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from data_processor import preprocess_to_wide_file  # type: ignore  # noqa: E402
from forecast_utils import run_model_comparison  # type: ignore  # noqa: E402
from download_data import download_economy_default  # type: ignore  # noqa: E402


def main():
    # Resolve paths relative to repo root
    script_dir = Path(__file__).resolve().parent          # .../src/economy
    repo_dir = script_dir.parent.parent                   # repo root

    raw_file = repo_dir / "data" / "economy_nuts2_all_columns.csv"
    wide_file = repo_dir / "data" / "economy_nuts2_wide.csv"
    output_dir = repo_dir / "results" / "economy" / "linear"

    # Ensure raw data exists
    if not raw_file.exists():
        print("Raw economy data not found, downloading from /economy endpoint...")
        download_economy_default()

    # Ensure preprocessed wide file exists
    if not wide_file.exists():
        print(f"Preprocessing raw data to wide format: {wide_file}")
        preprocess_to_wide_file(str(raw_file), str(wide_file))

    data_file = str(wide_file)

    run_model_comparison(
        model_class=Ridge,
        model_name="linear",
        data_file=data_file,
        output_dir=str(output_dir),
        alpha=1.0,
    )


if __name__ == "__main__":
    main()

