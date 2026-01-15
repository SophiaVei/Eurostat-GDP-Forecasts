from pathlib import Path
import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# Common paths
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent      # .../src
REPO_DIR = SCRIPT_DIR.parent                      # repo root
DATA_DIR = REPO_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------
# Base API endpoint
# ----------------------------
BASE_URL = "https://skillscapes.csd.auth.gr:22223"

# All columns listed in the /economy docs
ALL_ECONOMY_COLUMNS = [
    "gdp_mio_eur",
    "gdp_eur_hab",
    "gfcf",
    "gva",
    "gva_sector_a",
    "gva_sector_bde",
    "gva_sector_c",
    "gva_sector_f",
    "gva_sector_ghij",
    "gva_sector_ghi",
    "gva_sector_j",
    "gva_sector_klmn",
    "gva_sector_k",
    "gva_sector_l",
    "gva_sector_mn",
    "gva_sector_opqrstu",
    "gva_sector_opq",
    "gva_sector_rstu",
]


def download_endpoint(endpoint: str, out_file: Path, params: dict | None = None) -> Path:
    """
    Generic downloader for Skillscapes API endpoints.

    Args:
        endpoint: Endpoint path without leading slash, e.g. 'economy', 'labour'.
        out_file: Full path to the CSV file to be created.
        params: Optional dict of query parameters. For /economy we usually
                include 'nuts_level' and an 'include' list. For other
                endpoints we can often omit 'include' and rely on API defaults.
    """
    url = f"{BASE_URL}/{endpoint}"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    params = params or {}

    r = requests.get(url, params=params, timeout=180, verify=False)
    print("Request URL:", r.url)
    print("HTTP status:", r.status_code)

    if not r.ok:
        print("Response text:", r.text[:2000])
        r.raise_for_status()

    data = r.json()

    # If dict-wrapped, unwrap common keys; otherwise assume it's already a list
    records = data
    if isinstance(data, dict):
        for key in ("results", "data", "items"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break

    df = pd.json_normalize(records)
    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"✅ Saved CSV: {out_file}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))

    return out_file


def download_economy_default() -> Path:
    """
    Convenience function to download the /economy endpoint with the
    standard column set for NUTS2 regions.
    """
    out_file = DATA_DIR / "economy_nuts2_all_columns.csv"
    params = {
        "nuts_level": 2,
        "include": ",".join(ALL_ECONOMY_COLUMNS),
        # Optional (uncomment if you want a year range)
        # "year_start": 2008,
        # "year_end": 2024,
    }
    return download_endpoint("economy", out_file, params)


if __name__ == "__main__":
    # Default behavior: download /economy with the pre-defined column list.
    download_economy_default()
