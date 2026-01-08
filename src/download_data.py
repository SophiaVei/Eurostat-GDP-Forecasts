from pathlib import Path
import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# Save into repo/data
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent      # .../src
REPO_DIR = SCRIPT_DIR.parent                      # repo root
DATA_DIR = REPO_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUT_FILE = DATA_DIR / "economy_nuts2_all_columns.csv"

# ----------------------------
# API endpoint
# ----------------------------
BASE_URL = "https://skillscapes.csd.auth.gr:22223"
URL = f"{BASE_URL}/economy"

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

params = {
    "nuts_level": 2,
    "include": ",".join(ALL_ECONOMY_COLUMNS),
    # Optional (uncomment if you want a year range)
    # "year_start": 2008,
    # "year_end": 2024,
}

r = requests.get(URL, params=params, timeout=180, verify=False)
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

df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ Saved CSV: {OUT_FILE}")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("Columns:", list(df.columns))
