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


# ----------------------------
# Column Definitions
# ----------------------------

ALL_ECONOMY_COLUMNS = [
    "gdp_mio_eur", "gdp_eur_hab", "gfcf", "gva",
    "gva_sector_a", "gva_sector_bde", "gva_sector_c", "gva_sector_f",
    "gva_sector_ghij", "gva_sector_ghi", "gva_sector_j",
    "gva_sector_klmn", "gva_sector_k", "gva_sector_l", "gva_sector_mn",
    "gva_sector_opqrstu", "gva_sector_opq", "gva_sector_rstu",
]

# Labour columns based on API docs
ALL_LABOUR_COLUMNS = [
    # Employment & Unemployment
    "labour_force", "total_employment", "total_employment_rate",
    "youth_employment", "youth_employment_rate",
    "unemployment", "unemployment_rate",
    "long_term_unemployment", "long_term_unemployment_rate",
    "youth_unemployment", "youth_unemployment_rate", "youth_long_term_unemployment_rate",
    # Employee types
    "employees", "self_employed", "self_employed_with_employees",
    "self_employed_without_employees", "contributing_family_members",
    # Work patterns
    "employment_part_time", "employment_full_time",
    "employment_part_time_pct", "employment_full_time_pct", "weekly_hours",
    # Contract types
    "permanent_employment", "permanent_employment_pct",
    "temporary_employment", "temporary_employment_pct",
    # Precarity
    "neets", "neets_pop_prc",
    "fca_epl", "fca_no_epl",
    "involuntary_part_time", "involuntary_part_time_pct",
    "involuntary_temporary", "involuntary_temporary_pct",
    # Material conditions
    "housing", "persons_low_work", "persons_risk_poverty", "deprivation",
]
# Add sectoral employment variants (sector * (abs, pct, lq))
_labour_sectors = ["a", "bde", "c", "f", "g", "h", "i", "jklmnu", "opq", "rst"]
for _s in _labour_sectors:
    ALL_LABOUR_COLUMNS.extend([f"sector_{_s}", f"sector_{_s}_pct", f"sector_{_s}_lq"])

# Add education/skills
ALL_LABOUR_COLUMNS.extend([
    "empl_rate_ED0-2", "empl_rate_ED3-4", "empl_rate_ED5-8",
    "skills_isco_0", "skills_isco_1_3", "skills_isco_4_5", "skills_isco_6_8", "skills_isco_9",
    "skills_isco_0_pct", "skills_isco_1_3_pct", "skills_isco_4_5_pct", "skills_isco_6_8_pct", "skills_isco_9_pct",
])

ALL_TOURISM_COLUMNS = [
    # Arrivals
    "arrivals", "arrivals_per_person", "arrivals_per_km2",
    # Capacity
    "establishments", "establishments_per_1k_persons", "establishments_per_km2",
    "bed_places", "bed_places_per_1k_persons", "bed_places_per_km2",
    # Nights
    "nights_spent", "nights_spent_per_person", "nights_spent_per_km2",
    # Short-stay
    "short_stay", "short_stay_per_person", "short_stay_per_km2",
    # Investment
    "gfcf_sector_ghi",
]

ALL_GREEK_TOURISM_COLUMNS = [
    # Key Financial
    "receipts", "expenditure_per_overnight_stay",
    # Hotel Arrivals/Overnights
    "hotels_foreign_arrivals", "hotels_domestic_arrivals", "hotels_total_arrivals",
    "hotels_foreign_overnights", "hotels_domestic_overnights", "hotels_total_overnights",
    "hotels_occupancy",
    "hotels_avg_duration_of_stay_foreign", "hotels_avg_duration_of_stay_domestic", "hotels_avg_duration_of_stay_total",
    # Per-capita/area variants for hotels
    "hotels_total_arrivals_per_person", "hotels_total_arrivals_per_km2",
    # Hotel Capacity
    "units", "rooms", "guest_beds",
    "guest_beds_per_person", "guest_beds_per_km2",
    # STR
    "short_stay_total_arrivals", "short_stay_total_overnights",
    "short_stay_total_arrivals_per_person", "short_stay_total_arrivals_per_km2",
    "STR_accommodation_beds",
    # Employment
    "employment_accommodation_catering", "employment_other", "employment_total", "employment_total_greece",
    # Turnover
    "turnover_catering", "turnover_accommodation", "turnover_total",
    # Context
    "population", "land_area",
]


def download_endpoint(endpoint: str, out_file: Path, params: dict | None = None) -> Path:
    """
    Generic downloader for Skillscapes API endpoints.

    Args:
        endpoint: Endpoint path without leading slash, e.g. 'economy', 'labour'.
        out_file: Full path to the CSV file to be created.
        params: Optional dict of query parameters.
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
    }
    return download_endpoint("economy", out_file, params)


def download_labour_default() -> Path:
    """
    Convenience function to download the /labour endpoint.
    """
    out_file = DATA_DIR / "labour_nuts2_all_columns.csv"
    params = {
        "nuts_level": 2,
        "include": ",".join(ALL_LABOUR_COLUMNS),
    }
    return download_endpoint("labour", out_file, params)


def download_tourism_default() -> Path:
    """
    Convenience function to download the /tourism endpoint.
    """
    out_file = DATA_DIR / "tourism_nuts2_all_columns.csv"
    params = {
        "nuts_level": 2,
        "include": ",".join(ALL_TOURISM_COLUMNS),
    }
    return download_endpoint("tourism", out_file, params)


def download_greek_tourism_default() -> Path:
    """
    Convenience function to download the /greek-tourism endpoint.
    """
    out_file = DATA_DIR / "greek_tourism_nuts2_all_columns.csv"
    params = {
        "nuts_level": 2,
        "include": ",".join(ALL_GREEK_TOURISM_COLUMNS),
    }
    return download_endpoint("greek-tourism", out_file, params)


if __name__ == "__main__":
    # Default behavior: download all datasets
    print("Downloading Economy...")
    download_economy_default()
    print("Downloading Labour...")
    download_labour_default()
    print("Downloading Tourism...")
    download_tourism_default()
    print("Downloading Greek Tourism...")
    download_greek_tourism_default()

