from pathlib import Path
import requests
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# Base API endpoint
# ----------------------------
BASE_URL = "https://skillscapes.csd.auth.gr:22223"


# ----------------------------
# Column Definitions (for reference or domain-wide fetches)
# ----------------------------

ALL_ECONOMY_COLUMNS = [
    "gdp_mio_eur", "gdp_eur_hab", "gfcf", "gva",
    "gva_sector_a", "gva_sector_bde", "gva_sector_c", "gva_sector_f",
    "gva_sector_ghij", "gva_sector_ghi", "gva_sector_j",
    "gva_sector_klmn", "gva_sector_k", "gva_sector_l", "gva_sector_mn",
    "gva_sector_opqrstu", "gva_sector_opq", "gva_sector_rstu",
]

ALL_LABOUR_COLUMNS = [
    "labour_force", "total_employment", "total_employment_rate",
    "youth_employment", "youth_employment_rate",
    "unemployment", "unemployment_rate",
    "long_term_unemployment", "long_term_unemployment_rate",
    "youth_unemployment", "youth_unemployment_rate", "youth_long_term_unemployment_rate",
    "employees", "self_employed", "self_employed_with_employees",
    "self_employed_without_employees", "contributing_family_members",
    "employment_part_time", "employment_full_time",
    "employment_part_time_pct", "employment_full_time_pct", "weekly_hours",
    "permanent_employment", "permanent_employment_pct",
    "temporary_employment", "temporary_employment_pct",
    "neets", "neets_pop_prc",
    "fca_epl", "fca_no_epl",
    "involuntary_part_time", "involuntary_part_time_pct",
    "involuntary_temporary", "involuntary_temporary_pct",
    "housing", "persons_low_work", "persons_risk_poverty", "deprivation",
]

ALL_TOURISM_COLUMNS = [
    "arrivals", "arrivals_per_person", "arrivals_per_km2",
    "establishments", "establishments_per_1k_persons", "establishments_per_km2",
    "bed_places", "bed_places_per_1k_persons", "bed_places_per_km2",
    "nights_spent", "nights_spent_per_person", "nights_spent_per_km2",
    "short_stay", "short_stay_per_person", "short_stay_per_km2",
    "gfcf_sector_ghi",
]

ALL_GREEK_TOURISM_COLUMNS = [
    "receipts", "expenditure_per_overnight_stay",
    "hotels_foreign_arrivals", "hotels_domestic_arrivals", "hotels_total_arrivals",
    "hotels_foreign_overnights", "hotels_domestic_overnights", "hotels_total_overnights",
    "hotels_occupancy",
    "hotels_avg_duration_of_stay_foreign", "hotels_avg_duration_of_stay_domestic", "hotels_avg_duration_of_stay_total",
    "hotels_total_arrivals_per_person", "hotels_total_arrivals_per_km2",
    "units", "rooms", "guest_beds",
    "guest_beds_per_person", "guest_beds_per_km2",
    "short_stay_total_arrivals", "short_stay_total_overnights",
    "short_stay_total_arrivals_per_person", "short_stay_total_arrivals_per_km2",
    "STR_accommodation_beds",
    "employment_accommodation_catering", "employment_other", "employment_total", "employment_total_greece",
    "turnover_catering", "turnover_accommodation", "turnover_total",
    "population", "land_area",
]

def fetch_live_data_as_df(endpoint: str, columns: list, nuts_level: int = 2) -> pd.DataFrame:
    """
    Fetches specific columns for a domain directly into a DataFrame.
    """
    # Skillscapes API uses hyphens in paths (e.g. greek-tourism)
    api_endpoint = endpoint.replace("_", "-")
    url = f"{BASE_URL}/{api_endpoint}"
    params = {
        "nuts_level": nuts_level,
        "include": ",".join(columns),
    }

    try:
        r = requests.get(url, params=params, timeout=120, verify=False)
        if not r.ok:
            return pd.DataFrame()
        
        data = r.json()
        records = data
        if isinstance(data, dict):
            for key in ("results", "data", "items"):
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
        
        return pd.json_normalize(records)
    except Exception as e:
        print(f"Fetch failed: {e}")
        return pd.DataFrame()
