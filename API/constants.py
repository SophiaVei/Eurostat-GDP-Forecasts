from typing import Literal

DomainLiteral = Literal["economy", "labour", "tourism", "greek_tourism"]

EconomyIndicator = Literal[
    "gdp_eur_hab", "gdp_mio_eur", "gfcf", "gva", "gva_sector_a", 
    "gva_sector_bde", "gva_sector_c", "gva_sector_f", "gva_sector_ghi", 
    "gva_sector_ghij", "gva_sector_j", "gva_sector_k", "gva_sector_klmn", 
    "gva_sector_l", "gva_sector_mn", "gva_sector_opq", "gva_sector_opqrstu", 
    "gva_sector_rstu"
]

LabourIndicator = Literal[
    "contributing_family_members", "deprivation", "empl_rate_ED0-2", 
    "empl_rate_ED3-4", "empl_rate_ED5-8", "employees", "employment_full_time", 
    "employment_full_time_pct", "employment_part_time", "employment_part_time_pct", 
    "fca_epl", "fca_no_epl", "housing", "involuntary_part_time", 
    "involuntary_part_time_pct", "involuntary_temporary", "involuntary_temporary_pct", 
    "labour_force", "long_term_unemployment", "long_term_unemployment_rate", 
    "neets", "neets_pop_prc", "permanent_employment", "permanent_employment_pct", 
    "persons_low_work", "persons_risk_poverty", "sector_a", "sector_a_lq", 
    "sector_a_pct", "sector_bde", "sector_bde_lq", "sector_bde_pct", 
    "sector_c", "sector_c_lq", "sector_c_pct", "sector_f", "sector_f_lq", 
    "sector_f_pct", "sector_g", "sector_g_lq", "sector_g_pct", "sector_h", 
    "sector_h_lq", "sector_h_pct", "sector_i", "sector_i_lq", "sector_i_pct", 
    "sector_jklmnu", "sector_jklmnu_lq", "sector_jklmnu_pct", "sector_opq", 
    "sector_opq_lq", "sector_opq_pct", "sector_rst", "sector_rst_lq", 
    "sector_rst_pct", "self_employed", "self_employed_with_employees", 
    "self_employed_without_employees", "skills_isco_0", "skills_isco_1_3", 
    "skills_isco_1_3_pct", "skills_isco_4_5", "skills_isco_4_5_pct", 
    "skills_isco_6_8", "skills_isco_6_8_pct", "skills_isco_9", 
    "skills_isco_9_pct", "temporary_employment", "temporary_employment_pct", 
    "total_employment", "total_employment_rate", "unemployment", 
    "unemployment_rate", "weekly_hours", "youth_employment", 
    "youth_employment_rate", "youth_long_term_unemployment_rate", 
    "youth_unemployment", "youth_unemployment_rate"
]

TourismIndicator = Literal[
    "arrivals", "arrivals_per_km2", "arrivals_per_person", "bed_places", 
    "bed_places_per_1k_persons", "bed_places_per_km2", "establishments", 
    "establishments_per_1k_persons", "establishments_per_km2", "gfcf_sector_ghi", 
    "nights_spent", "nights_spent_per_km2", "nights_spent_per_person", 
    "short_stay", "short_stay_per_km2", "short_stay_per_person"
]

GreekTourismIndicator = Literal[
    "STR_accommodation_beds", "employment_accommodation_catering", 
    "employment_other", "employment_total", "employment_total_greece", 
    "expenditure_per_overnight_stay", "guest_beds", "guest_beds_per_km2", 
    "guest_beds_per_person", "hotels_avg_duration_of_stay_domestic", 
    "hotels_avg_duration_of_stay_foreign", "hotels_avg_duration_of_stay_total", 
    "hotels_domestic_arrivals", "hotels_domestic_overnights", 
    "hotels_foreign_arrivals", "hotels_foreign_overnights", 
    "hotels_total_arrivals", "hotels_total_arrivals_per_km2", 
    "hotels_total_arrivals_per_person", "hotels_total_overnights", 
    "land_area", "population", "receipts", "rooms", 
    "short_stay_total_arrivals", "short_stay_total_arrivals_per_km2", 
    "short_stay_total_arrivals_per_person", "short_stay_total_overnights", 
    "turnover_accommodation", "turnover_catering", "turnover_total", "units"
]
