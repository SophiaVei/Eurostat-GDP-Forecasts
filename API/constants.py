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

NutsCode = Literal[
    'AL01', 'AL02', 'AL03', 'ALZZ', 'AT11', 'AT12', 'AT13', 'AT21', 'AT22', 'AT31', 
    'AT32', 'AT33', 'AT34', 'ATZZ', 'BE10', 'BE21', 'BE22', 'BE23', 'BE24', 'BE25', 
    'BE31', 'BE32', 'BE33', 'BE34', 'BE35', 'BEZZ', 'BG31', 'BG32', 'BG33', 'BG34', 
    'BG41', 'BG42', 'BGZZ', 'CH01', 'CH02', 'CH03', 'CH04', 'CH05', 'CH06', 'CH07', 
    'CY00', 'CYZZ', 'CZ01', 'CZ02', 'CZ03', 'CZ04', 'CZ05', 'CZ06', 'CZ07', 'CZ08', 
    'CZZZ', 'DE11', 'DE12', 'DE13', 'DE14', 'DE21', 'DE22', 'DE23', 'DE24', 'DE25', 
    'DE26', 'DE27', 'DE30', 'DE40', 'DE50', 'DE60', 'DE71', 'DE72', 'DE73', 'DE80', 
    'DE91', 'DE92', 'DE93', 'DE94', 'DEA1', 'DEA2', 'DEA3', 'DEA4', 'DEA5', 'DEB1', 
    'DEB2', 'DEB3', 'DEC1', 'DED2', 'DED4', 'DED5', 'DEE0', 'DEF0', 'DEG0', 'DEZZ', 
    'DK01', 'DK02', 'DK03', 'DK04', 'DK05', 'DKZZ', 'EE00', 'EEZZ', 'EL30', 'EL41', 
    'EL42', 'EL43', 'EL51', 'EL52', 'EL53', 'EL54', 'EL61', 'EL62', 'EL63', 'EL64', 
    'EL65', 'ELZZ', 'ES11', 'ES12', 'ES13', 'ES21', 'ES22', 'ES23', 'ES24', 'ES30', 
    'ES41', 'ES42', 'ES43', 'ES51', 'ES52', 'ES53', 'ES61', 'ES62', 'ES63', 'ES64', 
    'ES70', 'ESZZ', 'FI19', 'FI1B', 'FI1C', 'FI1D', 'FI20', 'FIZZ', 'FR10', 'FRB0', 
    'FRC1', 'FRC2', 'FRD1', 'FRD2', 'FRE1', 'FRE2', 'FRF1', 'FRF2', 'FRF3', 'FRG0', 
    'FRH0', 'FRI1', 'FRI2', 'FRI3', 'FRJ1', 'FRJ2', 'FRK1', 'FRK2', 'FRL0', 'FRM0', 
    'FRY1', 'FRY2', 'FRY3', 'FRY4', 'FRY5', 'FRZZ', 'HR02', 'HR03', 'HR05', 'HR06', 
    'HRZZ', 'HU11', 'HU12', 'HU21', 'HU22', 'HU23', 'HU31', 'HU32', 'HU33', 'HUZZ', 
    'IE04', 'IE05', 'IE06', 'IEZZ', 'IS00', 'ITC1', 'ITC2', 'ITC3', 'ITC4', 'ITF1', 
    'ITF2', 'ITF3', 'ITF4', 'ITF5', 'ITF6', 'ITG1', 'ITG2', 'ITH1', 'ITH2', 'ITH3', 
    'ITH4', 'ITH5', 'ITI1', 'ITI2', 'ITI3', 'ITI4', 'ITZZ', 'LT01', 'LT02', 'LU00', 
    'LV00', 'LVZZ', 'ME00', 'MK00', 'MT00', 'MTZZ', 'NL11', 'NL12', 'NL13', 'NL21', 
    'NL22', 'NL23', 'NL32', 'NL34', 'NL41', 'NL42', 'NLZZ', 'NO02', 'NO06', 'NO07', 
    'NO08', 'NO09', 'NO0A', 'NO0B', 'NOZZ', 'PL21', 'PL22', 'PL41', 'PL42', 'PL43', 
    'PL51', 'PL52', 'PL61', 'PL62', 'PL63', 'PL71', 'PL72', 'PL81', 'PL82', 'PL84', 
    'PL91', 'PL92', 'PLZZ', 'PT11', 'PT15', 'PT20', 'PT30', 'PTZZ', 'RO11', 'RO12', 
    'RO21', 'RO22', 'RO31', 'RO32', 'RO41', 'RO42', 'ROZZ', 'RS11', 'RS12', 'RS21', 
    'RS22', 'RSZZ', 'SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33', 
    'SEZZ', 'SI03', 'SI04', 'SK01', 'SK02', 'SK03', 'SK04', 'TR10', 'TR21', 'TR22', 
    'TR31', 'TR32', 'TR33', 'TR41', 'TR42', 'TR51', 'TR52', 'TR61', 'TR62', 'TR63', 
    'TR71', 'TR72', 'TR81', 'TR82', 'TR83', 'TR90', 'TRA1', 'TRA2', 'TRB1', 'TRB2', 
    'TRC1', 'TRC2', 'TRC3'
]
