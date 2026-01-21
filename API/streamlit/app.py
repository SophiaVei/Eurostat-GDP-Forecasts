import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Eurostat Regional Analytics",
    layout="wide",
)

# --- MODERN BEAUTIFIED CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Header styling */
    h1 {
        color: #1e3a8a;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    
    /* Glassmorphism containers */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar beautification */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    .sidebar-title {
        color: #1e3a8a;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #64748b;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #1e3a8a !important;
        border-bottom-color: #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)

# --- API CONFIG ---
API_BASE_URL = "https://skillscapes.csd.auth.gr:22226"
SKILLSCAPES_API = "https://skillscapes.csd.auth.gr:22223"

# --- DESCRIPTIONS DATA ---
INDICATOR_DESCRIPTIONS = {
    "economy": {
        "gdp_eur_hab": "GDP per inhabitant in euros",
        "gdp_mio_eur": "GDP in million euros (absolute)",
        "gfcf": "Gross Fixed Capital Formation, total (absolute)",
        "gva": "Gross Value Added, total in million euros (absolute)",
        "gva_sector_a": "GVA for sector A (Agriculture, forestry and fishing)",
        "gva_sector_bde": "GVA for sectors B-E (Industry except construction)",
        "gva_sector_c": "GVA for sector C (Manufacturing)",
        "gva_sector_f": "GVA for sector F (Construction)",
        "gva_sector_ghij": "GVA for sectors G-J (Trade, transport, accommodation, information)",
        "gva_sector_ghi": "GVA for sectors G-I (Trade, transport, accommodation)",
        "gva_sector_j": "GVA for sector J (Information and communication)",
        "gva_sector_klmn": "GVA for sectors K-N (Financial and business services)",
        "gva_sector_k": "GVA for sector K (Financial and insurance activities)",
        "gva_sector_l": "GVA for sector L (Real estate activities)",
        "gva_sector_mn": "GVA for sectors M-N (Professional, scientific, technical and administrative services)",
        "gva_sector_opq": "GVA for sectors O-Q (Public admin, education, health)",
        "gva_sector_opqrstu": "GVA for sectors O-U (Public services and other)",
        "gva_sector_rstu": "GVA for sectors R-U (Arts, other services, households)"
    },
    "labour": {
        "contributing_family_members": "Employment as contributing family members",
        "deprivation": "Material deprivation (2021+)",
        "empl_rate_ED0-2": "Employment rate (Educational attainment ED0-2)",
        "empl_rate_ED3-4": "Employment rate (Educational attainment ED3-4)",
        "empl_rate_ED5-8": "Employment rate (Educational attainment ED5-8)",
        "employees": "Total number of employees",
        "employment_full_time": "Total full-time employment",
        "employment_full_time_pct": "Full-time employment as percentage of total",
        "employment_part_time": "Total part-time employment",
        "employment_part_time_pct": "Part-time employment as percentage of total",
        "fca_epl": "Very Fragile Conditional Autonomy rates with EPL",
        "fca_no_epl": "Very Fragile Conditional Autonomy rates without EPL",
        "housing": "Housing conditions (2021+)",
        "involuntary_part_time": "Involuntary part-time employment",
        "involuntary_part_time_pct": "Involuntary part-time employment as percentage of total",
        "involuntary_temporary": "Involuntary temporary employment",
        "involuntary_temporary_pct": "Involuntary temporary employment as percentage of total",
        "labour_force": "Total labour force",
        "long_term_unemployment": "Total long-term unemployment",
        "long_term_unemployment_rate": "Long-term unemployment rate",
        "neets": "Total NEETs (Not in Education, Employment, or Training)",
        "neets_pop_prc": "NEETs as percentage of population (ages 15-29)",
        "permanent_employment": "Total permanent employment",
        "permanent_employment_pct": "Permanent employment as percentage of total",
        "persons_low_work": "Persons living in low work intensity households (2021+)",
        "persons_risk_poverty": "Persons at risk of poverty (2021+)",
        "sector_a": "Employment in Agriculture (Sector A)",
        "sector_a_lq": "Location Quotient for Agriculture (Sector A)",
        "sector_a_pct": "Employment in Agriculture (% of total)",
        "sector_bde": "Employment in Mining, quarrying, power (Sectors B-E)",
        "sector_bde_lq": "Location Quotient for Mining, quarrying, power (Sectors B-E)",
        "sector_bde_pct": "Employment in Mining, quarrying, power (% of total)",
        "sector_c": "Employment in Manufacturing (Sector C)",
        "sector_c_lq": "Location Quotient for Manufacturing (Sector C)",
        "sector_c_pct": "Employment in Manufacturing (% of total)",
        "sector_f": "Employment in Construction (Sector F)",
        "sector_f_lq": "Location Quotient for Construction (Sector F)",
        "sector_f_pct": "Employment in Construction (% of total)",
        "sector_g": "Employment in Commerce (Sector G)",
        "sector_g_lq": "Location Quotient for Commerce (Sector G)",
        "sector_g_pct": "Employment in Commerce (% of total)",
        "sector_h": "Employment in Transportation (Sector H)",
        "sector_h_lq": "Location Quotient for Transportation (Sector H)",
        "sector_h_pct": "Employment in Transportation (% of total)",
        "sector_i": "Employment in Accommodation & catering (Sector I)",
        "sector_i_lq": "Location Quotient for Accommodation & catering (Sector I)",
        "sector_i_pct": "Employment in Accommodation & catering (% of total)",
        "sector_jklmnu": "Employment in Knowledge economy (Sectors J-N)",
        "sector_jklmnu_lq": "Location Quotient for Knowledge economy (Sectors J-N)",
        "sector_jklmnu_pct": "Employment in Knowledge economy (% of total)",
        "sector_opq": "Employment in Public admin, education, healthcare (Sectors O-Q)",
        "sector_opq_lq": "Location Quotient for Public admin, education, healthcare (Sectors O-Q)",
        "sector_opq_pct": "Employment in Public admin, education, healthcare (% of total)",
        "sector_rst": "Employment in Other services (Sectors R-T)",
        "sector_rst_lq": "Location Quotient for Other services (Sectors R-T)",
        "sector_rst_pct": "Employment in Other services (% of total)",
        "self_employed": "Total self-employed persons",
        "self_employed_with_employees": "Self-employed with employees",
        "self_employed_without_employees": "Self-employed without employees",
        "skills_isco_0": "Employment in Skills ISCO 0 (Armed forces)",
        "skills_isco_1_3": "Employment in Skills ISCO 1-3 (High)",
        "skills_isco_1_3_pct": "Skills ISCO 1-3 as percentage of total",
        "skills_isco_4_5": "Employment in Skills ISCO 4-5 (Medium non-manual)",
        "skills_isco_4_5_pct": "Skills ISCO 4-5 as percentage of total",
        "skills_isco_6_8": "Employment in Skills ISCO 6-8 (Medium manual)",
        "skills_isco_6_8_pct": "Skills ISCO 6-8 as percentage of total",
        "skills_isco_9": "Employment in Skills ISCO 9 (Low)",
        "skills_isco_9_pct": "Skills ISCO 9 as percentage of total",
        "temporary_employment": "Total temporary employment",
        "temporary_employment_pct": "Temporary employment as percentage of total",
        "total_employment": "Total employment",
        "total_employment_rate": "Total employment rate",
        "unemployment": "Total unemployment",
        "unemployment_rate": "Total unemployment rate",
        "weekly_hours": "Average weekly hours worked",
        "youth_employment": "Total youth employment (ages 15-29)",
        "youth_employment_rate": "Youth employment rate",
        "youth_long_term_unemployment_rate": "Youth long-term unemployment rate",
        "youth_unemployment": "Total youth unemployment",
        "youth_unemployment_rate": "Youth unemployment rate"
    },
    "tourism": {
        "arrivals": "Arrivals at tourist accommodation establishments, total",
        "arrivals_per_km2": "Arrivals per square kilometer",
        "arrivals_per_person": "Arrivals per capita",
        "bed_places": "Bed-places in tourist accommodation, total",
        "bed_places_per_1k_persons": "Bed-places per 1,000 persons",
        "bed_places_per_km2": "Bed-places per square kilometer",
        "establishments": "Number of tourist accommodation establishments, total",
        "establishments_per_1k_persons": "Establishments per 1,000 persons",
        "establishments_per_km2": "Establishments per square kilometer",
        "gfcf_sector_ghi": "Gross Fixed Capital Formation for NACE sectors G-I, total",
        "nights_spent": "Nights spent at tourist accommodation establishments, total",
        "nights_spent_per_km2": "Nights spent per square kilometer",
        "nights_spent_per_person": "Nights spent per capita",
        "short_stay": "Short-stay nights (collaborative platforms), total",
        "short_stay_per_km2": "Short-stay nights per square kilometer",
        "short_stay_per_person": "Short-stay nights per capita"
    },
    "greek_tourism": {
        "STR_accommodation_beds": "Short-term rental accommodation bed capacity",
        "employment_accommodation_catering": "Employment in accommodation & catering sector",
        "employment_other": "Employment in other sectors",
        "employment_total": "Total employment in region",
        "employment_total_greece": "Total employment in Greece (reference)",
        "expenditure_per_overnight_stay": "Average expenditure per overnight stay",
        "guest_beds": "Total number of guest beds",
        "guest_beds_per_km2": "Guest beds per square kilometer",
        "guest_beds_per_person": "Guest beds per capita",
        "hotels_avg_duration_of_stay_domestic": "Average stay duration (Domestic hotel guests)",
        "hotels_avg_duration_of_stay_foreign": "Average stay duration (Foreign hotel guests)",
        "hotels_avg_duration_of_stay_total": "Average stay duration (All hotel guests)",
        "hotels_domestic_arrivals": "Hotel arrivals: Domestic guests",
        "hotels_domestic_overnights": "Hotel overnights: Domestic guests",
        "hotels_foreign_arrivals": "Hotel arrivals: Foreign guests",
        "hotels_foreign_overnights": "Hotel overnights: Foreign guests",
        "hotels_total_arrivals": "Total hotel arrivals",
        "hotels_total_arrivals_per_km2": "Hotel arrivals per square kilometer",
        "hotels_total_arrivals_per_person": "Hotel arrivals per capita",
        "hotels_total_overnights": "Total hotel overnights",
        "population": "Regional population",
        "receipts": "Tourism receipts in million euros",
        "rooms": "Total number of hotel rooms",
        "short_stay_total_arrivals": "Total short-stay arrivals (collaborative economy)",
        "short_stay_total_arrivals_per_km2": "Short-stay arrivals per square kilometer",
        "short_stay_total_arrivals_per_person": "Short-stay arrivals per capita",
        "short_stay_total_overnights": "Total short-stay overnights (collaborative economy)",
        "turnover_accommodation": "Turnover in accommodation services",
        "turnover_catering": "Turnover in catering services",
        "turnover_total": "Total tourism industry turnover",
        "units": "Number of hotel units"
    }
}

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_metadata():
    try:
        r = requests.get(f"{API_BASE_URL}/forecast/metadata", verify=False)
        if r.ok: return r.json()
    except: pass
    return {}

@st.cache_data
def fetch_historical_data(domain, indicator):
    endpoint = domain.replace("_", "-")
    url = f"{SKILLSCAPES_API}/{endpoint}"
    params = {"nuts_level": 2, "include": indicator}
    try:
        r = requests.get(url, params=params, verify=False)
        if r.ok: return pd.json_normalize(r.json())
    except: pass
    return pd.DataFrame()

def fetch_forecast_data(domain, indicator, nuts_code=None):
    url = f"{API_BASE_URL}/forecast/{domain}/{indicator}"
    params = {"nuts_code": nuts_code} if nuts_code else {}
    try:
        r = requests.get(url, params=params, verify=False)
        if r.ok: return r.json()
    except: pass
    return None

# --- MAIN APP ---
tab1, tab2 = st.tabs(["Analysis Dashboard", "Indicator Reference"])

with tab1:
    # Sidebar Filters
    st.sidebar.markdown('<p class="sidebar-title">Analytics Filters</p>', unsafe_allow_html=True)
    metadata = get_metadata()

    if not metadata:
        st.error("Could not connect to the Forecasting API.")
        st.stop()

    domain = st.sidebar.selectbox("Domain", options=list(metadata.keys()), format_func=lambda x: x.replace("_", " ").title())
    indicator = st.sidebar.selectbox("Indicator", options=metadata[domain])

    historical_df = fetch_historical_data(domain, indicator)
    if historical_df.empty:
        st.warning(f"No historical data for {indicator}.")
        st.stop()

    available_nuts = sorted(historical_df['geo'].unique().tolist())
    nuts_code = st.sidebar.selectbox("Region (NUTS)", options=available_nuts)

    # Content
    st.title("Eurostat Regional Forecasts")
    
    desc = INDICATOR_DESCRIPTIONS.get(domain, {}).get(indicator, "Regional indicator analysis.")
    st.markdown(f"**{indicator}**: {desc} (Region: **{nuts_code}**)")

    region_df = historical_df[historical_df['geo'] == nuts_code].sort_values('year')
    forecast_json = fetch_forecast_data(domain, indicator, nuts_code)

    if forecast_json and "data" in forecast_json:
        forecast_df = pd.DataFrame(forecast_json['data'])
        model_name = forecast_df['model'].iloc[0] if not forecast_df.empty else "Model"
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        last_val = region_df[indicator].iloc[-1]
        next_val = forecast_df['value'].iloc[0]
        delta = ((next_val - last_val) / last_val) * 100
        
        m1.metric("Current Value", f"{last_val:,.2f}")
        m2.metric("Forecast (2024)", f"{next_val:,.2f}", f"{delta:+.2f}%")
        m3.metric("Prediction Model", model_name)

        # Plotly Go Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=region_df['year'], y=region_df[indicator], 
            name='Historical', mode='lines+markers',
            line=dict(color='#1e3a8a', width=3), marker=dict(size=8)
        ))

        if not region_df.empty:
            connect_x = [region_df['year'].iloc[-1]] + forecast_df['year'].tolist()
            connect_y = [region_df[indicator].iloc[-1]] + forecast_df['value'].tolist()
            marker_sizes = [0] + [8] * len(forecast_df)
            
            fig.add_trace(go.Scatter(
                x=connect_x, y=connect_y,
                name=f'Forecast ({model_name})', mode='lines+markers',
                line=dict(color='#ef4444', width=3, dash='dash'),
                marker=dict(size=marker_sizes, color='#ef4444')
            ))

        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="Year", yaxis_title="Value",
            template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Data Table"):
            h_table = region_df[['year', indicator]].rename(columns={indicator: 'Value'})
            h_table['Type'] = 'Historical'
            f_table = forecast_df[['year', 'value']].rename(columns={'value': 'Value'})
            f_table['Type'] = 'Forecast'
            st.dataframe(pd.concat([h_table, f_table]).sort_values('year', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.error("Forecast data unavailable.")

with tab2:
    st.title("Indicator Reference Guide")
    st.markdown("Detailed definitions of all technical indicators available in this system.")
    
    # Flatten descriptions for a table
    all_rows = []
    for dom, indicators in INDICATOR_DESCRIPTIONS.items():
        for k, v in indicators.items():
            all_rows.append({"Domain": dom.replace("_", " ").title(), "Key": k, "Description": v})
    
    ref_df = pd.DataFrame(all_rows)
    search = st.text_input("Search indicators...", placeholder="e.g. GDP, Tourism...")
    if search:
        ref_df = ref_df[ref_df['Description'].str.contains(search, case=False) | ref_df['Key'].str.contains(search, case=False)]
    
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
