from fastapi import APIRouter
from typing import Literal
from services.etl_service import update_datasets
from services.forecast_service import run_forecasts, get_forecasts, get_api_metadata
from constants import DomainLiteral, EconomyIndicator, LabourIndicator, TourismIndicator, GreekTourismIndicator

data_router = APIRouter(prefix="/data", tags=["Data"])
forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@data_router.post("/update")
def trigger_update_datasets():
    return update_datasets()

@forecast_router.post("/run")
def trigger_forecast():
    return run_forecasts()

@forecast_router.get("/metadata")
def get_metadata():
    """Returns a list of all available domains and their corresponding indicators."""
    return get_api_metadata()

# Domain-Specific Routes for refined indicator dropdowns
@forecast_router.get("/economy/{indicator}")
def get_economy_forecast(indicator: EconomyIndicator):
    """Retrieve forecasts for specific Economy indicators."""
    return get_forecasts("economy", indicator)

@forecast_router.get("/labour/{indicator}")
def get_labour_forecast(indicator: LabourIndicator):
    """Retrieve forecasts for specific Labour indicators."""
    return get_forecasts("labour", indicator)

@forecast_router.get("/tourism/{indicator}")
def get_tourism_forecast(indicator: TourismIndicator):
    """Retrieve forecasts for specific Tourism indicators."""
    return get_forecasts("tourism", indicator)

@forecast_router.get("/greek_tourism/{indicator}")
def get_greek_tourism_forecast(indicator: GreekTourismIndicator):
    """Retrieve forecasts for specific Greek Tourism indicators."""
    return get_forecasts("greek_tourism", indicator)

# Generic fallback route
@forecast_router.get("/{domain}/{indicator}")
def get_indicator_forecast(domain: DomainLiteral, indicator: str):
    """
    Generic endpoint for cross-domain forecast retrieval.
    Consult /forecast/metadata for available indicator strings.
    """
    return get_forecasts(domain, indicator)

@forecast_router.get("/{domain}")
def get_domain_forecasts(domain: DomainLiteral):
    """Retrieve all forecasts for a specific domain."""
    return get_forecasts(domain)
