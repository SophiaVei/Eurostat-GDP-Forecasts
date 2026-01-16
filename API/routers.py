from fastapi import APIRouter, Query
from typing import Literal, Optional
from services.forecast_service import get_forecasts, get_api_metadata
from constants import (
    DomainLiteral, EconomyIndicator, LabourIndicator, TourismIndicator, GreekTourismIndicator,
    EconomyNutsCode, LabourNutsCode, TourismNutsCode, GreekTourismNutsCode
)
from schemas import ForecastResponse

forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@forecast_router.get("/metadata")
def get_metadata():
    """Returns a list of all available domains and their corresponding indicators."""
    return get_api_metadata()

# Domain-Specific Routes with specific Indicator and NUTS dropdowns
@forecast_router.get("/economy/{indicator}", response_model=ForecastResponse)
def get_economy_forecast(
    indicator: EconomyIndicator, 
    nuts_code: Optional[EconomyNutsCode] = Query(None, description="Optional NUTS 2 code to filter results.")
):
    """Retrieve forecasts for specific Economy indicators."""
    return get_forecasts("economy", indicator, nuts_code)

@forecast_router.get("/labour/{indicator}", response_model=ForecastResponse)
def get_labour_forecast(
    indicator: LabourIndicator, 
    nuts_code: Optional[LabourNutsCode] = Query(None, description="Optional NUTS 2 code to filter results.")
):
    """Retrieve forecasts for specific Labour indicators."""
    return get_forecasts("labour", indicator, nuts_code)

@forecast_router.get("/tourism/{indicator}", response_model=ForecastResponse)
def get_tourism_forecast(
    indicator: TourismIndicator, 
    nuts_code: Optional[TourismNutsCode] = Query(None, description="Optional NUTS 2 code to filter results.")
):
    """Retrieve forecasts for specific Tourism indicators."""
    return get_forecasts("tourism", indicator, nuts_code)

@forecast_router.get("/greek_tourism/{indicator}", response_model=ForecastResponse)
def get_greek_tourism_forecast(
    indicator: GreekTourismIndicator, 
    nuts_code: Optional[GreekTourismNutsCode] = Query(None, description="Optional Greek NUTS 2 code to filter results.")
):
    """Retrieve forecasts for specific Greek Tourism indicators."""
    return get_forecasts("greek_tourism", indicator, nuts_code)
