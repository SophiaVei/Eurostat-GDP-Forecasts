from fastapi import APIRouter
from .services.etl_service import update_datasets
from .services.forecast_service import run_forecasts

data_router = APIRouter(prefix="/data", tags=["Data"])
forecast_router = APIRouter(prefix="/forecast", tags=["Forecast"])

@data_router.post("/update")
def trigger_update_datasets():
    return update_datasets()

@forecast_router.post("/run")
def trigger_forecast():
    return run_forecasts()

@forecast_router.get("/{domain}")
def get_domain_forecasts(domain: str):
    """Get all forecasts for a specific domain (economy, labour, tourism, greek_tourism)."""
    from .services.forecast_service import get_forecasts
    return get_forecasts(domain)

@forecast_router.get("/{domain}/{indicator}")
def get_indicator_forecast(domain: str, indicator: str):
    """Get forecasts for a specific indicator within a domain."""
    from .services.forecast_service import get_forecasts
    return get_forecasts(domain, indicator)
