import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000"

def test_forecast_year(domain, indicator, nuts_code="EL30"):
    url = f"{BASE_URL}/forecast/{domain}/{indicator}"
    try:
        logger.info(f"Testing {domain}/{indicator} for {nuts_code}...")
        response = requests.get(url, params={"nuts_code": nuts_code}, verify=False)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch data: {response.text}")
            return
            
        data = response.json()
        if "data" not in data or not data["data"]:
            logger.warning("No data returned.")
            return

        for result in data["data"]:
            geo = result.get("geo")
            year = result.get("year")
            val = result.get("value")
            model = result.get("model")
            logger.info(f"Result for {geo}: Year={year}, Value={val}, Model={model}")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    print("Verifying Forecast Years...")
    # AL01 should be 2022 if last data is 2021
    test_forecast_year("economy", "gdp_eur_hab", "AL01")
    # EL30 might be 2024 or 2023
    test_forecast_year("economy", "gdp_eur_hab", "EL30")
