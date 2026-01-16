# API Operational Instructions (Stateless)

## 1. Environment Activation

### Option A: Local Deployment (No Docker)
Ensure you are in the `API/` directory:
```powershell
cd API
pip install -r requirements.txt
python main.py
```

### Option B: Docker Deployment
The API can be started as a standalone container:
```powershell
cd API
docker-compose up --build
```

## 2. Standard Workflow


### Step 1: Discover Available Models
Check the metadata endpoint to see which indicators are supported for each domain.
`GET http://localhost:8000/forecast/metadata`

### Step 2: Request a Forecast
Request a forecast for a specific indicator. The API will fetch the latest data from the Skillscapes API and generate a prediction for the next year on-the-fly.

- **Example**: `GET http://localhost:8000/forecast/economy/gdp_eur_hab`

### Step 3: Filter by Region (Optional)
You can restrict the forecast to a specific NUTS 2 area by adding the `nuts_code` parameter.

- **Example**: `GET http://localhost:8000/forecast/economy/gdp_eur_hab?nuts_code=EL52`

## 3. Performance & Latency
Since the API performs real-time data ingestion for every request:
- **Fetch Time**: The API calls the external Skillscapes service, which can take **5-15 seconds** depending on the volume of historical data.
- **Inference**: Once data is received, the prediction is generated in under **100ms** (except for heavy models like TimesFM).

## 4. Response Format
The API returns a JSON object containing the prediction metadata and an array of regional results:

```json
{
  "domain": "economy",
  "indicator": "gdp_eur_hab",
  "count": 1,
  "data": [
    {
      "geo": "EL52",
      "indicator": "gdp_eur_hab",
      "year": 2024,
      "value": 15600.25,
      "model": "Tournament_Winner",
      "run_at": "2026-01-16T..."
    }
  ]
}
```
