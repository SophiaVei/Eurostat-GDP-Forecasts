# API Operational Instructions

## 1. Environment Activation
The forecasting service requires a connection to a MongoDB instance. 

### Option A: Local Deployment
Ensure MongoDB is running locally, then execute the following from the root directory:
```powershell
python -m API.main
```

### Option B: Docker Deployment (Containerized)
The following command initializes both the API and a dedicated MongoDB container:
```powershell
docker-compose up --build
```

## 2. Standard Workflow

### Phase 1: Data Synchronization
Refreshes the historical database with the latest available data through the Skillscapes API.
```powershell
curl -X POST http://localhost:8000/data/update
```
Historical records are stored with the attribute `type: "history"`.

### Phase 2: Generating Forecasts
Runs inference using pre-trained models. This step populates the database with predictions for the next logical year.
```powershell
curl -X POST http://localhost:8000/forecast/run
```
Predictions are stored with the attribute `type: "forecast"`.

### Phase 3: Data Access
Forecasts can be retrieved via standard GET requests:
- **Domain Level**: `GET http://localhost:8000/forecast/labour`
- **Indicator Level**: `GET http://localhost:8000/forecast/labour/labour_force`

## 3. Database Schema Reference
The MongoDB `skillscapes` database contains domain-specific collections. A typical forecast document follows this structure:

```json
{
  "geo": "EL52",
  "type": "forecast",
  "indicator": "labour_force",
  "year": 2025,
  "value": 123.45,
  "model": "Ensemble",
  "run_at": "2026-01-15T..."
}
```
Records can be inspected directly through MongoDB Compass for manual verification.
