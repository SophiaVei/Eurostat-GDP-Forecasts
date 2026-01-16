# Eurostat Forecast API Service

This directory contains the production-ready FastAPI service for served pre-trained Eurostat forecasting models.

## 1. Overview
The service interacts with a MongoDB instance to manage both regional history and future-looking forecasts for four domains: Economy, Labour, Tourism, and Greek Tourism.

## 2. Infrastructure
- **FastAPI**: Application entry point in `main.py`.
- **Database**: MongoDB connection management in `database.py`.
- **Pre-trained Models**: Binary `.pkl` snapshots stored in `models/`.
- **Business Logic**: ETL and Inference services in the `services/` directory.

## 3. Deployment
The service is now strictly standalone. You can run it in two ways:

### Option A: Via Project Root (Shortcut)
```bash
# Execute from the project root directory
docker-compose up --build
```

### Option B: Standalone Handover (Deployment)
If you move only this `API/` folder to a new server:
```bash
cd API
docker compose up --build
```
The service will be accessible at `http://localhost:8000`.

### 3. Usage & Endpoints
The following endpoints are available for data management and forecast retrieval:

#### Metadata & Discovery
- **GET `/forecast/metadata`**: Lists all available domains and their corresponding indicators.
- **Domain-Specific Discovery**: Use specialized routes like `/forecast/economy/{indicator}` or `/forecast/labour/{indicator}`. These routes provide **pre-filled dropdown menus** in the Swagger UI for all available indicators within that domain.

#### Data ETL
- **POST `/data/update`**: Refreshes MongoDB with the latest Eurostat figures.
- **POST `/forecast/run`**: Generates predictions for the next logical year.

#### Forecast Retrieval
- **GET `/forecast/economy/{indicator}`**: Returns forecasts for a specific economy indicator (with dropdown).
- **GET `/forecast/labour/{indicator}`**: Returns forecasts for a specific labour indicator (with dropdown).
- **GET `/forecast/tourism/{indicator}`**: Returns forecasts for a specific tourism indicator (with dropdown).
- **GET `/forecast/greek_tourism/{indicator}`**: Returns forecasts for a specific greek_tourism indicator (with dropdown).
- **GET `/forecast/{domain}`**: Returns all forecasts for a specific domain.
- **GET `/forecast/{domain}/{indicator}`**: Generic endpoint for all indicators.

## 4. Operation Lifecycle
Operations are designed for periodic execution (e.g., monthly):
1. **Data Ingestion**: `POST /data/update` - Refreshes MongoDB with the latest Eurostat figures.
2. **Inference Execution**: `POST /forecast/run` - Generates predictions for the next logical year.
3. **Retrieval**: `GET /forecast/{domain}` - Retrieves generated data for consumption.

To update the models (requires the full repository):
```bash
# From project root
python src/export_models_to_api.py
```
This script retrains selected models and refreshes the binary snapshots in `API/models/`.
