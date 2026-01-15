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
The recommended deployment method is via Docker, which orchestrates both the API and the MongoDB container:
```bash
# Execute from the project root directory
docker-compose up --build
```
The service will be accessible at `http://localhost:8000`.

## 4. Operation Lifecycle
Operations are designed for periodic execution (e.g., monthly):
1. **Data Ingestion**: `POST /data/update` - Refreshes MongoDB with the latest Eurostat figures.
2. **Inference Execution**: `POST /forecast/run` - Generates predictions for the next logical year.
3. **Retrieval**: `GET /forecast/{domain}` - Retrieves generated data for consumption.

## 5. Maintenance
To update the underlying models after fresh experiments:
```bash
python src/export_models_to_api.py
```
This script retrains the selected "winner" models on the full dataset and refreshes the binary snapshots in `API/models/`.
