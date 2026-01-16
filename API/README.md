# Eurostat Forecast API Service (Stateless)

This directory contains a production-ready, stateless FastAPI service for generating on-demand Eurostat forecasts.

## 1. Overview
The service generates future-looking forecasts for four domains: **Economy, Labour, Tourism, and Greek Tourism**. 

Unlike a traditional database-backed service, this API is **purely stateless**:
- It fetches real-time historical data from the Skillscapes API for every request.
- It performs in-memory preprocessing and inference using pre-trained `.pkl` models.
- It returns results immediately without requiring a local database (No MongoDB needed).

## 2. Infrastructure
- **FastAPI**: Application entry point in `main.py`.
- **Pre-trained Models**: Binary `.pkl` snapshots stored in `models/`.
- **In-Memory Logic**: Specialized logic for data ingestion and transformation in `src_shared/`.
- **Stateless Inference**: Handled via `services/forecast_service.py`.

## 3. Quick Start

### Docker Deployment (Recommended)
```bash
cd API
docker compose up --build
```
The service will be accessible at `http://localhost:8000/docs`.

### Local Deployment
```bash
cd API
pip install -r requirements.txt
python main.py
```

## 4. Usage & Endpoints

### Metadata & Discovery
- **GET `/forecast/metadata`**: Lists all available domains and their corresponding indicators.

### On-Demand Forecasting
Use the domain-specific routes to generate forecasts. These endpoints feature **dropdown menus** in the Swagger UI for both **Indicators** and **NUTS 2 Codes**:
- **GET `/forecast/economy/{indicator}`**
- **GET `/forecast/labour/{indicator}`**
- **GET `/forecast/tourism/{indicator}`**
- **GET `/forecast/greek_tourism/{indicator}`**

**Optional Parameter**: Add `?nuts_code=EL52` to any request to filter results for a specific region.

## 5. Maintenance
To update the underlying models (requires the full repository):
```bash
# From project root
python src/export_models_to_api.py
```
This script retrains the winner models and refreshes the binary snapshots in `API/models/`.
