# Standalone API Deployment Guide

This directory is a self-contained version of the Eurostat Forecasting API, designed for easy deployment without the parent repository.

## Directory Structure
- `data/`: Contains raw Eurostat CSV files used for historical data seeding.
- `models/`: Contains pre-trained `.pkl` models for the 4 domains.
- `src_shared/`: Core logic for data processing, downloading, and forecasting.
- `main.py`: API entry point.

## Deployment via Docker (Recommended)

1.  **Build and Start**:
    Run the following command from within this `API/` folder:
    ```bash
    docker compose up --build -d
    ```
2.  **Verify**:
    Open `http://localhost:8000/docs` to see the Swagger UI.

## Manual Setup (Alternative)

1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Variables**:
    - `MONGO_URI`: URI for your MongoDB instance (default: `mongodb://localhost:27017`)
    - `DB_NAME`: Database name (default: `skillscapes`)
3.  **Run API**:
    ```bash
    python main.py
    ```

## Post-Deployment Workflow
1.  **Sync Data**: Execute **POST `/data/update`**. This fetches live values from Eurostat and populates the database.
2.  **Generate Forecasts**: Execute **POST `/forecast/run`**. This runs the pre-trained models against the synced data.
3.  **Retrieve Results**: Use the **GET `/forecast/{domain}`** or **GET `/forecast/{domain}/{indicator}`** endpoints.
