# Eurostat Forecast API

This folder contains the FastAPI service for forecasting Eurostat indicators using pretrained models and MongoDB.

## 1. Setup
Ensure you have `mongodb` installed and running.
Install dependencies:
```bash
pip install fastapi uvicorn pymongo pandas numpy joblib scikit-learn xgboost
```
(TimesFM/PyTorch optional but recommended if models use it)

## 2. Directory Structure
- `models/`: Contains the `.pkl` models exported from your experiments.
- `services/`: Business logic for ETL (Download) and Forecasting.
- `routers.py`: API Endpoints.
- `main.py`: App entry point.
- `database.py`: MongoDB connection.

## 3. Running the API
Start the server:
```bash
# Run from root directory
python -m API.main
```

## 3. Deployment with Docker (Easiest)
If you have Docker installed, you can start everything (including MongoDB) with a single command:
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

## 4. Usage
### Step 1: Update Data (Monthly)
Triggers a download of the latest data from Eurostat and saves it to MongoDB.
```bash
curl -X POST http://localhost:8000/data/update
```

### Step 2: Run Forecasts
Uses the models in `API/models/` to predict the **Next Year** based on the data in MongoDB.
```bash
curl -X POST http://localhost:8000/forecast/run
```

### Step 3: View Forecasts
Retrieve the generated forecasts for each domain following your update.
```bash
# Get all economy forecasts
curl http://localhost:8000/forecast/economy

# Get specific labour indicator
curl http://localhost:8000/forecast/labour/labour_force
```

## 5. Model Management
To update the models (e.g., if you ran new experiments), run:
```bash
python src/export_models_to_api.py
```
This will re-train the winners on the full dataset and save new pickles to `API/models/`.
