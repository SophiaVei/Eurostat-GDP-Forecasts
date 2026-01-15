# API Usage Instructions

## 1. Setup
Make sure you have MongoDB running locally. The API will connect to:
- **Database**: `skillscapes`
- **Collections**: `economy`, `labour`, `tourism`, `greek_tourism`

## 2. Running the API
Open a terminal in the root folder (`Eurostat-GDP-Forecasts`) and run:
```powershell
python -m API.main
```
You will see logs indicating the server started at `http://0.0.0.0:8000`.

## 3. Workflow Steps

### Step A: Update Data (Monthly)
This command triggers the "ETL" process:
1.  Downloads latest data from Eurostat.
2.  Saves raw historical data into the `skillscapes` database, organized by collection (e.g. `economy`).
3.  Each document is marked with `type: "history"`.

**Run Command (PowerShell):**
```powershell
curl -X POST http://localhost:8000/data/update
```
*(Or navigate to http://localhost:8000/docs and click "Try it out" on `/data/update`)*

### Step B: Forecast (Monthly)
This command triggers the prediction engine:
1.  Loads the pretrained models from `API/models/`.
2.  Reads the "history" data from MongoDB.
3.  Predicts the next year (e.g. 2025).
4.  Saves the results into the **same** collections, marked with `type: "forecast"`.

**Run Command (PowerShell):**
```powershell
curl -X POST http://localhost:8000/forecast/run
```

### Step C: View Results (API)
Instead of checking MongoDB directly, you can now view the results via the API:
- To see all labour forecasts: `http://localhost:8000/forecast/labour`
- To see a specific indicator: `http://localhost:8000/forecast/labour/labour_force`

**Run Command (PowerShell):**
```powershell
curl http://localhost:8000/forecast/labour
```

## 4. Viewing Results (MongoDB Compass)
1.  Open **MongoDB Compass**.
2.  Connect to `local` (default).
3.  Find the database named **`skillscapes`**.
4.  Click on a collection, for example **`labour`**.
5.  You will see documents. You can filter them:
    - To see history: `{ "type": "history" }`
    - To see forecasts: `{ "type": "forecast" }`
6.  A forecast document looks like:
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
