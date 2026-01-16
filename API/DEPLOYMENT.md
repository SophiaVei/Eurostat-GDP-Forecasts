# Standalone API Deployment Guide (Stateless)

This directory is a self-contained, stateless version of the Eurostat Forecasting API. It is designed for easy "plug-and-play" deployment on any machine.

## Technical Components
- **`models/`**: Binary pre-trained model snapshots.
- **`src_shared/`**: Real-time data transformation and fetching utilities.
- **`services/`**: On-demand inference orchestration.
- **`main.py`**: Stateless FastAPI application.

## 1. Handover Instructions
To share this API with a colleague, simply send them the contents of this `API/` folder. They do **not** need the rest of the repository.

## 2. Deployment via Docker (Recommended)

1.  **Start the Service**:
    Run from within the `API/` folder:
    ```bash
    docker compose up --build -d
    ```
2.  **Access the API**:
    Open `http://localhost:8000/docs` to see the interactive documentation.

## 3. Deployment without Docker

1.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run with Uvicorn**:
    ```bash
    python main.py
    ```

## 4. Key Deployment Features
- **Zero Config**: No database URI or credentials required.
- **Auto-Discovery**: Indicators and NUTS codes are automatically provided as dropdowns in the browser.
- **Cross-Platform**: Uses `pathlib` for hardware-agnostic pathing (runs on Windows, Linux, or MacOS).

## 5. External Dependencies
The API requires an outbound internet connection to access:
- `https://skillscapes.csd.auth.gr:22223` (Live historical data source).
