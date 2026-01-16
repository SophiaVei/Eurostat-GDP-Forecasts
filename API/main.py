from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import db
from routers import data_router, forecast_router

from apscheduler.schedulers.background import BackgroundScheduler
from services.etl_service import update_datasets
from services.forecast_service import run_forecasts

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.connect()
    
    # Setup Background Scheduler for monthly updates
    scheduler = BackgroundScheduler()
    # Schedule for the 1st of every month at midnight
    scheduler.add_job(update_datasets, 'cron', day=1, hour=0, minute=0)
    scheduler.add_job(run_forecasts, 'cron', day=1, hour=0, minute=10) # 10 mins after ETL starts
    scheduler.start()
    
    yield
    scheduler.shutdown()
    db.close()

app = FastAPI(title="Eurostat Forecasting API", lifespan=lifespan)

app.include_router(data_router)
app.include_router(forecast_router)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Eurostat Forecasting API is ready."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
