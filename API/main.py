from fastapi import FastAPI
from contextlib import asynccontextmanager
from .database import db
from .routers import data_router, forecast_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.connect()
    yield
    db.close()

app = FastAPI(title="Eurostat Forecasting API", lifespan=lifespan)

app.include_router(data_router)
app.include_router(forecast_router)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Eurostat Forecasting API is ready."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API.main:app", host="0.0.0.0", port=8000, reload=True)
