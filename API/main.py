from fastapi import FastAPI
from contextlib import asynccontextmanager
from routers import forecast_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # No database or scheduler needed in stateless mode
    yield

app = FastAPI(title="Eurostat Forecasting API", lifespan=lifespan)

app.include_router(forecast_router)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Eurostat Forecasting API (Stateless) is ready."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
