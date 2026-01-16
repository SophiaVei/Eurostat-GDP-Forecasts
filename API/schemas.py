from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ForecastResult(BaseModel):
    geo: str
    indicator: str
    year: int
    value: float
    model: str
    run_at: str

class ForecastResponse(BaseModel):
    domain: str
    indicator: str
    count: int
    data: List[ForecastResult]

class ErrorResponse(BaseModel):
    error: str
