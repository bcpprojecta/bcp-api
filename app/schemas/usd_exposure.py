# filename: stuff/app/schemas/usd_exposure.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
import uuid

# --- Input Schemas ---

class UsdExposureInputBase(BaseModel):
    """Input fields needed for calculation."""
    input_total_assets: Optional[float] = Field(None, alias='totalAssets')
    input_total_liabilities: Optional[float] = Field(None, alias='totalLiabilities')
    input_total_capital: Optional[float] = Field(None, alias='totalCapital')

    class Config:
        allow_population_by_field_name = True
        orm_mode = False

class UsdExposureInputCreate(UsdExposureInputBase):
    """Data needed when creating a new record."""
    reporting_date: date

# --- Output Schemas ---

class UsdExposureResultBase(UsdExposureInputBase): # Include inputs if desired
    """Base schema for representing results."""
    reporting_date: date
    usd_exposure: float # Exposure is likely always calculable

class UsdExposureResult(UsdExposureResultBase):
    """Full result schema including database fields."""
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime

    class Config:
        orm_mode = True 