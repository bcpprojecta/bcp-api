# filename: stuff/app/schemas/liquidity_ratio.py

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime # Import datetime
import uuid

# --- Input Schemas ---

class LiquidityRatioInputBase(BaseModel):
    """Input fields needed for calculation (matching table columns)."""
    input_cash: Optional[float] = Field(None, alias='1010')
    input_mlp_hqla1_govt: Optional[float] = Field(None, alias='1040')
    input_sdebt_instruments: Optional[float] = Field(None, alias='1060')
    input_mlp_hqla1_cmb: Optional[float] = Field(None, alias='1078')
    input_mlp_hqla2b: Optional[float] = Field(None, alias='1079')
    input_total_assets: Optional[float] = Field(None, alias='1100') # Note: Frontend used this for core_cash_codes, double check logic
    input_borrowings: Optional[float] = Field(None, alias='2050')
    input_member_deposits: Optional[float] = Field(None, alias='2180')
    input_code_2255: Optional[float] = Field(None, alias='2255')
    input_code_2295: Optional[float] = Field(None, alias='2295')

    class Config:
        allow_population_by_field_name = True # Allow using aliases like '1010' from frontend
        orm_mode = False # This is for input, not mapping from DB object

class LiquidityRatioInputCreate(LiquidityRatioInputBase):
    """Data needed when creating a new record."""
    reporting_date: date

# --- Output Schemas ---

class LiquidityRatioResultBase(LiquidityRatioInputBase): # Include input fields in output if desired
    """Base schema for representing results, including calculated ratios."""
    reporting_date: date
    statutory_ratio: Optional[float] = None
    core_ratio: Optional[float] = None
    total_ratio: Optional[float] = None

class LiquidityRatioResult(LiquidityRatioResultBase):
    """Full result schema including database fields."""
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime

    class Config:
        orm_mode = True # Allow mapping from DB object 