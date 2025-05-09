from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
import uuid # Import uuid module

class SummaryItemBase(BaseModel):
    # Fields from your latest list
    previous_balance: Optional[float] = Field(default=None, alias="Previous Balance")
    opening_balance: Optional[float] = Field(default=None, alias="Opening Balance")
    net_activity: Optional[float] = Field(default=None, alias="Net Activity")
    # We will map "Closing Balance" from DB to closing_balance_usd for frontend consistency
    closing_balance_usd: Optional[float] = Field(default=None, alias="Closing Balance") 
    reporting_date: Optional[date] = Field(default=None, alias="Reporting Date")
    currency: Optional[str] = Field(default=None, alias="currency")
    
    user_id: Optional[uuid.UUID] = Field(default=None, alias="user_id") 
    upload_metadata_id: Optional[uuid.UUID] = Field(default=None, alias="upload_metadata_id")

    class Config:
        from_attributes = True
        populate_by_name = True
        # Pydantic v2 can automatically convert UUID to string if needed during serialization to JSON
        # However, FastAPI handles this correctly by default for response_model

class SummaryItem(SummaryItemBase):
    id: int = Field(..., alias="id") # id is an int and required
    created_at: Optional[datetime] = Field(default=None, alias="created_at")

# No computed fields needed now as we have direct mappings for Reporting Date and Closing Balance

class SummaryItemCreate(SummaryItemBase):
    # Potentially, if you were to create items via API
    pass

# You might also want a model for what's returned by the API,
# which might be slightly different if you transform data.
# For now, SummaryItem is used for response_model. 