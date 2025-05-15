# filename: stuff/app/routers/usd_exposure.py

from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from typing import List, Optional
from gotrue.types import User

from ..dependencies import get_supabase_client, get_current_user
from ..schemas.usd_exposure import UsdExposureInputCreate, UsdExposureResult, UsdExposureResultBase

router = APIRouter(
    prefix="/usd-exposure",
    tags=["USD Exposure"],
)

TABLE_NAME = "usd_exposure_results"

def calculate_exposure(inputs: dict) -> float:
    """Helper function to calculate USD exposure.
    All input values are treated as absolute positive values.
    Formula: abs(Total Assets) - abs(Total Liabilities) - abs(Total Capital)
    """
    assets = abs(inputs.get('totalAssets', 0) or 0)
    liabilities = abs(inputs.get('totalLiabilities', 0) or 0)
    capital = abs(inputs.get('totalCapital', 0) or 0)
    return assets - liabilities - capital

@router.post("/", response_model=UsdExposureResult, status_code=status.HTTP_201_CREATED)
async def create_usd_exposure_entry(
    payload: UsdExposureInputCreate,
    current_user: User = Depends(get_current_user),
    db: Client = Depends(get_supabase_client),
):
    """
    Calculates USD exposure based on input and stores the entry for the current user.
    """
    input_values_for_calc = payload.dict(by_alias=True, exclude_none=True, exclude={"reporting_date"})
    if 'reporting_date' in input_values_for_calc:
         del input_values_for_calc['reporting_date']
         
    exposure = calculate_exposure(input_values_for_calc)

    db_payload = {
        "user_id": str(current_user.id),
        "reporting_date": payload.reporting_date.isoformat(),
        "usd_exposure": exposure,
        "input_total_assets": payload.input_total_assets,
        "input_total_liabilities": payload.input_total_liabilities,
        "input_total_capital": payload.input_total_capital,
    }
    db_payload_cleaned = {k: v for k, v in db_payload.items() if v is not None}

    try:
        response = db.table(TABLE_NAME).insert(db_payload_cleaned).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save USD exposure data: No data returned from insert.")
        
        created_record = response.data[0]
        return UsdExposureResult(**created_record)

    except Exception as e:
        print(f"Error inserting USD exposure data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save USD exposure data: {str(e)}")


@router.get("/latest", response_model=Optional[UsdExposureResult])
async def get_latest_usd_exposure_entry(
    current_user: User = Depends(get_current_user),
    db: Client = Depends(get_supabase_client),
):
    """
    Retrieves the latest USD exposure entry for the current user.
    """
    try:
        response = (
            db.table(TABLE_NAME)
            .select("*")
            .eq("user_id", str(current_user.id))
            .order("created_at", desc=True)
            .limit(1)
            .maybe_single()
            .execute()
        )
        
        if response.data:
            return UsdExposureResult(**response.data)
        return None
        
    except Exception as e:
        print(f"Error fetching latest USD exposure data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch latest USD exposure data.") 