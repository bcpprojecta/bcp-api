# filename: stuff/app/routers/liquidity_ratios.py

from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from typing import List, Optional
from gotrue.types import User # For current_user type hint

from ..dependencies import get_supabase_client, get_current_user # Assuming get_supabase_client gives a user-context client
from ..schemas.liquidity_ratio import LiquidityRatioInputCreate, LiquidityRatioResult, LiquidityRatioResultBase

router = APIRouter(
    prefix="/liquidity-ratios",
    tags=["Liquidity Ratios"],
)

TABLE_NAME = "liquidity_ratio_results"

def calculate_ratios(inputs: dict) -> dict:
    """
    Helper function to calculate ratios based on input values.
    This logic should mirror what's in the frontend if possible,
    or be the authoritative calculation.
    """
    values = {k: (v if v is not None else 0) for k, v in inputs.items()}

    statutory_liquidity_codes = ['1010', '1040', '1060', '1078', '1079']
    deposit_and_debt_codes = ['2180', '2050', '2255', '2295']
    # In frontend, '1100' (Total Assets) was used for core_cash_codes.
    # This might be a direct input or a sum. If '1100' is 'input_total_assets'
    # then that should be used. Let's assume 'input_total_assets' is the correct field name.
    core_cash_source_code = '1100' # Alias for input_total_assets
    borrowings_codes = ['2050']
    member_deposits_codes = ['2180']

    sum_values = lambda codes: sum(values.get(code, 0) for code in codes)

    liquidity_available = sum_values(statutory_liquidity_codes)
    deposits_and_debt = sum_values(deposit_and_debt_codes)
    
    # Use the alias for 'input_total_assets' if that's what '1100' represents
    total_cash_liquid = values.get(core_cash_source_code, 0)
    
    borrowings = sum_values(borrowings_codes)
    member_deposits = sum_values(member_deposits_codes)

    statutory_ratio = (liquidity_available / deposits_and_debt) if deposits_and_debt != 0 else 0
    core_ratio = ((total_cash_liquid - borrowings) / member_deposits) if member_deposits != 0 else 0
    total_ratio = (total_cash_liquid / member_deposits) if member_deposits != 0 else 0
    
    return {
        "statutory_ratio": statutory_ratio,
        "core_ratio": core_ratio,
        "total_ratio": total_ratio,
    }

@router.post("/", response_model=LiquidityRatioResult, status_code=status.HTTP_201_CREATED)
async def create_liquidity_ratio_entry(
    payload: LiquidityRatioInputCreate,
    current_user: User = Depends(get_current_user),
    db: Client = Depends(get_supabase_client), # Using user-context client
):
    """
    Calculates liquidity ratios based on input and stores the entry for the current user.
    """
    input_values_for_calc = payload.dict(by_alias=True, exclude_none=True, exclude={"reporting_date"})
    # Remove 'reporting_date' if it's part of input_values_for_calc from .dict() and not meant for calculation input
    if 'reporting_date' in input_values_for_calc:
        del input_values_for_calc['reporting_date']
        
    calculated = calculate_ratios(input_values_for_calc)

    # Prepare data for Supabase insertion (column names should match DB, not aliases)
    db_payload = {
        "user_id": str(current_user.id),
        "reporting_date": payload.reporting_date.isoformat(),
        "statutory_ratio": calculated["statutory_ratio"],
        "core_ratio": calculated["core_ratio"],
        "total_ratio": calculated["total_ratio"],
        # Map input fields from schema to DB column names
        "input_cash": payload.input_cash,
        "input_mlp_hqla1_govt": payload.input_mlp_hqla1_govt,
        "input_sdebt_instruments": payload.input_sdebt_instruments,
        "input_mlp_hqla1_cmb": payload.input_mlp_hqla1_cmb,
        "input_mlp_hqla2b": payload.input_mlp_hqla2b,
        "input_total_assets": payload.input_total_assets,
        "input_borrowings": payload.input_borrowings,
        "input_member_deposits": payload.input_member_deposits,
        "input_code_2255": payload.input_code_2255,
        "input_code_2295": payload.input_code_2295,
    }
    # Remove None values so they don't overwrite DB defaults or existing NULLs if that's intended
    db_payload_cleaned = {k: v for k, v in db_payload.items() if v is not None}


    try:
        response = db.table(TABLE_NAME).insert(db_payload_cleaned).execute()
        if not response.data:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save liquidity ratio data: No data returned from insert.")
        
        # The response.data from insert usually contains a list with the inserted record(s)
        created_record = response.data[0]
        return LiquidityRatioResult(**created_record) # Validate and return using Pydantic model

    except Exception as e:
        print(f"Error inserting liquidity ratio data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save liquidity ratio data: {str(e)}")


@router.get("/latest", response_model=Optional[LiquidityRatioResult])
async def get_latest_liquidity_ratio_entry(
    current_user: User = Depends(get_current_user),
    db: Client = Depends(get_supabase_client),
):
    """
    Retrieves the latest liquidity ratio entry for the current user.
    """
    try:
        response = (
            db.table(TABLE_NAME)
            .select("*") # Select all columns defined in LiquidityRatioResult
            .eq("user_id", str(current_user.id))
            .order("created_at", desc=True) # Order by creation time to get the latest
            .limit(1)
            .maybe_single() # Returns one record or None, doesn't raise error if not found
            .execute()
        )
        
        if response.data:
            return LiquidityRatioResult(**response.data)
        return None # No record found for the user
        
    except Exception as e:
        print(f"Error fetching latest liquidity ratio data: {e}")
        # It's better not to expose raw error messages for GET requests if they might contain sensitive info
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch latest liquidity ratio data.") 