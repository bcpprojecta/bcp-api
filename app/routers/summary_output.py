from fastapi import APIRouter, HTTPException, Depends
# supabase_py might be an older or different library than the official 'supabase'
# Ensure you have the correct one installed. 'supabase' is the current official one.
# from supabase_py import create_client, Client # type: ignore 
from supabase import create_client, Client # Official library
import os
from typing import List, Optional
from ..schemas.summary_item import SummaryItem # Updated import
from pydantic import BaseModel # Ensure BaseModel is imported if used directly in this file, though SummaryItem handles it
from ..dependencies import get_supabase_client, get_current_user # Import get_current_user
from gotrue.types import User # Import User type for Depends annotation

# --- Supabase Configuration ---
# It's highly recommended to use environment variables for sensitive data
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Use service key for backend operations

if not SUPABASE_URL or not SUPABASE_KEY:
    # This will prevent the app from starting if credentials are not set,
    # which is good for backend services.
    # For development, you might want to handle this differently or use mock data.
    print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables.")
    # raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables.")
    # For now, let's allow it to proceed so the app can start, but log an error.
    # In a real scenario, you'd likely want to raise an error or have a fallback.
    supabase_client: Optional[Client] = None
else:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        supabase_client = None

router = APIRouter(
    prefix="/summary-output", # Consistent with frontend page name
    tags=["USD Summary Output"],
)

# Dependency to get Supabase client
async def get_supabase_client() -> Client:
    if supabase_client is None:
        # This will be caught by FastAPI and return a 503 error
        # if Supabase client wasn't initialized.
        raise HTTPException(status_code=503, detail="Supabase client not initialized. Check server environment variables and logs.")
    return supabase_client

# Columns to select from Supabase, matching Pydantic aliases or target field names if no alias
# Make sure these are the EXACT column names in your Supabase table.
COLUMNS_TO_SELECT = (
    "id, user_id, upload_metadata_id, "
    "\"Previous Balance\", \"Opening Balance\", \"Net Activity\", "
    "\"Closing Balance\", \"Reporting Date\", created_at, currency"
)

@router.get("/usd", response_model=List[SummaryItem])
async def get_usd_summary_data(
    skip: int = 0, 
    limit: int = 10, 
    db: Client = Depends(get_supabase_client),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve USD summary data from Supabase table "Summary_output(USD)" FOR THE CURRENT USER.
    Supports pagination using skip and limit.
    """
    if not current_user or not current_user.id:
        # This check might be redundant if get_current_user raises error, but belt-and-suspenders
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not authenticate user.")
        
    try:
        table_name = "Summary_output(USD)" # ADJUST IF NECESSARY
        
        # Fetch data from Supabase
        # ADD .eq() to filter by user_id
        query = (
            db.table(table_name)
            .select(COLUMNS_TO_SELECT)
            .eq("user_id", str(current_user.id)) # FILTER ADDED HERE - ensure user_id type matches
            .order("Reporting Date", desc=True)
            .offset(skip)
            .limit(limit)
        )
        response = query.execute()

        # The official supabase-py client returns data in response.data
        api_response_data = response.data

        if not isinstance(api_response_data, list):
            print(f"Unexpected data format from Supabase: {api_response_data}")
            raise HTTPException(status_code=500, detail="Unexpected data format from Supabase.")
        
        return api_response_data
    
    except Exception as e:
        print(f"Error fetching data from Supabase for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch USD summary data: {str(e)}")

# Example: How to use this router in your main.py
# from .routers import summary_output
# app.include_router(summary_output.router) 