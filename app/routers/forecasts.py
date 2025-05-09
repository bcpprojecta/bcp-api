from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
import uuid

from ..dependencies import get_supabase_client
from ..security import get_current_active_user
from ..schemas.auth_schemas import User
from ..core.forecasting import generate_forecast_for_user # Assuming this is an async function
from supabase import Client

router = APIRouter(
    prefix="/forecasts",
    tags=["Forecasts"],
    responses={404: {"description": "Not found"}},
)

class ForecastRequest(BaseModel):
    currency: str = Field(..., example="CAD")
    forecast_anchor_date: Optional[date] = Field(None, description="Date to start forecasting from. Defaults to today if None.")

class ForecastJobResponse(BaseModel):
    job_id: uuid.UUID
    message: str
    status: str
    currency: str
    requested_anchor_date: Optional[date] = None

@router.post("/", response_model=ForecastJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_forecast_job(
    request_data: ForecastRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client)
):
    """
    Creates a new forecast job and queues it for processing.
    """
    job_id = uuid.uuid4()
    user_id = current_user.id
    created_at = datetime.utcnow()

    try:
        # 1. Insert job into forecast_jobs table
        job_data = {
            "id": str(job_id),
            "user_id": str(user_id),
            "status": "queued",
            "currency": request_data.currency.upper(),
            "requested_anchor_date": request_data.forecast_anchor_date.isoformat() if request_data.forecast_anchor_date else None,
            "created_at": created_at.isoformat(),
            "error_message": None,
            "started_at": None,
            "completed_at": None
        }
        insert_response = db.table("forecast_jobs").insert(job_data).execute()

        if hasattr(insert_response, 'error') and insert_response.error:
            print(f"Error inserting forecast job: {insert_response.error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not create forecast job in database: {insert_response.error.message if hasattr(insert_response.error, 'message') else insert_response.error}"
            )
        if not insert_response.data:
             print(f"Warning: Insert into forecast_jobs for job {job_id} did not return data. Assuming success if no error.")
        
        # 2. Add the forecast generation to background tasks
        background_tasks.add_task(
            generate_forecast_for_user,
            db=db,
            user_id=user_id, # Pass UUID object
            job_id=job_id,   # Pass UUID object
            currency=request_data.currency.upper(),
            forecast_anchor_date=request_data.forecast_anchor_date # Pass date object or None
            # training_window_days and forecast_horizon_days will use defaults in generate_forecast_for_user
        )

        return ForecastJobResponse(
            job_id=job_id,
            message="Forecast job created and queued for processing.",
            status="queued",
            currency=request_data.currency.upper(),
            requested_anchor_date=request_data.forecast_anchor_date
        )

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTPException
    except Exception as e:
        print(f"An unexpected error occurred while creating forecast job: {e}")
        # Attempt to update job status to failed if it was inserted
        try:
            db.table("forecast_jobs").update({
                "status": "failed",
                "error_message": f"Failed during job creation: {type(e).__name__} - {str(e)}",
                "completed_at": datetime.utcnow().isoformat()
            }).eq("id", str(job_id)).execute()
        except Exception as update_e:
            print(f"CRITICAL: Failed to update job status to failed during error handling: {update_e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

class ForecastJobStatusResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    status: str
    currency: str
    requested_anchor_date: Optional[date] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@router.get("/jobs/{job_id}", response_model=ForecastJobStatusResponse)
async def get_forecast_job_status(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client)
):
    """
    Retrieves the status and details of a specific forecast job.
    Ensures the job belongs to the current authenticated user.
    """
    try:
        response = (
            db.table("forecast_jobs")
            .select("*") 
            .eq("id", str(job_id))
            .eq("user_id", str(current_user.id))
            .maybe_single()
            .execute()
        )

        if response.data:
            return ForecastJobStatusResponse(**response.data)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Forecast job with ID '{job_id}' not found or does not belong to the current user."
            )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error fetching forecast job status for job_id {job_id}: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while fetching job status: {str(e)}"
        )

class ForecastResultRow(BaseModel):
    job_id: uuid.UUID
    user_id: uuid.UUID
    forecast_run_timestamp: datetime
    Currency: str 
    Date: date 
    Forecasted_Amount: Optional[float] = Field(None, alias="Forecasted Amount")
    Forecasted_CashBalance: Optional[float] = Field(None, alias="Forecasted CashBalance")
    actual_cash_balance: Optional[float] = None

    class Config:
        populate_by_name = True

@router.get("/results/{job_id}", response_model=List[ForecastResultRow])
async def get_forecast_results(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client)
):
    """
    Retrieves the forecast results for a specific completed job.
    Ensures the job belongs to the current authenticated user.
    """
    try:
        job_response = (
            db.table("forecast_jobs")
            .select("id, user_id, status")
            .eq("id", str(job_id))
            .eq("user_id", str(current_user.id))
            .maybe_single()
            .execute()
        )

        if not job_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Forecast job with ID '{job_id}' not found or does not belong to the current user."
            )
        
        results_response = (
            db.table("forecast_results")
            .select("*")
            .eq("job_id", str(job_id))
            .order("Date", desc=False) 
            .execute()
        )

        if results_response.data:
            parsed_results = [ForecastResultRow.model_validate(row) for row in results_response.data]
            return parsed_results
        else:
            return [] 

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error fetching forecast results for job_id {job_id}: {type(e).__name__} - {str(e)}")
        # traceback.print_exc() # Uncomment for full traceback during debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while fetching job results: {str(e)}"
        )

@router.get("/jobs/", response_model=List[ForecastJobStatusResponse])
async def list_forecast_jobs_for_user(
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client),
    skip: int = 0,
    limit: int = 100
):
    """
    Retrieves a list of forecast jobs for the current authenticated user.
    """
    try:
        response = (
            db.table("forecast_jobs")
            .select("*")
            .eq("user_id", str(current_user.id))
            .order("created_at", desc=True)
            .offset(skip)
            .limit(limit)
            .execute()
        )
        
        if response.data is not None:
            return [ForecastJobStatusResponse(**job_data) for job_data in response.data]
        return []
    except Exception as e:
        print(f"Error listing forecast jobs for user {current_user.id}: {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while listing jobs: {str(e)}"
        )

# TODO: Add an endpoint to get forecast results by job_id (once completed) 