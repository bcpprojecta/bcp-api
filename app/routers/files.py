from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Query
from pydantic import BaseModel, Field
from supabase import Client
from postgrest import APIError
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date
from io import BytesIO, StringIO
import pandas as pd # Ensure pandas is imported
import uuid # For generating UUIDs or handling them if needed for new records

from ..dependencies import get_supabase_client, get_supabase_service_client # อาจจะยังไม่ต้องการ service client ที่นี่
from ..schemas.auth_schemas import User # schema สำหรับ User object
from ..security import get_current_active_user # Dependency สำหรับเอา user ปัจจุบัน
from ..schemas.file_schemas import UploadedFileMetadataItem, FileTypeEnum # Import FileTypeEnum

# Import parser functions from file_processing.py
from ..core import file_processing
# Import the new .041 parser
from ..core.raw_transaction_parser import process_041_content
from ..core import forecasting # Correct module for generate_forecast_for_user
from ..core import raw_transaction_parser # Ensure this is used if needed

router = APIRouter(
    prefix="/files",
    tags=["File Management"]
)

# อาจจะสร้าง schema สำหรับ response การอัปโหลด
class FileUploadResponse(BaseModel):
    message: str
    file_id: str # UUID of the record in uploaded_files_metadata
    original_filename: str
    storage_path: str
    content_type: Optional[str] = None
    processing_status: str # Add processing status to response
    processing_message: Optional[str] = None # Add potential message from processing

# Pydantic model for a single row of forecast data
class ForecastDataRow(BaseModel):
    Date: date
    forecasted_amount: Optional[float] = Field(default=None, alias="Forecasted Amount")
    forecasted_cash_balance: Optional[float] = Field(default=None, alias="Forecasted Cash Balance")
    actual_cash_balance: Optional[float] = Field(default=None, alias="Actual Cash Balance")

    class Config:
        # Pydantic V2 style config
        populate_by_name = True # Replaces allow_population_by_field_name and handles aliases correctly
        from_attributes = True # Replaces orm_mode

# Helper function to insert parsed data into a Supabase table
async def insert_parsed_data(
    db: Client, 
    table_name: str, # This will be the new Supabase table name like "Summary_output", "Output"
    parsed_df: pd.DataFrame, 
    user_id: uuid.UUID, 
    file_metadata_id: uuid.UUID # This is the PK from uploaded_files_metadata
):
    if parsed_df is None or parsed_df.empty:
        print(f"No data to insert into {table_name}.")
        return
    
    df_to_insert = parsed_df.copy()
    df_to_insert['user_id'] = user_id
    df_to_insert['upload_metadata_id'] = file_metadata_id

    # Standardize to lowercase 'currency' if it exists, for consistent logging/checking
    # This does not affect the actual database insertion if db schema uses 'currency' (lowercase)
    # and the specific df (summary_df or transaction_df) has been prepared with 'currency' (lowercase) before calling this function.
    cols_to_check_for_logging = df_to_insert.columns.tolist()
    if 'Currency' in cols_to_check_for_logging and 'currency' not in cols_to_check_for_logging:
        # This case should ideally be handled before calling insert_parsed_data
        # by ensuring parsed_df has lowercase 'currency'.
        # For logging purposes here, we might temporarily use it if present.
        currency_col_for_logging = 'Currency' 
    elif 'currency' in cols_to_check_for_logging:
        currency_col_for_logging = 'currency'
    else:
        currency_col_for_logging = None

    if currency_col_for_logging:
        print(f"[insert_parsed_data for {table_name}] '{currency_col_for_logging}' column exists. Sample values: {df_to_insert[currency_col_for_logging].unique().tolist()[:5]}")
    else:
        # Only critical if the target table actually requires a currency column (e.g. Output, Summary_output)
        if table_name in ["Output", "Summary_output"]:
             print(f"⚠️ [insert_parsed_data for {table_name}] Neither 'Currency' nor 'currency' column found for logging.")

    # Convert datetime columns to ISO format string, Supabase prefers this
    for col in df_to_insert.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, Asia/Bangkok]']).columns:
        # Ensure timezone is handled or stripped if Supabase expects naive UTC or specific format
        if df_to_insert[col].dt.tz is not None:
            df_to_insert[col] = df_to_insert[col].dt.tz_convert('UTC') # Convert to UTC
            df_to_insert[col] = df_to_insert[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%f+00:00') # Explicit UTC offset
        else:
            df_to_insert[col] = df_to_insert[col].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ') # Naive, assume UTC by Supabase

    if 'user_id' in df_to_insert.columns:
        df_to_insert['user_id'] = df_to_insert['user_id'].astype(str)
    if 'upload_metadata_id' in df_to_insert.columns:
        df_to_insert['upload_metadata_id'] = df_to_insert['upload_metadata_id'].astype(str)

    df_to_insert = df_to_insert.astype(object).where(pd.notnull(df_to_insert), None)
    records_to_insert = df_to_insert.to_dict(orient='records')

    try:
        print(f"Attempting to insert {len(records_to_insert)} records into Supabase table: {table_name}. Sample: {records_to_insert[0] if records_to_insert else 'N/A'}")
        insert_response = db.table(table_name).insert(records_to_insert).execute()
        if hasattr(insert_response, 'data') and insert_response.data:
            print(f"Successfully inserted {len(insert_response.data)} records into {table_name}.")
        elif hasattr(insert_response, 'error') and insert_response.error:
            print(f"Error inserting data into {table_name}: {insert_response.error}")
            raise Exception(f"DB insert error in {table_name}: {insert_response.error.message if hasattr(insert_response.error, 'message') else insert_response.error}")
        else:
            if not (hasattr(insert_response, 'error') and insert_response.error):
                 print(f"Successfully inserted records into {table_name} (count from response may vary based on settings).")
            else:
                 print(f"Unknown issue inserting data into {table_name}. Response: {insert_response}")
                 raise Exception(f"Unknown DB insert error in {table_name}.")
    except Exception as e:
        print(f"❌ Exception during data insertion into {table_name}: {type(e).__name__} - {str(e)}")
        raise

async def process_file_content_and_store(
    db: Client,
    user_id: uuid.UUID,
    file_metadata_id: uuid.UUID,
    file_type: FileTypeEnum,
    currency: Optional[str],
    file_content_bytes: bytes,
    background_tasks: BackgroundTasks # For potential future use
):
    processing_status = "processing_failed"
    processing_message = "An unexpected error occurred during file processing."
    parsed_df = None
    target_table_name = None

    try:
        file_stream: Union[BytesIO, StringIO]
        is_excel = file_type.value.endswith('_template')

        # Prepare primary file stream (BytesIO for binary, StringIO for text)
        if is_excel or file_type in [FileTypeEnum.CAD_TRANSACTION_RAW, FileTypeEnum.USD_TRANSACTION_RAW]: # Ensure BytesIO for raw transaction parser too
            file_stream = BytesIO(file_content_bytes)
        else:
            try:
                text_content = file_content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                text_content = file_content_bytes.decode('latin-1')
            file_stream = StringIO(text_content)

        # --- Logic for processing .041 files (CAD_SUMMARY_RAW or USD_SUMMARY_RAW) ---
        if file_type == FileTypeEnum.CAD_SUMMARY_RAW or file_type == FileTypeEnum.USD_SUMMARY_RAW:
            summary_parsed_successfully = False
            transaction_parsed_successfully = False
            summary_target_table = "Summary_output" if file_type == FileTypeEnum.CAD_SUMMARY_RAW else "Summary_output(USD)"
            
            # 1. Parse and Insert Summary Data
            print(f"[process_file_content_and_store] Attempting to parse SUMMARY data for {file_type.value}")
            current_summary_stream: Union[StringIO, BytesIO]
            if isinstance(file_stream, BytesIO):
                try: text_content_for_summary = file_content_bytes.decode('utf-8')
                except UnicodeDecodeError: text_content_for_summary = file_content_bytes.decode('latin-1')
                current_summary_stream = StringIO(text_content_for_summary)
            else: 
                file_stream.seek(0)
                current_summary_stream = file_stream

            summary_df = file_processing.parse_summary_file_from_content(current_summary_stream)
            if summary_df is not None and not summary_df.empty:
                print(f"[process_file_content_and_store] SUMMARY data parsed. Shape: {summary_df.shape}, Columns: {summary_df.columns.tolist()}")
                if summary_target_table == "Summary_output":
                    if currency: summary_df['currency'] = currency.upper()
                    else: summary_df['currency'] = None 
                    print(f"[process_file_content_and_store] Added/updated 'currency' column in summary_df for {summary_target_table}. Value: {summary_df['currency'].unique().tolist()[:1]}")
                
                await delete_existing_data_by_metadata_id(db, summary_target_table, file_metadata_id)
                await insert_parsed_data(db, summary_target_table, summary_df, user_id, file_metadata_id)
                summary_parsed_successfully = True
                print(f"[process_file_content_and_store] SUMMARY data for {file_type.value} processed and stored in {summary_target_table}.")
            else:
                print(f"[process_file_content_and_store] SUMMARY data parsing failed or returned empty for {file_type.value}.")

            # 2. Parse and Insert Transaction Data (from the same .041 file)
            print(f"[process_file_content_and_store] Attempting to parse TRANSACTION data for {file_type.value} (for Output table)")
            current_transaction_stream = BytesIO(file_content_bytes) 
            
            transaction_df = raw_transaction_parser.process_041_content(
                file_content_stream=current_transaction_stream, 
                db_client=db, 
                currency_code=currency
            )
            if transaction_df is not None and not transaction_df.empty:
                print(f"[process_file_content_and_store] TRANSACTION data parsed. Shape: {transaction_df.shape}, Columns: {transaction_df.columns.tolist()}")
                if 'currency' not in transaction_df.columns:
                     print(f"CRITICAL ⚠️ [process_file_content_and_store] 'currency' column still MISSING from transaction_df after call to process_041_content for Output table.")
                if 'transaction_category' not in transaction_df.columns:
                    print(f"⚠️ [process_file_content_and_store] 'transaction_category' column not found in transaction_df. Will be NULL if not present in records.")

                output_target_table = "Output"
                await delete_existing_data_by_metadata_id(db, output_target_table, file_metadata_id)
                await insert_parsed_data(db, output_target_table, transaction_df, user_id, file_metadata_id)
                transaction_parsed_successfully = True
                print(f"[process_file_content_and_store] TRANSACTION data for {file_type.value} processed and stored in Output table.")
            else:
                print(f"[process_file_content_and_store] TRANSACTION data parsing failed or returned empty for {file_type.value}.")

            if summary_parsed_successfully and transaction_parsed_successfully:
                processing_status = "processed_successfully"
                processing_message = f"File ({file_type.value}) processed: Summary and Transaction data stored."
            elif summary_parsed_successfully:
                processing_status = "processed_summary_only"
                processing_message = f"File ({file_type.value}) processed: Summary data stored, but Transaction data failed or was empty."
            elif transaction_parsed_successfully:
                processing_status = "processed_transactions_only"
                processing_message = f"File ({file_type.value}) processed: Transaction data stored, but Summary data failed or was empty."
            else:
                processing_status = "processing_failed"
                processing_message = f"File ({file_type.value}) processing failed for both Summary and Transaction data."
            print(f"[process_file_content_and_store] Final status for {file_type.value}: {processing_status} - {processing_message}")
            # No specific parsed_df or target_table_name to return here as we handled multiple inserts
            # The function will proceed to update metadata with the final status and message
            parsed_df = None # Clear parsed_df as its role is now ambiguous
            target_table_name = None # Clear as well
        
        # --- Original logic for CAD_TRANSACTION_RAW / USD_TRANSACTION_RAW might be redundant or need review ---
        # For now, if CAD_SUMMARY_RAW / USD_SUMMARY_RAW handles both, these specific _TRANSACTION_RAW types
        # might not be strictly needed from frontend for .041 files, or could imply processing *only* transactions.
        elif file_type == FileTypeEnum.CAD_TRANSACTION_RAW or file_type == FileTypeEnum.USD_TRANSACTION_RAW:
            # This block will now primarily handle cases where ONLY transaction data is expected to be parsed and stored.
            print(f"[process_file_content_and_store] Calling process_041_content for (specific transaction type): {file_type.value}")
            if not isinstance(file_stream, BytesIO):
                file_stream_for_041 = BytesIO(file_content_bytes)
            else:
                file_stream_for_041 = file_stream
            file_stream_for_041.seek(0)

            # Pass the 'currency' from the upload to process_041_content as 'currency_code'
            parsed_df = raw_transaction_parser.process_041_content(
                file_content_stream=file_stream_for_041, 
                db_client=db, 
                currency_code=currency # Pass the currency variable from the outer scope
            )
            target_table_name = "Output"
            if parsed_df is not None and not parsed_df.empty:
                print(f"[process_file_content_and_store] TRANSACTION data (specific type) parsed. Shape: {parsed_df.shape}, Columns: {parsed_df.columns.tolist()}")
                # Standardize currency and transaction_category columns for Output
                if 'currency' not in parsed_df.columns:
                    if 'Currency' in parsed_df.columns:
                        parsed_df.rename(columns={'Currency': 'currency'}, inplace=True)
                    elif currency:
                        parsed_df['currency'] = currency.upper()
                if 'transaction_category' not in parsed_df.columns:
                    if 'Transaction Category' in parsed_df.columns:
                        parsed_df.rename(columns={'Transaction Category': 'transaction_category'}, inplace=True)
                
                await delete_existing_data_by_metadata_id(db, target_table_name, file_metadata_id)
                await insert_parsed_data(db, target_table_name, parsed_df, user_id, file_metadata_id)
                processing_status = "processed_successfully" 
                processing_message = f"Transaction data ({file_type.value}) processed and stored."
            else:
                processing_status = "processing_failed"
                processing_message = f"Transaction data parsing ({file_type.value}) failed or was empty."
            print(f"[process_file_content_and_store] Status for {file_type.value}: {processing_status} - {processing_message}")

        elif file_type == FileTypeEnum.LIQUIDITY_TEMPLATE:
            # Ensure it's BytesIO for Excel parser
            if not isinstance(file_stream, BytesIO):
                 file_stream = BytesIO(file_content_bytes) # Re-create as BytesIO if it was StringIO
            parsed_df = file_processing.parse_liquidity_template_from_content(file_stream)
            target_table_name = "liquidity_ratios" # Correct - Matches screenshot
            if parsed_df is not None and not parsed_df.empty:
                await delete_existing_data_by_metadata_id(db, target_table_name, file_metadata_id)
                await insert_parsed_data(db, target_table_name, parsed_df, user_id, file_metadata_id)
                processing_status = "processed_successfully"
                processing_message = f"Liquidity ratios data ({file_type.value}) processed and stored."
            else:
                processing_status = "processing_failed"
                processing_message = f"Liquidity ratios data parsing ({file_type.value}) failed or was empty."
            print(f"[process_file_content_and_store] Status for {file_type.value}: {processing_status} - {processing_message}")
            parsed_df = None 
            target_table_name = None 
        
        elif file_type == FileTypeEnum.USD_EXPOSURE_TEMPLATE:
            if not isinstance(file_stream, BytesIO):
                 file_stream = BytesIO(file_content_bytes)
            parsed_df = file_processing.parse_usd_exposure_template_from_content(file_stream)
            target_table_name = "usd_exposure_values" # Correct - Matches screenshot
            if parsed_df is not None and not parsed_df.empty:
                await delete_existing_data_by_metadata_id(db, target_table_name, file_metadata_id)
                await insert_parsed_data(db, target_table_name, parsed_df, user_id, file_metadata_id)
                processing_status = "processed_successfully"
                processing_message = f"USD exposure data ({file_type.value}) processed and stored."
            else:
                processing_status = "processing_failed"
                processing_message = f"USD exposure data parsing ({file_type.value}) failed or was empty."
            print(f"[process_file_content_and_store] Status for {file_type.value}: {processing_status} - {processing_message}")
            parsed_df = None 
            target_table_name = None 
        
        elif file_type == FileTypeEnum.MAPPING_TABLE:
            # For MAPPING_TABLE, we might just store the file metadata
            # or parse and store its content if needed.
            # For now, let's assume it's metadata storage and optional parsing.
            # If it's an Excel file, it needs BytesIO
            if not isinstance(file_stream, BytesIO) and file_processing.is_excel_file_signature(file_content_bytes): # Add a helper in file_processing to check if excel
                 file_stream = BytesIO(file_content_bytes)
            # Example: parsed_df = file_processing.parse_mapping_table_from_content(file_stream)
            # target_table_name = "user_specific_mappings" or "transaction_code_mappings" (if overwriting global)
            processing_status = "processed_successfully" # Or "processing_skipped" if no parsing logic yet
            processing_message = f"File type {file_type.value} received. Specific parsing/storage logic can be added here."
            print(processing_message)
            # No parsed_df or target_table_name needed if just storing metadata
            # Update metadata directly here if no further data insertion
            # await delete_existing_data_by_metadata_id(db, target_table_name, file_metadata_id) # REMOVE if target_table_name is None or unreliable here
            db.table("uploaded_files_metadata").update({
                "processing_status": processing_status,
                "processing_message": processing_message if processing_status != "processed_successfully" else None
            }).eq("id", str(file_metadata_id)).execute()
            return processing_status, processing_message

        else:
            # This case should ideally not be reached if all FileTypeEnum members are handled above.
            # If it is, it means a new FileTypeEnum member was added without corresponding logic here.
            processing_status = "processing_failed" 
            processing_message = f"Unhandled FileTypeEnum member: '{file_type.value}'. No processing defined."
            print(processing_message)
            # Update metadata status for unhandled enum members
            # await delete_existing_data_by_metadata_id(db, target_table_name, file_metadata_id) # REMOVE if target_table_name is None or unreliable here
            db.table("uploaded_files_metadata").update({
                "processing_status": processing_status,
                "processing_message": processing_message
            }).eq("id", str(file_metadata_id)).execute()
            return processing_status, processing_message

        # --- Update metadata for the processed file ---
        # This block is crucial and should always attempt to run to reflect the outcome of parsing.
        try:
            print(f"[process_file_content_and_store] Attempting to update final metadata for file_id {file_metadata_id} with status: {processing_status}")
            db.table("uploaded_files_metadata").update({
                "processing_status": processing_status,
                "processing_message": processing_message if "failed" in processing_status or "warning" in processing_status or "only" in processing_status or processing_message else None # Ensure message is stored if present
            }).eq("id", str(file_metadata_id)).execute()
            print(f"[process_file_content_and_store] Successfully updated final metadata for file_id {file_metadata_id}.")
        except Exception as meta_update_e:
            print(f"CRITICAL: [process_file_content_and_store] Failed to update final metadata for file_id {file_metadata_id} after processing: {meta_update_e}")

    except Exception as e:
        processing_status = "processing_failed"
        processing_message = f"Error during file processing for {file_type.value}: {type(e).__name__} - {str(e)}"
        print(f"❌ {processing_message}")
        # Log the full traceback for server-side debugging
        import traceback
        traceback.print_exc()

    return processing_status, processing_message

# --- ฟังก์ชันใหม่สำหรับ Background Task ---
async def run_file_processing_background(
    db: Client, # Pass the Supabase client (consider lifecycle implications)
    user_id: uuid.UUID,
    file_metadata_id: uuid.UUID,
    file_type: FileTypeEnum,
    currency: Optional[str],
    storage_path: str,
    forecast_date: Optional[datetime] # Add forecast_date
):
    """
    Background task to process the uploaded file.
    Downloads the file from storage, calls the appropriate parser,
    and updates the metadata status.
    """
    print(f"[BG Task {file_metadata_id}] Starting processing for {storage_path}")
    processing_status = "processing_failed" # Default status if something goes wrong early
    processing_message = "Background processing task failed unexpectedly."

    try:
        # 1. Update status to 'processing'
        db.table("uploaded_files_metadata").update({
            "processing_status": "processing",
            "forecast_date": forecast_date.isoformat() if forecast_date else None # Store forecast_date
            }).eq("id", str(file_metadata_id)).execute()

        # 2. Download file content from storage
        print(f"[BG Task {file_metadata_id}] Downloading file from {storage_path}...")
        try:
            # Assumes bucket name is 'uploads'
            file_content_bytes = db.storage.from_("uploads").download(path=storage_path)
            if not file_content_bytes:
                raise Exception("Downloaded file content is empty or None.")
            print(f"[BG Task {file_metadata_id}] File downloaded successfully ({len(file_content_bytes)} bytes).")
        except Exception as download_e:
            print(f"❌ [BG Task {file_metadata_id}] Failed to download file from storage: {download_e}")
            processing_status = "download_failed" # More specific status
            processing_message = f"Failed to download file from storage: {str(download_e)}"
            # Update metadata and exit task - NO delete_existing_data here as target_table is unknown
            db.table("uploaded_files_metadata").update({
                "processing_status": processing_status,
                "processing_message": processing_message
            }).eq("id", str(file_metadata_id)).execute()
            return # Stop processing if download fails

        # 3. Call the main processing function (which updates status internally on success/failure)
        # Note: We pass a dummy BackgroundTasks object as the original function expects it,
        # but we don't add further nested tasks from here.
        dummy_bg_tasks = BackgroundTasks()
        final_status, final_message = await process_file_content_and_store(
            db=db,
            user_id=user_id,
            file_metadata_id=file_metadata_id,
            file_type=file_type,
            currency=currency,
            file_content_bytes=file_content_bytes, # Use downloaded bytes
            background_tasks=dummy_bg_tasks
        )
        print(f"[BG Task {file_metadata_id}] process_file_content_and_store finished with status: {final_status}, message: {final_message}")
        # The status should have already been updated inside process_file_content_and_store,
        # but we can do it again here to be absolutely sure or if process_file_content_and_store failed to update.
        # However, the primary update responsibility is now more robustly within process_file_content_and_store's own final try-except.
        # If process_file_content_and_store raised an exception caught by the outer try-except here, 'final_status' won't be set.
        # So, the metadata update in the outer 'except Exception as bg_e' block will handle that.

    except Exception as bg_e:
        # Catch any unexpected errors during the background task execution itself
        # This includes errors from process_file_content_and_store if it raised an exception
        processing_status = "background_task_failed" # More specific status
        processing_message = f"Unexpected error in background task: {type(bg_e).__name__} - {str(bg_e)}"
        print(f"❌ [BG Task {file_metadata_id}] {processing_message}")
        import traceback
        traceback.print_exc()
        # Attempt to update metadata one last time with the failure details
        # NO delete_existing_data here
        try:
            db.table("uploaded_files_metadata").update({
                "processing_status": processing_status,
                "processing_message": processing_message
            }).eq("id", str(file_metadata_id)).execute()
        except Exception as final_meta_e:
            print(f"CRITICAL: [BG Task {file_metadata_id}] Failed to update metadata after background task error: {final_meta_e}")


# --- แก้ไข Endpoint /upload ---
@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_data_file(
    file_type: FileTypeEnum = Form(...),
    currency: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Uploads a data file. If metadata for the same user, file_type, and name (with date prefix if applicable)
    exists, its ID is reused to overwrite associated data. Otherwise, new metadata is created.
    Processing happens in the background.
    """
    user_id = current_user.id
    original_filename = file.filename
    content_type = file.content_type
    initial_status = "processing_queued"

    if file_type == FileTypeEnum.CAD_SUMMARY_RAW or file_type == FileTypeEnum.CAD_TRANSACTION_RAW:
         if currency != 'CAD':
             print(f"Warning: Currency mismatch for {file_type.value}. Received {currency}, expected CAD. Overriding to CAD.")
             currency = 'CAD'
    elif file_type == FileTypeEnum.USD_SUMMARY_RAW or file_type == FileTypeEnum.USD_TRANSACTION_RAW:
         if currency != 'USD':
             print(f"Warning: Currency mismatch for {file_type.value}. Received {currency}, expected USD. Overriding to USD.")
             currency = 'USD'
    elif file_type in [FileTypeEnum.LIQUIDITY_TEMPLATE, FileTypeEnum.USD_EXPOSURE_TEMPLATE, FileTypeEnum.MAPPING_TABLE]:
        currency = None

    print(f"Received upload request: user_id={user_id}, file_type='{file_type.value}', currency='{currency}', filename='{original_filename}'")

    # Read file content once for date extraction and upload
    file_content_bytes = await file.read()
    file_size_bytes = len(file_content_bytes)

    # Determine storage_path with potential date prefix
    safe_original_filename = "".join(c if c.isalnum() or c in ('.', '-', '_') else '_' for c in original_filename)
    storage_path_base = f"{user_id}/{file_type.value}"
    storage_path: str
    forecast_date_dt: Optional[datetime] = None

    if file_type in [FileTypeEnum.CAD_SUMMARY_RAW, FileTypeEnum.USD_SUMMARY_RAW, FileTypeEnum.CAD_TRANSACTION_RAW, FileTypeEnum.USD_TRANSACTION_RAW]:
        try:
            temp_stream = BytesIO(file_content_bytes)
            forecast_date_dt = file_processing.get_forecast_date_from_content(temp_stream)
            temp_stream.close()
            if forecast_date_dt:
                print(f"Extracted forecast_date: {forecast_date_dt} for {original_filename}")
                date_prefix = forecast_date_dt.strftime("%Y-%m-%d")
                storage_path = f"{storage_path_base}/{date_prefix}_{safe_original_filename}"
            else:
                print(f"No forecast_date found in header for {original_filename}, using default storage path (no date prefix).")
                storage_path = f"{storage_path_base}/{safe_original_filename}"
        except Exception as e:
            print(f"⚠️ Could not extract forecast_date for {original_filename}: {e}. Using default storage path (no date prefix).")
            storage_path = f"{storage_path_base}/{safe_original_filename}"
    else:
        storage_path = f"{storage_path_base}/{safe_original_filename}"
    
    print(f"Determined storage_path: {storage_path}")

    # Upload to Supabase Storage (overwrite if path exists)
    try:
        storage_response = db.storage.from_("uploads").upload(
            path=storage_path,
            file=file_content_bytes, # Use the already read bytes
            file_options={"content-type": content_type or 'application/octet-stream', "upsert": "true"}
        )
        print(f"Supabase storage response: {storage_response}")
    except Exception as e:
        print(f"❌ Error uploading to Supabase Storage: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file to storage: {str(e)}")

    # Check for existing metadata for this specific storage_path and user
    existing_metadata_response = db.table("uploaded_files_metadata")\
                                  .select("id")\
                                  .eq("user_id", str(user_id))\
                                  .eq("storage_path", storage_path)\
                                  .maybe_single()\
                                  .execute()
    
    file_metadata_id_to_use: uuid.UUID
    current_utc_time = datetime.utcnow()

    if hasattr(existing_metadata_response, 'data') and existing_metadata_response.data:
        existing_metadata_id_str = existing_metadata_response.data.get('id')
        file_metadata_id_to_use = uuid.UUID(existing_metadata_id_str)
        print(f"Found existing metadata (ID: {file_metadata_id_to_use}) for storage_path: {storage_path}. Will update it.")
        metadata_to_update = {
            "original_filename": original_filename, # Keep original filename in metadata, even if storage_path has date prefix
            "file_type": file_type.value,
            "currency": currency,
            "file_size_bytes": file_size_bytes,
            "content_type": content_type,
            "upload_timestamp": current_utc_time.isoformat(),
            "processing_status": initial_status,
            "processing_message": None,
            "forecast_date": forecast_date_dt.isoformat() if forecast_date_dt else None
        }
        try:
            update_response = db.table("uploaded_files_metadata")\
                                .update(metadata_to_update)\
                                .eq("id", str(file_metadata_id_to_use))\
                                .execute()
            if not (hasattr(update_response, 'data') and update_response.data):
                 print(f"Warning: Update metadata for ID {file_metadata_id_to_use} might not have been successful or returned no data. Response: {update_response}")
            print(f"✅ Successfully updated existing metadata (ID: {file_metadata_id_to_use}), status: {initial_status}")
        except Exception as e:
            print(f"❌ Error updating existing metadata (ID: {file_metadata_id_to_use}): {type(e).__name__} - {str(e)}")
            pass # Or raise HTTPException
    else:
        file_metadata_id_to_use = uuid.uuid4()
        print(f"No existing metadata for storage_path: {storage_path}. Creating new metadata with ID: {file_metadata_id_to_use}.")
        metadata_to_insert = {
            "id": str(file_metadata_id_to_use),
            "user_id": str(user_id),
            "original_filename": original_filename,
            "storage_path": storage_path, # This now includes date_prefix if applicable
            "file_type": file_type.value,
            "currency": currency,
            "file_size_bytes": file_size_bytes,
            "content_type": content_type,
            "upload_timestamp": current_utc_time.isoformat(),
            "processing_status": initial_status,
            "forecast_date": forecast_date_dt.isoformat() if forecast_date_dt else None
        }
        try:
            insert_response = db.table("uploaded_files_metadata").insert(metadata_to_insert).execute()
            if not (hasattr(insert_response, 'data') and insert_response.data and len(insert_response.data) > 0):
                 try:
                     db.storage.from_("uploads").remove([storage_path])
                     print(f"Cleaned up uploaded file from storage: {storage_path} due to failed metadata insert.")
                 except Exception as cleanup_e:
                     print(f"Error cleaning up storage after failed metadata insert: {cleanup_e}")
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save file metadata after successful upload.")
            print(f"✅ Successfully inserted new metadata (ID: {file_metadata_id_to_use}), status: {initial_status}")
        except Exception as e:
            try:
                db.storage.from_("uploads").remove([storage_path])
                print(f"Cleaned up uploaded file from storage: {storage_path} due to failed metadata insert.")
            except Exception as cleanup_e:
                print(f"Error cleaning up storage after failed metadata insert: {cleanup_e}")
            print(f"❌ Error inserting new metadata: {type(e).__name__} - {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save new file metadata: {str(e)}")

    background_tasks.add_task(
        run_file_processing_background,
        db=db,
        user_id=user_id,
        file_metadata_id=file_metadata_id_to_use,
        file_type=file_type,
        currency=currency,
        storage_path=storage_path, # Pass the potentially date-prefixed storage_path
        forecast_date=forecast_date_dt
    )
    print(f"✅ Queued background task for file ID: {file_metadata_id_to_use}")

    return FileUploadResponse(
        message="File received and queued for processing.",
        file_id=str(file_metadata_id_to_use),
        original_filename=original_filename,
        storage_path=storage_path,
        content_type=content_type,
        processing_status=initial_status
    )

@router.get("/history", response_model=List[UploadedFileMetadataItem])
async def get_user_upload_history(
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client) # This client should now be user-authenticated
):
    """
    Retrieves the upload history for the currently authenticated user.
    """
    user_id = current_user.id
    print(f"Fetching upload history for user_id: {user_id}")

    try:
        # Query the uploaded_files_metadata table
        # Filter by user_id and order by upload_timestamp descending
        response = (
            db.table("uploaded_files_metadata")
            .select(
                "id, user_id, original_filename, storage_path, file_type, currency, "
                "upload_timestamp, processing_status, processing_message, "
                "file_size_bytes, content_type, forecast_date"
                ) 
            .eq("user_id", str(user_id)) 
            .order("upload_timestamp", desc=True)
            .execute()
        )

        # print(f"Supabase history response: {response}") # For debugging

        if response.data:
            # Pydantic should automatically validate and convert the list of dicts
            # Ensure the UploadedFileMetadataItem schema matches these fields
            return response.data
        else:
            return [] # Return an empty list if no files are found

    except Exception as e:
        print(f"Error fetching upload history for user {user_id}: {type(e).__name__} - {str(e)}")
        # You might want to inspect the error more closely if it's from Supabase
        # e.g., if hasattr(e, 'message') and hasattr(e, 'details') for GoTrueApiError/StorageApiError
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve upload history."
        )

# --- Schema for Forecast Request ---
class ForecastRequest(BaseModel):
    currency: str # "CAD" or "USD"
    forecast_anchor_date: Optional[date] = None # Optional: YYYY-MM-DD
    training_window_days: Optional[int] = 730
    forecast_horizon_days: Optional[int] = 35

class ForecastResponse(BaseModel):
    message: str
    currency: str
    forecast_rows_generated: int
    # Optionally, you could return a summary or a path to the full forecast if stored

# --- Endpoint to Trigger Forecast Generation ---
@router.post("/forecast/generate", response_model=ForecastResponse, status_code=status.HTTP_200_OK)
async def generate_user_forecast(
    request_data: ForecastRequest,
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client)
):
    """
    Generates a cash flow forecast for the user based on their historical data.
    Results are stored in the 'full_forecast_output' table.
    """
    user_id = current_user.id
    currency = request_data.currency.upper()
    
    if currency not in ["CAD", "USD"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid currency specified. Must be CAD or USD.")

    print(f"Received forecast generation request for user {user_id}, currency {currency}, anchor: {request_data.forecast_anchor_date}")

    try:
        # Call generate_forecast_for_user from the correct module: forecasting
        forecast_df = await forecasting.generate_forecast_for_user(
            db=db, 
            user_id=user_id, 
            currency=currency,
            forecast_anchor_date=request_data.forecast_anchor_date,
            training_window_days=request_data.training_window_days or 730,
            forecast_horizon_days=request_data.forecast_horizon_days or 35
        )

        if forecast_df is None or forecast_df.empty:
            print(f"Forecast generation returned no data for user {user_id}, currency {currency}.")
            # Return a specific response or raise HTTP exception if no data means failure here
            # For now, let's assume it might be valid if no forecastable data, but log it.
            # If it's an error, the function would have printed it.
            return ForecastResponse(
                message=f"Forecast generation completed but produced no data for {currency}. This might be due to insufficient historical data or other internal reasons.",
                currency=currency,
                forecast_rows_generated=0
            )

        # 2. Prepare data for insertion (add user_id)
        df_to_insert = forecast_df.copy()
        df_to_insert['user_id'] = str(user_id) # Add user_id as string for Supabase
        
        # Ensure the currency column from forecasting.py is present and correctly cased for the insert
        # forecasting.py should add 'currency' (lowercase) with uppercase value (e.g. CAD)
        if 'currency' not in df_to_insert.columns:
            print(f"CRITICAL ⚠️: 'currency' column MISSING from forecast_df in router. Attempting to add.")
            df_to_insert['currency'] = currency # currency is already request_data.currency.upper()
        else:
            print(f"✅ 'currency' column found in forecast_df from forecasting.py. Values: {df_to_insert['currency'].unique().tolist()[:5]}")
            # Ensure it's uppercase as expected by the rest of the logic/DB query (if not already)
            df_to_insert['currency'] = df_to_insert['currency'].str.upper()

        # Ensure Date is string and other numerics are float/None
        if 'Date' in df_to_insert.columns:
            df_to_insert['Date'] = df_to_insert['Date'].astype(str)
        
        numeric_cols = ['Forecasted Amount', 'Forecasted Cash Balance', 'Actual Cash Balance']
        for col in numeric_cols:
            if col in df_to_insert.columns:
                df_to_insert[col] = pd.to_numeric(df_to_insert[col], errors='coerce').astype(object).where(pd.notnull(df_to_insert[col]), None)

        # 3. Delete old forecast data for this user and currency
        print(f"Deleting old forecast data for user {user_id}, currency {currency} from full_forecast_output...")
        delete_response = (
            db.table("full_forecast_output")
            .delete()
            .eq("user_id", str(user_id))
            .eq("currency", currency) # Ensure we filter by the correct currency column name and value
            .execute()
        )
        if hasattr(delete_response, 'error') and delete_response.error:
            print(f"⚠️ Error deleting old forecast data: {delete_response.error}. Proceeding with insert anyway.")
            # Not raising an exception here, as the insert is more critical
        else:
            # Check how many rows were deleted (if API provides this info)
            # data_len = len(delete_response.data) if hasattr(delete_response, 'data') else "unknown"
            # print(f"Successfully deleted {data_len} old forecast records.")
            pass # Supabase delete usually doesn't return count unless specific select is done

        # 4. Insert new forecast data
        print(f"Columns in df_to_insert before to_dict: {df_to_insert.columns.tolist()}")
        print(f"Sample record from df_to_insert before to_dict: {df_to_insert.head(1).to_dict(orient='records') if not df_to_insert.empty else 'Empty df'}")
        records_to_insert = df_to_insert.to_dict(orient='records')
        
        print(f"Inserting {len(records_to_insert)} new forecast records for user {user_id}, currency {currency} into full_forecast_output...")
        insert_response = db.table("full_forecast_output").insert(records_to_insert).execute()

        if hasattr(insert_response, 'error') and insert_response.error:
            print(f"❌ Error inserting forecast data: {insert_response.error}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to store forecast results: {insert_response.error.message}")
        
        # num_inserted = len(insert_response.data) if hasattr(insert_response, 'data') else len(records_to_insert)
        # The response from Supabase insert might not contain all inserted rows by default, depending on `returning` option.
        # We'll use len(records_to_insert) as the intended number.
        num_inserted = len(records_to_insert)
        print(f"✅ Successfully inserted {num_inserted} forecast records.")

        return ForecastResponse(
            message=f"Forecast generation for {request_data.currency.upper()} completed. {len(forecast_df) if forecast_df is not None else 0} rows generated.",
            currency=request_data.currency.upper(),
            forecast_rows_generated=len(forecast_df) if forecast_df is not None else 0
        )

    except HTTPException as http_exc: # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        print(f"❌ Unexpected error during forecast generation for user {user_id}, currency {currency}: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while generating the forecast.")

@router.get("/forecast/latest", response_model=List[ForecastDataRow])
async def get_latest_forecast_data(
    currency: str = Query(..., description="The currency for which to fetch the forecast (e.g., CAD or USD)"),
    current_user: User = Depends(get_current_active_user),
    db: Client = Depends(get_supabase_client)
):
    try:
        print(f"Fetching latest forecast data for user {current_user.id}, currency: {currency.upper()}")
        response = (
            db.table("full_forecast_output")
            .select('"Date", "Forecasted Amount", "Forecasted Cash Balance", "Actual Cash Balance"') 
            .eq("user_id", str(current_user.id))
            .eq("currency", currency.upper())
            .order('"Date"', desc=False)
            .execute()
        )

        if response.data:
            print(f"Found {len(response.data)} forecast records.")
            # Convert list of dicts to list of ForecastDataRow instances
            # Pydantic will use the aliases for field names from the database
            result = [ForecastDataRow.model_validate(row) for row in response.data]
            return result
        else:
            print(f"No forecast data found for user {current_user.id}, currency: {currency.upper()}")
            return [] # Return empty list if no data

    except Exception as e:
        print(f"❌ Error fetching latest forecast data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching forecast data: {str(e)}"
        )

# Note: The old /forecasts/ router and its endpoints are assumed to be removed or commented out.

async def delete_existing_data_by_metadata_id(
    db: Client, 
    table_name: str, 
    file_metadata_id: uuid.UUID
):
    """Deletes records from the specified table that match the file_metadata_id."""
    # Consistently use 'upload_metadata_id' as the column name for the foreign key
    foreign_key_column_name = "upload_metadata_id"
    try:
        print(f"[delete_existing_data_by_metadata_id] Attempting to delete data from '{table_name}' for {foreign_key_column_name}: {file_metadata_id}")
        delete_result = (
            db.table(table_name)
            .delete()
            .eq(foreign_key_column_name, str(file_metadata_id)) # <<< USE THE CONSISTENT COLUMN NAME
            .execute()
        )
        # Supabase delete often returns a list of the deleted records in `data`
        # For PostgREST, if `Prefer: return=representation` is set, `data` contains deleted items.
        # Otherwise, `data` might be empty, and success is indicated by lack of error.
        if hasattr(delete_result, 'data') and delete_result.data:
            print(f"[delete_existing_data_by_metadata_id] Successfully deleted {len(delete_result.data)} record(s) from '{table_name}' for {foreign_key_column_name}: {file_metadata_id}")
        elif hasattr(delete_result, 'count') and delete_result.count is not None: # Some clients might return a count
             print(f"[delete_existing_data_by_metadata_id] Successfully deleted {delete_result.count} record(s) from '{table_name}' for {foreign_key_column_name}: {file_metadata_id}")
        else:
            # If no data/count and no error, assume success but log it for verification
            print(f"[delete_existing_data_by_metadata_id] Executed delete for '{table_name}' for {foreign_key_column_name}: {file_metadata_id}. Response did not explicitly state count of deleted rows, check for errors.")
        
        # You might want to check for errors in delete_result more explicitly if your client version provides them
        if hasattr(delete_result, 'error') and delete_result.error:
            print(f"ERROR [delete_existing_data_by_metadata_id] Error during deletion from '{table_name}': {delete_result.error}")
            # Decide if this should raise an exception
            # raise Exception(f"Error deleting from {table_name}: {delete_result.error}")

    except APIError as e:
        print(f"ERROR [delete_existing_data_by_metadata_id] Supabase APIError deleting from '{table_name}' for {foreign_key_column_name}: {file_metadata_id}. Error: {e}")
        # Depending on desired behavior, you might re-raise or handle
        raise
    except Exception as e:
        print(f"ERROR [delete_existing_data_by_metadata_id] Unexpected error deleting from '{table_name}' for {foreign_key_column_name}: {file_metadata_id}. Error: {type(e).__name__} - {e}")
        # Depending on desired behavior, you might re-raise or handle
        raise
