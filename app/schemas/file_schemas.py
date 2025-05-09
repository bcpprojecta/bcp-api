from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid
from enum import Enum

class FileTypeEnum(str, Enum):
    CAD_TRANSACTION_RAW = "CAD_TRANSACTION_RAW"
    USD_TRANSACTION_RAW = "USD_TRANSACTION_RAW"
    CAD_SUMMARY_RAW = "CAD_SUMMARY_RAW"
    USD_SUMMARY_RAW = "USD_SUMMARY_RAW"
    LIQUIDITY_TEMPLATE = "LIQUIDITY_TEMPLATE"
    USD_EXPOSURE_TEMPLATE = "USD_EXPOSURE_TEMPLATE"
    MAPPING_TABLE = "MAPPING_TABLE"

class UploadedFileMetadataItem(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    original_filename: str
    storage_path: str
    file_type: Optional[FileTypeEnum] = None
    currency: Optional[str] = None
    file_size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    upload_timestamp: datetime
    processing_status: Optional[str] = "pending"
    processing_message: Optional[str] = None
    forecast_date: Optional[datetime] = None

    class Config:
        from_attributes = True
        use_enum_values = True

# You might also want a schema for a list of these items
# class UploadedFileHistoryResponse(BaseModel):
#     files: List[UploadedFileMetadataItem]
#     total_files: int 