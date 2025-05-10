from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from ..dependencies import get_supabase_service_client
from ..schemas.auth_schemas import UserCreate, User
from ..security import require_admin_role
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr
from uuid import UUID

router = APIRouter(
    prefix="/admin", # กำหนด prefix สำหรับ admin routes ทั้งหมด
    tags=["Admin Management"] # กำหนด tag สำหรับกลุ่มนี้
)

# Pydantic schema for user data returned by /admin/users
class AdminUserView(BaseModel):
    id: UUID
    email: EmailStr
    created_at: datetime
    last_sign_in_at: Optional[datetime] = None
    user_metadata: Optional[dict] = {}

    class Config:
        from_attributes = True

@router.post(
    "/create-user", # path จะกลายเป็น /admin/create-user
    status_code=status.HTTP_201_CREATED,
    response_model=User
    # ไม่ต้องใส่ dependencies=[Depends(require_admin_role)] ที่นี่แล้ว ถ้าจะใส่ใน parameter
)
async def admin_create_new_user(
    user_data: UserCreate,
    db_service_client: Client = Depends(get_supabase_service_client),
    current_admin: User = Depends(require_admin_role) # Enforces admin role
):
    # ... (เนื้อหาฟังก์ชัน admin_create_new_user เหมือนเดิม) ...
    try:
        new_user_payload = {
            "email": user_data.email,
            "password": user_data.password,
            "email_confirm": True,
        }
        if user_data.user_metadata:
            new_user_payload["options"] = {
                "data": user_data.user_metadata
            }
        response_supabase_user = db_service_client.auth.admin.create_user(new_user_payload)
        created_user_for_response = None
        if hasattr(response_supabase_user, 'id') and hasattr(response_supabase_user, 'email'):
            created_user_for_response = User.model_validate(response_supabase_user)
        elif hasattr(response_supabase_user, 'user') and response_supabase_user.user:
            created_user_for_response = User.model_validate(response_supabase_user.user)

        if created_user_for_response:
            return created_user_for_response
        else:
            print(f"Admin create user unexpected response structure: {response_supabase_user}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user: Unexpected response structure from authentication service."
            )
    except Exception as e:
        error_message = str(e)
        print(f"Admin create user error: Type='{type(e).__name__}', Message='{error_message}'")
        is_user_conflict = False
        detail_msg_for_client = f"Could not create user: {error_message}"
        status_code_for_client = status.HTTP_400_BAD_REQUEST
        try:
            import json
            if error_message.startswith("{") and error_message.endswith("}"):
                error_data = json.loads(error_message)
                actual_supabase_msg = error_data.get('msg', error_data.get('message', ''))
                if "User already registered" in actual_supabase_msg or \
                   "already registered" in actual_supabase_msg.lower() or \
                   "user_already_exists" in actual_supabase_msg.lower():
                    is_user_conflict = True
        except json.JSONDecodeError:
            if "User already registered" in error_message or \
               "user with this email address has already been registered" in error_message.lower() or \
               "User already exists" in error_message:
                is_user_conflict = True
        if not is_user_conflict and hasattr(e, 'message') and isinstance(e.message, str):
            if "User already registered" in e.message or \
               "user with this email address has already been registered" in e.message.lower() or \
               "User already exists" in e.message:
                is_user_conflict = True
        if hasattr(e, 'status') and isinstance(e.status, int):
            if not is_user_conflict or e.status != 400 and e.status != 422 :
                 status_code_for_client = e.status
        if is_user_conflict:
            print(f"Conflict: User with email '{user_data.email}' already exists.")
            detail_msg_for_client = f"User with email '{user_data.email}' already exists."
            status_code_for_client = status.HTTP_409_CONFLICT
        raise HTTPException(
            status_code=status_code_for_client,
            detail=detail_msg_for_client
        )

@router.get(
    "/users",
    response_model=List[AdminUserView]
)
async def list_all_users(
    supabase: Client = Depends(get_supabase_service_client),
    current_admin: User = Depends(require_admin_role) # Enforces admin role for this endpoint
):
    try:
        # Fetch all users. Supabase's list_users returns a paged response by default (50 users per page).
        # For simplicity, we'll fetch the first page or all if it's a simple list.
        # Based on the log, list_users() directly returns the list of User objects.
        users_list = supabase.auth.admin.list_users()

        # The users_list should directly be a list of user objects.
        # If users_list is None or not a list (though unlikely based on logs if it prints a list), 
        # Pydantic validation will fail or an error might occur earlier.
        # We can add a simple check if it can be None.
        if users_list is None:
            print("Supabase list_users returned None.")
            return [] # Return empty list if None
        
        # Pydantic will validate each item in the list against AdminUserView.
        # Ensure that the fields in Supabase User objects match AdminUserView or use aliases.
        # Based on the provided log, fields like id, email, created_at, last_sign_in_at, user_metadata exist.
        return users_list

    except Exception as e:
        # Log the detailed error for debugging
        print(f"Error listing users: Type='{type(e).__name__}', Message='{str(e)}'")
        # Provide a generic error message to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while fetching the list of users."
        )
