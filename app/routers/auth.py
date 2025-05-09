# stuff/app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from supabase import Client #, PostgrestAPIResponse # PostgrestAPIResponse อาจจะไม่จำเป็นถ้า v2 คืน user obj โดยตรง
# from supabase_py_async.lib.client_async_options import ClientOptions # <--- COMMENT THIS OUT
# from httpx import HTTPStatusError # <--- COMMENT THIS OUT for now, or try to install httpx if needed by your supabase version
import os

from ..dependencies import get_supabase_anon_client, get_supabase_service_client, get_supabase_client
from ..schemas.auth_schemas import Token, UserCreate, User
# from ..security import require_admin_role # Moved to admin router
from ..security import get_current_active_user # CORRECT RELATIVE IMPORT from security.py

router = APIRouter()

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Client = Depends(get_supabase_anon_client)
):
    """
    Logs in a user and returns an access token.
    FastAPI's OAuth2PasswordRequestForm expects 'username' and 'password'.
    We will use 'username' as the email.
    """
    print(f"Login attempt for username (email): {form_data.username}")
    try:
        response = db.auth.sign_in_with_password({
            "email": form_data.username,
            "password": form_data.password
        })

        # Check if the response indicates a successful login
        # Supabase-py v2 might raise an exception on failure, which is caught below.
        # For v1 or if no exception, check the response structure.
        if not response or not hasattr(response, 'session') or not response.session or not response.session.access_token:
            # This path might be taken if sign_in_with_password doesn't raise an error
            # but returns a non-session object or a session without an access token.
            print(f"Login failed: No session or access token in Supabase response. Response: {response}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password (no session returned).",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # print(f"Supabase sign_in response (session): {response.session}") # Debugging
        return Token(
            access_token=response.session.access_token,
            token_type="bearer",
            refresh_token=response.session.refresh_token if response.session.refresh_token else "", # Ensure refresh_token is a string
            expires_in=response.session.expires_in if response.session.expires_in is not None else 3600 # Provide default if None
        )
    except Exception as e:
        # Log the actual error from Supabase or other issues
        error_message = str(e)
        print(f"Error during Supabase sign_in_with_password: {type(e).__name__} - {error_message}")
        
        # Try to parse Supabase/GoTrue specific errors for more user-friendly messages
        detail_message = "Incorrect email or password."
        status_code_to_return = status.HTTP_401_UNAUTHORIZED

        # GoTrueApiError from supabase-py often has a message and status attribute
        if hasattr(e, 'message') and isinstance(getattr(e, 'message'), str):
            if "Invalid login credentials" in e.message:
                detail_message = "Invalid login credentials. Please check your email and password."
            elif "Email not confirmed" in e.message:
                detail_message = "Email not confirmed. Please check your inbox for a confirmation email."
            # Add more specific GoTrue error messages if needed
            else:
                detail_message = e.message # Use the message from the exception if it's somewhat descriptive
        
        if hasattr(e, 'status') and isinstance(getattr(e, 'status'), int):
            # Use status from exception if available and seems like an HTTP status
            if 400 <= e.status < 500:
                status_code_to_return = e.status

        raise HTTPException(
            status_code=status_code_to_return,
            detail=detail_message,
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current authenticated user.
    """
    return current_user

# --- Admin Create User Endpoint Removed --- 
# The /admin/create-user endpoint has been moved to stuff/app/routers/admin.py
