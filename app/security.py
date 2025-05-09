from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import Client
# from jose import JWTError, jwt # Not needed if using db.auth.get_user()

from .dependencies import get_supabase_client
from .schemas.auth_schemas import User, TokenData # Import User schema

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_active_user(
    token: str = Depends(oauth2_scheme),
    db: Client = Depends(get_supabase_client)
) -> User: # Return our Pydantic User model
    """
    Dependency to get the current active user from the token.
    Verifies the token using Supabase client.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # db.auth.get_user(token) verifies the token and returns the user object
        # In supabase-py v1.x, this returns a UserResponse object with a .user attribute
        # In supabase-py v2.x, this might return the user object directly or handle errors differently
        
        # print(f"Token received by get_current_active_user: {token[:20]}...") # Debug: print part of token
        user_response_or_user = db.auth.get_user(token)
        # print(f"Type of response from db.auth.get_user: {type(user_response_or_user)}")
        # print(f"Content of response from db.auth.get_user: {user_response_or_user}")


        # Adapt based on the actual structure of user_response_or_user
        # For Supabase-py v1.x (often returns UserResponse):
        if hasattr(user_response_or_user, 'user') and user_response_or_user.user:
            supabase_user = user_response_or_user.user
        # For Supabase-py v2.x (might return user object directly):
        elif hasattr(user_response_or_user, 'id'): # Check for a common user attribute
            supabase_user = user_response_or_user
        else:
            # print("get_user did not return a valid user object or UserResponse.")
            raise credentials_exception
        
        if not supabase_user:
            # print("supabase_user object is None after trying to access it.")
            raise credentials_exception

        # Convert Supabase user object to our Pydantic User model
        # Ensure attributes match the User Pydantic model
        pydantic_user = User.model_validate(supabase_user) # Pydantic v2
        # pydantic_user = User.from_orm(supabase_user) # Pydantic v1
        
        # print(f"Current user (Pydantic): {pydantic_user.model_dump_json(indent=2)}")
        return pydantic_user

    except Exception as e:
        print(f"Error in get_current_active_user: {type(e).__name__} - {str(e)}")
        # Example: Supabase client might raise specific auth errors
        # if "Invalid JWT" in str(e) or "Token expired" in str(e):
        #     raise credentials_exception
        raise credentials_exception


async def require_admin_role(current_user: User = Depends(get_current_active_user)):
    """
    Dependency to ensure the current user has the 'admin' role
    in their user_metadata.
    """
    if not current_user.user_metadata or current_user.user_metadata.get("role") != "admin":
        # print(f"Admin role check failed for user: {current_user.email}, metadata: {current_user.user_metadata}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not permitted: Requires admin role."
        )
    # print(f"Admin role check passed for user: {current_user.email}")
    return current_user
