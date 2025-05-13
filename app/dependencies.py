import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from supabase import create_client, Client
from dotenv import load_dotenv
from supabase.lib.client_options import ClientOptions
from gotrue.types import User

# Determine the project root directory (stuff/) assuming this file is in stuff/app/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Construct the path to the .env file
dotenv_path = os.path.join(project_root, '.env')

# Load environment variables from .env file
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"Warning: .env file not found at {dotenv_path}. Environment variables might not be loaded.")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") # This is your PUBLIC ANON KEY
# SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # Keep for service client

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase URL or Public Anon Key not found in environment variables.")

# New dependency for a simple anonymous client (for login endpoint)
async def get_supabase_anon_client() -> Client:
    """
    Returns a Supabase client instance initialized with the public anonymous key.
    This client is suitable for operations like login/signup that don't require a user session yet.
    """
    # print("get_supabase_anon_client: Creating anon client") # Uncomment for debugging
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return client

async def get_supabase_client(token: str = Depends(oauth2_scheme)) -> Client:
    """
    Returns a Supabase client instance initialized to act as the authenticated user.
    The client's session is set using the provided JWT access token.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated (token missing)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    client = create_client(SUPABASE_URL, SUPABASE_KEY) # Start with anon key
    try:
        # print(f"get_supabase_client: Attempting to set session with token: {token[:20]}...") # Uncomment for debugging
        client.auth.set_session(access_token=token, refresh_token="dummy_refresh_token_placeholder")
        # print("get_supabase_client: Session supposedly set.") # Uncomment for debugging
    except Exception as e:
        print(f"CRITICAL: get_supabase_client - Error during client.auth.set_session: {type(e).__name__} - {str(e)}")
        # Consider raising an error if session setting fails, as RLS might not work.
        pass 
    return client

async def get_supabase_service_client() -> Client:
    """
    Returns a Supabase client instance initialized with the SERVICE_ROLE_KEY.
    USE WITH CAUTION: This client bypasses Row Level Security (RLS).
    """
    SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase Service Role Key (SUPABASE_SERVICE_ROLE_KEY) not found.")
    service_client = create_client(SUPABASE_URL, SERVICE_ROLE_KEY)
    return service_client

# Example of how you might want to handle ClientOptions if you were using supabase-py v2.x features
# from supabase.lib.client_options import ClientOptions
# async def get_supabase_client_v2_example(token: str = Depends(oauth2_scheme)) -> Client:
#     headers = {
#         "apikey": SUPABASE_KEY,
#         "Authorization": f"Bearer {token}"
#     }
#     # Note: For v2, how the token is used for auth state (for db.auth.get_user()) vs
#     # just RLS on PostgREST/Storage needs careful checking of library docs.
#     # ClientOptions is generally for things like custom fetch, auto-refresh, etc.
#     # Setting the session via client.auth.set_session is still the most direct way
#     # to make the client instance user-aware.
#     options = ClientOptions(headers=headers)
#     return create_client(SUPABASE_URL, SUPABASE_KEY, options=options)

# New dependency to get the current authenticated user object
async def get_current_user(client: Client = Depends(get_supabase_client)) -> User:
    """
    Depends on get_supabase_client to get an authenticated client,
    then retrieves the user object associated with the session.
    Raises HTTPException 401 if the user cannot be retrieved (invalid token/session).
    """
    try:
        user_response = client.auth.get_user()
        # print(f"get_current_user: Raw response from get_user(): {user_response}") # Debugging
        
        # Check if user data is present in the response
        # Adjust based on the actual structure of user_response if needed
        if user_response and hasattr(user_response, 'user') and user_response.user:
             # print(f"get_current_user: User found: {user_response.user.id}") # Debugging
             return user_response.user # Return the User object
        else:
             # print("get_current_user: No user found in response.") # Debugging
             raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials or user not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
    except Exception as e:
        # Catch potential exceptions during the get_user call
        print(f"CRITICAL: get_current_user - Error calling client.auth.get_user(): {type(e).__name__} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
