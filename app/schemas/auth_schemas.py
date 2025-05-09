    # stuff/app/schemas/auth_schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime

class UserLogin(BaseModel): # <--- ตรวจสอบว่ามี class นี้
    email: EmailStr
    password: str

class Token(BaseModel): # <--- ตรวจสอบว่ามี class นี้
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None

class TokenData(BaseModel): # For decoding token's payload
    sub: Optional[str] = None # Subject (usually user_id)
    # aud: Optional[str] = None # Audience
    # exp: Optional[int] = None # Expiration time
    email: Optional[EmailStr] = None # From user_metadata or identity
    # role: Optional[str] = None # Standard role claim if Supabase adds it
    user_metadata: Optional[Dict[str, Any]] = {} # Our custom data including role

class User(BaseModel): # Schema for user object returned by Supabase (e.g. from get_user())
    id: str
    email: Optional[EmailStr] = None
    app_metadata: Optional[Dict[str, Any]] = {}
    user_metadata: Optional[Dict[str, Any]] = {} # This will contain our role
    created_at: Optional[datetime] = None
    # Add other fields you expect from Supabase user object if needed

    class Config:
        # orm_mode = True # For Pydantic v1
        from_attributes = True # For Pydantic v2+

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    user_metadata: Optional[Dict[str, Any]] = None