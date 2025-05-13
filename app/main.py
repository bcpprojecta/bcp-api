from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# --- เพิ่ม import สำหรับ BackgroundTasks ถ้าจะใช้ ---
# from fastapi import BackgroundTasks
# ---------------------------------------------------
from .routers import auth, admin, files, forecasts # <--- Import forecasts router
from .routers import summary_output # <--- ADD THIS LINE
from .routers import liquidity_ratios # ADD THIS
from .routers import usd_exposure     # ADD THIS
# from .routers import forecast # Keep these commented for now

app = FastAPI(
    title="BCP Forecasting API",
    description="API for BCP data processing and forecasting.",
    version="0.1.0"
)

# Add CORS middleware configuration
origins = [
    "http://localhost:3000",  # For local development
    "https://bcp-front.vercel.app",  # For production frontend
    "http://127.0.0.1:3000", # Might be needed too
    # Add other origins if necessary (e.g., deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Configuration ---

app.include_router(auth.router, prefix="/auth", tags=["Authentication"]) # <--- Include the auth router
app.include_router(admin.router) # <--- Include the admin router (prefix="/admin", tags=["Admin Management"] are set in admin.py)
app.include_router(files.router) # <--- Include the files router (prefix="/files", tags=["File Management"] included in router definition)
app.include_router(summary_output.router) # <--- ADD THIS LINE
app.include_router(liquidity_ratios.router) # ADD THIS
app.include_router(usd_exposure.router)     # ADD THIS
# app.include_router(forecasts.router) # Comment out this line
# app.include_router(forecast.router, prefix="/forecast", tags=["Forecasting"])

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the BCP Forecasting API!"}

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "ok", "message": "API is healthy"}