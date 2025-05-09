from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# --- เพิ่ม import สำหรับ BackgroundTasks ถ้าจะใช้ ---
# from fastapi import BackgroundTasks
# ---------------------------------------------------
from .routers import auth, admin, files, forecasts # <--- Import forecasts router
from .routers import summary_output # <--- ADD THIS LINE
# from .routers import forecast # Keep these commented for now

app = FastAPI(
    title="BCP Forecasting API",
    description="API for BCP data processing and forecasting.",
    version="0.1.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",  # Origin of your Next.js frontend during development
    "http://127.0.0.1:3000", # Might be needed too
    # Add other origins if necessary (e.g., deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Allows specific origins
    allow_credentials=True, # Allows cookies to be included in requests (if you use them)
    allow_methods=["*"],    # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allows all headers
)
# --- End CORS Configuration ---

app.include_router(auth.router, prefix="/auth", tags=["Authentication"]) # <--- Include the auth router
app.include_router(admin.router) # <--- Include the admin router (prefix="/admin", tags=["Admin Management"] are set in admin.py)
app.include_router(files.router) # <--- Include the files router (prefix="/files", tags=["File Management"] included in router definition)
app.include_router(summary_output.router) # <--- ADD THIS LINE
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