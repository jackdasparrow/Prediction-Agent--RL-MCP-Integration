"""
FastAPI Server
Exposes MCP-style endpoints with JWT authentication and rate limiting.
"""

import os
import time
import psutil
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from pydantic import BaseModel
import logging
from collections import defaultdict
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.mcp_adapter import (
    MCPAdapter, 
    PredictRequest, 
    ScanAllRequest, 
    AnalyzeRequest,
    MCPResponse
)
from core.data_ingest import DataIngestion
from core.features import FeaturePipeline
from core.models.baseline_lightgbm import BaselineLightGBM
from core.models.rl_agent import LinUCBAgent, ThompsonSamplingAgent, DQNAgent

# ------------------- Logging -------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/api_server.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------- JWT Config -------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change_this_secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

security = HTTPBearer()

# ------------------- Rate Limit -------------------
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
rate_limit_store = defaultdict(list)

# ------------------- FastAPI -------------------
app = FastAPI(
    title="Prediction Agent API",
    description="RL-powered prediction agent with MCP-style endpoints",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------- Models -------------------
class TokenRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    system: dict
    models_loaded: bool

# ------------------- Global State -------------------
class AppState:
    def __init__(self):
        self.mcp_adapter: Optional[MCPAdapter] = None
        self.start_time = time.time()
        self.request_count = 0

app_state = AppState()

# ------------------- Auth -------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# ------------------- Rate Limiting -------------------
def check_rate_limit(request: Request):
    client_ip = request.client.host
    now = time.time()
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded")
    rate_limit_store[client_ip].append(now)

# ------------------- Startup -------------------
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting Prediction Agent API...")

        # Create directories
        for d in ["logs", "models", "data/cache", "data/features"]:
            os.makedirs(d, exist_ok=True)

        # Load baseline model
        baseline_model = BaselineLightGBM(model_dir="./models", task="classification", model_name="lightgbm-v1")
        try:
            baseline_model.load()
            logger.info("Baseline model loaded")
        except FileNotFoundError:
            logger.warning("Baseline model not found")

        # Initialize feature pipeline
        feature_pipeline = FeaturePipeline(feature_store_dir="./data/features")

        # Load feature store to determine n_features
        try:
            feature_store_df = feature_pipeline.load_feature_store()
            # Get feature columns (exclude metadata columns)
            feature_cols = [col for col in feature_store_df.columns if col not in 
                           ['symbol', 'source', 'fetch_timestamp', 'target', 'target_direction']]
            n_features = len(feature_cols)
            logger.info(f"Feature store loaded: {n_features} features")
        except Exception as e:
            n_features = 50
            logger.warning(f"Feature store empty or missing: {e}. Defaulting n_features=50")

        # Initialize RL agent
        agent_type = os.getenv("RL_AGENT_TYPE", "linucb")
        if agent_type == "linucb":
            agent = LinUCBAgent(n_features=n_features)
            try:
                agent.load('linucb_agent.pkl')
                logger.info("LinUCB agent loaded successfully")
            except FileNotFoundError:
                logger.warning("LinUCB agent not found, using untrained agent")
        elif agent_type == "thompson":
            agent = ThompsonSamplingAgent(n_features=n_features)
            try:
                agent.load('thompson_agent.pkl')
                logger.info("Thompson agent loaded successfully")
            except FileNotFoundError:
                logger.warning("Thompson agent not found, using untrained agent")
        else:
            device = "cuda" if os.getenv("GPU_DEVICE") else "cpu"
            agent = DQNAgent(state_dim=n_features, action_dim=3, device=device)
            try:
                agent.load('dqn_agent.pt')
                logger.info("DQN agent loaded successfully")
            except FileNotFoundError:
                logger.warning("DQN agent not found, using untrained agent")

        # Initialize MCPAdapter
        app_state.mcp_adapter = MCPAdapter(agent, baseline_model, feature_pipeline)
        logger.info("[OK] Prediction Agent API started")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

# ------------------- Shutdown -------------------
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API...")

# ------------------- Endpoints -------------------
@app.post("/auth/token", response_model=TokenResponse)
async def login(request: TokenRequest):
    if not request.username or not request.password:
        raise HTTPException(400, "Username/password required")
    token = create_access_token({"sub": request.username})
    return TokenResponse(access_token=token, expires_in=JWT_EXPIRE_MINUTES*60)

@app.get("/tools/health", response_model=HealthResponse)
async def health_check():
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        gpu_info = {"available": False}
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=time.time() - app_state.start_time,
            system={"cpu_percent": cpu, "memory_percent": memory.percent, "request_count": app_state.request_count, "gpu": gpu_info},
            models_loaded=app_state.mcp_adapter is not None
        )
    except Exception as e:
        return HealthResponse(status="degraded", timestamp=datetime.now().isoformat(), uptime_seconds=0, system={"error": str(e)}, models_loaded=False)

@app.post("/tools/predict", response_model=MCPResponse)
async def predict(request: PredictRequest, token: dict = Depends(verify_token), rate_limit: None = Depends(check_rate_limit)):
    if not app_state.mcp_adapter:
        raise HTTPException(503, "Service not initialized")
    app_state.request_count += 1
    return app_state.mcp_adapter.predict(request)

@app.post("/tools/scan_all", response_model=MCPResponse)
async def scan_all(request: ScanAllRequest, token: dict = Depends(verify_token), rate_limit: None = Depends(check_rate_limit)):
    if not app_state.mcp_adapter:
        raise HTTPException(503, "Service not initialized")
    app_state.request_count += 1
    return app_state.mcp_adapter.scan_all(request)

@app.post("/tools/analyze", response_model=MCPResponse)
async def analyze(request: AnalyzeRequest, token: dict = Depends(verify_token), rate_limit: None = Depends(check_rate_limit)):
    if not app_state.mcp_adapter:
        raise HTTPException(503, "Service not initialized")
    app_state.request_count += 1
    return app_state.mcp_adapter.analyze(request)

# ------------------- Run Server -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload to avoid logging issues
        log_level="warning",  # Reduce log level
        access_log=False,  # Disable access logging
        log_config=None  # Disable uvicorn's logging config to avoid Windows issues
    )
