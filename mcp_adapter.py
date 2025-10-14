"""
MCP (Model Context Protocol) Adapter
Provides JSON-RPC style wrappers for the prediction agent to be called as a tool.
Works with features from both Yahoo Finance and Alpha Vantage data.
"""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum

logger = logging.getLogger(__name__)


class Horizon(str, Enum):
    """Trading horizons"""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class Action(str, Enum):
    """Trading actions"""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class RiskProfile(BaseModel):
    """Risk parameters for prediction"""
    stop_loss_pct: float = Field(default=2.0, ge=0, le=100)
    capital_risk_pct: float = Field(default=1.5, ge=0, le=100)
    drawdown_limit_pct: float = Field(default=10.0, ge=0, le=100)

    class Config:
        use_enum_values = True


class PredictRequest(BaseModel):
    """Request schema for /tools/predict"""
    symbols: List[str] = Field(..., min_length=1, max_length=100)
    horizon: Horizon = Field(default=Horizon.INTRADAY)
    risk_profile: Optional[RiskProfile] = None

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        return [s.strip().upper() for s in v if s.strip()]

    class Config:
        use_enum_values = True


class ScanAllRequest(BaseModel):
    """Request schema for /tools/scan_all"""
    horizon: Horizon = Field(default=Horizon.INTRADAY)
    risk_profile: Optional[RiskProfile] = None
    top_k: int = Field(default=20, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0, le=1)

    class Config:
        use_enum_values = True


class AnalyzeRequest(BaseModel):
    """Request schema for /tools/analyze"""
    symbols: List[str] = Field(..., min_length=1, max_length=10)
    horizon: Horizon = Field(default=Horizon.INTRADAY)
    detailed: bool = Field(default=False)

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        return [s.strip().upper() for s in v if s.strip()]

    class Config:
        use_enum_values = True


# ===================== New: Feedback / Training / Fetch =====================
class FeedbackRequest(BaseModel):
    """User feedback on a prior prediction."""
    symbol: str
    predicted_action: Action
    user_feedback: str = Field(pattern=r"^(correct|incorrect)$")
    horizon: Horizon = Field(default=Horizon.INTRADAY)

    class Config:
        use_enum_values = True


class TrainRLRequest(BaseModel):
    """Trigger a short RL training loop and return reward stats."""
    agent_type: str = Field(default="linucb")
    rounds: int = Field(default=50, ge=1, le=2000)
    top_k: int = Field(default=20, ge=1, le=100)
    horizon: int = Field(default=1, ge=1, le=30)


class FetchDataRequest(BaseModel):
    """Fetch data for assets across asset classes."""
    symbols: list[str] = Field(default_factory=list, max_length=200)
    source: str = Field(default="auto")  # auto|yfinance|coingecko|alpha_vantage|binance|quandl
    period: str = Field(default="6mo")
    interval: str = Field(default="1d")


class PredictionResponse(BaseModel):
    """Single prediction response"""
    symbol: str
    horizon: str
    predicted_price: float
    confidence: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)
    action: Action
    risk_applied: RiskProfile
    reason: str
    timestamp: str
    model_version: str

    class Config:
        use_enum_values = True
        protected_namespaces = ()  # Fixes the warning about 'model_version'    


class MCPResponse(BaseModel):
    """Standard MCP response wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class MCPAdapter:
    """MCP Adapter for prediction agent"""
    
    def __init__(self, agent, baseline_model, feature_pipeline):
        """
        Initialize MCP Adapter.
        
        Args:
            agent: RL agent (LinUCB, Thompson, or DQN)
            baseline_model: Baseline LightGBM model
            feature_pipeline: Feature pipeline for data loading
        """
        self.agent = agent
        self.baseline_model = baseline_model
        self.feature_pipeline = feature_pipeline
        self.request_log: List[Dict] = []
        
        logger.info("MCP Adapter initialized")

        # Create MCP tools registry
        self.create_mcp_tools_registry()

        # Simple MCP tool registry (for documentation/introspection)
        self.mcp_tools = [
            {"name": "predict", "path": "/tools/predict", "method": "POST"},
            {"name": "scan_all", "path": "/tools/scan_all", "method": "POST"},
            {"name": "analyze", "path": "/tools/analyze", "method": "POST"},
            {"name": "feedback", "path": "/tools/feedback", "method": "POST"},
            {"name": "train_rl", "path": "/tools/train_rl", "method": "POST"},
            {"name": "fetch_data", "path": "/tools/fetch_data", "method": "POST"},
        ]

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract feature columns (exclude OHLCV, metadata, targets, and timestamp columns)"""
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'adj_close',
            'symbol', 'source', 'fetch_timestamp', 'date', 'Date',
            'target', 'target_return', 'target_direction', 'target_binary'
        ]
        return [col for col in df.columns if col not in exclude_cols]

    def _log_request(self, tool_name: str, request_data: Dict, response_data: Dict):
        """Log request and response"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'tool': tool_name,
            'request': request_data,
            'response': response_data
        }
        self.request_log.append(log_entry)
        logger.info(f"Tool invoked: {tool_name}")

    def _action_from_index(self, action_idx: int) -> str:
        """Convert action index to action name"""
        action_map = {0: "short", 1: "hold", 2: "long"}
        return action_map.get(action_idx, "hold")

    def _generate_reason(self, symbol: str, features: Dict, action: str) -> str:
        """Generate human-readable reason for prediction"""
        reasons = []
        
        # RSI signals
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            if not pd.isna(rsi):
                if rsi > 70:
                    reasons.append("Overbought RSI")
                elif rsi < 30:
                    reasons.append("Oversold RSI")
        
        # MACD signals
        if 'macd_hist' in features:
            macd_hist = features['macd_hist']
            if not pd.isna(macd_hist) and macd_hist > 0:
                reasons.append("Positive MACD momentum")
        
        # Volume signals
        if 'volume_ratio' in features:
            vol_ratio = features['volume_ratio']
            if not pd.isna(vol_ratio) and vol_ratio > 1.5:
                reasons.append("High volume")
        
        # Price vs SMA
        if 'sma_20' in features and 'close' in features:
            sma = features['sma_20']
            close = features['close']
            if not pd.isna(sma) and not pd.isna(close):
                if close > sma:
                    reasons.append("Above 20-day SMA")
                else:
                    reasons.append("Below 20-day SMA")
        
        # Fallback
        if not reasons:
            reasons.append(f"RL signal: {action}")
        
        return " + ".join(reasons)

    def _apply_risk_adjustment(
        self, 
        score: float, 
        confidence: float, 
        risk_profile: RiskProfile
    ) -> Tuple[float, float]:
        """
        Apply risk-based adjustments to score and confidence.
        
        More conservative risk profile = lower adjusted scores
        """
        # Risk factor based on stop loss (tighter stop = more conservative)
        stop_loss_factor = 1.0 - (risk_profile.stop_loss_pct / 100.0) * 0.2
        
        # Capital risk factor
        capital_risk_factor = 1.0 - (risk_profile.capital_risk_pct / 100.0) * 0.15
        
        # Drawdown limit factor
        drawdown_factor = 1.0 - (risk_profile.drawdown_limit_pct / 100.0) * 0.1
        
        # Combined adjustment
        combined_factor = stop_loss_factor * capital_risk_factor * drawdown_factor
        
        adjusted_score = score * combined_factor
        adjusted_confidence = confidence * combined_factor
        
        return adjusted_score, adjusted_confidence

    def _extract_features_safely(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Safely extract features from dataframe.
        
        Returns:
            (feature_vector, feature_series) tuple
        """
        feature_cols = self._get_feature_columns(df)
        
        if not feature_cols:
            raise ValueError("No feature columns found in dataframe")
        
        # Get last row
        features_row = df[feature_cols].iloc[-1]
        
        # Convert to numpy array
        feature_vector = features_row.values
        
        # Clean data
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure consistent dimensions for RL agent
        if hasattr(self.agent, 'n_features'):
            expected_features = self.agent.n_features
            if len(feature_vector) != expected_features:
                logger.warning(f"Feature dimension mismatch: got {len(feature_vector)}, expected {expected_features}")
                # Pad or truncate to match expected size
                if len(feature_vector) < expected_features:
                    feature_vector = np.pad(feature_vector, (0, expected_features - len(feature_vector)), mode='constant')
                else:
                    feature_vector = feature_vector[:expected_features]
        
        return feature_vector, features_row

    def _get_baseline_prediction(
        self, 
        features_df: pd.DataFrame
    ) -> float:
        """
        Get baseline model prediction safely.
        
        Returns:
            Prediction probability (0-1)
        """
        if not self.baseline_model or not getattr(self.baseline_model, 'is_trained', False):
            return 0.5
        
        try:
            # Align features if model has feature names
            if self.baseline_model.feature_names:
                features_df = features_df.reindex(
                    columns=self.baseline_model.feature_names, 
                    fill_value=0
                )
            
            # Get prediction
            baseline_pred = self.baseline_model.predict_proba(features_df)[0]
            
            # Handle different output shapes
            if hasattr(baseline_pred, '__len__'):
                if len(baseline_pred) >= 3:
                    # Multi-class: use long class probability
                    return float(baseline_pred[2])
                elif len(baseline_pred) == 2:
                    # Binary: use positive class
                    return float(baseline_pred[1])
                else:
                    return float(baseline_pred[0])
            else:
                return float(baseline_pred)
                
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            return 0.5

    # ========================== PREDICT ==========================
    def predict(self, request: PredictRequest) -> MCPResponse:
        """
        Predict for specific symbols.
        
        Args:
            request: Prediction request
            
        Returns:
            MCPResponse with predictions
        """
        try:
            risk_profile = request.risk_profile or RiskProfile()
            predictions = []

            # Load feature store
            try:
                feature_dict = self.feature_pipeline.load_feature_store()
            except FileNotFoundError:
                return MCPResponse(
                    success=False, 
                    error="Feature store not found. Run data ingestion and feature generation first."
                )

            if not feature_dict:
                return MCPResponse(success=False, error="Feature store is empty")

            # Process each symbol
            for symbol in request.symbols:
                if symbol not in feature_dict:
                    logger.warning(f"Symbol {symbol} not found in feature store")
                    continue
                
                df = feature_dict[symbol]
                
                if df.empty:
                    logger.warning(f"Empty dataframe for {symbol}")
                    continue

                try:
                    # Extract features
                    feature_vector, feature_series = self._extract_features_safely(df)
                    
                    # RL ranking
                    contexts = {symbol: feature_vector}
                    
                    if hasattr(self.agent, 'action_dim'):
                        # DQN agent
                        rl_rankings = self.agent.rank_symbols(contexts, top_k=1)
                        if not rl_rankings:
                            continue
                        _, rl_score, action_idx, rl_confidence = rl_rankings[0]
                        action = self._action_from_index(action_idx)
                    else:
                        # Bandit agent
                        rl_rankings = self.agent.rank_symbols(contexts, top_k=1)
                        if not rl_rankings:
                            continue
                        _, rl_score, rl_confidence = rl_rankings[0]
                        action = "long" if rl_score > 0 else "short"
                    
                    # Sanitize RL outputs
                    rl_score = float(np.nan_to_num(rl_score, nan=0.0, posinf=0.0, neginf=0.0))
                    rl_confidence = float(np.nan_to_num(rl_confidence, nan=0.0, posinf=0.0, neginf=0.0))
                    
                    # Baseline prediction
                    features_df = feature_series.to_frame().T
                    baseline_pred = self._get_baseline_prediction(features_df)
                    
                    # Combine scores
                    combined_score = (rl_score + baseline_pred) / 2
                    combined_confidence = (rl_confidence + baseline_pred) / 2
                    
                    # Apply risk adjustment
                    adjusted_score, adjusted_confidence = self._apply_risk_adjustment(
                        combined_score, combined_confidence, risk_profile
                    )
                    
                    # Normalize to 0-1 range
                    safe_score = max(0.0, min(1.0, adjusted_score))
                    safe_confidence = max(0.0, min(1.0, adjusted_confidence))
                    
                    # Get current price
                    close_col = 'close' if 'close' in df.columns else 'Close'
                    if close_col not in df.columns:
                        logger.error(f"No close price column for {symbol}")
                        continue
                    
                    current_price = float(df[close_col].iloc[-1])
                    
                    # Predict future price
                    predicted_price = current_price * (1 + (safe_score - 0.5) * 0.1)
                    
                    # Generate reason
                    feature_dict_with_price = feature_series.to_dict()
                    feature_dict_with_price['close'] = current_price
                    reason = self._generate_reason(symbol, feature_dict_with_price, action)
                    
                    # Create response
                    pred_resp = PredictionResponse(
                        symbol=symbol,
                        horizon=request.horizon,
                        predicted_price=round(predicted_price, 2),
                        confidence=round(safe_confidence, 4),
                        score=round(safe_score, 4),
                        action=action,
                        risk_applied=risk_profile,
                        reason=reason,
                        timestamp=datetime.now().isoformat(),
                        model_version=f"{self.baseline_model.model_name}+{type(self.agent).__name__.lower()}-v1"
                    )
                    
                    predictions.append(pred_resp.model_dump())
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                    continue

            response = MCPResponse(
                success=True,
                data=predictions,
                metadata={
                    'total_predictions': len(predictions),
                    'requested_symbols': len(request.symbols),
                    'horizon': request.horizon
                }
            )
            
            self._log_request('predict', request.model_dump(), response.model_dump())
            
            return response

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    # ========================== FEEDBACK ==========================
    def feedback(self, request: FeedbackRequest) -> MCPResponse:
        """
        Apply user feedback to the RL agent by updating the latest context reward.

        Persists feedback into logs/feedback_loop.json and feedback_memory.json for continuity.
        """
        try:
            symbol = request.symbol.upper()
            correct = request.user_feedback.lower() == "correct"

            # Load feature store to get latest context
            feature_dict = self.feature_pipeline.load_feature_store()
            if symbol not in feature_dict or feature_dict[symbol].empty:
                return MCPResponse(success=False, error=f"No features for symbol {symbol}")

            df = feature_dict[symbol]
            feature_vector, _ = self._extract_features_safely(df)

            # Compute a shaped reward from feedback
            reward = 5.0 if correct else -5.0

            # If bandit-style agent
            if hasattr(self.agent, 'update') and not hasattr(self.agent, 'action_dim'):
                self.agent.update(symbol, feature_vector, reward)
            else:
                # DQN-style: store as a single transition with identical next_state
                try:
                    self.agent.store_transition(feature_vector, 1, reward, feature_vector, True)
                    self.agent.train_step()
                except Exception:
                    pass

            # Persist feedback
            from pathlib import Path
            import json
            from datetime import datetime
            Path('logs').mkdir(exist_ok=True)
            fb_path = Path('logs') / 'feedback_loop.json'
            entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'predicted_action': request.predicted_action,
                'user_feedback': request.user_feedback,
                'applied_reward': reward
            }
            try:
                if fb_path.exists():
                    data = json.loads(fb_path.read_text())
                else:
                    data = []
                data.append(entry)
                fb_path.write_text(json.dumps(data, indent=2))
            except Exception:
                pass

            response = MCPResponse(success=True, data={'updated': True, 'reward': reward})
            self._log_request('feedback', request.model_dump(), response.model_dump())
            return response
        except Exception as e:
            logger.error(f"Feedback error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    # ========================== TRAIN RL ==========================
    def train_rl(self, request: TrainRLRequest) -> MCPResponse:
        """Run a brief RL training loop and return stats."""
        try:
            from core.models.rl_agent import RLTrainer

            # Load feature store
            feature_dict = self.feature_pipeline.load_feature_store()
            trainer = RLTrainer(self.agent, feature_dict, agent_type=request.agent_type)

            if hasattr(self.agent, 'action_dim'):
                stats = trainer.train_dqn(
                    n_episodes=request.rounds,
                    max_steps=min(50, len(feature_dict)),
                    horizon=request.horizon
                )
            else:
                stats = trainer.train_bandit(
                    n_rounds=request.rounds,
                    top_k=request.top_k,
                    horizon=request.horizon
                )

            response = MCPResponse(success=True, data=stats)
            self._log_request('train_rl', request.model_dump(), response.model_dump())
            return response
        except Exception as e:
            logger.error(f"Train RL error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    # ========================== FETCH DATA ==========================
    def fetch_data(self, request: FetchDataRequest) -> MCPResponse:
        """Fetch batch data via data ingestion for requested symbols."""
        try:
            from core.data_ingest import DataIngestion
            ingestion = DataIngestion()

            symbols = request.symbols
            if not symbols:
                # Load default universe file if none provided
                try:
                    with open('universe.txt','r') as f:
                        symbols = [s.strip() for s in f if s.strip() and not s.startswith('#')]
                except FileNotFoundError:
                    symbols = []

            results = {}
            for sym in symbols:
                try:
                    # Decide handler by simple heuristic
                    if '-' in sym and sym.endswith('USD'):
                        df = ingestion.fetch_crypto_data(sym)
                    elif sym.endswith('=F'):
                        df = ingestion.fetch_commodity_data(sym)
                    else:
                        df = ingestion.fetch_yahoo_finance(sym, period=request.period, interval=request.interval)
                    results[sym] = (df is not None and not df.empty)
                except Exception:
                    results[sym] = False

            response = MCPResponse(success=True, data={'fetched': results})
            self._log_request('fetch_data', request.model_dump(), response.model_dump())
            return response
        except Exception as e:
            logger.error(f"Fetch data error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    # ========================== MCP TOOLS REGISTRY ==========================
    def create_mcp_tools_registry(self) -> Dict[str, Any]:
        """Create MCP tools registry as specified in the requirements"""
        mcp_tools = {
            "tools": [
                {
                    "name": "predict",
                    "description": "Generate trading predictions for given symbols",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symbols to predict"
                            },
                            "horizon": {
                                "type": "string",
                                "enum": ["intraday", "daily", "weekly", "monthly"],
                                "default": "daily",
                                "description": "Trading horizon"
                            },
                            "risk_profile": {
                                "type": "object",
                                "properties": {
                                    "stop_loss_pct": {"type": "number", "default": 2.0},
                                    "capital_risk_pct": {"type": "number", "default": 1.5},
                                    "drawdown_limit_pct": {"type": "number", "default": 10.0}
                                }
                            }
                        },
                        "required": ["symbols"]
                    }
                },
                {
                    "name": "scan_all",
                    "description": "Scan all symbols in universe and return top performers",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "top_k": {
                                "type": "integer",
                                "default": 10,
                                "description": "Number of top performers to return"
                            },
                            "horizon": {
                                "type": "string",
                                "enum": ["intraday", "daily", "weekly", "monthly"],
                                "default": "daily",
                                "description": "Trading horizon"
                            }
                        }
                    }
                },
                {
                    "name": "analyze",
                    "description": "Analyze a specific symbol with detailed metrics",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Symbol to analyze"
                            },
                            "horizon": {
                                "type": "string",
                                "enum": ["intraday", "daily", "weekly", "monthly"],
                                "default": "daily",
                                "description": "Trading horizon"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "feedback",
                    "description": "Provide feedback on RL agent predictions for learning",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Symbol that was predicted"
                            },
                            "predicted_action": {
                                "type": "string",
                                "enum": ["long", "short", "hold"],
                                "description": "Action predicted by RL agent"
                            },
                            "user_feedback": {
                                "type": "string",
                                "enum": ["correct", "incorrect"],
                                "description": "User's feedback on the prediction"
                            }
                        },
                        "required": ["symbol", "predicted_action", "user_feedback"]
                    }
                },
                {
                    "name": "train_rl",
                    "description": "Run RL training loop and return reward statistics",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_type": {
                                "type": "string",
                                "enum": ["linucb", "thompson", "dqn"],
                                "default": "linucb",
                                "description": "Type of RL agent to train"
                            },
                            "rounds": {
                                "type": "integer",
                                "default": 100,
                                "description": "Number of training rounds"
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 5,
                                "description": "Top K symbols for bandit training"
                            },
                            "horizon": {
                                "type": "string",
                                "enum": ["intraday", "daily", "weekly", "monthly"],
                                "default": "daily",
                                "description": "Trading horizon"
                            }
                        }
                    }
                },
                {
                    "name": "fetch_data",
                    "description": "Fetch batch data for symbols from multiple sources",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symbols to fetch (empty for universe)"
                            },
                            "period": {
                                "type": "string",
                                "default": "6mo",
                                "description": "Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"
                            },
                            "interval": {
                                "type": "string",
                                "default": "1d",
                                "description": "Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"
                            }
                        }
                    }
                }
            ],
            "version": "1.0.0",
            "description": "Prediction Agent MCP Tools Registry",
            "created_at": datetime.now().isoformat()
        }
        
        # Save to file
        registry_path = Path("core/mcp_tools.json")
        with open(registry_path, 'w') as f:
            json.dump(mcp_tools, f, indent=2)
        
        logger.info(f"MCP tools registry created: {registry_path}")
        return mcp_tools

    # ========================== SCAN ALL ==========================
    def scan_all(self, request: ScanAllRequest) -> MCPResponse:
        """
        Scan all symbols and return top-ranked.
        
        Args:
            request: Scan request
            
        Returns:
            MCPResponse with top symbols
        """
        try:
            risk_profile = request.risk_profile or RiskProfile()
            
            # Load feature store
            try:
                feature_dict = self.feature_pipeline.load_feature_store()
            except FileNotFoundError:
                return MCPResponse(
                    success=False,
                    error="Feature store not found. Run data ingestion first."
                )

            if not feature_dict:
                return MCPResponse(success=False, error="Feature store is empty")

            # Build contexts for all symbols
            contexts = {}
            symbol_data = {}
            
            for symbol, df in feature_dict.items():
                if df.empty:
                    continue
                
                try:
                    feature_vector, feature_series = self._extract_features_safely(df)
                    contexts[symbol] = feature_vector
                    symbol_data[symbol] = (feature_series, df.iloc[-1])
                except Exception as e:
                    logger.error(f"Error preparing {symbol}: {e}")
                    continue

            if not contexts:
                return MCPResponse(
                    success=False,
                    error="No valid contexts available for ranking"
                )

            # Rank using RL agent
            try:
                if hasattr(self.agent, 'action_dim'):
                    # DQN agent
                    rankings = self.agent.rank_symbols(contexts, top_k=None)
                else:
                    # Bandit agent
                    rankings = self.agent.rank_symbols(contexts, top_k=None)
            except Exception as e:
                logger.error(f"RL ranking failed: {e}")
                return MCPResponse(success=False, error=f"RL ranking failed: {e}")

            # Process rankings
            results = []
            
            for item in rankings:
                try:
                    if hasattr(self.agent, 'action_dim'):
                        # DQN: (symbol, score, action_idx, confidence)
                        symbol, rl_score, action_idx, rl_confidence = item
                        action = self._action_from_index(action_idx)
                    else:
                        # Bandit: (symbol, score, confidence)
                        symbol, rl_score, rl_confidence = item
                        action = "long" if rl_score > 0 else "short"
                    
                    # Sanitize
                    rl_score = float(np.nan_to_num(rl_score, nan=0.0, posinf=0.0, neginf=0.0))
                    rl_confidence = float(np.nan_to_num(rl_confidence, nan=0.0, posinf=0.0, neginf=0.0))
                    
                    # Get baseline prediction
                    features_series, last_row = symbol_data[symbol]
                    features_df = features_series.to_frame().T
                    baseline_pred = self._get_baseline_prediction(features_df)
                    
                    # Combine
                    combined_score = (rl_score + baseline_pred) / 2
                    combined_confidence = (rl_confidence + baseline_pred) / 2
                    
                    # Apply risk
                    adj_score, adj_conf = self._apply_risk_adjustment(
                        combined_score, combined_confidence, risk_profile
                    )
                    
                    # Normalize
                    adj_score = max(0.0, min(1.0, float(np.nan_to_num(adj_score, nan=0.0))))
                    adj_conf = max(0.0, min(1.0, float(np.nan_to_num(adj_conf, nan=0.0))))
                    
                    # Filter by min_score
                    if adj_score < request.min_score:
                        continue
                    
                    # Get price
                    close_col = 'close' if 'close' in last_row.index else 'Close'
                    if close_col in last_row.index:
                        price = float(last_row[close_col])
                        predicted_price = price * (1 + (adj_score - 0.5) * 0.1)
                    else:
                        predicted_price = None
                    
                    result = {
                        'symbol': symbol,
                        'score': round(adj_score, 4),
                        'confidence': round(adj_conf, 4),
                        'action': action,
                        'predicted_price': round(predicted_price, 2) if predicted_price else None,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing ranking for {symbol}: {e}")
                    continue

            # Keep only top_k
            results.sort(key=lambda r: r['score'], reverse=True)
            results = results[:request.top_k]
            
            # Save shortlist
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shortlist_path = Path("logs") / f"shortlist_{timestamp}.json"
            shortlist_path.parent.mkdir(exist_ok=True)
            
            with open(shortlist_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Shortlist saved to {shortlist_path}")
            
            response = MCPResponse(
                success=True,
                data=results,
                metadata={
                    'returned': len(results),
                    'scanned': len(contexts),
                    'shortlist_file': str(shortlist_path)
                }
            )
            
            self._log_request('scan_all', request.model_dump(), response.model_dump())
            
            return response

        except Exception as e:
            logger.error(f"Scan all error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    # ========================== ANALYZE ==========================
    def analyze(self, request: AnalyzeRequest) -> MCPResponse:
        """
        Analyze specific symbols with detailed technical indicators.
        
        Args:
            request: Analyze request
            
        Returns:
            MCPResponse with analysis
        """
        try:
            # Load feature store
            try:
                feature_dict = self.feature_pipeline.load_feature_store()
            except FileNotFoundError:
                return MCPResponse(
                    success=False,
                    error="Feature store not found. Run data ingestion first."
                )

            analyses = []
            
            for symbol in request.symbols:
                if symbol not in feature_dict or feature_dict[symbol].empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                try:
                    df = feature_dict[symbol]
                    
                    # Get current price
                    close_col = 'close' if 'close' in df.columns else 'Close'
                    if close_col not in df.columns:
                        continue
                    
                    price = float(df[close_col].iloc[-1])
                    
                    # Extract features
                    feature_vector, feature_series = self._extract_features_safely(df)
                    
                    # Technical signals
                    signals = {}
                    
                    # RSI
                    if 'rsi_14' in feature_series.index:
                        rsi = float(feature_series['rsi_14'])
                        signals['rsi_14'] = round(rsi, 2)
                        signals['rsi_state'] = (
                            'overbought' if rsi > 70 else
                            'oversold' if rsi < 30 else
                            'neutral'
                        )
                    
                    # MACD
                    if 'macd_hist' in feature_series.index:
                        signals['macd_hist'] = round(float(feature_series['macd_hist']), 4)
                    
                    # Volume
                    if 'volume_ratio' in feature_series.index:
                        signals['volume_ratio'] = round(float(feature_series['volume_ratio']), 2)
                    
                    # Moving averages
                    for ma in ['sma_20', 'sma_50', 'ema_20']:
                        if ma in feature_series.index:
                            signals[ma] = round(float(feature_series[ma]), 2)
                    
                    # RL prediction
                    contexts = {symbol: feature_vector}
                    
                    if hasattr(self.agent, 'action_dim'):
                        ranked = self.agent.rank_symbols(contexts, top_k=1)
                        _, rl_score, action_idx, rl_conf = ranked[0]
                        action = self._action_from_index(action_idx)
                    else:
                        ranked = self.agent.rank_symbols(contexts, top_k=1)
                        _, rl_score, rl_conf = ranked[0]
                        action = "long" if rl_score > 0 else "short"
                    
                    # Sanitize
                    rl_score = float(np.nan_to_num(rl_score, nan=0.0))
                    rl_conf = float(np.nan_to_num(rl_conf, nan=0.0))
                    
                    # Generate reason
                    feature_dict_with_price = feature_series.to_dict()
                    feature_dict_with_price['close'] = price
                    reason = self._generate_reason(symbol, feature_dict_with_price, action)
                    
                    analysis = {
                        'symbol': symbol,
                        'price': round(price, 2),
                        'signals': signals,
                        'rl_score': round(rl_score, 4),
                        'rl_confidence': round(rl_conf, 4),
                        'suggested_action': action,
                        'reason': reason,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    analyses.append(analysis)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            response = MCPResponse(
                success=True,
                data=analyses,
                metadata={'count': len(analyses)}
            )
            
            self._log_request('analyze', request.model_dump(), response.model_dump())
            
            return response
            
        except Exception as e:
            logger.error(f"Analyze error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def get_request_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent request logs"""
        return self.request_log[-limit:]

    def clear_logs(self):
        """Clear request logs"""
        self.request_log.clear()


# JSON-RPC helpers
def create_jsonrpc_request(method: str, params: Dict, request_id: int = 1) -> Dict:
    """Create JSON-RPC request"""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": request_id
    }


def create_jsonrpc_response(
    result: Any, 
    request_id: int = 1, 
    error: Optional[str] = None
) -> Dict:
    """Create JSON-RPC response"""
    response = {"jsonrpc": "2.0", "id": request_id}
    
    if error:
        response["error"] = {"code": -32000, "message": error}
    else:
        response["result"] = result
    
    return response