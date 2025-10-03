"""
MCP (Model Context Protocol) Adapter
Provides JSON-RPC style wrappers for the prediction agent to be called as a tool.
"""

import json
import pandas as pd
import numpy as np
import logging
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
        self.agent = agent
        self.baseline_model = baseline_model
        self.feature_pipeline = feature_pipeline
        self.request_log: List[Dict] = []

    def _log_request(self, tool_name: str, request_data: Dict, response_data: Dict):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'tool': tool_name,
            'request': request_data,
            'response': response_data
        }
        self.request_log.append(log_entry)
        logger.info(f"Tool invoked: {tool_name}")

    def _action_from_index(self, action_idx: int) -> str:
        action_map = {0: "short", 1: "hold", 2: "long"}
        return action_map.get(action_idx, "hold")

    def _generate_reason(self, symbol: str, features: Dict, action: str) -> str:
        reasons = []
        if 'RSI_14' in features:
            rsi = features['RSI_14']
            if rsi > 70:
                reasons.append("Overbought RSI")
            elif rsi < 30:
                reasons.append("Oversold RSI")
        if 'MACD_hist' in features and features['MACD_hist'] > 0:
            reasons.append("Positive MACD")
        if 'volume_ratio' in features and features['volume_ratio'] > 1.5:
            reasons.append("High volume")
        if 'SMA_20' in features and 'Close' in features:
            if features['Close'] > features['SMA_20']:
                reasons.append("Above SMA")
        if not reasons:
            reasons.append(f"RL signal: {action}")
        return " + ".join(reasons)

    def _apply_risk_adjustment(self, score: float, confidence: float, risk_profile: RiskProfile) -> Tuple[float, float]:
        risk_factor = 1.0 - (risk_profile.stop_loss_pct / 100.0) * 0.1
        adjusted_score = score * risk_factor
        adjusted_confidence = confidence * (1.0 - risk_profile.capital_risk_pct / 100.0)
        return adjusted_score, adjusted_confidence

    # ========================== PREDICT ==========================
    def predict(self, request: PredictRequest) -> MCPResponse:
        try:
            risk_profile = request.risk_profile or RiskProfile()
            predictions = []

            # Load feature store
            try:
                feature_store_df = self.feature_pipeline.load_feature_store()
                if isinstance(feature_store_df, dict):
                    feature_store = feature_store_df
                else:
                    if 'symbol' not in feature_store_df.columns:
                        return MCPResponse(success=False, error="Feature store missing 'symbol' column")
                    feature_store = {sym: df.drop(columns=['symbol'], errors='ignore') for sym, df in feature_store_df.groupby('symbol')}
            except FileNotFoundError:
                return MCPResponse(success=False, error="Feature store not found. Run data ingestion first.")

            if not feature_store:
                return MCPResponse(success=False, error="Feature store is empty")

            for symbol in request.symbols:
                if symbol not in feature_store:
                    logger.warning(f"Symbol {symbol} not found in feature store")
                    continue
                df = feature_store[symbol]
                if df.empty:
                    continue

                # Keep 2D DataFrame for model
                feature_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume','source','fetch_timestamp','target','target_direction']]
                features_df = df[feature_cols].iloc[[-1]]
                # Clean data: ensure numeric, handle NaN/inf
                try:
                    features_df = features_df.apply(pd.to_numeric, errors='coerce') if 'pd' in globals() else features_df
                except Exception:
                    pass
                features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)
                features_series = features_df.iloc[0]
                feature_vector = np.nan_to_num(features_df.values[0], nan=0.0, posinf=0.0, neginf=0.0)

                # RL ranking
                contexts = {symbol: feature_vector}
                try:
                    rl_rankings = self.agent.rank_symbols(contexts, top_k=1)
                except Exception as e:
                    logger.error(f"RL ranking failed for {symbol}: {e}")
                    continue
                if not rl_rankings:
                    continue

                if hasattr(self.agent, 'action_dim'):
                    _, rl_score, action_idx, rl_confidence = rl_rankings[0]
                    action = self._action_from_index(action_idx)
                else:
                    _, rl_score, rl_confidence = rl_rankings[0]
                    action = "long" if rl_score > 0 else "short"

                # Baseline prediction safely
                try:
                    if self.baseline_model and getattr(self.baseline_model, 'is_trained', False):
                        if self.baseline_model.feature_names:
                            features_df_ordered = features_df.reindex(columns=self.baseline_model.feature_names, fill_value=0)
                            baseline_pred = self.baseline_model.predict_proba(features_df_ordered)[0]
                        else:
                            baseline_pred = self.baseline_model.predict_proba(features_df)[0]
                        # For multiclass, use long class (index 2) probability if available, else mean
                        if hasattr(baseline_pred, '__len__') and len(baseline_pred) >= 3:
                            baseline_pred = float(baseline_pred[2])
                        elif hasattr(baseline_pred, '__len__') and len(baseline_pred) == 2:
                            baseline_pred = float(baseline_pred[1])
                        else:
                            baseline_pred = float(baseline_pred)
                    else:
                        baseline_pred = 0.5
                except Exception as e:
                    logger.error(f"Baseline prediction failed for {symbol}: {e}")
                    baseline_pred = 0.5

                combined_score = (rl_score + baseline_pred)/2
                combined_confidence = (rl_confidence + baseline_pred)/2
                adjusted_score, adjusted_confidence = self._apply_risk_adjustment(combined_score, combined_confidence, risk_profile)
                # Sanitize values to avoid NaN/inf in API response
                adjusted_score = float(np.nan_to_num(adjusted_score, nan=0.0, posinf=0.0, neginf=0.0))
                adjusted_confidence = float(np.nan_to_num(adjusted_confidence, nan=0.0, posinf=0.0, neginf=0.0))
                safe_score = max(0.0, min(1.0, adjusted_score))
                safe_confidence = max(0.0, min(1.0, adjusted_confidence))

                close_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
                if close_col is None:
                    raise ValueError("Missing Close/close column")
                current_price = df[close_col].iloc[-1]
                predicted_price = current_price * (1 + safe_score * 0.05)

                feature_dict = features_series.to_dict()
                feature_dict['Close'] = current_price
                reason = self._generate_reason(symbol, feature_dict, action)

                pred_resp = PredictionResponse(
                    symbol=symbol,
                    horizon=request.horizon,
                    predicted_price=round(predicted_price,2),
                    confidence=safe_confidence,
                    score=safe_score,
                    action=action,
                    risk_applied=risk_profile,
                    reason=reason,
                    timestamp=datetime.now().isoformat(),
                    model_version=f"{self.baseline_model.model_name}+{type(self.agent).__name__.lower()}-v1"
                )

                predictions.append(pred_resp.model_dump())

            return MCPResponse(success=True, data=predictions, metadata={'total_predictions': len(predictions)})

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    # ========================== SCAN ALL ==========================
    def scan_all(self, request: ScanAllRequest) -> MCPResponse:
        try:
            risk_profile = request.risk_profile or RiskProfile()
            # Load feature store
            try:
                feature_store_df = self.feature_pipeline.load_feature_store()
                if 'symbol' not in feature_store_df.columns:
                    return MCPResponse(success=False, error="Feature store missing 'symbol' column")
                grouped = {sym: df.drop(columns=['symbol'], errors='ignore') for sym, df in feature_store_df.groupby('symbol')}
            except FileNotFoundError:
                return MCPResponse(success=False, error="Feature store not found. Run data ingestion first.")

            if not grouped:
                return MCPResponse(success=False, error="Feature store is empty")

            # Build contexts
            contexts = {}
            latest_rows = {}
            for symbol, df in grouped.items():
                if df.empty:
                    continue
                feature_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume','source','fetch_timestamp','target','target_direction']]
                row = df[feature_cols].iloc[[-1]]
                # Clean numeric
                try:
                    row = row.apply(pd.to_numeric, errors='coerce') if 'pd' in globals() else row
                except Exception:
                    pass
                row = row.replace([np.inf, -np.inf], 0).fillna(0)
                contexts[symbol] = np.nan_to_num(row.values[0], nan=0.0, posinf=0.0, neginf=0.0)
                latest_rows[symbol] = (row.iloc[0], df.iloc[-1])

            if not contexts:
                return MCPResponse(success=False, error="No contexts available for ranking")

            # Rank using RL agent
            try:
                rankings = self.agent.rank_symbols(contexts, top_k=None)
            except Exception as e:
                logger.error(f"RL ranking failed: {e}")
                return MCPResponse(success=False, error=f"RL ranking failed: {e}")

            results = []
            for item in rankings:
                if hasattr(self.agent, 'action_dim'):
                    symbol, rl_score, action_idx, rl_confidence = item
                    action = self._action_from_index(action_idx)
                else:
                    symbol, rl_score, rl_confidence = item
                    action = "long" if rl_score > 0 else "short"

                # Baseline blending
                features_series, last_row = latest_rows[symbol]
                features_df = features_series.to_frame().T
                try:
                    if self.baseline_model and getattr(self.baseline_model, 'is_trained', False):
                        if self.baseline_model.feature_names:
                            features_df_ordered = features_df.reindex(columns=self.baseline_model.feature_names, fill_value=0)
                            baseline_pred = self.baseline_model.predict_proba(features_df_ordered)[0]
                        else:
                            baseline_pred = self.baseline_model.predict_proba(features_df)[0]
                        if hasattr(baseline_pred, '__len__') and len(baseline_pred) >= 3:
                            baseline_pred = float(baseline_pred[2])
                        elif hasattr(baseline_pred, '__len__') and len(baseline_pred) == 2:
                            baseline_pred = float(baseline_pred[1])
                        else:
                            baseline_pred = float(baseline_pred)
                    else:
                        baseline_pred = 0.5
                except Exception as e:
                    logger.error(f"Baseline prediction failed for {symbol}: {e}")
                    baseline_pred = 0.5

                combined_score = (rl_score + baseline_pred)/2
                combined_confidence = (rl_confidence + baseline_pred)/2
                adj_score, adj_conf = self._apply_risk_adjustment(combined_score, combined_confidence, risk_profile)
                # Sanitize
                adj_score = float(np.nan_to_num(adj_score, nan=0.0, posinf=0.0, neginf=0.0))
                adj_conf = float(np.nan_to_num(adj_conf, nan=0.0, posinf=0.0, neginf=0.0))
                adj_score = max(0.0, min(1.0, adj_score))
                adj_conf = max(0.0, min(1.0, adj_conf))

                if adj_score < request.min_score:
                    continue

                close_col = 'Close' if 'Close' in last_row.index else ('close' if 'close' in last_row.index else None)
                price = float(last_row[close_col]) if close_col else None
                predicted_price = price * (1 + adj_score * 0.05) if price is not None else None

                result = {
                    'symbol': symbol,
                    'score': adj_score,
                    'confidence': adj_conf,
                    'action': action,
                    'predicted_price': round(float(predicted_price), 2) if predicted_price is not None else None,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

            # Keep top_k
            results.sort(key=lambda r: r['score'], reverse=True)
            results = results[: request.top_k]

            return MCPResponse(success=True, data=results, metadata={'returned': len(results)})

        except Exception as e:
            logger.error(f"Scan all error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    # ========================== ANALYZE ==========================
    def analyze(self, request: AnalyzeRequest) -> MCPResponse:
        try:
            # Load feature store
            try:
                feature_store_df = self.feature_pipeline.load_feature_store()
                if 'symbol' not in feature_store_df.columns:
                    return MCPResponse(success=False, error="Feature store missing 'symbol' column")
                feature_store = {sym: df.drop(columns=['symbol'], errors='ignore') for sym, df in feature_store_df.groupby('symbol')}
            except FileNotFoundError:
                return MCPResponse(success=False, error="Feature store not found. Run data ingestion first.")

            analyses = []
            for symbol in request.symbols:
                if symbol not in feature_store or feature_store[symbol].empty:
                    continue
                df = feature_store[symbol]
                close_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
                price = float(df[close_col].iloc[-1]) if close_col else None

                # Signals
                signals = {}
                colmap = {c.lower(): c for c in df.columns}
                # RSI
                rsi_col = colmap.get('rsi_14')
                if rsi_col:
                    rsi = float(df[rsi_col].iloc[-1])
                    signals['rsi'] = rsi
                    signals['rsi_state'] = 'overbought' if rsi > 70 else ('oversold' if rsi < 30 else 'neutral')
                # MACD
                macd_hist_col = colmap.get('macd_hist')
                if macd_hist_col:
                    signals['macd_hist'] = float(df[macd_hist_col].iloc[-1])
                # Volume ratio
                vol_ratio_col = colmap.get('volume_ratio')
                if vol_ratio_col:
                    signals['volume_ratio'] = float(df[vol_ratio_col].iloc[-1])

                # Build context and get RL suggestion
                feature_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume','source','fetch_timestamp','target','target_direction']]
                context = df[feature_cols].iloc[-1].values
                context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
                if hasattr(self.agent, 'action_dim'):
                    ranked = self.agent.rank_symbols({symbol: context}, top_k=1)
                    _, rl_score, action_idx, rl_conf = ranked[0]
                    action = self._action_from_index(action_idx)
                else:
                    ranked = self.agent.rank_symbols({symbol: context}, top_k=1)
                    _, rl_score, rl_conf = ranked[0]
                    action = "long" if rl_score > 0 else "short"
                # Sanitize RL outputs
                rl_score = float(np.nan_to_num(rl_score, nan=0.0, posinf=0.0, neginf=0.0))
                rl_conf = float(np.nan_to_num(rl_conf, nan=0.0, posinf=0.0, neginf=0.0))

                feature_series = df[feature_cols].iloc[-1]
                feature_dict = feature_series.to_dict()
                if price is not None:
                    feature_dict['Close'] = price
                reason = self._generate_reason(symbol, feature_dict, action)

                analyses.append({
                    'symbol': symbol,
                    'price': price,
                    'signals': signals,
                    'rl_score': float(rl_score),
                    'rl_confidence': float(rl_conf),
                    'suggested_action': action,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                })

            return MCPResponse(success=True, data=analyses, metadata={'count': len(analyses)})
        except Exception as e:
            logger.error(f"Analyze error: {e}", exc_info=True)
            return MCPResponse(success=False, error=str(e))

    def get_request_logs(self, limit: int = 100) -> List[Dict]:
        return self.request_log[-limit:]

    def clear_logs(self):
        self.request_log.clear()


# JSON-RPC helpers
def create_jsonrpc_request(method: str, params: Dict, request_id: int = 1) -> Dict:
    return {"jsonrpc":"2.0","method":method,"params":params,"id":request_id}

def create_jsonrpc_response(result: Any, request_id: int = 1, error: Optional[str] = None) -> Dict:
    response = {"jsonrpc":"2.0","id":request_id}
    if error:
        response["error"] = {"code":-32000,"message":error}
    else:
        response["result"] = result
    return response
