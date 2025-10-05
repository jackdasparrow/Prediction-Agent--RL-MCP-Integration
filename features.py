"""
Feature Engineering Pipeline
Computes technical indicators and statistical features for market data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Try to import TA-Lib, fall back to ta if not available
try:
    import talib
    USE_TALIB = True
except ImportError:
    import ta
    USE_TALIB = False
    logging.warning("TA-Lib not found, using 'ta' library instead")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Generates technical analysis features and statistical features from OHLCV data.
    Supports both TA-Lib and 'ta' library for flexibility.
    Works with Yahoo Finance and Alpha Vantage data.
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        feature_store_dir: str = "data/features"
    ):
        """
        Initialize FeaturePipeline.
        
        Args:
            cache_dir: Directory containing cached OHLCV data
            feature_store_dir: Directory to save computed features
        """
        self.cache_dir = Path(cache_dir)
        self.feature_store_dir = Path(feature_store_dir)
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_talib = USE_TALIB
        logger.info(f"Using {'TA-Lib' if self.use_talib else 'ta library'} for indicators")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names from different data sources.
        Works with both Yahoo Finance and Alpha Vantage.
        """
        df = df.copy()
        
        # Mapping for various column name formats
        column_mapping = {
            # Yahoo Finance format (Title Case)
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            
            # Alpha Vantage format (already lowercase in our ingestion)
            # But handle if someone uses raw API
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume',
        }
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        # Ensure lowercase
        df.columns = df.columns.str.lower()
        
        return df
    
    def compute_sma(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Compute Simple Moving Averages."""
        for period in periods:
            if len(df) < period:
                logger.warning(f"Insufficient data for SMA_{period} (need {period}, have {len(df)})")
                df[f'sma_{period}'] = np.nan
                continue
                
            if self.use_talib:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            else:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def compute_ema(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """Compute Exponential Moving Averages."""
        for period in periods:
            if len(df) < period:
                logger.warning(f"Insufficient data for EMA_{period}")
                df[f'ema_{period}'] = np.nan
                continue
                
            if self.use_talib:
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            else:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def compute_rsi(self, df: pd.DataFrame, periods: List[int] = [14, 21]) -> pd.DataFrame:
        """Compute Relative Strength Index."""
        for period in periods:
            if len(df) < period + 1:
                logger.warning(f"Insufficient data for RSI_{period}")
                df[f'rsi_{period}'] = np.nan
                continue
                
            if self.use_talib:
                df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
            else:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 1e-10)
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD (Moving Average Convergence Divergence)."""
        if len(df) < 26:
            logger.warning("Insufficient data for MACD (need 26 rows)")
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_hist'] = np.nan
            return df
            
        if self.use_talib:
            macd, signal, hist = talib.MACD(
                df['close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
        else:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def compute_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        if len(df) < period:
            logger.warning(f"Insufficient data for Bollinger Bands (need {period})")
            df['bb_upper'] = np.nan
            df['bb_middle'] = np.nan
            df['bb_lower'] = np.nan
            df['bb_width'] = np.nan
            return df
            
        if self.use_talib:
            upper, middle, lower = talib.BBANDS(
                df['close'],
                timeperiod=period,
                nbdevup=std,
                nbdevdn=std,
                matype=0
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
        else:
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['bb_middle'] + (rolling_std * std)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * std)
        
        # Bollinger Band Width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        
        return df
    
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average True Range."""
        if len(df) < period:
            logger.warning(f"Insufficient data for ATR (need {period})")
            df['atr'] = np.nan
            return df
            
        if self.use_talib:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        else:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=period).mean()
        return df
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages
        if len(df) >= 20:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        else:
            df['volume_sma_20'] = np.nan
            df['volume_ratio'] = np.nan
        
        # On-Balance Volume (OBV)
        if self.use_talib:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        else:
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['obv'] = obv
        
        return df
    
    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based statistical features."""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['price_change_20'] = df['close'].pct_change(periods=20)
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        
        # Close position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Rolling volatility
        if len(df) >= 10:
            df['volatility_10'] = df['price_change'].rolling(window=10).std()
        else:
            df['volatility_10'] = np.nan
            
        if len(df) >= 20:
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
        else:
            df['volatility_20'] = np.nan
        
        return df
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators."""
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            if len(df) < period:
                df[f'roc_{period}'] = np.nan
                continue
                
            if self.use_talib:
                df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
            else:
                df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
        
        # Momentum
        if len(df) >= 10:
            if self.use_talib:
                df['momentum'] = talib.MOM(df['close'], timeperiod=10)
            else:
                df['momentum'] = df['close'] - df['close'].shift(10)
        else:
            df['momentum'] = np.nan
        
        return df
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all computed features
        """
        logger.info(f"Computing features for {len(df)} rows")
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Verify required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Verify minimum data
        if len(df) < 50:
            logger.warning(f"Limited data ({len(df)} rows). Features may not be reliable.")
        
        # Compute all feature groups
        try:
            df = self.compute_sma(df)
            df = self.compute_ema(df)
            df = self.compute_rsi(df)
            df = self.compute_macd(df)
            df = self.compute_bollinger_bands(df)
            df = self.compute_atr(df)
            df = self.compute_volume_features(df)
            df = self.compute_price_features(df)
            df = self.compute_momentum_features(df)
        except Exception as e:
            logger.error(f"Error computing features: {e}", exc_info=True)
            raise
        
        # Handle NaN values
        initial_rows = len(df)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['open', 'high', 'low', 'close', 'volume', 
                        'symbol', 'source', 'fetch_timestamp', 'adj_close']]
        
        # Drop rows where ALL features are NaN (typically first few rows)
        df = df.dropna(subset=feature_cols, how='all')
        
        # Forward fill then backward fill remaining NaN
        df[feature_cols] = df[feature_cols].ffill().bfill()
        
        # Replace any remaining inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with insufficient data")
        
        # Generate target variables
        df = self.generate_targets(df)
        
        logger.info(f"Feature computation complete: {len(df)} rows, {df.shape[1]} features")
        
        return df
    
    def generate_targets(self, df: pd.DataFrame, lookahead_days: int = 5) -> pd.DataFrame:
        """
        Generate target variables for supervised learning.
        
        Args:
            df: DataFrame with OHLCV and features
            lookahead_days: Number of days to look ahead for target
            
        Returns:
            DataFrame with target columns added
        """
        df = df.copy()
        
        # Ensure we have enough data
        if len(df) < lookahead_days + 1:
            logger.warning(f"Insufficient data for target generation (need {lookahead_days + 1}, have {len(df)})")
            df['target_return'] = np.nan
            df['target_direction'] = 1
            df['target_binary'] = 0
            return df
        
        # Calculate future returns
        df['future_close'] = df['close'].shift(-lookahead_days)
        df['target_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Generate classification targets
        # 0: short (negative return < -2%), 1: hold (-2% to 2%), 2: long (positive return > 2%)
        df['target_direction'] = 1  # Default to hold
        df.loc[df['target_return'] < -0.02, 'target_direction'] = 0  # Short
        df.loc[df['target_return'] > 0.02, 'target_direction'] = 2   # Long
        
        # Generate binary target (up/down)
        df['target_binary'] = (df['target_return'] > 0).astype(int)
        
        # Clean up temporary columns
        df = df.drop(['future_close'], axis=1)
        
        return df
    
    def process_universe(
        self,
        raw_data: Dict[str, pd.DataFrame],
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process raw market data and compute features for entire universe.
        Works with both Yahoo Finance and Alpha Vantage data.
        
        Args:
            raw_data: Dictionary mapping symbols to raw OHLCV DataFrames
            save: Whether to save processed features to feature store
            
        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        logger.info(f"Processing features for {len(raw_data)} symbols")
        
        results = {}
        all_features = []
        failed_symbols = []
        
        for symbol, df in raw_data.items():
            try:
                logger.info(f"Processing features for {symbol}")
                
                if df.empty:
                    logger.warning(f"Empty dataframe for {symbol}, skipping")
                    failed_symbols.append(symbol)
                    continue
                
                # Compute all features
                feature_df = self.compute_all_features(df)
                
                # Add metadata
                feature_df['symbol'] = symbol
                
                results[symbol] = feature_df
                all_features.append(feature_df)
                
                logger.info(f"Generated {len(feature_df.columns)} features for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}", exc_info=True)
                failed_symbols.append(symbol)
                continue
        
        # Save to feature store if requested
        if save and all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            store_path = self.feature_store_dir / "feature_store.parquet"
            combined_df.to_parquet(store_path, compression='snappy')
            logger.info(f"Feature store saved: {store_path} ({len(combined_df)} rows)")
        
        # Summary
        logger.info(f"Successfully processed {len(results)}/{len(raw_data)} symbols")
        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
        
        return results
    
    def load_feature_store(self) -> Dict[str, pd.DataFrame]:
        """
        Load the combined feature store and return as dictionary.
        
        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        store_path = self.feature_store_dir / "feature_store.parquet"
        
        if not store_path.exists():
            raise FileNotFoundError(f"Feature store not found: {store_path}")
        
        df = pd.read_parquet(store_path)
        logger.info(f"Loaded feature store: {len(df)} rows, {df.shape[1]} columns")
        
        # Convert to dictionary grouped by symbol
        if 'symbol' not in df.columns:
            logger.warning("Feature store missing 'symbol' column")
            return {'UNKNOWN': df}
        
        feature_dict = {}
        for symbol, group in df.groupby('symbol'):
            feature_dict[symbol] = group.drop(columns=['symbol'], errors='ignore')
        
        logger.info(f"Feature store contains {len(feature_dict)} symbols")
        
        return feature_dict


def main():
    """Test feature pipeline."""
    print("\n" + "="*60)
    print("FEATURE PIPELINE TEST")
    print("="*60)
    
    # Check for cached data
    cache_dir = Path("data/cache")
    cache_files = list(cache_dir.glob("*_yfinance.parquet")) + \
                  list(cache_dir.glob("*_alphavantage.parquet"))
    
    if not cache_files:
        print("\n[ERROR] No cached data found!")
        print(f"Cache directory: {cache_dir}")
        print("\nPlease run data ingestion first:")
        print("  python fetch_more_data.py")
        return
    
    print(f"\n[1] Found {len(cache_files)} cached files")
    
    # Load cached data
    print("\n[2] Loading cached data...")
    raw_data = {}
    
    for cache_file in cache_files:
        try:
            symbol = cache_file.stem.split('_')[0]
            df = pd.read_parquet(cache_file)
            raw_data[symbol] = df
            print(f"    Loaded {symbol}: {len(df)} rows")
        except Exception as e:
            print(f"    Failed to load {cache_file.name}: {e}")
    
    if not raw_data:
        print("\n[ERROR] No data could be loaded!")
        return
    
    # Initialize pipeline
    print("\n[3] Initializing feature pipeline...")
    pipeline = FeaturePipeline()
    
    # Process data
    print("\n[4] Computing features...")
    print("="*60)
    results = pipeline.process_universe(raw_data, save=True)
    
    # Show results
    print("\n" + "="*60)
    print("FEATURE COMPUTATION RESULTS")
    print("="*60)
    
    if results:
        print(f"\n[OK] Successfully processed {len(results)} symbols:")
        
        for symbol, df in list(results.items())[:5]:  # Show first 5
            print(f"\n  {symbol}:")
            print(f"    - Rows: {len(df)}")
            print(f"    - Features: {df.shape[1]}")
            
            # Show sample features
            feature_cols = [col for col in df.columns if col not in 
                          ['symbol', 'open', 'high', 'low', 'close', 'volume', 
                           'source', 'fetch_timestamp', 'adj_close']]
            print(f"    - Sample features: {', '.join(feature_cols[:5])}...")
        
        if len(results) > 5:
            print(f"\n  ... and {len(results) - 5} more symbols")
        
        # Load and display feature store info
        print("\n" + "="*60)
        print("FEATURE STORE INFO")
        print("="*60)
        
        feature_store = pipeline.load_feature_store()
        total_rows = sum(len(df) for df in feature_store.values())
        
        print(f"\n[OK] Feature store loaded:")
        print(f"  - Total rows: {total_rows}")
        print(f"  - Symbols: {len(feature_store)}")
        
        # Show feature categories
        print("\n[OK] Feature categories:")
        sample_df = next(iter(feature_store.values()))
        feature_cols = [col for col in sample_df.columns if col not in 
                       ['source', 'fetch_timestamp', 'adj_close', 
                        'target_return', 'target_direction', 'target_binary']]
        
        categories = {
            'SMA': [c for c in feature_cols if c.startswith('sma_')],
            'EMA': [c for c in feature_cols if c.startswith('ema_')],
            'RSI': [c for c in feature_cols if c.startswith('rsi_')],
            'MACD': [c for c in feature_cols if 'macd' in c],
            'Bollinger': [c for c in feature_cols if 'bb_' in c],
            'Volume': [c for c in feature_cols if 'volume' in c or 'obv' in c],
            'Price': [c for c in feature_cols if 'price' in c or 'hl_range' in c or 'close_position' in c],
            'Momentum': [c for c in feature_cols if 'roc_' in c or 'momentum' in c],
            'Other': [c for c in feature_cols if c.startswith('atr') or 'volatility' in c]
        }
        
        for category, features in categories.items():
            if features:
                print(f"  - {category}: {len(features)} features")
        
        print("\n" + "="*60)
        print("[OK] Feature pipeline test complete!")
        print("="*60)
        print(f"\nFeature store saved to: data/features/feature_store.parquet")
        print("You can now run: python core/models/baseline_lightgbm.py")
        
    else:
        print("\n[ERROR] No features were computed!")


if __name__ == "__main__":
    main()