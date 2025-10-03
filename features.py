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
    
    def compute_sma(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Compute Simple Moving Averages."""
        for period in periods:
            if self.use_talib:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            else:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def compute_ema(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Compute Exponential Moving Averages."""
        for period in periods:
            if self.use_talib:
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            else:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    def compute_rsi(self, df: pd.DataFrame, periods: List[int] = [14]) -> pd.DataFrame:
        """Compute Relative Strength Index."""
        for period in periods:
            if self.use_talib:
                df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
            else:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD (Moving Average Convergence Divergence)."""
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
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average True Range."""
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
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
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
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Close position within range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Rolling volatility
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        return df
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum indicators."""
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            if self.use_talib:
                df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
            else:
                df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
        
        # Momentum
        if self.use_talib:
            df['momentum'] = talib.MOM(df['close'], timeperiod=10)
        else:
            df['momentum'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with all computed features
        """
        logger.info(f"Computing features for {len(df)} rows")
        
        # Ensure required columns are lowercase
        df = df.copy()
        
        # Check if we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        # Rename columns to lowercase if needed
        rename_map = {}
        for req_col in required_cols:
            if req_col not in df.columns and req_col in df_cols_lower:
                rename_map[df_cols_lower[req_col]] = req_col
        
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(f"Renamed columns: {rename_map}")
        
        # Verify we have minimum required data
        if len(df) < 50:
            logger.warning(f"Very few rows ({len(df)}). Features may not be reliable.")
        
        # Compute all feature groups
        df = self.compute_sma(df)
        df = self.compute_ema(df)
        df = self.compute_rsi(df)
        df = self.compute_macd(df)
        df = self.compute_bollinger_bands(df)
        df = self.compute_atr(df)
        df = self.compute_volume_features(df)
        df = self.compute_price_features(df)
        df = self.compute_momentum_features(df)
        
        # Drop rows with NaN values (from rolling windows)
        # Keep at least the beginning of data for context
        initial_rows = len(df)
        
        # Drop only rows where ALL feature columns are NaN
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'source', 'fetch_timestamp']]
        df = df.dropna(subset=feature_cols, how='all')
        
        # Then drop rows with any remaining NaN in critical features
        critical_features = ['close', 'volume']
        df = df.dropna(subset=critical_features)
        
        # Fill remaining NaN with forward fill then backward fill
        df = df.ffill().bfill()
        
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")
        
        # Generate target variables for supervised learning
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
    
    def process_cached_data(
        self,
        symbols: Optional[List[str]] = None,
        save_to_store: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all cached data and compute features.
        
        Args:
            symbols: List of symbols to process (None = all cached symbols)
            save_to_store: Save combined feature store to disk
            
        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        # Get cached files
        cache_files = list(self.cache_dir.glob("*_yfinance.parquet"))
        
        if not cache_files:
            logger.warning(f"No cached data found in {self.cache_dir}")
            return {}
        
        # Filter by symbols if provided
        if symbols:
            cache_files = [f for f in cache_files if any(f.stem.startswith(s) for s in symbols)]
        
        logger.info(f"Processing {len(cache_files)} cached files")
        
        results = {}
        all_features = []
        
        for cache_file in cache_files:
            symbol = cache_file.stem.split('_')[0]
            
            try:
                # Load cached data
                df = pd.read_parquet(cache_file)
                logger.info(f"Processing {symbol}: {len(df)} rows")
                
                # Compute features
                features_df = self.compute_all_features(df)
                
                # Add symbol column
                features_df['symbol'] = symbol
                
                results[symbol] = features_df
                all_features.append(features_df)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Save combined feature store
        if save_to_store and all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            store_path = self.feature_store_dir / "feature_store.parquet"
            combined_df.to_parquet(store_path, compression='snappy')
            logger.info(f"Feature store saved: {store_path} ({len(combined_df)} rows)")
        
        return results
    
    def process_universe(
        self,
        raw_data: Dict[str, pd.DataFrame],
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process raw market data and compute features for entire universe.
        
        Args:
            raw_data: Dictionary mapping symbols to raw OHLCV DataFrames
            save: Whether to save processed features to feature store
            
        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        logger.info(f"Processing features for {len(raw_data)} symbols")
        
        results = {}
        all_features = []
        
        for symbol, df in raw_data.items():
            try:
                logger.info(f"Processing features for {symbol}")
                
                # Ensure required columns exist and are properly named
                df_clean = df.copy()
                
                # Standardize column names (yfinance uses uppercase)
                column_mapping = {
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                }
                df_clean = df_clean.rename(columns=column_mapping)
                
                # Ensure we have the required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df_clean.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {symbol}, skipping")
                    continue
                
                # Compute all features
                feature_df = self.compute_all_features(df_clean)
                
                # Add metadata
                feature_df['symbol'] = symbol
                
                results[symbol] = feature_df
                all_features.append(feature_df)
                
                logger.info(f"Generated {len(feature_df.columns)} features for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        # Save to feature store if requested
        if save and all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            store_path = self.feature_store_dir / "feature_store.parquet"
            combined_df.to_parquet(store_path, compression='snappy')
            logger.info(f"Feature store saved: {store_path} ({len(combined_df)} rows)")
        
        # Set feature_cols attribute for external access
        if results:
            sample_df = next(iter(results.values()))
            self.feature_cols = [col for col in sample_df.columns if col not in ['symbol']]
        else:
            self.feature_cols = []
        
        logger.info(f"Successfully processed {len(results)} symbols")
        return results
    
    def load_feature_store(self) -> pd.DataFrame:
        """Load the combined feature store."""
        store_path = self.feature_store_dir / "feature_store.parquet"
        
        if not store_path.exists():
            raise FileNotFoundError(f"Feature store not found: {store_path}")
        
        df = pd.read_parquet(store_path)
        logger.info(f"Loaded feature store: {len(df)} rows, {df.shape[1]} columns")
        
        return df


def main():
    """Test feature pipeline."""
    print("\n" + "="*60)
    print("FEATURE PIPELINE TEST")
    print("="*60)
    
    # Check for cached data
    cache_dir = Path("data/cache")
    cache_files = list(cache_dir.glob("*_yfinance.parquet"))
    
    if not cache_files:
        print("\n[ERROR] No cached data found!")
        print(f"Cache directory: {cache_dir}")
        print("\nPlease run data_ingest.py first:")
        print("  python core/data_ingest.py")
        return
    
    print(f"\n[1] Found {len(cache_files)} cached files")
    
    # Initialize pipeline
    print("\n[2] Initializing feature pipeline...")
    pipeline = FeaturePipeline()
    
    # Process data
    print("\n[3] Computing features...")
    print("="*60)
    results = pipeline.process_cached_data(save_to_store=True)
    
    # Show results
    print("\n" + "="*60)
    print("FEATURE COMPUTATION RESULTS")
    print("="*60)
    
    if results:
        print(f"\n[OK] Successfully processed {len(results)} symbols:")
        
        for symbol, df in results.items():
            print(f"\n  {symbol}:")
            print(f"    - Rows: {len(df)}")
            print(f"    - Features: {df.shape[1]}")
            
            # Show sample features
            feature_cols = [col for col in df.columns if col not in ['symbol', 'open', 'high', 'low', 'close', 'volume']]
            print(f"    - Sample features: {', '.join(feature_cols[:5])}...")
        
        # Load and display feature store info
        print("\n" + "="*60)
        print("FEATURE STORE INFO")
        print("="*60)
        
        feature_store = pipeline.load_feature_store()
        print(f"\n[OK] Feature store loaded:")
        print(f"  - Total rows: {len(feature_store)}")
        print(f"  - Total features: {feature_store.shape[1]}")
        print(f"  - Symbols: {feature_store['symbol'].nunique()}")
        print(f"  - Date range: {feature_store.index.min()} to {feature_store.index.max()}")
        
        # Show feature categories
        print("\n[OK] Feature categories:")
        feature_cols = [col for col in feature_store.columns if col not in ['symbol', 'source', 'fetch_timestamp']]
        
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