"""
Data Ingestion Module
Fetches market data from Yahoo Finance and Quandl with caching and rate limiting.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles fetching and caching of market data from multiple sources.
    Supports Yahoo Finance and Quandl with rate limiting and retry logic.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize DataIngestion with cache directory.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quandl_api_key = os.getenv("QUANDL_API_KEY")
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        self.max_retries = 3
        self.base_backoff = 2  # seconds
        
    def _rate_limit(self):
        """Implement rate limiting with exponential backoff."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_path(self, symbol: str, source: str) -> Path:
        """Generate cache file path for a symbol."""
        return self.cache_dir / f"{symbol}_{source}.parquet"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)
    
    def fetch_yahoo_finance(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance with caching.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force fetch even if cache exists
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        cache_path = self._get_cache_path(symbol, "yfinance")
        
        # Check cache
        if not force_refresh and self._is_cache_valid(cache_path):
            logger.info(f"Loading {symbol} from cache")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Cache read failed for {symbol}: {e}")
        
        # Fetch with retries
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                logger.info(f"Fetching {symbol} from Yahoo Finance (attempt {attempt + 1}/{self.max_retries})")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                
                # Add metadata
                df['symbol'] = symbol
                df['source'] = 'yfinance'
                df['fetch_timestamp'] = datetime.now()
                
                # Cache the data
                df.to_parquet(cache_path, compression='snappy')
                logger.info(f"Successfully fetched and cached {symbol} ({len(df)} rows)")
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    backoff_time = self.base_backoff * (2 ** attempt)
                    logger.info(f"Retrying after {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    logger.error(f"Failed to fetch {symbol} after {self.max_retries} attempts")
                    return None
    
    def fetch_quandl(
        self,
        dataset_code: str,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from Quandl with caching.
        
        Args:
            dataset_code: Quandl dataset code (e.g., 'WIKI/AAPL')
            symbol: Stock ticker for caching
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: Force fetch even if cache exists
            
        Returns:
            DataFrame with data or None if fetch fails
        """
        if not self.quandl_api_key:
            logger.warning("Quandl API key not found. Set QUANDL_API_KEY in .env")
            return None
        
        cache_path = self._get_cache_path(symbol, "quandl")
        
        # Check cache
        if not force_refresh and self._is_cache_valid(cache_path):
            logger.info(f"Loading {symbol} from Quandl cache")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Quandl cache read failed for {symbol}: {e}")
        
        # Note: Quandl API has been deprecated. Using alternative approach
        logger.warning("Quandl API is deprecated. Consider using alternative data sources.")
        return None
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        source: str = "yfinance",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with progress tracking.
        
        Args:
            symbols: List of ticker symbols
            source: Data source ('yfinance' or 'quandl')
            **kwargs: Additional arguments for fetch methods
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"Fetching {total} symbols from {source}")
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Progress: {i}/{total} - {symbol}")
            
            if source == "yfinance":
                df = self.fetch_yahoo_finance(symbol, **kwargs)
            elif source == "quandl":
                df = self.fetch_quandl(symbol=symbol, **kwargs)
            else:
                logger.error(f"Unknown source: {source}")
                continue
            
            if df is not None:
                results[symbol] = df
            
            # Progress indicator
            progress_pct = (i / total) * 100
            logger.info(f"Overall progress: {progress_pct:.1f}% complete")
        
        logger.info(f"Successfully fetched {len(results)}/{total} symbols")
        return results
    
    def get_cached_data(self, symbol: str, source: str = "yfinance") -> Optional[pd.DataFrame]:
        """
        Retrieve cached data without fetching.
        
        Args:
            symbol: Stock ticker symbol
            source: Data source
            
        Returns:
            Cached DataFrame or None
        """
        cache_path = self._get_cache_path(symbol, source)
        
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.error(f"Error reading cache for {symbol}: {e}")
        
        return None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            symbol: Clear cache for specific symbol, or all if None
        """
        if symbol:
            for cache_file in self.cache_dir.glob(f"{symbol}_*.parquet"):
                cache_file.unlink()
                logger.info(f"Cleared cache for {symbol}")
        else:
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info("Cleared all cache")

    def get_universe_data(
        self,
        universe_file: Optional[str] = None,
        default_symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for entire universe of symbols.
        
        Args:
            universe_file: Path to file containing symbol list (one per line)
            default_symbols: Default symbol list if no file provided
            
        Returns:
            Dictionary of DataFrames
        """
        if universe_file and os.path.exists(universe_file):
            with open(universe_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
        elif default_symbols:
            symbols = default_symbols
        else:
            # Default universe: mix of stocks and indices
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # US tech
                'RELIANCE.NS', 'TCS.NS', 'INFY.NS',  # Indian stocks
                '^GSPC', '^DJI', '^IXIC'  # Indices
            ]
        
        logger.info(f"Fetching data for {len(symbols)} symbols")
        return self.fetch_multiple_symbols(symbols, source="yfinance")


def main():
    """Test data ingestion with sample symbols."""
    print("\n" + "="*60)
    print("DATA INGESTION TEST")
    print("="*60)
    
    # Initialize ingestion
    ingestion = DataIngestion()
    
    # Test symbols
    test_symbols = ["AAPL", "MSFT", "TSLA"]
    
    print(f"\nTesting with symbols: {', '.join(test_symbols)}")
    print("-" * 60)
    
    # Fetch data
    results = ingestion.fetch_multiple_symbols(
        symbols=test_symbols,
        source="yfinance",
        period="1mo",
        interval="1d"
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for symbol, df in results.items():
        if df is not None:
            print(f"\n[OK] {symbol}:")
            print(f"  - Rows: {len(df)}")
            print(f"  - Date range: {df.index.min()} to {df.index.max()}")
            print(f"  - Columns: {', '.join(df.columns.tolist())}")
            
            # Show sample data
            print("\n  Sample data (last 3 rows):")
            print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(3).to_string())
        else:
            print(f"\n[FAILED] {symbol}: No data retrieved")
    
    print("\n" + "="*60)
    print("[OK] Data ingestion test complete!")
    print("="*60)


if __name__ == "__main__":
    main()