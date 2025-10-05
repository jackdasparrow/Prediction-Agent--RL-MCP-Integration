"""
Data Ingestion Module
Fetches market data from Yahoo Finance and Alpha Vantage.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
import logging

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Multi-source data ingestion with caching and rate limiting."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.max_retries = 3
        self.base_backoff = 3
        
        # Alpha Vantage: 5 calls/min for free tier
        self.alpha_vantage_calls = []
        self.alpha_vantage_limit = 5
        
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _alpha_vantage_rate_limit(self):
        """Rate limit for Alpha Vantage (5 calls/min)"""
        now = time.time()
        self.alpha_vantage_calls = [t for t in self.alpha_vantage_calls if now - t < 60]
        
        if len(self.alpha_vantage_calls) >= self.alpha_vantage_limit:
            wait_time = 60 - (now - self.alpha_vantage_calls[0])
            logger.info(f"Alpha Vantage rate limit reached. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time + 1)
            self.alpha_vantage_calls = []
        
        self.alpha_vantage_calls.append(now)
    
    def _get_cache_path(self, symbol: str, source: str) -> Path:
        """Generate cache file path"""
        return self.cache_dir / f"{symbol}_{source}.parquet"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache is still valid"""
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
        """Fetch from Yahoo Finance with improved error handling and fallback"""
        cache_path = self._get_cache_path(symbol, "yfinance")
        
        if not force_refresh and self._is_cache_valid(cache_path):
            logger.info(f"Loading {symbol} from cache")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    wait = self.base_backoff * (2 ** attempt)
                    logger.info(f"Waiting {wait}s before retry...")
                    time.sleep(wait)
                
                self._rate_limit()
                logger.info(f"Fetching {symbol} from Yahoo Finance (attempt {attempt + 1}/{self.max_retries})")
                
                # Try multiple approaches for better reliability
                df = None
                
                # Method 1: Direct yfinance with custom session
                try:
                    session = requests.Session()
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    })
                    
                    ticker = yf.Ticker(symbol, session=session)
                    df = ticker.history(period=period, interval=interval, timeout=30)
                    
                except Exception as e1:
                    logger.warning(f"Method 1 failed for {symbol}: {e1}")
                    
                    # Method 2: Try with different period
                    try:
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(period="2y", interval="1d", timeout=30)
                        if not df.empty:
                            # Take only the last year
                            df = df.tail(252)  # ~1 year of trading days
                    except Exception as e2:
                        logger.warning(f"Method 2 failed for {symbol}: {e2}")
                        
                        # Method 3: Try with max period
                        try:
                            ticker = yf.Ticker(symbol)
                            df = ticker.history(period="max", interval="1d", timeout=30)
                            if not df.empty:
                                # Take only the last year
                                df = df.tail(252)
                        except Exception as e3:
                            logger.warning(f"Method 3 failed for {symbol}: {e3}")
                            continue
                
                if df is None or df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Validate data quality
                if len(df) < 10:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                    continue
                
                # Clean and prepare data
                df = df.dropna()
                if df.empty:
                    logger.warning(f"All data is NaN for {symbol}")
                    continue
                
                df['symbol'] = symbol
                df['source'] = 'yfinance'
                df['fetch_timestamp'] = datetime.now()
                
                # Save to cache
                df.to_parquet(cache_path)
                logger.info(f"Successfully cached {symbol} ({len(df)} rows)")
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                if attempt == self.max_retries - 1:
                    return None
        
        return None
    
    def fetch_alpha_vantage(
        self,
        symbol: str,
        outputsize: str = "full",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch from Alpha Vantage
        Free tier: 5 calls/minute, 500 calls/day
        Get API key from: https://www.alphavantage.co/support/#api-key
        """
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not found. Set ALPHA_VANTAGE_API_KEY in .env")
            return None
        
        cache_path = self._get_cache_path(symbol, "alphavantage")
        
        if not force_refresh and self._is_cache_valid(cache_path):
            logger.info(f"Loading {symbol} from cache")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    wait = self.base_backoff * (2 ** attempt)
                    time.sleep(wait)
                
                self._alpha_vantage_rate_limit()
                logger.info(f"Fetching {symbol} from Alpha Vantage (attempt {attempt + 1}/{self.max_retries})")
                
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "outputsize": outputsize,
                    "apikey": self.alpha_vantage_key
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                    return None
                
                if "Note" in data:
                    logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                    time.sleep(60)
                    continue
                
                if "Time Series (Daily)" not in data:
                    logger.error(f"No time series data for {symbol}")
                    return None
                
                time_series = data["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(time_series, orient='index')
                
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.apply(pd.to_numeric)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                df['symbol'] = symbol
                df['source'] = 'alphavantage'
                df['fetch_timestamp'] = datetime.now()
                
                df.to_parquet(cache_path)
                logger.info(f"Successfully cached {symbol} from Alpha Vantage ({len(df)} rows)")
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
                if attempt == self.max_retries - 1:
                    return None
        
        return None
    
    def fetch_symbol(
        self,
        symbol: str,
        source: str = "auto",
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data with automatic source selection
        
        Args:
            symbol: Symbol to fetch
            source: "yahoo", "alphavantage", or "auto"
            **kwargs: Additional arguments for specific sources
        """
        if source == "auto":
            source = "alphavantage" if self.alpha_vantage_key else "yahoo"
        
        if source == "yahoo":
            return self.fetch_yahoo_finance(symbol, **kwargs)
        elif source == "alphavantage":
            return self.fetch_alpha_vantage(symbol, **kwargs)
        else:
            logger.error(f"Unknown source: {source}")
            return None
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        source: str = "auto",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple symbols with progress tracking"""
        results = {}
        total = len(symbols)
        
        logger.info(f"Fetching {total} symbols from {source}")
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Progress: {i}/{total} - {symbol}")
            
            df = self.fetch_symbol(symbol, source=source, **kwargs)
            
            if df is not None:
                results[symbol] = df
            
            progress = (i / total) * 100
            logger.info(f"Overall progress: {progress:.1f}% complete")
        
        logger.info(f"Successfully fetched {len(results)}/{total} symbols")
        return results
    
    def get_universe_data(
        self,
        universe_file: Optional[str] = None,
        default_symbols: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for entire universe"""
        if universe_file and os.path.exists(universe_file):
            with open(universe_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        elif default_symbols:
            symbols = default_symbols
        else:
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
                '^GSPC', '^DJI', '^IXIC'
            ]
        
        logger.info(f"Fetching data for {len(symbols)} symbols")
        return self.fetch_multiple_symbols(symbols, source="auto")


def main():
    """Test data ingestion"""
    print("\n" + "="*60)
    print("DATA INGESTION TEST")
    print("="*60)
    
    ingestion = DataIngestion()
    test_symbols = ["AAPL", "MSFT"]
    
    print(f"\nTesting with symbols: {', '.join(test_symbols)}")
    print("-" * 60)
    
    source = "alphavantage" if ingestion.alpha_vantage_key else "yahoo"
    print(f"Using source: {source}")
    
    results = ingestion.fetch_multiple_symbols(test_symbols, source=source)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for symbol, df in results.items():
        if df is not None:
            print(f"\n[OK] {symbol}:")
            print(f"  - Rows: {len(df)}")
            print(f"  - Date range: {df.index.min()} to {df.index.max()}")
            print(f"  - Source: {df['source'].iloc[0]}")
        else:
            print(f"\n[FAILED] {symbol}")
    
    print("\n" + "="*60)
    print("[OK] Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()  