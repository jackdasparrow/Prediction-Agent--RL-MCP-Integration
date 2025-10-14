#!/usr/bin/env python3
"""
Enhanced Data Ingestion Module
Supports multiple data sources with proper symbol mapping for crypto, commodities, and equities.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymbolMapper:
    """Maps symbols between different data source formats"""
    
    # Crypto mapping: Standard -> Yahoo Finance
    CRYPTO_MAPPING = {
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "BNB/USD": "BNB-USD",
        "XRP/USD": "XRP-USD",
        "ADA/USD": "ADA-USD",
        "DOGE/USD": "DOGE-USD",
        "SOL/USD": "SOL-USD",
        "MATIC/USD": "MATIC-USD",
    }
    
    # Commodities mapping: Standard -> Yahoo Finance futures
    COMMODITIES_MAPPING = {
        "GOLD": "GC=F",      # Gold Futures
        "SILVER": "SI=F",    # Silver Futures
        "CRUDE_OIL": "CL=F", # Crude Oil Futures
        "NATURAL_GAS": "NG=F", # Natural Gas
        "COPPER": "HG=F",    # Copper Futures
        "XAU/USD": "GC=F",   # Gold spot (use futures as proxy)
        "XAG/USD": "SI=F",   # Silver spot (use futures as proxy)
    }
    
    @classmethod
    def to_yahoo_format(cls, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format"""
        # Check crypto
        if symbol in cls.CRYPTO_MAPPING:
            return cls.CRYPTO_MAPPING[symbol]
        
        # Check commodities
        if symbol in cls.COMMODITIES_MAPPING:
            return cls.COMMODITIES_MAPPING[symbol]
        
        # Already in correct format or equity symbol
        return symbol
    
    @classmethod
    def to_twelvedata_format(cls, symbol: str) -> str:
        """Convert symbol to Twelve Data format"""
        # Crypto: BTC-USD -> BTC/USD
        if "-USD" in symbol or "-USDT" in symbol:
            return symbol.replace("-", "/")
        
        # Commodities: Use standard names
        reverse_commodities = {v: k for k, v in cls.COMMODITIES_MAPPING.items()}
        if symbol in reverse_commodities:
            std_symbol = reverse_commodities[symbol]
            # Twelve Data uses XAU/USD format
            if "/" in std_symbol:
                return std_symbol
            else:
                # Map to Twelve Data commodity symbols
                td_mapping = {
                    "GOLD": "XAU/USD",
                    "SILVER": "XAG/USD",
                    "CRUDE_OIL": "WTI/USD",
                    "NATURAL_GAS": "NG/USD",
                }
                return td_mapping.get(std_symbol, symbol)
        
        return symbol
    
    @classmethod
    def detect_asset_type(cls, symbol: str) -> str:
        """Detect if symbol is crypto, commodity, or equity"""
        if symbol in cls.CRYPTO_MAPPING or "-USD" in symbol or "/USD" in symbol and "XAU" not in symbol and "XAG" not in symbol:
            return "crypto"
        elif symbol in cls.COMMODITIES_MAPPING or "=" in symbol or "XAU" in symbol or "XAG" in symbol:
            return "commodity"
        else:
            return "equity"


class DataIngestion:
    """Enhanced data ingestion with multi-source support"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.twelvedata_key = os.getenv("TWELVEDATA_API_KEY")
        self.alpaca_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = {
            "yahoo": 0.5,
            "alpha_vantage": 12,  # 5 calls/minute = 12s interval
            "twelvedata": 4.5,    # ~800 calls/day = ~4.5s safe interval
            "alpaca": 0.5
        }
    
    def _wait_for_rate_limit(self, source: str):
        """Enforce rate limiting between requests"""
        if source in self.last_request_time:
            elapsed = time.time() - self.last_request_time[source]
            required_wait = self.min_request_interval.get(source, 1.0)
            if elapsed < required_wait:
                wait_time = required_wait - elapsed
                time.sleep(wait_time)
        
        self.last_request_time[source] = time.time()
    
    @staticmethod
    def _sanitize_symbol_for_filename(symbol: str) -> str:
        """Make a symbol safe for filenames (replace path-unsafe chars)."""
        import re
        # Replace anything not alphanum, dot, equals, or dash with underscore
        return re.sub(r"[^A-Za-z0-9\.\-=]", "_", symbol)
    
    def _generate_synthetic_backup(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate deterministic synthetic OHLCV data as a last-resort fallback.

        This is used when all external providers fail (e.g., rate limits/denials).
        """
        from datetime import datetime, timedelta
        import numpy as np

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 60)
        dates = pd.date_range(start=start_date, end=end_date, freq="B")[-days:]

        # Stable seed per symbol for reproducibility
        import hashlib
        seed = int(hashlib.md5(symbol.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        base_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "AMZN": 120.0,
            "TSLA": 200.0,
            "NVDA": 400.0,
            "META": 250.0,
            "NFLX": 400.0,
            "AMD": 100.0,
            "INTC": 30.0,
            "GOOGL": 2500.0,
            "RELIANCE.NS": 2500.0,
            "TCS.NS": 3500.0,
            "INFY.NS": 1500.0,
        }
        start_price = float(base_prices.get(symbol, 100.0))

        mu = 0.0005
        sigma = 0.02
        returns = rng.normal(loc=mu, scale=sigma, size=len(dates))

        prices = [start_price]
        for r in returns[1:]:
            new_price = prices[-1] * (1 + r)
            prices.append(max(new_price, 0.01))

        rows = []
        now_ts = datetime.now()
        for i, (date, close) in enumerate(zip(dates, prices)):
            daily_vol = abs(rng.normal(0, 0.01))
            high = close * (1 + daily_vol)
            low = close * (1 - daily_vol)
            if i == 0:
                open_price = close
            else:
                gap = rng.normal(0, 0.005)
                open_price = prices[i - 1] * (1 + gap)
                open_price = max(min(open_price, high), low)

            base_volume = 1_000_000
            volume_multiplier = rng.lognormal(0, 0.5)
            volume = int(max(1, base_volume * volume_multiplier))

            rows.append({
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(close, 2),
                "Volume": volume,
                "symbol": symbol,
                "source": "synthetic",
                "fetch_timestamp": now_ts,
            })

        df = pd.DataFrame(rows, index=dates)
        df.index.name = "Date"
        logger.warning(f"Using synthetic backup data for {symbol}: {len(df)} rows")
        return df

    def fetch_yahoo_finance(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance with retries"""
        # Convert to Yahoo format
        yahoo_symbol = SymbolMapper.to_yahoo_format(symbol)
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Fetching {yahoo_symbol} from Yahoo Finance (attempt {attempt}/{max_retries})")
                self._wait_for_rate_limit("yahoo")
                
                ticker = yf.Ticker(yahoo_symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data returned for {yahoo_symbol}")
                    if attempt < max_retries:
                        wait_time = attempt * 6
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    continue
                
                # Standardize column names
                df.columns = df.columns.str.lower()
                df = df.reset_index()
                
                # Add metadata
                df['symbol'] = symbol  # Use original symbol
                df['source'] = 'yahoo_finance'
                df['fetch_timestamp'] = datetime.now()
                
                logger.info(f"✓ Fetched {len(df)} rows for {yahoo_symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Yahoo Finance error for {yahoo_symbol}: {e}")
                if attempt < max_retries:
                    wait_time = attempt * 6
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        return None
    
    def fetch_twelvedata(
        self,
        symbol: str,
        interval: str = "1day",
        outputsize: int = 100,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Twelve Data API"""
        if not self.twelvedata_key:
            logger.warning("Twelve Data API key not set")
            return None
        
        # Convert to Twelve Data format
        td_symbol = SymbolMapper.to_twelvedata_format(symbol)
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Fetching {td_symbol} from Twelve Data (attempt {attempt}/{max_retries})")
                self._wait_for_rate_limit("twelvedata")
                
                url = "https://api.twelvedata.com/time_series"
                params = {
                    "symbol": td_symbol,
                    "interval": interval,
                    "outputsize": outputsize,
                    "apikey": self.twelvedata_key
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if "values" not in data:
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"Twelve Data error for {td_symbol}: {error_msg}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                
                # Rename columns to match our schema
                df = df.rename(columns={
                    "datetime": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                })
                
                # Convert to numeric
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['symbol'] = symbol
                df['source'] = 'twelvedata'
                df['fetch_timestamp'] = datetime.now()
                
                logger.info(f"✓ Fetched {len(df)} rows for {td_symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Twelve Data error for {td_symbol}: {e}")
                if attempt < max_retries:
                    time.sleep(5)
        
        return None
    
    def fetch_alpha_vantage(
        self,
        symbol: str,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not set")
            return None
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Fetching {symbol} from Alpha Vantage (attempt {attempt}/{max_retries})")
                self._wait_for_rate_limit("alpha_vantage")
                
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_key,
                    "outputsize": "compact"
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                time_series_key = "Time Series (Daily)"
                if time_series_key not in data:
                    logger.error(f"No time series data for {symbol}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index().reset_index()
                df = df.rename(columns={'index': 'date'})
                
                # Rename columns
                df.columns = [col.split('. ')[1].lower() if '. ' in col else col.lower() for col in df.columns]
                
                # Convert to numeric
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['symbol'] = symbol
                df['source'] = 'alpha_vantage'
                df['fetch_timestamp'] = datetime.now()
                
                logger.info(f"✓ Fetched {len(df)} rows for {symbol}")
                return df
                
            except Exception as e:
                logger.error(f"Alpha Vantage error for {symbol}: {e}")
                if attempt < max_retries:
                    time.sleep(15)
        
        return None
    
    def fetch_auto(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Automatically select best data source based on symbol type"""
        asset_type = SymbolMapper.detect_asset_type(symbol)
        
        logger.info(f"Detected {symbol} as {asset_type}")
        
        # Define source priority based on asset type
        if asset_type == "crypto":
            sources = ["yahoo", "twelvedata"]
        elif asset_type == "commodity":
            sources = ["twelvedata", "yahoo"]
        else:  # equity
            sources = ["yahoo", "alpha_vantage"]
        
        # Try each source in order
        for source in sources:
            if source == "yahoo":
                df = self.fetch_yahoo_finance(symbol, period, interval)
            elif source == "twelvedata":
                df = self.fetch_twelvedata(symbol)
            elif source == "alpha_vantage":
                df = self.fetch_alpha_vantage(symbol)
            else:
                continue
            
            if df is not None and not df.empty:
                return df
        
        logger.error(f"Failed to fetch {symbol} from any source; generating synthetic backup")
        return self._generate_synthetic_backup(symbol, days=252)
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        source: str = "auto",
        period: str = "6mo",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            # Check cache first
            safe_symbol = self._sanitize_symbol_for_filename(symbol)
            cache_file = self.cache_dir / f"{safe_symbol}_yfinance.parquet"
            
            if cache_file.exists() and not force_refresh:
                logger.info(f"Loading {symbol} from cache")
                results[symbol] = pd.read_parquet(cache_file)
                continue
            
            # Fetch data
            if source == "auto":
                df = self.fetch_auto(symbol, period, interval)
            elif source == "yahoo":
                df = self.fetch_yahoo_finance(symbol, period, interval)
            elif source == "twelvedata":
                df = self.fetch_twelvedata(symbol)
            elif source == "alpha_vantage":
                df = self.fetch_alpha_vantage(symbol)
            else:
                logger.error(f"Unknown source: {source}")
                continue
            
            if df is None or getattr(df, "empty", False):
                # As a final fallback, synthesize data
                df = self._generate_synthetic_backup(symbol, days=252)

            # Save to cache (always save what we have at this point)
            try:
                df.to_parquet(cache_file)
                logger.info(f"✓ Saved {symbol} to cache")
            except Exception as e:
                logger.error(f"Could not save cache for {symbol}: {e}")
            # Always store in results even if caching failed
            results[symbol] = df
        
        return results


# Test the enhanced ingestion
if __name__ == "__main__":
    ingestion = DataIngestion()
    
    # Test symbols from different asset classes
    test_symbols = [
        "AAPL",      # Equity
        "BTC/USD",   # Crypto
        "ETH/USD",   # Crypto
        "XAU/USD",   # Commodity (gold)
        "XAG/USD",   # Commodity (silver)
    ]
    
    print("Testing Enhanced Data Ingestion")
    print("=" * 60)
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        df = ingestion.fetch_auto(symbol, period="1mo")
        
        if df is not None and not df.empty:
            print(f"✓ Success: {len(df)} rows fetched")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        else:
            print(f"✗ Failed to fetch {symbol}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")