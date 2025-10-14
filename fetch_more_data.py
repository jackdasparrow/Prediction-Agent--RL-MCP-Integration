#!/usr/bin/env python3
"""
core/fetch_more_data.py - Enhanced Data Fetching Script

Fetches historical market data with improved error handling and multiple data sources.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_ingest import DataIngestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_universe(file_path: str = "universe.txt") -> list:
    """Load symbols from universe file"""
    universe_path = Path(file_path)
    
    if not universe_path.exists():
        logger.error(f"Universe file not found: {file_path}")
        return []
    
    with open(universe_path, 'r') as f:
        symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    return symbols


def main():
    """Main data fetching routine"""
    print("\n" + "="*60)
    print("FETCHING HISTORICAL MARKET DATA")
    print("="*60)
    
    # Load symbols
    symbols = load_universe()
    
    if not symbols:
        logger.error("No symbols found in universe.txt")
        return
    
    print(f"\n[1] Loaded {len(symbols)} symbols from universe.txt")
    print(f"    Symbols: {', '.join(symbols[:5])}")
    if len(symbols) > 5:
        print(f"             ...and {len(symbols) - 5} more")
    
    # Initialize data ingestion
    print("\n[2] Initializing data ingestion...")
    ingestion = DataIngestion()
    
    # Check API keys
    has_alpha = bool(ingestion.alpha_vantage_key)
    has_twelve = bool(ingestion.twelvedata_key)
    has_alpaca = bool(ingestion.alpaca_key and ingestion.alpaca_secret)
    
    print(f"    Providers detected:")
    print(f"      - Alpha Vantage: {'YES' if has_alpha else 'NO'}")
    print(f"      - Twelve Data  : {'YES' if has_twelve else 'NO'}")
    print(f"      - Alpaca SDK   : {'YES (fallback to Yahoo if NO)' if has_alpaca else 'NO'}")
    print(f"      - Yahoo Finance: YES")
    print(f"    Routing: AUTO (per-symbol selection across providers)")
    
    # Estimate time
    avg_time_per_symbol = 8  # seconds
    total_time = (len(symbols) * avg_time_per_symbol) / 60
    print(f"    Estimated time: {total_time:.1f} minutes")
    
    # Fetch data
    print(f"\n[3] Starting data fetch...")
    print(f"    This will:")
    print(f"    - Fetch data for {len(symbols)} symbols")
    print(f"    - Cache data to data/cache/")
    print(f"    - Use AUTO routing (Twelve Data, Alpaca/Yahoo, Alpha Vantage, Yahoo)")
    
    # Auto-continue
    print("\nAuto-continuing...\n")
    print("="*60)
    
    start_time = datetime.now()
    
    # Fetch all symbols
    results = ingestion.fetch_multiple_symbols(
        symbols,
        source="auto",
        period="1y",
        force_refresh=True
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    # Results
    print("\n" + "="*60)
    print("FETCH RESULTS")
    print("="*60)
    
    successful = []
    failed = []
    
    for symbol in symbols:
        if symbol in results:
            df = results[symbol]
            # Resolve date range from column or index
            if 'Date' in df.columns:
                dates = pd.to_datetime(df['Date'], errors='coerce')
            elif 'date' in df.columns:
                dates = pd.to_datetime(df['date'], errors='coerce')
            else:
                dates = df.index if isinstance(df.index, pd.DatetimeIndex) else None

            if dates is not None and len(dates) > 0:
                try:
                    dmin = pd.to_datetime(dates.min()).strftime('%Y-%m-%d')
                    dmax = pd.to_datetime(dates.max()).strftime('%Y-%m-%d')
                    date_range = f"{dmin} to {dmax}"
                except Exception:
                    date_range = "unknown"
            else:
                date_range = "unknown"

            source = df['source'].iloc[0] if 'source' in df.columns else 'unknown'
            print(f"[OK] {symbol:<12} - {len(df):>4} rows ({date_range}) [{source}]")
            successful.append(symbol)
        else:
            print(f"[FAIL] {symbol:<10} - No data")
            failed.append(symbol)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successful: {len(successful)}/{len(symbols)}")
    print(f"Failed:     {len(failed)}/{len(symbols)}")
    print(f"Time:       {elapsed:.1f} minutes")
    print(f"Data saved: data/cache/")
    
    if failed:
        print(f"\n[WARNING] Failed symbols: {', '.join(failed)}")
        print(f"\nPossible reasons:")
        print(f"  - Invalid symbol")
        print(f"  - Symbol not available on source")
        print(f"  - Rate limit exceeded")
        print(f"  - Network issues")
    
    # Next steps
    if successful:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("Data fetched successfully! Run these commands:")
        print("")
        print("1. Feature engineering:")
        print("   python core/features.py")
        print("")
        print("2. Train baseline model:")
        print("   python core/models/baseline_lightgbm.py")
        print("")
        print("3. Train RL agent:")
        print("   python train_pipeline.py")
        print("")
        print("4. Start API server:")
        print("   python api/server.py")
    else:
        print("\n[ERROR] All symbols failed to fetch!")
        print("Please check:")
        print("  1. Internet connection")
        print("  2. API keys in .env file")
        print("  3. Symbol validity")


if __name__ == "__main__":
    main()