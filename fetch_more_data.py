"""
Fetch More Historical Data
Downloads historical data from Yahoo Finance and Alpha Vantage.
Supports automatic source selection and batch processing.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add core to path
sys.path.append(str(Path(__file__).parent))

from data_ingest import DataIngestion


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_section(number, text):
    """Print section header"""
    print(f"\n[{number}] {text}")


def create_default_universe():
    """Create default universe file with diverse symbols"""
    default_symbols = [
        # US Tech Giants
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet
        "AMZN",   # Amazon
        "NVDA",   # NVIDIA
        "META",   # Meta
        "TSLA",   # Tesla
        
        # US Tech/Semiconductors
        "AMD",    # AMD
        "INTC",   # Intel
        "NFLX",   # Netflix
        "ADBE",   # Adobe
        
        # US Finance
        "JPM",    # JPMorgan
        "BAC",    # Bank of America
        "GS",     # Goldman Sachs
        "V",      # Visa
        "MA",     # Mastercard
        
        # US Healthcare
        "JNJ",    # Johnson & Johnson
        "UNH",    # UnitedHealth
        "PFE",    # Pfizer
        
        # US Consumer
        "WMT",    # Walmart
        "HD",     # Home Depot
        "MCD",    # McDonald's
        
        # Indices
        "^GSPC",  # S&P 500
        "^DJI",   # Dow Jones
        "^IXIC",  # NASDAQ
    ]
    
    return default_symbols


def estimate_time(n_symbols, source):
    """Estimate fetch time based on source and symbol count"""
    if source == "alphavantage":
        # Alpha Vantage: 5 calls/min = 12s per symbol
        return n_symbols * 12 / 60
    else:
        # Yahoo Finance: ~2s per symbol
        return n_symbols * 2 / 60


def main():
    """Fetch historical market data"""
    print_banner("FETCHING HISTORICAL MARKET DATA")
    
    # Read or create universe file
    universe_file = Path("universe.txt")
    
    if not universe_file.exists():
        print("\n⚠️  universe.txt not found!")
        print("Creating default universe file...")
        
        default_symbols = create_default_universe()
        universe_file.write_text("\n".join(default_symbols))
        
        print(f"✓ Created universe.txt with {len(default_symbols)} symbols")
        print("\nYou can edit universe.txt to add/remove symbols")
        print("Format: One symbol per line, lines starting with # are ignored")
    
    # Read symbols from universe file
    with open(universe_file, 'r') as f:
        symbols = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]
    
    if not symbols:
        print("\n❌ No symbols found in universe.txt!")
        print("Add symbols (one per line) and try again")
        return
    
    print_section(1, f"Loaded {len(symbols)} symbols from universe.txt")
    print(f"    Symbols: {', '.join(symbols[:5])}")
    if len(symbols) > 5:
        print(f"             ...and {len(symbols) - 5} more")
    
    # Initialize data ingestion
    print_section(2, "Initializing data ingestion...")
    ingestion = DataIngestion()
    
    # Determine data source
    if ingestion.alpha_vantage_key:
        source = "alphavantage"
        print("    Source: Alpha Vantage (API key detected)")
        print(f"    Rate limit: 5 calls/min, 500 calls/day")
    else:
        source = "yahoo"
        print("    Source: Yahoo Finance (no API key needed)")
        print(f"    Rate limit: ~2 seconds per symbol")
    
    # Estimate time
    estimated_minutes = estimate_time(len(symbols), source)
    print(f"    Estimated time: {estimated_minutes:.1f} minutes")
    
    # Confirm with user
    print_section(3, "Starting data fetch...")
    print("    This will:")
    print(f"    - Fetch data for {len(symbols)} symbols")
    print(f"    - Cache data to data/cache/")
    print(f"    - Use {source.upper()} as data source")
    print()
    
    confirm = input("Continue? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("Aborted.")
        return
    
    print("\n" + "=" * 60)
    
    # Fetch data
    start_time = datetime.now()
    
    if source == "yahoo":
        results = ingestion.fetch_multiple_symbols(
            symbols=symbols,
            source="yahoo",
            period="1y",      # 1 year of data
            interval="1d",    # Daily data
            force_refresh=True
        )
    else:
        results = ingestion.fetch_multiple_symbols(
            symbols=symbols,
            source="alphavantage",
            outputsize="full",  # Full historical data
            force_refresh=True
        )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 60
    
    # Show results
    print_banner("FETCH RESULTS")
    
    successful = []
    failed = []
    
    for symbol in symbols:
        if symbol in results and results[symbol] is not None:
            df = results[symbol]
            try:
                start_date = df.index.min().strftime("%Y-%m-%d")
                end_date = df.index.max().strftime("%Y-%m-%d")
                source_name = df['source'].iloc[0]
                rows = len(df)
            except Exception:
                start_date = str(df.index.min())
                end_date = str(df.index.max())
                source_name = "unknown"
                rows = len(df)
            
            print(f"✓ {symbol:10s} - {rows:4d} rows ({start_date} to {end_date}) [{source_name}]")
            successful.append(symbol)
        else:
            print(f"✗ {symbol:10s} - No data")
            failed.append(symbol)
    
    # Summary
    print_banner("SUMMARY")
    print(f"Successful: {len(successful)}/{len(symbols)}")
    print(f"Failed:     {len(failed)}/{len(symbols)}")
    print(f"Time:       {elapsed:.1f} minutes")
    print(f"Data saved: data/cache/")
    
    if failed:
        print(f"\n⚠️  Failed symbols: {', '.join(failed[:10])}")
        if len(failed) > 10:
            print(f"   ...and {len(failed) - 10} more")
        
        print("\nPossible reasons:")
        print("  - Invalid symbol")
        print("  - Symbol not available on source")
        print("  - Rate limit exceeded")
        print("  - Network issues")
        
        if source == "alphavantage":
            print("\nNote: Alpha Vantage free tier has limits:")
            print("  - 5 calls per minute")
            print("  - 500 calls per day")
    
    # Next steps
    if successful:
        print_banner("NEXT STEPS")
        print("Data fetched successfully! Run these commands:")
        print()
        print("  1. python core/features.py")
        print("     Generate technical indicators")
        print()
        print("  2. python core/models/baseline_lightgbm.py")
        print("     Train baseline model")
        print()
        print("  3. python train_rl_agent.py")
        print("     Train RL agent")
        print()
        print("  4. python api/server.py")
        print("     Start API server")
        
        # Cache info
        print_banner("CACHE INFO")
        cache_dir = Path("data/cache")
        cache_files = list(cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        
        print(f"Cached files: {len(cache_files)}")
        print(f"Total size:   {total_size:.2f} MB")
        print(f"Location:     {cache_dir.absolute()}")
        
    else:
        print_banner("ERROR")
        print("❌ No data was fetched successfully!")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify symbols in universe.txt are valid")
        print("  3. If using Alpha Vantage:")
        print("     - Verify API key in .env")
        print("     - Check daily limit (500 calls)")
        print("  4. Try with fewer symbols first")
        print()
        print("For testing, try these symbols:")
        print("  AAPL, MSFT, GOOGL")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("Partial data may have been cached")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()