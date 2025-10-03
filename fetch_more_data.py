"""
Fetch More Historical Data
Downloads more historical data to ensure sufficient samples for training.
"""

import sys

# safe printing helper: use sys.__stdout__ if sys.stdout is closed
def safe_print(*args, **kwargs):
    """
    Print to sys.stdout if available, otherwise fallback to sys.__stdout__.
    Keeps same signature as built-in print.
    """
    file = kwargs.pop("file", None)
    out = sys.stdout
    try:
        if file is not None:
            print(*args, file=file, **kwargs)
            return
        if getattr(sys, "stdout", None) is None or getattr(sys.stdout, "closed", False):
            # fallback to original stdout provided by Python runtime
            print(*args, file=sys.__stdout__, **kwargs)
        else:
            print(*args, file=sys.stdout, **kwargs)
    except Exception:
        # ultimate fallback to original stdout (avoid raising)
        try:
            print(*args, file=sys.__stdout__, **kwargs)
        except Exception:
            # swallow to avoid crashing the script due to printing
            pass

safe_print("stdout closed?", getattr(sys.stdout, "closed", False))
safe_print("stdout is", sys.stdout)
from pathlib import Path

# Fix Unicode encoding for Windows console
if sys.platform == "win32":
    import io

    # Re-wrap stdout/stderr to support utf-8 in Windows consoles.
    # This is safe because safe_print handles closed stdout gracefully.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add core to path
sys.path.append(str(Path(__file__).parent / "core"))

from data_ingest import DataIngestion


def main():
    """Fetch more historical data for training."""
    safe_print("\n" + "=" * 60)
    safe_print("FETCHING MORE HISTORICAL DATA")
    safe_print("=" * 60)

    # Read universe file
    universe_file = Path("universe.txt")

    if not universe_file.exists():
        safe_print("\n[ERROR] universe.txt not found!")
        safe_print("Creating a default universe file...")

        # Create default universe
        default_symbols = [
            "AAPL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "AMD",
            "INTC",
            "GOOGL",
        ]

        universe_file.write_text("\n".join(default_symbols))
        safe_print(f"[OK] Created universe.txt with {len(default_symbols)} symbols")

    # Read symbols
    symbols = [line.strip() for line in universe_file.read_text().splitlines() if line.strip()]
    safe_print(f"\n[1] Loading {len(symbols)} symbols from universe.txt")
    safe_print(f"    Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

    # Initialize data ingestion
    ingestion = DataIngestion()

    # Fetch data with longer period
    safe_print("\n[2] Fetching 6 months of daily data...")
    safe_print("    This may take a few minutes...")
    safe_print("=" * 60)

    results = ingestion.fetch_multiple_symbols(
        symbols=symbols,
        source="yfinance",
        period="6mo",  # 6 months of data
        interval="1d",
        force_refresh=True,  # Force refresh to get latest data
    )

    # Show results
    safe_print("\n" + "=" * 60)
    safe_print("FETCH RESULTS")
    safe_print("=" * 60)

    successful = 0
    failed = 0

    for symbol in symbols:
        if symbol in results and results[symbol] is not None:
            df = results[symbol]
            # df index may or may not be datetime; guard formatting
            try:
                start = df.index.min().strftime("%Y-%m-%d")
                end = df.index.max().strftime("%Y-%m-%d")
            except Exception:
                start = str(df.index.min())
                end = str(df.index.max())
            safe_print(
                f"[OK] {symbol:12s} - {len(df):4d} rows ({start} to {end})"
            )
            successful += 1
        else:
            safe_print(f"[FAILED] {symbol:12s} - No data")
            failed += 1

    safe_print("\n" + "=" * 60)
    safe_print(f"[OK] Data fetch complete!")
    safe_print(f"    Successful: {successful}/{len(symbols)}")
    safe_print(f"    Failed: {failed}/{len(symbols)}")
    safe_print("=" * 60)

    if successful > 0:
        safe_print("\nNext steps:")
        safe_print("  1. python core/features.py        # Generate features")
        safe_print("  2. python core/models/baseline_lightgbm.py  # Train model")
    else:
        safe_print("\n[ERROR] No data was fetched successfully!")
        safe_print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()