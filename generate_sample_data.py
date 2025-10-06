#!/usr/bin/env python3
"""
scripts/generate_sample_data.py

Generate synthetic OHLCV market data for testing and save to data/cache/.
- Default behavior: reads symbols from `universe.txt` (one symbol per line).
- Options:
    --symbol SYMBOL        Generate for a single symbol (can be used instead of universe.txt)
    --days N               Number of trading days to generate per symbol (default: 252)
    --outdir PATH          Output directory (default: data/cache)
    --force                Overwrite existing parquet files
"""

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so `import core` works when invoked from anywhere.
# This assumes scripts/ is directly under repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# (Optional) If you have project modules to import, do it after the sys.path change:
# from core import some_module

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate_sample_data")


def stable_seed_from_symbol(symbol: str) -> int:
    """Return a deterministic seed for a symbol, stable across platforms/runs."""
    h = hashlib.md5(symbol.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # use first 8 hex chars -> 32-bit int


def generate_sample_data(symbol: str, days: int = 252) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for `symbol` covering `days` trading days.
    Returns a pandas DataFrame indexed by Date (business days).
    """
    logger.info("Generating sample data for %s (days=%d)", symbol, days)

    # Create date range (trading days only). Add a margin to ensure we get `days` business days.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 60)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")[-days:]

    # Deterministic seed per symbol
    seed = stable_seed_from_symbol(symbol)
    rng = np.random.default_rng(seed)

    # Starting price table (realistic examples)
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

    # Simulate daily returns (slight upward bias)
    mu = 0.0005
    sigma = 0.02
    returns = rng.normal(loc=mu, scale=sigma, size=len(dates))

    prices = [start_price]
    for r in returns[1:]:
        new_price = prices[-1] * (1 + r)
        prices.append(max(new_price, 0.01))

    prices = np.array(prices)

    rows = []
    now_ts = datetime.now()
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Intraday volatility
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

        rows.append(
            {
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(close, 2),
                "Volume": volume,
                "symbol": symbol,
                "source": "synthetic",
                "fetch_timestamp": now_ts,
            }
        )

    df = pd.DataFrame(rows, index=dates)
    df.index.name = "Date"
    logger.info("Generated %d rows for %s", len(df), symbol)
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic market data")
    p.add_argument("--symbol", help="Generate for a single symbol (overrides universe.txt)")
    p.add_argument("--days", type=int, default=252, help="Number of trading days to generate (default 252)")
    p.add_argument("--outdir", default="data/cache", help="Output directory for parquet files")
    p.add_argument("--force", action="store_true", help="Overwrite existing parquet files")
    p.add_argument("--universe", default="universe.txt", help="Universe file path (one symbol per line)")
    return p.parse_args()


def load_universe(path: Path):
    if not path.exists():
        logger.warning("Universe file %s not found", path)
        return []
    with path.open("r", encoding="utf-8") as f:
        symbols = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return symbols


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.symbol:
        symbols = [args.symbol.strip()]
    else:
        symbols = load_universe(Path(args.universe))

    if not symbols:
        logger.error("No symbols found to generate (provide --symbol or create universe.txt)")
        return

    successful = 0
    failed = 0

    for symbol in symbols:
        try:
            target_path = outdir / f"{symbol}_yfinance.parquet"
            if target_path.exists() and not args.force:
                logger.info("Skipping %s (exists). Use --force to overwrite.", target_path)
                successful += 1
                continue

            df = generate_sample_data(symbol, days=args.days)
            # Use pyarrow / fastparquet backend; make sure dependencies are installed
            df.to_parquet(target_path)
            logger.info("✓ %s: saved %d rows to %s", symbol, len(df), target_path)
            successful += 1
        except Exception as exc:  # keep individual symbol failures from stopping the batch
            logger.exception("✗ Failed to generate %s: %s", symbol, exc)
            failed += 1

    logger.info("=== DONE ===")
    logger.info("Successful: %d, Failed: %d, Total: %d", successful, failed, len(symbols))
    if successful:
        logger.info("Next steps (examples):")
        logger.info("  - python -c \"from core.features import FeaturePipeline; ...\"")
        logger.info("  - python core/models/baseline_lightgbm.py")
        logger.info("  - python api/server.py")


if __name__ == "__main__":
    main()
    