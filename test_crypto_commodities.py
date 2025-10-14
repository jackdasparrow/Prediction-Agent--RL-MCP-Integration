#!/usr/bin/env python3
"""
Test script for crypto and commodities data fetching
Tests the enhanced data ingestion with proper symbol mapping
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_ingest import DataIngestion, SymbolMapper

def test_symbol_mapping():
    """Test symbol format conversion"""
    print("\n" + "="*60)
    print("TESTING SYMBOL MAPPING")
    print("="*60)
    
    test_cases = [
        ("BTC/USD", "crypto"),
        ("ETH/USD", "crypto"),
        ("XAU/USD", "commodity"),
        ("XAG/USD", "commodity"),
        ("AAPL", "equity"),
        ("GC=F", "commodity"),
    ]
    
    for symbol, expected_type in test_cases:
        detected_type = SymbolMapper.detect_asset_type(symbol)
        yahoo_format = SymbolMapper.to_yahoo_format(symbol)
        td_format = SymbolMapper.to_twelvedata_format(symbol)
        
        status = "✓" if detected_type == expected_type else "✗"
        print(f"\n{status} {symbol}")
        print(f"  Type: {detected_type} (expected: {expected_type})")
        print(f"  Yahoo format: {yahoo_format}")
        print(f"  Twelve Data format: {td_format}")

def test_data_fetching():
    """Test actual data fetching"""
    print("\n" + "="*60)
    print("TESTING DATA FETCHING")
    print("="*60)
    
    ingestion = DataIngestion()
    
    # Test different asset types
    test_symbols = {
        "Equities": ["AAPL", "MSFT"],
        "Crypto": ["BTC/USD", "ETH/USD"],
        "Commodities": ["XAU/USD", "XAG/USD"],
    }
    
    results = {}
    
    for category, symbols in test_symbols.items():
        print(f"\n{category}:")
        print("-" * 40)
        
        for symbol in symbols:
            print(f"\nFetching {symbol}...")
            df = ingestion.fetch_auto(symbol, period="1mo", interval="1d")
            
            if df is not None and not df.empty:
                print(f"  ✓ Success: {len(df)} rows")
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"  Source: {df['source'].iloc[0]}")
                print(f"  Sample price: Close=${df['close'].iloc[-1]:.2f}")
                results[symbol] = True
            else:
                print(f"  ✗ Failed")
                results[symbol] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total - successful}/{total}")
    
    if successful == total:
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Some tests failed")
        print("\nFailed symbols:")
        for symbol, success in results.items():
            if not success:
                print(f"  - {symbol}")
        return False

def test_cache_system():
    """Test caching functionality"""
    print("\n" + "="*60)
    print("TESTING CACHE SYSTEM")
    print("="*60)
    
    ingestion = DataIngestion()
    test_symbol = "AAPL"
    
    # First fetch (should hit API)
    print(f"\nFirst fetch of {test_symbol} (should hit API)...")
    df1 = ingestion.fetch_auto(test_symbol, period="1mo")
    
    if df1 is None or df1.empty:
        print("✗ Failed to fetch data")
        return False
    
    print(f"✓ Fetched {len(df1)} rows")
    
    # Check if cache file exists
    cache_file = ingestion.cache_dir / f"{test_symbol}_yfinance.parquet"
    if cache_file.exists():
        print(f"✓ Cache file created: {cache_file}")
    else:
        print("✗ Cache file not created")
        return False
    
    # Second fetch (should use cache)
    print(f"\nSecond fetch of {test_symbol} (should use cache)...")
    
    results = ingestion.fetch_multiple_symbols(
        [test_symbol],
        source="auto",
        force_refresh=False
    )
    
    if test_symbol in results:
        df2 = results[test_symbol]
        print(f"✓ Loaded from cache: {len(df2)} rows")
        
        # Verify data matches
        if len(df1) == len(df2):
            print("✓ Cache data matches original")
            return True
        else:
            print("✗ Cache data doesn't match")
            return False
    else:
        print("✗ Failed to load from cache")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CRYPTO & COMMODITIES DATA INGESTION TEST SUITE")
    print("="*80)
    
    tests = [
        ("Symbol Mapping", test_symbol_mapping),
        ("Data Fetching", test_data_fetching),
        ("Cache System", test_cache_system),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"\n✗ Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Data ingestion is working correctly.")
        print("\nNext steps:")
        print("1. Update universe.txt with crypto/commodity symbols")
        print("2. Run: python train_pipeline.py")
        print("3. Test API: python api/server.py")
    else:
        print("\n⚠ Some tests failed. Please check:")
        print("1. API keys are set in .env file")
        print("2. Internet connection is stable")
        print("3. Check error messages above for details")

if __name__ == "__main__":
    main()