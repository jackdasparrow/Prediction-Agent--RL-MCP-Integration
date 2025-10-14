#!/usr/bin/env python3
"""
Test Pipeline Script
Tests the complete prediction agent pipeline to ensure everything works.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        # Core modules
        from core.data_ingest import DataIngestion
        from core.features import FeaturePipeline
        from core.models.baseline_lightgbm import BaselineLightGBM
        from core.models.rl_agent import LinUCBAgent, ThompsonSamplingAgent, DQNAgent, RLTrainer
        from core.mcp_adapter import MCPAdapter
        
        logger.info("[OK] Core modules imported successfully")
        
        # External dependencies
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        import torch
        import sklearn
        import ta
        import yfinance as yf
        import requests
        
        logger.info("[OK] External dependencies imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"[ERROR] Import failed: {e}")
        return False

def test_data_ingestion():
    """Test data ingestion with a small sample"""
    logger.info("Testing data ingestion...")
    
    try:
        from core.data_ingest import DataIngestion
        
        ingestion = DataIngestion()
        
        # Test with a single symbol
        test_symbol = "AAPL"
        logger.info(f"Fetching data for {test_symbol}...")
        
        df = ingestion.fetch_yahoo_finance(test_symbol, period="5d", interval="1d")
        
        if df is not None and not df.empty:
            logger.info(f"[OK] Data ingestion successful: {len(df)} rows for {test_symbol}")
            return True
        else:
            logger.error("[ERROR] Data ingestion failed: No data returned")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Data ingestion failed: {e}")
        return False

def test_feature_pipeline():
    """Test feature engineering pipeline"""
    logger.info("Testing feature pipeline...")
    
    try:
        from core.features import FeaturePipeline
        from core.data_ingest import DataIngestion
        
        # Get some test data
        ingestion = DataIngestion()
        df = ingestion.fetch_yahoo_finance("AAPL", period="30d", interval="1d")
        
        if df is None or df.empty:
            logger.error("✗ No test data available")
            return False
        
        # Test feature pipeline
        pipeline = FeaturePipeline()
        feature_df = pipeline.compute_all_features(df)
        
        if feature_df is not None and not feature_df.empty:
            logger.info(f"[OK] Feature pipeline successful: {len(feature_df.columns)} features generated")
            return True
        else:
            logger.error("[ERROR] Feature pipeline failed: No features generated")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Feature pipeline failed: {e}")
        return False

def test_models():
    """Test model initialization"""
    logger.info("Testing model initialization...")
    
    try:
        from core.models.baseline_lightgbm import BaselineLightGBM
        from core.models.rl_agent import LinUCBAgent, ThompsonSamplingAgent, DQNAgent
        
        # Test baseline model
        baseline = BaselineLightGBM(task='classification', model_name='test-model')
        logger.info("[OK] Baseline model initialized")
        
        # Test RL agents
        n_features = 50  # Dummy feature count
        
        linucb = LinUCBAgent(n_features=n_features)
        logger.info("[OK] LinUCB agent initialized")
        
        thompson = ThompsonSamplingAgent(n_features=n_features)
        logger.info("[OK] Thompson Sampling agent initialized")
        
        dqn = DQNAgent(state_dim=n_features, action_dim=3, device='cpu')
        logger.info("[OK] DQN agent initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Model initialization failed: {e}")
        return False

def test_api_server():
    """Test API server imports (without starting server)"""
    logger.info("Testing API server imports...")
    
    try:
        # Test if we can import the server module
        import api.server
        logger.info("[OK] API server module imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] API server import failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("PREDICTION AGENT PIPELINE TEST")
    logger.info("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Ingestion Test", test_data_ingestion),
        ("Feature Pipeline Test", test_feature_pipeline),
        ("Model Initialization Test", test_models),
        ("API Server Test", test_api_server),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("[SUCCESS] All tests passed! The pipeline is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Data ingestion: python core/fetch_more_data.py")
        logger.info("2. Feature engineering: python core/features.py")
        logger.info("3. Train baseline model: python core/models/baseline_lightgbm.py")
        logger.info("4. Train RL agent: python train_pipeline.py")
        logger.info("5. Start API server: python api/server.py")
    else:
        logger.error("[ERROR] Some tests failed. Please fix the issues before proceeding.")
        logger.error("\nCommon fixes:")
        logger.error("- Install missing dependencies: pip install -r requirements.txt")
        logger.error("- Check Python version compatibility")
        logger.error("- Verify all files are in correct locations")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
