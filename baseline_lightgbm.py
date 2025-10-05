"""
Baseline LightGBM Model
Traditional machine learning baseline for comparison with RL agents.
Works with features from both Yahoo Finance and Alpha Vantage data.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BaselineLightGBM:
    """Baseline LightGBM model for traditional ML predictions."""
    
    def __init__(
        self, 
        task: str = 'classification', 
        model_name: str = 'lightgbm-v1', 
        model_dir: str = './models'
    ):
        """
        Initialize LightGBM model.
        
        Args:
            task: 'classification' or 'regression'
            model_name: Name for saving/loading
            model_dir: Directory for model artifacts
        """
        self.task = task
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.feature_importance_df = None
        self.is_trained = False
        
        # LightGBM parameters
        self.params = {
            'objective': 'multiclass' if task == 'classification' else 'regression',
            'num_class': 3 if task == 'classification' else None,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'verbosity': -1,
            'metric': 'multi_logloss' if task == 'classification' else 'rmse',
            'min_data_in_leaf': 20,
            'max_depth': 7
        }
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract feature columns from dataframe.
        Excludes OHLCV, metadata, and target columns.
        """
        exclude_cols = [
            # OHLCV
            'open', 'high', 'low', 'close', 'volume', 'adj_close',
            # Metadata
            'symbol', 'source', 'fetch_timestamp',
            # Targets
            'target', 'target_return', 'target_direction', 'target_binary'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: Optional[List[str]] = None,
        target_col: str = 'target_direction',
        test_size: float = 0.2,
        use_time_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training with proper time-series handling.
        
        Args:
            df: Feature dataframe
            feature_cols: List of feature column names (auto-detect if None)
            target_col: Target column name
            test_size: Test set proportion
            use_time_split: Use time-series split instead of random
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Auto-detect features if not provided
        if feature_cols is None:
            feature_cols = self._get_feature_columns(df)
            logger.info(f"Auto-detected {len(feature_cols)} feature columns")
        
        # Validate target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=[target_col])
        logger.info(f"Rows after removing NaN targets: {len(df_clean)}")
        
        # Extract features and target
        X = df_clean[feature_cols].copy()
        y = df_clean[target_col].copy()
        
        # Handle any remaining NaN in features
        if X.isna().any().any():
            logger.warning("NaN values found in features, filling with 0")
            X = X.fillna(0)
        
        # Handle inf values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split data
        if use_time_split:
            # Time-series split (train on past, test on future)
            split_idx = int(len(X) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            logger.info("Using time-series split")
        else:
            # Random split (not recommended for time-series)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            logger.info("Using random split")
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train target distribution:\n{y_train.value_counts()}")
        logger.info(f"Test target distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: Optional[pd.DataFrame] = None, 
        y_val: Optional[pd.Series] = None, 
        num_boost_round: int = 200,
        early_stopping_rounds: int = 50
    ):
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            num_boost_round: Maximum boosting rounds
            early_stopping_rounds: Early stopping patience
        """
        logger.info("Starting training...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        # Add validation set if provided
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
            logger.info("Using validation set for early stopping")
        
        # Callbacks for training
        callbacks = []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        
        # Train model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.is_trained = True
        
        # Compute feature importance
        self.feature_importance_df = self.get_feature_importance()
        
        logger.info("Training complete!")
        logger.info(f"Best iteration: {self.model.best_iteration}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features dataframe
            
        Returns:
            Predictions (probabilities for classification, values for regression)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature alignment
        X_aligned = self._align_features(X)
        
        return self.model.predict(X_aligned, num_iteration=self.model.best_iteration)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Features dataframe
            
        Returns:
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        return self.predict(X)
    
    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predicted classes (classification only).
        
        Args:
            X: Features dataframe
            
        Returns:
            Predicted class labels
        """
        if self.task != 'classification':
            raise ValueError("predict_class only available for classification")
        
        probas = self.predict(X)
        
        if len(probas.shape) > 1:
            return np.argmax(probas, axis=1)
        else:
            return (probas > 0.5).astype(int)
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align features with trained model's expected features.
        
        Args:
            X: Input features
            
        Returns:
            Aligned features
        """
        if self.feature_names is None:
            logger.warning("No feature names stored, using input as-is")
            return X
        
        # Reindex to match training features
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        
        # Check for missing features
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            logger.warning(f"Missing {len(missing)} features, filled with 0: {list(missing)[:5]}...")
        
        return X_aligned
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        if self.task == 'classification':
            y_pred_proba = self.predict_proba(X_test)
            y_pred = self.predict_class(X_test)
            
            # Basic metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # AUC score
            try:
                n_classes = len(np.unique(y_test))
                if n_classes > 2:
                    metrics['auc'] = roc_auc_score(
                        y_test, y_pred_proba, 
                        multi_class='ovr',
                        average='weighted'
                    )
                else:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics['auc'] = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"\nConfusion Matrix:\n{cm}")
            
            # Per-class metrics
            for class_idx in range(n_classes):
                class_mask = y_test == class_idx
                if class_mask.sum() > 0:
                    class_acc = (y_pred[class_mask] == class_idx).mean()
                    metrics[f'class_{class_idx}_accuracy'] = class_acc
        
        else:  # Regression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            y_pred = self.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        # Log metrics
        logger.info("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: 'gain', 'split', or 'cover'
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def save(self, filepath: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save model (auto-generated if None)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = self.model_dir / f"{self.model_name}.pkl"
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'params': self.params,
            'task': self.task,
            'model_name': self.model_name,
            'trained_at': datetime.now().isoformat(),
            'feature_importance': self.feature_importance_df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save feature importance separately
        if self.feature_importance_df is not None:
            imp_path = self.model_dir / f"{self.model_name}_feature_importance.csv"
            self.feature_importance_df.to_csv(imp_path, index=False)
            logger.info(f"Feature importance saved to {imp_path}")
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """
        Load a trained model.
        
        Args:
            filepath: Path to load model from (auto-generated if None)
        """
        if filepath is None:
            filepath = self.model_dir / f"{self.model_name}.pkl"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.params = model_data.get('params', self.params)
        self.task = model_data.get('task', self.task)
        self.feature_importance_df = model_data.get('feature_importance')
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Trained at: {model_data.get('trained_at', 'unknown')}")
        logger.info(f"Features: {len(self.feature_names)}")


def main():
    """Train baseline model on feature store."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from core.features import FeaturePipeline
    
    print("\n" + "="*60)
    print("BASELINE LIGHTGBM TRAINING")
    print("="*60)
    
    # Load feature store
    print("\n[1] Loading feature store...")
    pipeline = FeaturePipeline()
    
    try:
        feature_dict = pipeline.load_feature_store()
        print(f"[OK] Loaded {len(feature_dict)} symbols")
    except FileNotFoundError:
        print("[ERROR] Feature store not found!")
        print("Please run: python core/features.py")
        return
    
    # Combine all data
    print("\n[2] Combining features from all symbols...")
    all_data = pd.concat(feature_dict.values(), ignore_index=True)
    print(f"[OK] Combined data shape: {all_data.shape}")
    
    # Initialize model
    print("\n[3] Initializing LightGBM model...")
    model = BaselineLightGBM(task='classification', model_name='lightgbm-v1')
    
    # Prepare data
    print("\n[4] Preparing train/test split...")
    X_train, X_test, y_train, y_test = model.prepare_data(
        all_data,
        target_col='target_direction',
        test_size=0.2,
        use_time_split=True
    )
    
    # Train model
    print("\n[5] Training model...")
    print("="*60)
    model.train(
        X_train, y_train, 
        X_val=X_test, y_val=y_test,
        num_boost_round=200,
        early_stopping_rounds=50
    )
    
    # Evaluate
    print("\n[6] Evaluating model...")
    print("="*60)
    metrics = model.evaluate(X_test, y_test)
    
    # Show top features
    print("\n[7] Top 10 Important Features:")
    print("="*60)
    top_features = model.get_feature_importance().head(10)
    print(top_features.to_string(index=False))
    
    # Save model
    print("\n[8] Saving model...")
    model.save()
    
    print("\n" + "="*60)
    print("[OK] Training complete!")
    print("="*60)
    print(f"\nModel saved to: models/{model.model_name}.pkl")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    if metrics['auc'] > 0:
        print(f"AUC: {metrics['auc']:.4f}")
    
    print("\nNext steps:")
    print("  1. python train_rl_agent.py    # Train RL agent")
    print("  2. python api/server.py         # Start API server")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()