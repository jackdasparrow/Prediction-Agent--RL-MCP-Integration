"""
Baseline LightGBM Model
Traditional machine learning baseline for comparison with RL agents.
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class BaselineLightGBM:
    """Baseline LightGBM model for traditional ML predictions."""
    
    def __init__(self, task: str = 'classification', model_name: str = 'lightgbm-baseline', model_dir: str = './models'):
        self.task = task
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        self.params = {
            'objective': 'multiclass' if task == 'classification' else 'regression',
            'num_class': 3 if task == 'classification' else None,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,  # Suppress all output
            'verbosity': -1,  # Also suppress verbosity
            'metric': 'multi_logloss' if task == 'classification' else 'rmse'
        }
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training."""
        clean_df = df[feature_cols + [target_col]].dropna()
        logger.info(f"Data shape after cleaning: {clean_df.shape}")
        
        X = clean_df[feature_cols]
        y = clean_df[target_col]
        self.feature_names = feature_cols
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None, num_boost_round: int = 100, early_stopping_rounds: int = 10):
        """Train the LightGBM model."""
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=num_boost_round
        )
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.predict(X)
    
    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        """Get predicted classes."""
        probas = self.predict(X)
        if len(probas.shape) > 1:
            return np.argmax(probas, axis=1)
        else:
            return (probas > 0.5).astype(int)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if self.task == 'classification':
            y_pred_proba = self.predict_proba(X_test)
            y_pred = self.predict_class(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            try:
                if len(np.unique(y_test)) > 2:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                else:
                    metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                metrics['auc'] = 0.0
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            y_pred = self.predict(X_test)
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance = self.model.feature_importance(importance_type='gain')
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, filepath: Optional[str] = None):
        """Save the trained model."""
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
            'trained_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.feature_names:
            feature_imp = self.get_feature_importance()
            imp_path = self.model_dir / f"{self.model_name}_feature_importance.csv"
            feature_imp.to_csv(imp_path, index=False)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """Load a trained model."""
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
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    logger.info("[OK] BaselineLightGBM class created successfully!")
