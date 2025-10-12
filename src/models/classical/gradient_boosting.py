"""
Gradient Boosting Model for Qrucible
Implements XGBoost/LightGBM for molecular property prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class GradientBoostingModel:
    """
    Gradient Boosting model for molecular property prediction
    """
    
    def __init__(self, config: Optional[Dict] = None, algorithm: str = 'xgboost'):
        """
        Initialize Gradient Boosting model
        
        Args:
            config: Configuration dictionary
            algorithm: 'xgboost' or 'lightgbm'
        """
        self.config = config or {}
        self.algorithm = algorithm
        self.logger = logging.getLogger(__name__)
        
        # Get model configuration
        gb_config = self.config.get('classical_models', {}).get('gradient_boosting', {})
        
        if algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
            
            self.model = xgb.XGBRegressor(
                n_estimators=gb_config.get('n_estimators', 200),
                learning_rate=gb_config.get('learning_rate', 0.1),
                max_depth=gb_config.get('max_depth', 6),
                subsample=gb_config.get('subsample', 0.8),
                colsample_bytree=gb_config.get('colsample_bytree', 0.8),
                random_state=gb_config.get('random_state', 42),
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        scale_features: bool = True,
        early_stopping_rounds: Optional[int] = 50
    ) -> 'GradientBoostingModel':
        """
        Train the Gradient Boosting model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target
            scale_features: Whether to scale features
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Training {self.algorithm} model...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val.values if isinstance(X_val, pd.DataFrame) else X_val)
        
        # Train with or without early stopping
        if X_val is not None and y_val is not None and early_stopping_rounds is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        self.logger.info(f"Model trained on {len(X_train)} samples")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        scale_features: bool = True
    ) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            scale_features: Whether to scale features
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if scale_features:
            X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        
        return predictions
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scale_features: bool = True
    ) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            cv: Number of CV folds
            scale_features: Whether to scale features
            
        Returns:
            Dictionary of CV scores
        """
        self.logger.info(f"Performing {cv}-fold cross-validation...")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if scale_features:
            X = self.scaler.fit_transform(X)
        
        # Calculate CV scores
        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        rmse_scores = np.sqrt(-scores)
        
        results = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'min_rmse': rmse_scores.min(),
            'max_rmse': rmse_scores.max()
        }
        
        self.logger.info(f"CV RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
        
        return results
    
    def get_feature_importance(self, top_n: Optional[int] = 20) -> pd.DataFrame:
        """
        Get feature importance scores
        
        Args:
            top_n: Number of top features to return (None for all)
            
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'config': self.config,
            'algorithm': self.algorithm
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GradientBoostingModel':
        """
        Load model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded GradientBoostingModel instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(
            config=model_data.get('config'),
            algorithm=model_data.get('algorithm', 'xgboost')
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        instance.logger.info(f"Model loaded from: {filepath}")
        
        return instance


def main():
    """Example usage"""
    import yaml
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from utils.metrics import RegressionMetrics
    
    # Load config
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Load processed data
    data_path = Path('data/processed/chembl_egfr_clean.csv')
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Prepare features
    feature_cols = ['mol_weight', 'logp', 'tpsa', 'num_h_donors', 'num_h_acceptors', 
                   'num_rotatable_bonds', 'n_heavy_atoms']
    
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(available_cols)} features: {available_cols}")
    
    X = df[available_cols]
    y = df['pIC50']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Initialize and train model
    print("\nTraining XGBoost model...")
    xgb_model = GradientBoostingModel(config, algorithm='xgboost')
    xgb_model.train(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("TRAINING SET METRICS")
    train_metrics = RegressionMetrics.calculate_all_metrics(y_train, y_pred_train)
    RegressionMetrics.print_metrics(train_metrics)
    
    print("\n" + "="*60)
    print("TEST SET METRICS")
    test_metrics = RegressionMetrics.calculate_all_metrics(y_test, y_pred_test)
    RegressionMetrics.print_metrics(test_metrics)
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 10 IMPORTANT FEATURES")
    print("="*60)
    importance_df = xgb_model.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:.<40} {row['importance']:.4f}")
    
    # Save model
    output_path = Path('results/models/xgboost_model.pkl')
    xgb_model.save(output_path)
    print(f"\n✓ Model saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()