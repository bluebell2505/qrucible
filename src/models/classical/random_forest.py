"""
Random Forest Model for Qrucible
Implements Random Forest regressor for molecular property prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler


class RandomForestModel:
    """
    Random Forest model for molecular property prediction
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Random Forest model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get model configuration
        rf_config = self.config.get('classical_models', {}).get('random_forest', {})
        
        # Initialize model with config parameters
        self.model = RandomForestRegressor(
            n_estimators=rf_config.get('n_estimators', 200),
            max_depth=rf_config.get('max_depth', 20),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=rf_config.get('random_state', 42),
            n_jobs=rf_config.get('n_jobs', -1),
            verbose=0
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_features: bool = True
    ) -> 'RandomForestModel':
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            scale_features: Whether to scale features
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Training Random Forest model...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
        
        # Train model
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
    
    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        Uses predictions from individual trees
        
        Args:
            X: Features to predict
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if scale_features:
            X = self.scaler.transform(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])
        
        # Calculate mean and std
        predictions = np.mean(tree_predictions, axis=0)
        uncertainties = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainties
    
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
    
    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3
    ) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search
            cv: Number of CV folds
            
        Returns:
            Dictionary with best parameters and scores
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        self.logger.info("Starting hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': np.sqrt(-grid_search.best_score_),
            'cv_results': grid_search.cv_results_
        }
        
        self.logger.info(f"Best parameters: {results['best_params']}")
        self.logger.info(f"Best RMSE: {results['best_score']:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return results
    
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
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestModel':
        """
        Load model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded RandomForestModel instance
        """
        model_data = joblib.load(filepath)
        
        instance = cls(config=model_data.get('config'))
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
    
    # Check if columns exist
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
    print("\nTraining Random Forest model...")
    rf_model = RandomForestModel(config)
    rf_model.train(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
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
    importance_df = rf_model.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:.<40} {row['importance']:.4f}")
    
    # Save model
    output_path = Path('results/models/random_forest_model.pkl')
    rf_model.save(output_path)
    print(f"\n✓ Model saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()