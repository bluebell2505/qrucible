"""
Data Splitting Module for Qrucible
Handles train/validation/test splits with stratification options
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from typing import Tuple, Optional, Dict
import logging


class DataSplitter:
    """
    Handles splitting of molecular datasets for training and evaluation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataSplitter with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get training configuration
        train_config = self.config.get('training', {})
        self.test_size = train_config.get('test_size', 0.2)
        self.validation_size = train_config.get('validation_size', 0.1)
        self.random_state = train_config.get('random_state', 42)
        self.cv_folds = train_config.get('cv_folds', 5)
    
    def train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'pIC50',
        feature_cols: Optional[list] = None,
        stratify: bool = False,
        n_bins: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (if None, use all except target)
            stratify: Whether to use stratified sampling
            n_bins: Number of bins for stratification (if stratify=True)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Prepare features and target
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Stratification for regression
        stratify_labels = None
        if stratify:
            # Bin continuous target for stratification
            stratify_labels = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            self.logger.info(f"Using stratified split with {n_bins} bins")
        
        # Perform split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )
        
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_val_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'pIC50',
        feature_cols: Optional[list] = None,
        stratify: bool = False,
        n_bins: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
               pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns
            stratify: Whether to use stratified sampling
            n_bins: Number of bins for stratification
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Prepare features and target
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Stratification labels
        stratify_labels = None
        if stratify:
            stratify_labels = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )
        
        # Second split: train vs val
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        
        if stratify:
            stratify_labels_temp = pd.qcut(
                y_temp, q=n_bins, labels=False, duplicates='drop'
            )
        else:
            stratify_labels_temp = None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_labels_temp
        )
        
        self.logger.info(
            f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_cv_folds(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratified: bool = False,
        n_bins: int = 5
    ):
        """
        Get cross-validation fold indices
        
        Args:
            X: Features DataFrame
            y: Target Series
            stratified: Whether to use stratified CV
            n_bins: Number of bins for stratification
            
        Yields:
            train_idx, val_idx for each fold
        """
        if stratified:
            # Create bins for stratification
            y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            splits = cv.split(X, y_binned)
        else:
            cv = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            splits = cv.split(X)
        
        self.logger.info(f"Generating {self.cv_folds}-fold CV splits")
        
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            self.logger.info(
                f"Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)}"
            )
            yield train_idx, val_idx
    
    def temporal_split(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str = 'pIC50',
        feature_cols: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data based on temporal ordering
        Useful for time-series or chronological data
        
        Args:
            df: Input DataFrame
            date_col: Name of date/time column
            target_col: Name of target column
            feature_cols: List of feature columns
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Sort by date
        df_sorted = df.sort_values(date_col)
        
        # Calculate split index
        split_idx = int(len(df_sorted) * (1 - self.test_size))
        
        # Split
        df_train = df_sorted.iloc[:split_idx]
        df_test = df_sorted.iloc[split_idx:]
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [col for col in df.columns 
                          if col not in [target_col, date_col]]
        
        X_train = df_train[feature_cols]
        X_test = df_test[feature_cols]
        y_train = df_train[target_col]
        y_test = df_test[target_col]
        
        self.logger.info(f"Temporal split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test


def main():
    """Example usage"""
    import yaml
    from pathlib import Path
    
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
    
    df = pd.read_csv(data_path)
    
    # Initialize splitter
    splitter = DataSplitter(config)
    
    # Example 1: Simple train/test split
    print("\n1. Simple Train/Test Split:")
    feature_cols = ['mol_weight', 'logp', 'tpsa', 'num_h_donors', 'num_h_acceptors']
    X_train, X_test, y_train, y_test = splitter.train_test_split(
        df, 
        target_col='pIC50',
        feature_cols=feature_cols
    )
    
    # Example 2: Train/Val/Test split
    print("\n2. Train/Val/Test Split:")
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
        df,
        target_col='pIC50',
        feature_cols=feature_cols,
        stratify=True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()