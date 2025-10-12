"""
Classical Model Training Script for Qrucible
Trains Random Forest and Gradient Boosting models

Usage:
    python scripts/train_classical.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data.splitter import DataSplitter
from models.classical.random_forest import RandomForestModel
from models.classical.gradient_boosting import GradientBoostingModel
from utils.metrics import RegressionMetrics
from utils.logger import setup_logger

# Setup logging
logger = setup_logger('train_classical')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train classical ML models for Qrucible'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to processed data file (optional, uses config if not provided)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['rf', 'xgb'],
        choices=['rf', 'xgb', 'ensemble'],
        help='Models to train (rf: Random Forest, xgb: XGBoost)'
    )
    
    parser.add_argument(
        '--cv',
        action='store_true',
        help='Perform cross-validation'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to CSV'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    config_file = PROJECT_ROOT / config_path
    
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_file}")
    return config


def load_data(config, data_path=None):
    """Load processed dataset"""
    if data_path is None:
        # Use config to determine data path
        target = config['data']['sources']['chembl']['target']
        data_path = PROJECT_ROOT / 'data' / 'processed' / f'chembl_{target.lower()}_clean.csv'
    else:
        data_path = PROJECT_ROOT / data_path
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run preprocessing first: python scripts/run_preprocessing.py")
        sys.exit(1)
    
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    return df


def prepare_features(df, config):
    """Prepare feature matrix and target"""
    # Define feature columns
    feature_cols = []
    
    # Add basic descriptors
    basic_features = ['mol_weight', 'logp', 'tpsa', 'num_h_donors', 'num_h_acceptors',
                     'num_rotatable_bonds', 'n_heavy_atoms', 'num_aromatic_rings',
                     'lipinski_violations']
    
    for col in basic_features:
        if col in df.columns:
            feature_cols.append(col)
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Prepare X and y
    X = df[feature_cols]
    y = df['pIC50']
    
    # Handle missing values
    if X.isnull().any().any():
        logger.warning("Found missing values, filling with median")
        X = X.fillna(X.median())
    
    return X, y, feature_cols


def plot_predictions(y_true, y_pred, title, output_path):
    """Plot predicted vs actual values"""
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    
    # Calculate metrics
    from scipy.stats import pearsonr
    r2 = RegressionMetrics.calculate_all_metrics(y_true, y_pred)['r2']
    rmse = RegressionMetrics.rmse(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    # Add metrics to plot
    plt.text(0.05, 0.95, 
             f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nPearson r = {pearson_r:.3f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to: {output_path}")
    plt.close()


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, config, args):
    """Train and evaluate a model"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*70}")
    
    # Train model
    model.train(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    logger.info(f"\n{model_name} - Training Set:")
    train_metrics = RegressionMetrics.calculate_all_metrics(y_train, y_pred_train)
    for metric, value in train_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\n{model_name} - Test Set:")
    test_metrics = RegressionMetrics.calculate_all_metrics(y_test, y_pred_test)
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Plot predictions
    figures_dir = PROJECT_ROOT / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plot_predictions(
        y_test, y_pred_test,
        f'{model_name} - Predicted vs Actual',
        figures_dir / f'{model_name.lower().replace(" ", "_")}_predictions.png'
    )
    
    # Feature importance
    logger.info(f"\n{model_name} - Top 10 Important Features:")
    importance_df = model.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:.<40} {row['importance']:.4f}")
    
    # Save model
    models_dir = PROJECT_ROOT / 'results' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f'{model_name.lower().replace(" ", "_")}.pkl'
    model.save(model_path)
    
    # Save predictions if requested
    if args.save_predictions:
        pred_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred_test,
            'residual': y_test - y_pred_test
        })
        pred_path = PROJECT_ROOT / 'results' / 'reports' / f'{model_name.lower().replace(" ", "_")}_predictions.csv'
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions to: {pred_path}")
    
    return test_metrics


def main():
    """Main execution function"""
    args = parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("QRUCIBLE - CLASSICAL MODEL TRAINING")
    logger.info("="*70 + "\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    df = load_data(config, args.data)
    
    # Prepare features
    X, y, feature_cols = prepare_features(df, config)
    
    # Split data
    logger.info("\nSplitting data...")
    splitter = DataSplitter(config)
    X_train, X_test, y_train, y_test = splitter.train_test_split(
        df=pd.concat([X, y], axis=1),
        target_col='pIC50',
        feature_cols=feature_cols,
        stratify=True
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Store results
    all_results = {}
    
    # Train Random Forest
    if 'rf' in args.models:
        rf_model = RandomForestModel(config)
        rf_results = train_and_evaluate(
            rf_model, 'Random Forest', 
            X_train, X_test, y_train, y_test,
            config, args
        )
        all_results['Random Forest'] = rf_results
    
    # Train XGBoost
    if 'xgb' in args.models:
        xgb_model = GradientBoostingModel(config, algorithm='xgboost')
        xgb_results = train_and_evaluate(
            xgb_model, 'XGBoost',
            X_train, X_test, y_train, y_test,
            config, args
        )
        all_results['XGBoost'] = xgb_results
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    
    summary_df = pd.DataFrame(all_results).T
    logger.info(f"\n{summary_df[['r2', 'rmse', 'mae', 'pearson_r']].to_string()}")
    
    # Save summary
    summary_path = PROJECT_ROOT / 'results' / 'reports' / 'training_summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path)
    logger.info(f"\nSaved training summary to: {summary_path}")
    
    logger.info("\n✅ Training Complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Review results in results/figures/")
    logger.info("  2. Check model performance in results/reports/")
    logger.info("  3. Proceed to explainability analysis")


if __name__ == "__main__":
    main()