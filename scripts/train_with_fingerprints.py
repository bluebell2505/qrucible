"""
Train models using molecular fingerprints
Expected improvement: Test R² from 0.38 to 0.50-0.65

Usage:
    python scripts/train_with_fingerprints.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from features.fingerprints import FingerprintGenerator
from models.classical.random_forest import RandomForestModel
from models.classical.gradient_boosting import GradientBoostingModel
from utils.metrics import RegressionMetrics
from utils.logger import setup_logger

logger = setup_logger('train_fingerprints')


def main():
    print("\n" + "="*70)
    print("TRAINING WITH MOLECULAR FINGERPRINTS")
    print("="*70 + "\n")
    
    # Load data
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'chembl_egfr_kinase_clean.csv'
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} compounds")
    
    # Check if fingerprints already exist
    morgan_path = PROJECT_ROOT / 'data' / 'processed' / 'morgan_fingerprints.csv'
    
    if morgan_path.exists():
        logger.info("Loading existing fingerprints...")
        fp_df = pd.read_csv(morgan_path)
    else:
        logger.info("Generating Morgan fingerprints (this may take 2-3 minutes)...")
        generator = FingerprintGenerator()
        
        fp_df = generator.fingerprints_to_dataframe(
            df['canonical_smiles'].tolist(),
            fp_type='morgan',
            radius=2,
            n_bits=2048,
            show_progress=True
        )
        
        # Save for future use
        fp_df.to_csv(morgan_path, index=False)
        logger.info(f"Saved fingerprints to: {morgan_path}")
    
    logger.info(f"Fingerprint shape: {fp_df.shape}")
    logger.info(f"Sparsity: {(fp_df == 0).sum().sum() / fp_df.size * 100:.1f}%")
    
    # Prepare features and target
    X = fp_df
    y = df['pIC50']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train Random Forest
    logger.info("\n" + "="*70)
    logger.info("Training Random Forest with Fingerprints")
    logger.info("="*70)
    
    rf_model = RandomForestModel({
        'classical_models': {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        }
    })
    
    rf_model.train(X_train, y_train, scale_features=False)  # Don't scale binary features
    
    # Predictions
    y_pred_train_rf = rf_model.predict(X_train, scale_features=False)
    y_pred_test_rf = rf_model.predict(X_test, scale_features=False)
    
    # Metrics
    train_metrics_rf = RegressionMetrics.calculate_all_metrics(y_train, y_pred_train_rf)
    test_metrics_rf = RegressionMetrics.calculate_all_metrics(y_test, y_pred_test_rf)
    
    logger.info("\nRandom Forest + Fingerprints:")
    logger.info(f"  Train R²: {train_metrics_rf['r2']:.4f}")
    logger.info(f"  Test R²:  {test_metrics_rf['r2']:.4f}")
    logger.info(f"  Test RMSE: {test_metrics_rf['rmse']:.4f}")
    logger.info(f"  Test MAE:  {test_metrics_rf['mae']:.4f}")
    logger.info(f"  Pearson r: {test_metrics_rf['pearson_r']:.4f}")
    
    # Save model
    model_path = PROJECT_ROOT / 'results' / 'models' / 'random_forest_fingerprints.pkl'
    rf_model.save(model_path)
    logger.info(f"\nSaved model to: {model_path}")
    
    # Train XGBoost
    logger.info("\n" + "="*70)
    logger.info("Training XGBoost with Fingerprints")
    logger.info("="*70)
    
    xgb_model = GradientBoostingModel({
        'classical_models': {
            'gradient_boosting': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        }
    })
    
    xgb_model.train(X_train, y_train, scale_features=False)
    
    # Predictions
    y_pred_train_xgb = xgb_model.predict(X_train, scale_features=False)
    y_pred_test_xgb = xgb_model.predict(X_test, scale_features=False)
    
    # Metrics
    train_metrics_xgb = RegressionMetrics.calculate_all_metrics(y_train, y_pred_train_xgb)
    test_metrics_xgb = RegressionMetrics.calculate_all_metrics(y_test, y_pred_test_xgb)
    
    logger.info("\nXGBoost + Fingerprints:")
    logger.info(f"  Train R²: {train_metrics_xgb['r2']:.4f}")
    logger.info(f"  Test R²:  {test_metrics_xgb['r2']:.4f}")
    logger.info(f"  Test RMSE: {test_metrics_xgb['rmse']:.4f}")
    logger.info(f"  Test MAE:  {test_metrics_xgb['mae']:.4f}")
    logger.info(f"  Pearson r: {test_metrics_xgb['pearson_r']:.4f}")
    
    # Save model
    model_path = PROJECT_ROOT / 'results' / 'models' / 'xgboost_fingerprints.pkl'
    xgb_model.save(model_path)
    logger.info(f"\nSaved model to: {model_path}")
    
    # Ensemble
    logger.info("\n" + "="*70)
    logger.info("Ensemble Predictions")
    logger.info("="*70)
    
    y_pred_ensemble = (y_pred_test_rf + y_pred_test_xgb) / 2
    ensemble_metrics = RegressionMetrics.calculate_all_metrics(y_test, y_pred_ensemble)
    
    logger.info("\nEnsemble (RF + XGBoost):")
    logger.info(f"  Test R²:  {ensemble_metrics['r2']:.4f}")
    logger.info(f"  Test RMSE: {ensemble_metrics['rmse']:.4f}")
    logger.info(f"  Test MAE:  {ensemble_metrics['mae']:.4f}")
    logger.info(f"  Pearson r: {ensemble_metrics['pearson_r']:.4f}")
    
    # Comparison with descriptor-based models
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*70)
    
    comparison = pd.DataFrame({
        'Model': [
            'RF (Descriptors)',
            'XGBoost (Descriptors)',
            'RF (Fingerprints)',
            'XGBoost (Fingerprints)',
            'Ensemble (Fingerprints)'
        ],
        'Test R²': [
            0.376,  # From previous run
            0.387,  # From previous run
            test_metrics_rf['r2'],
            test_metrics_xgb['r2'],
            ensemble_metrics['r2']
        ],
        'Test RMSE': [
            0.961,
            0.952,
            test_metrics_rf['rmse'],
            test_metrics_xgb['rmse'],
            ensemble_metrics['rmse']
        ],
        'Pearson r': [
            0.615,
            0.623,
            test_metrics_rf['pearson_r'],
            test_metrics_xgb['pearson_r'],
            ensemble_metrics['pearson_r']
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Save comparison
    output_path = PROJECT_ROOT / 'results' / 'reports' / 'fingerprint_comparison.csv'
    comparison.to_csv(output_path, index=False)
    logger.info(f"\nSaved comparison to: {output_path}")
    
    # Plot comparison
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(comparison))
    ax.bar(x, comparison['Test R²'], color=['steelblue', 'steelblue', 'coral', 'coral', 'green'],
           edgecolor='black', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison['Model'], rotation=45, ha='right')
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.axhline(y=0.6, color='r', linestyle='--', label='Target (0.60)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = PROJECT_ROOT / 'results' / 'figures' / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to: {plot_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review plots in results/figures/")
    print("  2. Analyze feature importance")
    print("  3. Try hyperparameter tuning if R² < 0.60")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()