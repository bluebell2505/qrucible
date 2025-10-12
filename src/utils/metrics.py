"""
Evaluation Metrics for Qrucible
Provides comprehensive metrics for model evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr


class RegressionMetrics:
    """
    Calculate comprehensive regression metrics
    """
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # R² Score
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Squared Error
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            metrics['mape'] = np.nan
        
        # Explained Variance
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # Pearson Correlation
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
        
        # Spearman Correlation
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p
        
        # Max Error
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # Median Absolute Error
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
        
        return metrics
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE"""
        return mean_absolute_percentage_error(y_true, y_pred)
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
        """
        Print metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics display
        """
        print("\n" + "=" * 60)
        print(f"{title:^60}")
        print("=" * 60)
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                if abs(value) < 0.01:
                    print(f"{metric_name:.<40} {value:.6f}")
                else:
                    print(f"{metric_name:.<40} {value:.4f}")
            else:
                print(f"{metric_name:.<40} {value}")
        
        print("=" * 60)


class UncertaintyMetrics:
    """
    Calculate uncertainty quantification metrics
    """
    
    @staticmethod
    def calculate_prediction_intervals(
        predictions: List[np.ndarray],
        confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals from ensemble predictions
        
        Args:
            predictions: List of prediction arrays from different models
            confidence: Confidence level for intervals
            
        Returns:
            Dictionary with mean, lower, and upper bounds
        """
        predictions_array = np.array(predictions)
        
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        z_score = 1.96  # For 95% confidence
        
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower': lower,
            'upper': upper
        }
    
    @staticmethod
    def calculate_calibration_error(
        y_true: np.ndarray,
        predictions: List[np.ndarray],
        n_bins: int = 10
    ) -> float:
        """
        Calculate calibration error for uncertainty estimates
        
        Args:
            y_true: True values
            predictions: List of predictions from ensemble
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration error
        """
        intervals = UncertaintyMetrics.calculate_prediction_intervals(predictions)
        
        mean_pred = intervals['mean']
        std_pred = intervals['std']
        
        # Calculate standardized errors
        errors = np.abs(y_true - mean_pred) / (std_pred + 1e-10)
        
        # Bin errors and calculate calibration
        bins = np.linspace(0, errors.max(), n_bins + 1)
        calibration_error = 0
        
        for i in range(n_bins):
            mask = (errors >= bins[i]) & (errors < bins[i + 1])
            if mask.sum() > 0:
                expected_prob = (i + 0.5) / n_bins
                observed_prob = mask.sum() / len(errors)
                calibration_error += abs(expected_prob - observed_prob)
        
        return calibration_error / n_bins


def main():
    """Example usage"""
    
    # Generate example data
    np.random.seed(42)
    y_true = np.random.randn(100) * 2 + 5
    y_pred = y_true + np.random.randn(100) * 0.5
    
    # Calculate metrics
    print("\nExample: Regression Metrics")
    metrics = RegressionMetrics.calculate_all_metrics(y_true, y_pred)
    RegressionMetrics.print_metrics(metrics, "Model Performance")
    
    # Uncertainty metrics
    print("\nExample: Uncertainty Quantification")
    predictions = [
        y_pred + np.random.randn(100) * 0.3,
        y_pred + np.random.randn(100) * 0.3,
        y_pred + np.random.randn(100) * 0.3
    ]
    
    intervals = UncertaintyMetrics.calculate_prediction_intervals(predictions)
    print(f"Mean prediction shape: {intervals['mean'].shape}")
    print(f"Average uncertainty (std): {intervals['std'].mean():.4f}")


if __name__ == "__main__":
    main()