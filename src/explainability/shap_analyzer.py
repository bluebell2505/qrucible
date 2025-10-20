"""
SHAP Explainability Module
Provides model interpretability and feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, List

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")


class SHAPAnalyzer:
    """
    SHAP-based model explainability
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model (sklearn compatible)
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.logger = logging.getLogger(__name__)
    
    def create_explainer(self, X_background: np.ndarray, explainer_type: str = 'tree'):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background dataset for explainer
            explainer_type: 'tree', 'linear', or 'kernel'
        """
        self.logger.info(f"Creating {explainer_type} explainer...")
        
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                X_background
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        self.logger.info("Explainer created successfully")
    
    def calculate_shap_values(
        self,
        X: np.ndarray,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values
        
        Args:
            X: Feature matrix
            max_samples: Maximum samples to explain (for computational efficiency)
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first")
        
        # Sample if needed
        if max_samples and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            self.logger.info(f"Sampling {max_samples} from {len(X)} instances")
        else:
            X_sample = X
        
        self.logger.info(f"Calculating SHAP values for {len(X_sample)} instances...")
        self.shap_values = self.explainer.shap_values(X_sample)
        
        return self.shap_values
    
    def plot_summary(
        self,
        X: Optional[np.ndarray] = None,
        plot_type: str = 'dot',
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP summary plot
        
        Args:
            X: Feature matrix (optional if using cached values)
            plot_type: 'dot', 'bar', or 'violin'
            max_display: Maximum features to display
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved summary plot to: {save_path}")
        
        return plt.gcf()
    
    def plot_feature_importance(
        self,
        top_n: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance based on mean |SHAP value|
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        # Calculate mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'Feature {i}' for i in range(len(importance))],
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importance_df)), importance_df['importance'], color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Mean |SHAP value|', fontsize=12)
        ax.set_title(f'Top {top_n} Important Features (SHAP)', fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved importance plot to: {save_path}")
        
        return fig, importance_df
    
    def explain_single_prediction(
        self,
        X_instance: np.ndarray,
        instance_name: str = "Instance"
    ) -> Dict:
        """
        Explain a single prediction
        
        Args:
            X_instance: Single instance to explain (1D array)
            instance_name: Name for the instance
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        # Ensure 2D shape
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            'prediction': self.model.predict(X_instance)[0]
        }
        
        # Top contributing features
        if self.feature_names:
            contributions = pd.DataFrame({
                'feature': self.feature_names,
                'value': X_instance[0],
                'shap_value': explanation['shap_values']
            })
            contributions['abs_shap'] = np.abs(contributions['shap_value'])
            contributions = contributions.sort_values('abs_shap', ascending=False)
            
            explanation['top_features'] = contributions.head(10)
        
        return explanation
    
    def plot_waterfall(
        self,
        X_instance: np.ndarray,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Create waterfall plot for single prediction
        
        Args:
            X_instance: Single instance
            max_display: Max features to display
            save_path: Path to save figure
        """
        if self.explainer is None:
            raise ValueError("Explainer not created")
        
        # Ensure 2D
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=X_instance[0],
                feature_names=self.feature_names
            ),
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved waterfall plot to: {save_path}")
        
        return plt.gcf()
    
    def get_feature_importance_dict(self) -> Dict[str, float]:
        """
        Get feature importance as dictionary
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        importance = np.abs(self.shap_values).mean(axis=0)
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}


class UncertaintyQuantifier:
    """
    Uncertainty quantification for predictions
    """
    
    def __init__(self, models: List):
        """
        Initialize with ensemble of models
        
        Args:
            models: List of trained models
        """
        self.models = models
        self.logger = logging.getLogger(__name__)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """
        Make predictions with uncertainty estimates
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Calculate mean and std
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    def calculate_confidence_intervals(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals
        
        Args:
            X: Feature matrix
            confidence: Confidence level (default 0.95)
            
        Returns:
            Dictionary with mean, lower, and upper bounds
        """
        mean_pred, std_pred = self.predict_with_uncertainty(X)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return {
            'mean': mean_pred,
            'lower': lower,
            'upper': upper,
            'std': std_pred
        }
    
    def identify_high_uncertainty(
        self,
        X: np.ndarray,
        threshold_percentile: float = 90
    ) -> np.ndarray:
        """
        Identify predictions with high uncertainty
        
        Args:
            X: Feature matrix
            threshold_percentile: Percentile for high uncertainty threshold
            
        Returns:
            Boolean array indicating high uncertainty instances
        """
        _, std_pred = self.predict_with_uncertainty(X)
        
        threshold = np.percentile(std_pred, threshold_percentile)
        high_uncertainty = std_pred > threshold
        
        self.logger.info(
            f"Found {high_uncertainty.sum()} high-uncertainty predictions "
            f"(>{threshold:.3f} std)"
        )
        
        return high_uncertainty


def main():
    """Example usage"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    import joblib
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Load model and data
    model_path = Path('results/models/random_forest.pkl')
    data_path = Path('data/processed/chembl_egfr_kinase_clean.csv')
    
    if not model_path.exists() or not data_path.exists():
        print("Model or data not found. Please train models first.")
        return
    
    # Load
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data.get('feature_names', None)
    
    df = pd.read_csv(data_path)
    
    # Prepare data
    feature_cols = ['mol_weight', 'logp', 'tpsa', 'num_h_donors', 
                   'num_h_acceptors', 'num_rotatable_bonds', 'n_heavy_atoms']
    X = df[feature_cols].values
    y = df['pIC50'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize SHAP analyzer
    print("\nInitializing SHAP analyzer...")
    analyzer = SHAPAnalyzer(model, feature_names=feature_cols)
    
    # Create explainer
    analyzer.create_explainer(X_train[:100], explainer_type='tree')
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = analyzer.calculate_shap_values(X_test[:200])
    
    # Plot summary
    print("Creating summary plot...")
    analyzer.plot_summary(X_test[:200], save_path='results/figures/shap_summary.png')
    
    # Plot feature importance
    print("Creating feature importance plot...")
    fig, importance_df = analyzer.plot_feature_importance(
        top_n=10,
        save_path='results/figures/shap_feature_importance.png'
    )
    
    print("\nTop 10 Important Features:")
    print(importance_df)
    
    # Explain single prediction
    print("\nExplaining single prediction...")
    explanation = analyzer.explain_single_prediction(X_test[0])
    print(f"Prediction: {explanation['prediction']:.2f}")
    print("\nTop contributing features:")
    print(explanation['top_features'])
    
    print("\n✓ SHAP analysis complete!")
    print("Plots saved to results/figures/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()