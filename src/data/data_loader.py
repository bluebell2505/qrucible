"""
Data Loader Module for Qrucible
Handles data acquisition from ChEMBL, BindingDB, and ZINC databases
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging
from tqdm import tqdm

# ChEMBL client
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logging.warning("ChEMBL client not available. Install with: pip install chembl-webresource-client")


class DataLoader:
    """
    Handles loading and initial processing of molecular datasets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataLoader with configuration
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.data_dir = Path(config['data']['raw_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def load_from_chembl(
        self,
        target_name: str,
        activity_threshold: Optional[float] = None,
        max_compounds: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data from ChEMBL database
        
        Args:
            target_name: Name of the target protein (e.g., 'EGFR')
            activity_threshold: Minimum pIC50 value to include
            max_compounds: Maximum number of compounds to retrieve
            
        Returns:
            DataFrame with columns: [molecule_chembl_id, canonical_smiles, pIC50, ...]
        """
        if not CHEMBL_AVAILABLE:
            raise ImportError("ChEMBL client not installed")
        
        self.logger.info(f"Fetching data from ChEMBL for target: {target_name}")
        
        # Search for target
        target_client = new_client.target
        targets = target_client.search(target_name)
        
        if not targets:
            raise ValueError(f"No targets found for: {target_name}")
        
        # Get first matching target
        target_chembl_id = targets[0]['target_chembl_id']
        self.logger.info(f"Target ID: {target_chembl_id}")
        
        # Fetch bioactivities
        activity_client = new_client.activity
        activities = activity_client.filter(
            target_chembl_id=target_chembl_id,
            standard_type="IC50",
            relation="=",
            assay_type="B"
        )
        
        # Convert to DataFrame
        records = []
        for i, act in enumerate(tqdm(activities, desc="Loading activities")):
            if max_compounds and i >= max_compounds:
                break
            
            # Extract relevant fields
            try:
                record = {
                    'molecule_chembl_id': act.get('molecule_chembl_id'),
                    'canonical_smiles': act.get('canonical_smiles'),
                    'standard_value': act.get('standard_value'),
                    'standard_units': act.get('standard_units'),
                    'standard_type': act.get('standard_type'),
                    'pchembl_value': act.get('pchembl_value'),
                    'assay_chembl_id': act.get('assay_chembl_id'),
                    'document_chembl_id': act.get('document_chembl_id')
                }
                
                # Calculate pIC50 if not available
                if record['standard_value'] and not record['pchembl_value']:
                    if record['standard_units'] == 'nM':
                        # pIC50 = -log10(IC50_M) = -log10(IC50_nM * 1e-9)
                        ic50_m = float(record['standard_value']) * 1e-9
                        record['pIC50'] = -np.log10(ic50_m)
                    else:
                        record['pIC50'] = None
                else:
                    record['pIC50'] = record['pchembl_value']
                
                records.append(record)
            except Exception as e:
                self.logger.warning(f"Error processing activity: {e}")
                continue
        
        df = pd.DataFrame(records)
        
        # Convert pIC50 to numeric, handling any non-numeric values
        df['pIC50'] = pd.to_numeric(df['pIC50'], errors='coerce')
        
        # Remove rows where pIC50 is NaN
        initial_count = len(df)
        df = df.dropna(subset=['pIC50'])
        self.logger.info(f"Removed {initial_count - len(df)} rows with invalid pIC50 values")
        
        # Filter by activity threshold
        if activity_threshold:
            df = df[df['pIC50'] >= activity_threshold]
            self.logger.info(f"Compounds after threshold filtering: {len(df)}")
        
        # Save to disk
        output_path = self.data_dir / f"chembl_{target_name.lower()}_raw.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(df)} compounds to {output_path}")
        
        return df
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with molecular data
        """
        self.logger.info(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        self.logger.info(f"Loaded {len(df)} compounds")
        return df
    
    def load_from_sdf(self, filepath: str) -> pd.DataFrame:
        """
        Load data from SDF file (for ZINC, BindingDB)
        
        Args:
            filepath: Path to SDF file
            
        Returns:
            DataFrame with molecular data
        """
        from rdkit import Chem
        from rdkit.Chem import PandasTools
        
        self.logger.info(f"Loading SDF from: {filepath}")
        df = PandasTools.LoadSDF(filepath)
        self.logger.info(f"Loaded {len(df)} compounds from SDF")
        return df
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'n_compounds': len(df),
            'n_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum()
        }
        
        if 'pIC50' in df.columns:
            info['activity_stats'] = {
                'mean': df['pIC50'].mean(),
                'std': df['pIC50'].std(),
                'min': df['pIC50'].min(),
                'max': df['pIC50'].max()
            }
        
        return info


def main():
    """Example usage"""
    import yaml
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize loader
    loader = DataLoader(config)
    
    # Load ChEMBL data
    df = loader.load_from_chembl(
        target_name=config['data']['sources']['chembl']['target'],
        activity_threshold=config['data']['sources']['chembl']['activity_threshold'],
        max_compounds=config['data']['sources']['chembl']['max_compounds']
    )
    
    # Print dataset info
    info = loader.get_dataset_info(df)
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()