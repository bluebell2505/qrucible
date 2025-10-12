"""
Data Preprocessing Module for Qrucible
Handles cleaning, standardization, and validation of molecular data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, SaltRemover, Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize


class MolecularPreprocessor:
    """
    Handles preprocessing and standardization of molecular structures
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize RDKit tools
        self.salt_remover = SaltRemover.SaltRemover()
        self.uncharger = rdMolStandardize.Uncharger()
        self.normalizer = rdMolStandardize.Normalizer()
        
        # Get preprocessing settings
        preproc_config = self.config.get('data', {}).get('preprocessing', {})
        self.remove_duplicates = preproc_config.get('remove_duplicates', True)
        self.remove_salts = preproc_config.get('remove_salts', True)
        self.neutralize = preproc_config.get('neutralize', True)
        self.standardize = preproc_config.get('standardize', True)
        self.remove_mixtures = preproc_config.get('remove_mixtures', True)
        self.min_heavy_atoms = preproc_config.get('min_heavy_atoms', 6)
        self.max_heavy_atoms = preproc_config.get('max_heavy_atoms', 50)
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete cleaning pipeline for molecular dataset
        
        Args:
            df: Input DataFrame with SMILES column
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Starting preprocessing. Initial size: {len(df)}")
        
        # Make a copy
        df_clean = df.copy()
        
        # Identify SMILES column
        smiles_col = self._find_smiles_column(df_clean)
        
        # Remove missing SMILES
        df_clean = df_clean.dropna(subset=[smiles_col])
        self.logger.info(f"After removing missing SMILES: {len(df_clean)}")
        
        # Standardize SMILES
        df_clean['mol'] = df_clean[smiles_col].apply(self._smiles_to_mol)
        df_clean = df_clean[df_clean['mol'].notna()]
        self.logger.info(f"After parsing SMILES: {len(df_clean)}")
        
        # Apply preprocessing steps
        if self.remove_salts:
            df_clean['mol'] = df_clean['mol'].apply(self._remove_salts)
        
        if self.neutralize:
            df_clean['mol'] = df_clean['mol'].apply(self._neutralize_charges)
        
        if self.standardize:
            df_clean['mol'] = df_clean['mol'].apply(self._standardize_mol)
        
        if self.remove_mixtures:
            df_clean = df_clean[~df_clean['mol'].apply(self._is_mixture)]
            self.logger.info(f"After removing mixtures: {len(df_clean)}")
        
        # Filter by molecular size
        df_clean['n_heavy_atoms'] = df_clean['mol'].apply(
            lambda m: m.GetNumHeavyAtoms() if m else 0
        )
        df_clean = df_clean[
            (df_clean['n_heavy_atoms'] >= self.min_heavy_atoms) &
            (df_clean['n_heavy_atoms'] <= self.max_heavy_atoms)
        ]
        self.logger.info(f"After size filtering: {len(df_clean)}")
        
        # Generate canonical SMILES
        df_clean['canonical_smiles'] = df_clean['mol'].apply(
            lambda m: Chem.MolToSmiles(m) if m else None
        )
        
        # Remove duplicates
        if self.remove_duplicates:
            initial_size = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['canonical_smiles'])
            self.logger.info(
                f"Removed {initial_size - len(df_clean)} duplicates. "
                f"Final size: {len(df_clean)}"
            )
        
        # Clean up
        df_clean = df_clean.drop(columns=['mol'])
        
        return df_clean
    
    def _find_smiles_column(self, df: pd.DataFrame) -> str:
        """Find the SMILES column in the DataFrame"""
        possible_names = ['smiles', 'SMILES', 'canonical_smiles', 
                         'Smiles', 'smiles_string']
        
        for col in possible_names:
            if col in df.columns:
                return col
        
        raise ValueError("No SMILES column found in DataFrame")
    
    def _smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit Mol object with error handling"""
        if pd.isna(smiles) or not isinstance(smiles, str):
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception as e:
            self.logger.debug(f"Error parsing SMILES '{smiles}': {e}")
            return None
    
    def _remove_salts(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        """Remove salts from molecule"""
        if mol is None:
            return None
        
        try:
            return self.salt_remover.StripMol(mol)
        except Exception as e:
            self.logger.debug(f"Error removing salts: {e}")
            return mol
    
    def _neutralize_charges(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        """Neutralize charged molecules"""
        if mol is None:
            return None
        
        try:
            return self.uncharger.uncharge(mol)
        except Exception as e:
            self.logger.debug(f"Error neutralizing: {e}")
            return mol
    
    def _standardize_mol(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        """Apply standardization rules"""
        if mol is None:
            return None
        
        try:
            return self.normalizer.normalize(mol)
        except Exception as e:
            self.logger.debug(f"Error standardizing: {e}")
            return mol
    
    def _is_mixture(self, mol: Optional[Chem.Mol]) -> bool:
        """Check if molecule is a mixture (contains '.')"""
        if mol is None:
            return True
        
        smiles = Chem.MolToSmiles(mol)
        return '.' in smiles
    
    def validate_molecule(self, mol: Chem.Mol) -> Dict[str, bool]:
        """
        Validate molecule against various criteria
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary of validation results
        """
        validations = {
            'is_valid': mol is not None,
            'has_no_explicit_H': True,
            'in_size_range': True,
            'no_radicals': True,
            'sanitizable': True
        }
        
        if mol is None:
            return {k: False for k in validations}
        
        try:
            # Check size
            n_atoms = mol.GetNumHeavyAtoms()
            validations['in_size_range'] = (
                self.min_heavy_atoms <= n_atoms <= self.max_heavy_atoms
            )
            
            # Check for radicals
            for atom in mol.GetAtoms():
                if atom.GetNumRadicalElectrons() > 0:
                    validations['no_radicals'] = False
                    break
            
            # Try to sanitize
            Chem.SanitizeMol(mol)
            validations['sanitizable'] = True
            
        except Exception as e:
            self.logger.debug(f"Validation error: {e}")
            validations['sanitizable'] = False
        
        return validations
    
    def add_basic_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic molecular properties to DataFrame
        
        Args:
            df: DataFrame with canonical_smiles column
            
        Returns:
            DataFrame with added properties
        """
        self.logger.info("Calculating basic molecular properties...")
        
        df = df.copy()
        
        # Convert to mol objects
        df['mol'] = df['canonical_smiles'].apply(self._smiles_to_mol)
        
        # Calculate properties
        df['mol_weight'] = df['mol'].apply(
            lambda m: Descriptors.MolWt(m) if m else None
        )
        df['logp'] = df['mol'].apply(
            lambda m: Crippen.MolLogP(m) if m else None
        )
        df['tpsa'] = df['mol'].apply(
            lambda m: Descriptors.TPSA(m) if m else None
        )
        df['num_h_donors'] = df['mol'].apply(
            lambda m: Descriptors.NumHDonors(m) if m else None
        )
        df['num_h_acceptors'] = df['mol'].apply(
            lambda m: Descriptors.NumHAcceptors(m) if m else None
        )
        df['num_rotatable_bonds'] = df['mol'].apply(
            lambda m: Descriptors.NumRotatableBonds(m) if m else None
        )
        
        # Lipinski's Rule of Five
        df['lipinski_violations'] = df.apply(
            lambda row: self._count_lipinski_violations(row), axis=1
        )
        
        # Clean up
        df = df.drop(columns=['mol'])
        
        self.logger.info("Properties calculated successfully")
        return df
    
    def _count_lipinski_violations(self, row: pd.Series) -> int:
        """Count Lipinski's Rule of Five violations"""
        violations = 0
        
        if pd.notna(row.get('mol_weight')) and row['mol_weight'] > 500:
            violations += 1
        if pd.notna(row.get('logp')) and row['logp'] > 5:
            violations += 1
        if pd.notna(row.get('num_h_donors')) and row['num_h_donors'] > 5:
            violations += 1
        if pd.notna(row.get('num_h_acceptors')) and row['num_h_acceptors'] > 10:
            violations += 1
        
        return violations


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
    
    # Initialize preprocessor
    preprocessor = MolecularPreprocessor(config)
    
    # Example: Load and clean data
    data_file = Path('data/raw/chembl_egfr_raw.csv')
    if data_file.exists():
        df = pd.read_csv(data_file)
        
        # Clean the data
        df_clean = preprocessor.clean_dataset(df)
        
        # Add basic properties
        df_clean = preprocessor.add_basic_properties(df_clean)
        
        # Save cleaned data
        output_path = Path('data/processed/chembl_egfr_clean.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        
        print(f"\nPreprocessing complete!")
        print(f"Original size: {len(df)}")
        print(f"Cleaned size: {len(df_clean)}")
        print(f"Saved to: {output_path}")
        
        # Show sample
        print("\nSample of cleaned data:")
        print(df_clean.head())
    else:
        print(f"Data file not found: {data_file}")
        print("Please run data_loader.py first to download data.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()