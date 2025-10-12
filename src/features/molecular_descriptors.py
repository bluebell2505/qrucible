"""
Molecular Descriptor Calculator for Qrucible
Calculates comprehensive 2D and 3D molecular descriptors
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf
from rdkit.Chem import rdMolDescriptors, Fragments
from rdkit.Chem import AllChem


class DescriptorCalculator:
    """
    Calculates molecular descriptors for drug discovery
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize descriptor calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get feature configuration
        feature_config = self.config.get('features', {})
        self.descriptor_2d_list = feature_config.get('descriptors_2d', [])
        self.use_3d = feature_config.get('descriptors_3d', {}).get('enabled', False)
    
    def calculate_2d_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate comprehensive 2D molecular descriptors
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary of descriptor values
        """
        if mol is None:
            return {}
        
        descriptors = {}
        
        try:
            # Basic molecular properties
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['ExactMolWt'] = Descriptors.ExactMolWt(mol)
            descriptors['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
            
            # Lipophilicity
            descriptors['LogP'] = Crippen.MolLogP(mol)
            descriptors['MolMR'] = Crippen.MolMR(mol)
            
            # Topological descriptors
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
            
            # Hydrogen bonding
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            
            # Rotatable bonds
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            
            # Ring information
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
            descriptors['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(mol)
            
            # Hybridization
            descriptors['FractionCsp3'] = Descriptors.FractionCSP3(mol)
            
            # Atom counts
            descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            descriptors['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
            descriptors['NumAtoms'] = mol.GetNumAtoms()
            
            # Valence electrons
            descriptors['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
            
            # Complexity measures
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            descriptors['BalabanJ'] = Descriptors.BalabanJ(mol)
            
            # Charge descriptors
            descriptors['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
            descriptors['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
            
            # Additional descriptors
            descriptors['Chi0'] = Descriptors.Chi0(mol)
            descriptors['Chi1'] = Descriptors.Chi1(mol)
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)
            
        except Exception as e:
            self.logger.warning(f"Error calculating descriptors: {e}")
        
        return descriptors
    
    def calculate_drug_likeness(self, mol: Chem.Mol) -> Dict[str, any]:
        """
        Calculate drug-likeness properties
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary with drug-likeness metrics
        """
        if mol is None:
            return {}
        
        metrics = {}
        
        try:
            # Lipinski's Rule of Five
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            lipinski_violations = 0
            if mw > 500: lipinski_violations += 1
            if logp > 5: lipinski_violations += 1
            if hbd > 5: lipinski_violations += 1
            if hba > 10: lipinski_violations += 1
            
            metrics['lipinski_violations'] = lipinski_violations
            metrics['lipinski_pass'] = lipinski_violations <= 1
            
            # Veber's Rule (oral bioavailability)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            
            veber_pass = (rotatable_bonds <= 10) and (tpsa <= 140)
            metrics['veber_pass'] = veber_pass
            
            # Ghose Filter
            ghose_pass = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6)
            metrics['ghose_pass'] = ghose_pass
            
            # QED (Quantitative Estimate of Drug-likeness)
            try:
                from rdkit.Chem import QED
                metrics['qed'] = QED.qed(mol)
            except:
                metrics['qed'] = None
            
        except Exception as e:
            self.logger.warning(f"Error calculating drug-likeness: {e}")
        
        return metrics
    
    def calculate_fingerprint_properties(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Calculate properties from molecular fingerprints
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dictionary with fingerprint-based properties
        """
        if mol is None:
            return {}
        
        properties = {}
        
        try:
            # Functional group counts
            properties['num_aromatic_carbocycles'] = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
            properties['num_aromatic_heterocycles'] = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
            properties['num_aliphatic_carbocycles'] = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
            properties['num_aliphatic_heterocycles'] = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
            
            # Bridgehead atoms
            properties['num_bridgehead_atoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            properties['num_spiro_atoms'] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            
        except Exception as e:
            self.logger.warning(f"Error calculating fingerprint properties: {e}")
        
        return properties
    
    def calculate_all_descriptors(self, smiles: str) -> Dict[str, any]:
        """
        Calculate all available descriptors for a SMILES string
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with all descriptors
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {'smiles': smiles, 'valid': False}
        
        # Combine all descriptor types
        all_descriptors = {'smiles': smiles, 'valid': True}
        all_descriptors.update(self.calculate_2d_descriptors(mol))
        all_descriptors.update(self.calculate_drug_likeness(mol))
        all_descriptors.update(self.calculate_fingerprint_properties(mol))
        
        return all_descriptors
    
    def calculate_descriptors_batch(
        self,
        smiles_list: List[str],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Calculate descriptors for a list of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with descriptors
        """
        from tqdm import tqdm
        
        results = []
        
        iterator = tqdm(smiles_list, desc="Calculating descriptors") if show_progress else smiles_list
        
        for smiles in iterator:
            descriptors = self.calculate_all_descriptors(smiles)
            results.append(descriptors)
        
        df = pd.DataFrame(results)
        
        # Remove invalid molecules
        valid_count = df['valid'].sum()
        df = df[df['valid'] == True].drop(columns=['valid'])
        
        self.logger.info(f"Calculated descriptors for {valid_count}/{len(smiles_list)} molecules")
        
        return df
    
    def add_descriptors_to_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'canonical_smiles'
    ) -> pd.DataFrame:
        """
        Add descriptors to an existing DataFrame
        
        Args:
            df: Input DataFrame with SMILES column
            smiles_col: Name of SMILES column
            
        Returns:
            DataFrame with added descriptors
        """
        self.logger.info(f"Adding descriptors to DataFrame with {len(df)} rows")
        
        # Calculate descriptors
        descriptors_df = self.calculate_descriptors_batch(df[smiles_col].tolist())
        
        # Merge with original DataFrame
        result_df = df.copy()
        
        # Drop SMILES column from descriptors if it exists
        if 'smiles' in descriptors_df.columns:
            descriptors_df = descriptors_df.drop(columns=['smiles'])
        
        # Concatenate
        result_df = pd.concat([result_df.reset_index(drop=True), 
                               descriptors_df.reset_index(drop=True)], axis=1)
        
        self.logger.info(f"Added {len(descriptors_df.columns)} descriptor columns")
        
        return result_df


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
    
    # Initialize calculator
    calculator = DescriptorCalculator(config)
    
    # Example 1: Single molecule
    print("\nExample 1: Single molecule")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    descriptors = calculator.calculate_all_descriptors(smiles)
    print(f"Calculated {len(descriptors)} descriptors for aspirin")
    print(f"MolWt: {descriptors.get('MolWt', 'N/A'):.2f}")
    print(f"LogP: {descriptors.get('LogP', 'N/A'):.2f}")
    print(f"Lipinski violations: {descriptors.get('lipinski_violations', 'N/A')}")
    
    # Example 2: Batch processing
    print("\nExample 2: Load and process dataset")
    data_path = Path('data/processed/chembl_egfr_clean.csv')
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} molecules")
        
        # Take a small sample for demonstration
        df_sample = df.head(100)
        
        # Calculate descriptors
        df_with_descriptors = calculator.add_descriptors_to_dataframe(df_sample)
        
        print(f"\nOriginal columns: {len(df_sample.columns)}")
        print(f"After descriptors: {len(df_with_descriptors.columns)}")
        print(f"Added columns: {len(df_with_descriptors.columns) - len(df_sample.columns)}")
        
        # Save
        output_path = Path('data/processed/chembl_egfr_with_descriptors.csv')
        df_with_descriptors.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
    else:
        print(f"Data file not found: {data_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()