"""
Molecular Fingerprint Generator for Qrucible
Generates various types of molecular fingerprints for ML models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit import DataStructs


class FingerprintGenerator:
    """
    Generates molecular fingerprints for machine learning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize fingerprint generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get fingerprint configuration
        fp_config = self.config.get('features', {}).get('fingerprints', {})
        self.morgan_config = fp_config.get('morgan', {})
        self.maccs_enabled = fp_config.get('maccs', {}).get('enabled', True)
    
    def generate_morgan_fingerprint(
        self,
        mol: Chem.Mol,
        radius: int = 2,
        n_bits: int = 2048,
        use_features: bool = False
    ) -> np.ndarray:
        """
        Generate Morgan (circular) fingerprint
        
        Args:
            mol: RDKit Mol object
            radius: Radius of the fingerprint
            n_bits: Number of bits in the fingerprint
            use_features: Use feature-based Morgan fingerprint
            
        Returns:
            Numpy array of fingerprint bits
        """
        if mol is None:
            return np.zeros(n_bits)
        
        try:
            if use_features:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius, nBits=n_bits, useFeatures=True
                )
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius, nBits=n_bits
                )
            
            # Convert to numpy array
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            
            return arr
        
        except Exception as e:
            self.logger.warning(f"Error generating Morgan fingerprint: {e}")
            return np.zeros(n_bits)
    
    def generate_maccs_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """
        Generate MACCS keys fingerprint (166 bits)
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Numpy array of fingerprint bits
        """
        if mol is None:
            return np.zeros(166)
        
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((166,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        
        except Exception as e:
            self.logger.warning(f"Error generating MACCS fingerprint: {e}")
            return np.zeros(166)
    
    def generate_topological_fingerprint(
        self,
        mol: Chem.Mol,
        n_bits: int = 2048
    ) -> np.ndarray:
        """
        Generate RDKit topological fingerprint
        
        Args:
            mol: RDKit Mol object
            n_bits: Number of bits in the fingerprint
            
        Returns:
            Numpy array of fingerprint bits
        """
        if mol is None:
            return np.zeros(n_bits)
        
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        
        except Exception as e:
            self.logger.warning(f"Error generating topological fingerprint: {e}")
            return np.zeros(n_bits)
    
    def generate_atom_pair_fingerprint(
        self,
        mol: Chem.Mol,
        n_bits: int = 2048
    ) -> np.ndarray:
        """
        Generate atom pair fingerprint
        
        Args:
            mol: RDKit Mol object
            n_bits: Number of bits in the fingerprint
            
        Returns:
            Numpy array of fingerprint bits
        """
        if mol is None:
            return np.zeros(n_bits)
        
        try:
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=n_bits
            )
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        
        except Exception as e:
            self.logger.warning(f"Error generating atom pair fingerprint: {e}")
            return np.zeros(n_bits)
    
    def generate_all_fingerprints(
        self,
        smiles: str,
        morgan_radius: int = 2,
        morgan_n_bits: int = 2048
    ) -> Dict[str, np.ndarray]:
        """
        Generate all available fingerprints for a molecule
        
        Args:
            smiles: SMILES string
            morgan_radius: Radius for Morgan fingerprint
            morgan_n_bits: Number of bits for Morgan fingerprint
            
        Returns:
            Dictionary with all fingerprint types
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {
                'morgan': np.zeros(morgan_n_bits),
                'maccs': np.zeros(166),
                'topological': np.zeros(2048),
                'atom_pair': np.zeros(2048)
            }
        
        fingerprints = {}
        
        # Generate each fingerprint type
        fingerprints['morgan'] = self.generate_morgan_fingerprint(
            mol, radius=morgan_radius, n_bits=morgan_n_bits
        )
        fingerprints['maccs'] = self.generate_maccs_fingerprint(mol)
        fingerprints['topological'] = self.generate_topological_fingerprint(mol)
        fingerprints['atom_pair'] = self.generate_atom_pair_fingerprint(mol)
        
        return fingerprints
    
    def fingerprints_to_dataframe(
        self,
        smiles_list: List[str],
        fp_type: str = 'morgan',
        show_progress: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate fingerprints for a list of SMILES and return as DataFrame
        
        Args:
            smiles_list: List of SMILES strings
            fp_type: Type of fingerprint ('morgan', 'maccs', 'topological', 'atom_pair', 'all')
            show_progress: Whether to show progress bar
            **kwargs: Additional arguments for fingerprint generation
            
        Returns:
            DataFrame with fingerprint columns
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(smiles_list, desc=f"Generating {fp_type} fingerprints") if show_progress else smiles_list
        
        for smiles in iterator:
            mol = Chem.MolFromSmiles(smiles)
            
            if fp_type == 'morgan':
                fp = self.generate_morgan_fingerprint(
                    mol,
                    radius=kwargs.get('radius', 2),
                    n_bits=kwargs.get('n_bits', 2048)
                )
            elif fp_type == 'maccs':
                fp = self.generate_maccs_fingerprint(mol)
            elif fp_type == 'topological':
                fp = self.generate_topological_fingerprint(
                    mol,
                    n_bits=kwargs.get('n_bits', 2048)
                )
            elif fp_type == 'atom_pair':
                fp = self.generate_atom_pair_fingerprint(
                    mol,
                    n_bits=kwargs.get('n_bits', 2048)
                )
            elif fp_type == 'all':
                all_fps = self.generate_all_fingerprints(smiles, **kwargs)
                # Concatenate all fingerprints
                fp = np.concatenate([all_fps[key] for key in all_fps])
            else:
                raise ValueError(f"Unknown fingerprint type: {fp_type}")
            
            results.append(fp)
        
        # Create DataFrame
        fp_array = np.array(results)
        
        # Generate column names
        if fp_type == 'all':
            col_names = []
            for key in ['morgan', 'maccs', 'topological', 'atom_pair']:
                n = len(all_fps[key])
                col_names.extend([f'{key}_{i}' for i in range(n)])
        else:
            col_names = [f'{fp_type}_{i}' for i in range(fp_array.shape[1])]
        
        df = pd.DataFrame(fp_array, columns=col_names)
        
        self.logger.info(f"Generated {fp_type} fingerprints: {df.shape}")
        
        return df
    
    def calculate_similarity(
        self,
        smiles1: str,
        smiles2: str,
        fp_type: str = 'morgan',
        metric: str = 'tanimoto'
    ) -> float:
        """
        Calculate similarity between two molecules
        
        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string
            fp_type: Type of fingerprint to use
            metric: Similarity metric ('tanimoto', 'dice', 'cosine')
            
        Returns:
            Similarity score (0-1)
        """
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        # Generate fingerprints
        if fp_type == 'morgan':
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        elif fp_type == 'maccs':
            fp1 = MACCSkeys.GenMACCSKeys(mol1)
            fp2 = MACCSkeys.GenMACCSKeys(mol2)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        # Calculate similarity
        if metric == 'tanimoto':
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        elif metric == 'dice':
            similarity = DataStructs.DiceSimilarity(fp1, fp2)
        elif metric == 'cosine':
            similarity = DataStructs.CosineSimilarity(fp1, fp2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return similarity
    
    def find_similar_molecules(
        self,
        query_smiles: str,
        smiles_list: List[str],
        threshold: float = 0.7,
        top_n: Optional[int] = None,
        fp_type: str = 'morgan'
    ) -> List[tuple]:
        """
        Find similar molecules to a query molecule
        
        Args:
            query_smiles: Query SMILES string
            smiles_list: List of SMILES to search
            threshold: Minimum similarity threshold
            top_n: Return only top N results (if None, return all above threshold)
            fp_type: Type of fingerprint to use
            
        Returns:
            List of (smiles, similarity) tuples
        """
        similarities = []
        
        for smiles in smiles_list:
            sim = self.calculate_similarity(query_smiles, smiles, fp_type=fp_type)
            if sim >= threshold:
                similarities.append((smiles, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N if specified
        if top_n is not None:
            similarities = similarities[:top_n]
        
        self.logger.info(f"Found {len(similarities)} similar molecules")
        
        return similarities


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
    
    # Initialize generator
    generator = FingerprintGenerator(config)
    
    # Example 1: Single molecule fingerprints
    print("\nExample 1: Generate fingerprints for aspirin")
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    fingerprints = generator.generate_all_fingerprints(aspirin)
    print(f"Morgan fingerprint: {len(fingerprints['morgan'])} bits, {fingerprints['morgan'].sum()} set")
    print(f"MACCS fingerprint: {len(fingerprints['maccs'])} bits, {fingerprints['maccs'].sum()} set")
    
    # Example 2: Similarity calculation
    print("\nExample 2: Calculate similarity")
    ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    similarity = generator.calculate_similarity(aspirin, ibuprofen)
    print(f"Similarity between aspirin and ibuprofen: {similarity:.3f}")
    
    # Example 3: Batch processing
    print("\nExample 3: Batch fingerprint generation")
    data_path = Path('data/processed/chembl_egfr_clean.csv')
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} molecules")
        
        # Take a small sample
        df_sample = df.head(100)
        
        # Generate Morgan fingerprints
        fp_df = generator.fingerprints_to_dataframe(
            df_sample['canonical_smiles'].tolist(),
            fp_type='morgan',
            radius=2,
            n_bits=2048
        )
        
        print(f"Generated fingerprints: {fp_df.shape}")
        print(f"Sparsity: {(fp_df == 0).sum().sum() / fp_df.size * 100:.1f}%")
        
        # Save
        output_path = Path('data/processed/chembl_egfr_fingerprints.csv')
        fp_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
    else:
        print(f"Data file not found: {data_path}")
    
    # Example 4: Find similar molecules
    print("\nExample 4: Find similar molecules")
    if data_path.exists():
        df = pd.read_csv(data_path)
        query = df['canonical_smiles'].iloc[0]
        candidates = df['canonical_smiles'].iloc[1:20].tolist()
        
        similar = generator.find_similar_molecules(
            query,
            candidates,
            threshold=0.5,
            top_n=5
        )
        
        print(f"\nTop 5 similar molecules to query:")
        for i, (smiles, sim) in enumerate(similar, 1):
            print(f"{i}. Similarity: {sim:.3f} - {smiles[:50]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()