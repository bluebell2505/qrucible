"""
Complete Preprocessing Pipeline for Qrucible
Combines data download and preprocessing in one script

Usage:
    python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path
import logging
import yaml
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data.data_loader import DataLoader
from data.preprocessor import MolecularPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete preprocessing pipeline"""
    
    print("\n" + "=" * 70)
    print("QRUCIBLE - PREPROCESSING PIPELINE")
    print("=" * 70 + "\n")
    
    # Load configuration
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create config/config.yaml first")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract settings
    target = config['data']['sources']['chembl']['target']
    threshold = config['data']['sources']['chembl']['activity_threshold']
    max_compounds = config['data']['sources']['chembl']['max_compounds']
    
    logger.info(f"Target: {target}")
    logger.info(f"Activity threshold: {threshold}")
    logger.info(f"Max compounds: {max_compounds}\n")
    
    # Step 1: Check if raw data exists
    # raw_file = PROJECT_ROOT / 'data' / 'raw' / f'chembl_{target.lower()}_raw.csv'
    target = 'egfr_kinase'
    raw_file = PROJECT_ROOT / 'data' / 'raw' / f'chembl_{target}_raw.csv'
    
    if raw_file.exists():
        logger.info(f"Found existing raw data: {raw_file}")
        logger.info("Loading from file...")
        df_raw = pd.read_csv(raw_file)
    else:
        logger.info("Raw data not found. Downloading from ChEMBL...")
        loader = DataLoader(config)
        df_raw = loader.load_from_chembl(
            target_name=target,
            activity_threshold=threshold,
            max_compounds=max_compounds
        )
    
    logger.info(f"Raw data size: {len(df_raw)} compounds\n")
    
    # Step 2: Preprocess data
    logger.info("Starting data preprocessing...")
    preprocessor = MolecularPreprocessor(config)
    df_clean = preprocessor.clean_dataset(df_raw)
    
    logger.info(f"Cleaned data size: {len(df_clean)} compounds")
    logger.info(f"Reduction: {(1 - len(df_clean)/len(df_raw)) * 100:.1f}%\n")
    
    # Step 3: Add molecular properties
    logger.info("Calculating molecular properties...")
    df_clean = preprocessor.add_basic_properties(df_clean)
    logger.info(f"Added {len(df_clean.columns)} total features\n")
    
    # Step 4: Save processed data
    processed_file = PROJECT_ROOT / 'data' / 'processed' / f'chembl_{target.lower()}_clean.csv'
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_file, index=False)
    
    logger.info(f"Saved processed data to: {processed_file}\n")
    
    # Step 5: Summary statistics
    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Original compounds:     {len(df_raw):>6}")
    print(f"  Cleaned compounds:      {len(df_clean):>6}")
    print(f"  Reduction:              {(1 - len(df_clean)/len(df_raw)) * 100:>5.1f}%")
    
    if 'pIC50' in df_clean.columns:
        print(f"\n🎯 Activity (pIC50):")
        print(f"  Mean:                   {df_clean['pIC50'].mean():>6.2f}")
        print(f"  Std:                    {df_clean['pIC50'].std():>6.2f}")
        print(f"  Range:                  {df_clean['pIC50'].min():.2f} - {df_clean['pIC50'].max():.2f}")
    
    if 'lipinski_violations' in df_clean.columns:
        drug_like = (df_clean['lipinski_violations'] <= 1).sum()
        print(f"\n💊 Drug-likeness:")
        print(f"  Drug-like (≤1 violation): {drug_like:>6} ({drug_like/len(df_clean)*100:.1f}%)")
        print(f"  Poor drug-like (>1):      {len(df_clean) - drug_like:>6} ({(len(df_clean)-drug_like)/len(df_clean)*100:.1f}%)")
    
    print(f"\n📁 Output Files:")
    print(f"  Raw:       {raw_file}")
    print(f"  Processed: {processed_file}")
    
    print("\n✅ Preprocessing Complete!")
    print("\n📝 Next Steps:")
    print("  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("  2. Feature engineering: notebooks/02_feature_engineering.ipynb")
    print("  3. Train models: python scripts/train_classical.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()