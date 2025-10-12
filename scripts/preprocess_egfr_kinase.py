"""
Direct preprocessing script for EGFR kinase data
Processes the downloaded data and prepares it for training

Usage:
    python preprocess_egfr_kinase.py
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data.preprocessor import MolecularPreprocessor

def main():
    print("\n" + "="*70)
    print("QRUCIBLE - DATA PREPROCESSING")
    print("="*70 + "\n")
    
    # Load config
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Load the raw data
    raw_file = PROJECT_ROOT / 'data' / 'raw' / 'chembl_egfr kinase_raw.csv'
    
    if not raw_file.exists():
        print(f"❌ Raw data file not found: {raw_file}")
        print("\nPlease run download first:")
        print('  python scripts/download_data.py --source chembl --target "EGFR kinase" --limit 20000 --threshold 5.0')
        return
    
    print(f"📁 Loading raw data from: {raw_file}")
    df_raw = pd.read_csv(raw_file)
    print(f"   Loaded {len(df_raw)} compounds\n")
    
    # Initialize preprocessor
    preprocessor = MolecularPreprocessor(config)
    
    # Step 1: Clean dataset
    print("🧹 Step 1: Cleaning dataset...")
    df_clean = preprocessor.clean_dataset(df_raw)
    print(f"   Cleaned: {len(df_clean)} compounds")
    print(f"   Reduction: {(1 - len(df_clean)/len(df_raw)) * 100:.1f}%\n")
    
    # Step 2: Add molecular properties
    print("🔬 Step 2: Calculating molecular properties...")
    df_clean = preprocessor.add_basic_properties(df_clean)
    print(f"   Added {len(df_clean.columns)} total features\n")
    
    # Step 3: Save processed data
    output_file = PROJECT_ROOT / 'data' / 'processed' / 'chembl_egfr_kinase_clean.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    
    print(f"💾 Saved processed data to: {output_file}\n")
    
    # Summary
    print("="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Original compounds:     {len(df_raw):>6,}")
    print(f"  Cleaned compounds:      {len(df_clean):>6,}")
    print(f"  Reduction:              {(1 - len(df_clean)/len(df_raw)) * 100:>5.1f}%")
    
    if 'pIC50' in df_clean.columns:
        print(f"\n🎯 Activity (pIC50):")
        print(f"  Mean:                   {df_clean['pIC50'].mean():>6.2f}")
        print(f"  Std:                    {df_clean['pIC50'].std():>6.2f}")
        print(f"  Min:                    {df_clean['pIC50'].min():>6.2f}")
        print(f"  Max:                    {df_clean['pIC50'].max():>6.2f}")
    
    if 'lipinski_violations' in df_clean.columns:
        drug_like = (df_clean['lipinski_violations'] <= 1).sum()
        print(f"\n💊 Drug-likeness:")
        print(f"  Drug-like (≤1 violation): {drug_like:>6,} ({drug_like/len(df_clean)*100:.1f}%)")
        print(f"  Poor drug-like (>1):      {len(df_clean) - drug_like:>6,} ({(len(df_clean)-drug_like)/len(df_clean)*100:.1f}%)")
    
    print(f"\n📁 Files:")
    print(f"  Raw:       {raw_file}")
    print(f"  Processed: {output_file}")
    
    print("\n✅ Preprocessing Complete!")
    print("\n📝 Next Steps:")
    print("  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("  2. Train models: python scripts/train_classical.py --data data/processed/chembl_egfr_kinase_clean.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()