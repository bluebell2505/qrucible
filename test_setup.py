"""
Setup Verification Script
Tests all imports and basic functionality

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test all module imports"""
    print("\n" + "="*70)
    print("TESTING MODULE IMPORTS")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd() / 'src'))
    
    # Test data modules
    try:
        from data.data_loader import DataLoader
        print("✓ data.data_loader")
        tests_passed += 1
    except Exception as e:
        print(f"✗ data.data_loader: {e}")
        tests_failed += 1
    
    try:
        from data.preprocessor import MolecularPreprocessor
        print("✓ data.preprocessor")
        tests_passed += 1
    except Exception as e:
        print(f"✗ data.preprocessor: {e}")
        tests_failed += 1
    
    try:
        from data.splitter import DataSplitter
        print("✓ data.splitter")
        tests_passed += 1
    except Exception as e:
        print(f"✗ data.splitter: {e}")
        tests_failed += 1
    
    # Test feature modules
    try:
        from features.molecular_descriptors import DescriptorCalculator
        print("✓ features.molecular_descriptors")
        tests_passed += 1
    except Exception as e:
        print(f"✗ features.molecular_descriptors: {e}")
        tests_failed += 1
    
    try:
        from features.fingerprints import FingerprintGenerator
        print("✓ features.fingerprints")
        tests_passed += 1
    except Exception as e:
        print(f"✗ features.fingerprints: {e}")
        tests_failed += 1
    
    # Test model modules
    try:
        from models.classical.random_forest import RandomForestModel
        print("✓ models.classical.random_forest")
        tests_passed += 1
    except Exception as e:
        print(f"✗ models.classical.random_forest: {e}")
        tests_failed += 1
    
    try:
        from models.classical.gradient_boosting import GradientBoostingModel
        print("✓ models.classical.gradient_boosting")
        tests_passed += 1
    except Exception as e:
        print(f"✗ models.classical.gradient_boosting: {e}")
        tests_failed += 1
    
    # Test utility modules
    try:
        from utils.logger import setup_logger
        print("✓ utils.logger")
        tests_passed += 1
    except Exception as e:
        print(f"✗ utils.logger: {e}")
        tests_failed += 1
    
    try:
        from utils.metrics import RegressionMetrics
        print("✓ utils.metrics")
        tests_passed += 1
    except Exception as e:
        print(f"✗ utils.metrics: {e}")
        tests_failed += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*70}\n")
    
    return tests_failed == 0


def test_dependencies():
    """Test key dependencies"""
    print("\n" + "="*70)
    print("TESTING DEPENDENCIES")
    print("="*70 + "\n")
    
    dependencies = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('rdkit', 'Chem'),
        ('yaml', 'yaml'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    for module_name, import_as in dependencies:
        try:
            if '.' in module_name:
                # Handle submodules
                parts = module_name.split('.')
                mod = __import__(module_name, fromlist=[parts[-1]])
            else:
                mod = __import__(module_name)
            
            # Get version if available
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {module_name:.<40} v{version}")
            tests_passed += 1
        except Exception as e:
            print(f"✗ {module_name:.<40} MISSING")
            tests_failed += 1
    
    # Test optional dependencies
    print("\nOptional Dependencies:")
    optional = [
        ('xgboost', 'xgb'),
        ('shap', 'shap'),
    ]
    
    for module_name, import_as in optional:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {module_name:.<40} v{version}")
        except:
            print(f"⚠ {module_name:.<40} Not installed (optional)")
    
    print(f"\n{'='*70}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*70}\n")
    
    return tests_failed == 0


def test_file_structure():
    """Test directory structure"""
    print("\n" + "="*70)
    print("TESTING FILE STRUCTURE")
    print("="*70 + "\n")
    
    required_dirs = [
        'src/data',
        'src/features',
        'src/models/classical',
        'src/utils',
        'scripts',
        'config',
        'data/raw',
        'data/processed',
        'results/models',
        'results/figures',
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
            tests_passed += 1
        else:
            print(f"✗ {dir_path} (missing)")
            tests_failed += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*70}\n")
    
    return tests_failed == 0


def test_config():
    """Test configuration file"""
    print("\n" + "="*70)
    print("TESTING CONFIGURATION")
    print("="*70 + "\n")
    
    import yaml
    
    config_path = Path('config/config.yaml')
    
    if not config_path.exists():
        print("✗ config/config.yaml not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print("✗ Config file is empty")
            return False
        
        required_sections = ['data', 'features', 'classical_models', 'training']
        missing = []
        
        for section in required_sections:
            if section in config and config[section] is not None:
                print(f"✓ Section '{section}' exists")
            else:
                print(f"✗ Section '{section}' missing or empty")
                missing.append(section)
        
        if missing:
            print(f"\n{'='*70}")
            print("Config validation FAILED")
            print(f"{'='*70}\n")
            return False
        
        print(f"\n{'='*70}")
        print("Config validation PASSED")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False


def test_data_availability():
    """Test if data files exist"""
    print("\n" + "="*70)
    print("TESTING DATA AVAILABILITY")
    print("="*70 + "\n")
    
    raw_data = Path('data/raw/chembl_egfr_kinase_raw.csv')
    processed_data = Path('data/processed/chembl_egfr_kinase_clean.csv')
    
    if raw_data.exists():
        print(f"✓ Raw data found: {raw_data}")
        print(f"  Size: {raw_data.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"⚠ Raw data not found: {raw_data}")
        print("  Run: python scripts/download_data.py")
    
    if processed_data.exists():
        print(f"✓ Processed data found: {processed_data}")
        print(f"  Size: {processed_data.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Try to load it
        try:
            import pandas as pd
            df = pd.read_csv(processed_data)
            print(f"  Compounds: {len(df)}")
            print(f"  Features: {len(df.columns)}")
        except Exception as e:
            print(f"  ⚠ Error reading file: {e}")
    else:
        print(f"⚠ Processed data not found: {processed_data}")
        print("  Run: python scripts/run_preprocessing.py")
    
    print(f"\n{'='*70}\n")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("QRUCIBLE SETUP VERIFICATION")
    print("="*70)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_dependencies()
    all_passed &= test_file_structure()
    all_passed &= test_config()
    all_passed &= test_imports()
    test_data_availability()  # Informational only
    
    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Setup is complete!")
        print("="*70)
        print("\nYou're ready to:")
        print("  1. Train models: python scripts/train_classical.py")
        print("  2. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    else:
        print("❌ SOME TESTS FAILED - Please fix the issues above")
        print("="*70)
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Create directories: python init_project.py")
        print("  - Fix config: python verify_config.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()