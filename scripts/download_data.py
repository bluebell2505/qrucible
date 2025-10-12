"""
Data Download Script for Qrucible
Downloads molecular data from ChEMBL database

Usage:
    python scripts/download_data.py --source chembl --target "EGFR kinase" --limit 30000
"""

import argparse
import sys
from pathlib import Path
import logging
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download molecular data for Qrucible'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='chembl',
        choices=['chembl', 'bindingdb', 'zinc'],
        help='Data source to download from'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='EGFR',
        help='Target protein name (e.g., "EGFR kinase", "p53")'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Maximum number of compounds to download'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=6.5,
        help='Activity threshold (pIC50) for filtering'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (optional, auto-generated if not provided)'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    config_file = PROJECT_ROOT / config_path
    
    # Default minimal config
    default_config = {
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'sources': {
                'chembl': {
                    'enabled': True,
                    'target': 'EGFR',
                    'activity_threshold': 6.5,
                    'max_compounds': 10000
                }
            }
        }
    }
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if config is valid
        if config is None or not isinstance(config, dict):
            logger.warning(f"Config file is empty or invalid, using defaults")
            config = default_config
        elif 'data' not in config or config['data'] is None:
            logger.warning(f"Config file missing 'data' section, using defaults")
            config = default_config
        else:
            logger.info(f"Loaded configuration from: {config_file}")
    else:
        logger.warning(f"Config file not found: {config_file}")
        logger.info("Using default configuration")
        config = default_config
    
    return config


def main():
    """Main execution function"""
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("QRUCIBLE - DATA DOWNLOAD SCRIPT")
    logger.info("=" * 70)
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info("=" * 70)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['data']['sources']['chembl']['target'] = args.target
    config['data']['sources']['chembl']['max_compounds'] = args.limit
    config['data']['sources']['chembl']['activity_threshold'] = args.threshold
    
    # Initialize data loader
    loader = DataLoader(config)
    
    try:
        if args.source == 'chembl':
            logger.info("Downloading from ChEMBL database...")
            df = loader.load_from_chembl(
                target_name=args.target,
                activity_threshold=args.threshold,
                max_compounds=args.limit
            )
            
            # Generate output filename if not provided
            if args.output is None:
                target_clean = args.target.lower().replace(' ', '_').replace('-', '_')
                output_file = PROJECT_ROOT / 'data' / 'raw' / f'chembl_{target_clean}_raw.csv'
            else:
                output_file = PROJECT_ROOT / args.output
            
            # Save data
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            
            logger.info("=" * 70)
            logger.info("DOWNLOAD COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Total compounds downloaded: {len(df)}")
            logger.info(f"Output file: {output_file}")
            
            # Display dataset info
            info = loader.get_dataset_info(df)
            logger.info("\nDataset Summary:")
            logger.info(f"  Compounds: {info['n_compounds']}")
            logger.info(f"  Features: {info['n_features']}")
            logger.info(f"  Duplicates: {info['duplicates']}")
            
            if 'activity_stats' in info:
                logger.info("\nActivity Statistics (pIC50):")
                logger.info(f"  Mean: {info['activity_stats']['mean']:.2f}")
                logger.info(f"  Std: {info['activity_stats']['std']:.2f}")
                logger.info(f"  Range: [{info['activity_stats']['min']:.2f}, {info['activity_stats']['max']:.2f}]")
            
            logger.info("\nNext steps:")
            logger.info("1. Review the downloaded data")
            logger.info(f"2. Run preprocessing: python src/data/preprocessor.py")
            logger.info("3. Or use the notebook: notebooks/01_data_exploration.ipynb")
            
        elif args.source == 'bindingdb':
            logger.error("BindingDB download not yet implemented")
            logger.info("Please download manually from: https://www.bindingdb.org/")
            sys.exit(1)
            
        elif args.source == 'zinc':
            logger.error("ZINC download not yet implemented")
            logger.info("Please download manually from: https://zinc.docking.org/")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()