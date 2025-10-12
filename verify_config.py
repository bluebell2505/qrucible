"""
Verify and fix configuration file
Run this if you're having config issues

Usage:
    python verify_config.py
"""

import yaml
from pathlib import Path

def create_default_config():
    """Create a working default config"""
    config = {
        'project': {
            'name': 'Qrucible',
            'version': '0.1.0',
            'description': 'Hybrid Quantum-Classical ML for Drug Discovery'
        },
        
        'data': {
            'raw_dir': 'data/raw',
            'processed_dir': 'data/processed',
            'interim_dir': 'data/interim',
            
            'sources': {
                'chembl': {
                    'enabled': True,
                    'target': 'EGFR',
                    'activity_threshold': 6.5,
                    'max_compounds': 10000
                },
                'bindingdb': {
                    'enabled': False,
                    'target': None
                },
                'zinc': {
                    'enabled': False,
                    'subset': 'lead-like'
                }
            },
            
            'preprocessing': {
                'remove_duplicates': True,
                'remove_salts': True,
                'neutralize': True,
                'standardize': True,
                'remove_mixtures': True,
                'min_heavy_atoms': 6,
                'max_heavy_atoms': 50
            }
        },
        
        'features': {
            'descriptors_2d': [
                'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                'NumRotatableBonds', 'NumAromaticRings', 'FractionCsp3',
                'MolMR', 'BalabanJ'
            ],
            
            'descriptors_3d': {
                'enabled': False,
                'conformer_generation': True
            },
            
            'fingerprints': {
                'morgan': {
                    'enabled': True,
                    'radius': 2,
                    'n_bits': 2048
                },
                'maccs': {
                    'enabled': True
                },
                'topological': {
                    'enabled': False
                }
            },
            
            'quantum_features': {
                'enabled': False
            }
        },
        
        'classical_models': {
            'random_forest': {
                'enabled': True,
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            
            'gradient_boosting': {
                'enabled': True,
                'algorithm': 'xgboost',
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            
            'ensemble': {
                'enabled': True,
                'method': 'weighted_average',
                'weights': [0.5, 0.5]
            }
        },
        
        'training': {
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'cv_folds': 5,
            'metrics': ['r2', 'mae', 'rmse', 'pearson']
        },
        
        'explainability': {
            'shap': {
                'enabled': True,
                'max_samples': 1000,
                'algorithm': 'tree'
            },
            'lime': {
                'enabled': True,
                'n_samples': 5000
            },
            'uncertainty': {
                'enabled': True,
                'confidence_threshold': 0.7
            }
        },
        
        'ranking': {
            'multi_objective': {
                'binding_affinity': {
                    'weight': 0.4,
                    'direction': 'maximize'
                },
                'drug_likeness': {
                    'weight': 0.2,
                    'direction': 'maximize',
                    'rules': 'lipinski'
                },
                'admet': {
                    'weight': 0.2,
                    'direction': 'maximize'
                },
                'confidence': {
                    'weight': 0.2,
                    'direction': 'maximize'
                }
            },
            'top_n_candidates': 20
        },
        
        'quantum': {
            'enabled': False,
            'backend': {
                'simulator': 'qasm_simulator',
                'real_hardware': False,
                'device_name': 'ibm_brisbane'
            },
            'algorithms': {
                'vqe': {
                    'optimizer': 'SLSQP',
                    'max_iterations': 1000,
                    'convergence_threshold': 1e-6
                },
                'qaoa': {
                    'p_layers': 3,
                    'optimizer': 'COBYLA'
                }
            },
            'error_mitigation': {
                'enabled': True,
                'method': 'readout'
            }
        },
        
        'active_learning': {
            'enabled': False,
            'iterations': 3,
            'selection_strategy': 'uncertainty',
            'batch_size': 10,
            'retrain_interval': 1
        },
        
        'visualization': {
            'plots_dir': 'results/figures',
            'dpi': 300,
            'format': 'png',
            'dashboard': {
                'enabled': False,
                'framework': 'streamlit',
                'port': 8501
            }
        },
        
        'results': {
            'models_dir': 'results/models',
            'reports_dir': 'results/reports',
            'benchmarks_dir': 'results/benchmarks',
            'save_predictions': True,
            'save_shap_values': True
        },
        
        'logging': {
            'level': 'INFO',
            'file': 'qrucible.log',
            'console': True
        }
    }
    
    return config


def main():
    print("\n" + "="*70)
    print("Configuration Verification Tool")
    print("="*70 + "\n")
    
    # Check if config exists
    config_path = Path('config/config.yaml')
    
    if not config_path.exists():
        print("❌ Config file not found!")
        print(f"Expected location: {config_path.absolute()}")
        
        # Create directory
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create config
        config = create_default_config()
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Created new config file: {config_path}")
        print("\nPlease review and edit the configuration as needed.")
        return
    
    # Try to load existing config
    print(f"Found config file: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print("❌ Config file is empty!")
            raise ValueError("Empty config")
        
        # Verify required sections
        required_sections = ['data', 'features', 'classical_models', 'training']
        missing_sections = []
        
        for section in required_sections:
            if section not in config or config[section] is None:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing required sections: {missing_sections}")
            raise ValueError("Missing sections")
        
        print("✅ Config file is valid!")
        print(f"\nConfiguration summary:")
        print(f"  - Target: {config['data']['sources']['chembl']['target']}")
        print(f"  - Max compounds: {config['data']['sources']['chembl']['max_compounds']}")
        print(f"  - Activity threshold: {config['data']['sources']['chembl']['activity_threshold']}")
        print(f"  - Models enabled: RF={config['classical_models']['random_forest']['enabled']}, "
              f"XGB={config['classical_models']['gradient_boosting']['enabled']}")
        
    except Exception as e:
        print(f"❌ Config file is invalid: {e}")
        print("\nCreating backup and regenerating config...")
        
        # Backup old config
        backup_path = config_path.with_suffix('.yaml.bak')
        if config_path.exists():
            config_path.rename(backup_path)
            print(f"  Backed up old config to: {backup_path}")
        
        # Create new config
        config = create_default_config()
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Created new config file: {config_path}")
        print("\nPlease review and edit the configuration as needed.")
    
    print("\n" + "="*70)
    print("Verification complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()