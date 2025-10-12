"""
Project Initialization Script
Creates all necessary directories and files for Qrucible

Usage:
    python init_project.py
"""

from pathlib import Path
import sys

def create_directory_structure():
    """Create the complete project directory structure"""
    
    directories = [
        # Data directories
        'data/raw',
        'data/processed',
        'data/interim',
        
        # Source code directories
        'src/data',
        'src/features',
        'src/models/classical',
        'src/models/quantum',
        'src/explainability',
        'src/ranking',
        'src/active_learning',
        'src/visualization',
        'src/utils',
        
        # Notebooks
        'notebooks',
        
        # Scripts
        'scripts',
        
        # Tests
        'tests/test_data',
        'tests/test_features',
        'tests/test_models',
        
        # Results
        'results/models',
        'results/figures',
        'results/reports',
        'results/benchmarks',
        
        # Configuration
        'config',
        
        # Logs
        'logs',
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("\n✓ All directories created!")


def create_init_files():
    """Create __init__.py files for Python packages"""
    
    packages = [
        'src',
        'src/data',
        'src/features',
        'src/models',
        'src/models/classical',
        'src/models/quantum',
        'src/explainability',
        'src/ranking',
        'src/active_learning',
        'src/visualization',
        'src/utils',
        'tests',
        'tests/test_data',
        'tests/test_features',
        'tests/test_models',
    ]
    
    print("\nCreating __init__.py files...")
    for package in packages:
        init_file = Path(package) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"  ✓ {init_file}")
    
    print("\n✓ All __init__.py files created!")


def create_gitkeep_files():
    """Create .gitkeep files for empty directories"""
    
    directories = [
        'data/raw',
        'data/processed',
        'data/interim',
        'results/models',
        'results/figures',
        'results/reports',
        'results/benchmarks',
        'logs',
    ]
    
    print("\nCreating .gitkeep files...")
    for directory in directories:
        gitkeep = Path(directory) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
            print(f"  ✓ {gitkeep}")
    
    print("\n✓ All .gitkeep files created!")


def create_gitignore():
    """Create .gitignore file"""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
qrucible_env*/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data - Keep structure but ignore large files
data/raw/*.csv
data/raw/*.sdf
data/raw/*.xlsx
data/processed/*.csv
data/interim/*.csv
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/interim/.gitkeep

# Models
results/models/*.pkl
results/models/*.h5
results/models/*.pt
results/models/*.joblib

# Figures (optional - uncomment to ignore)
# results/figures/*.png
# results/figures/*.jpg
# results/figures/*.pdf

# Logs
logs/*.log
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Quantum
qiskit.cfg
"""
    
    gitignore_file = Path('.gitignore')
    if not gitignore_file.exists():
        with open(gitignore_file, 'w') as f:
            f.write(gitignore_content)
        print("\n✓ .gitignore created!")
    else:
        print("\n⚠ .gitignore already exists, skipping...")


def create_readme():
    """Create basic README.md"""
    
    readme_content = """# Qrucible

**Hybrid Quantum-Classical Machine Learning Framework for Drug Discovery**

## Overview

Qrucible combines classical machine learning with quantum computing to accelerate molecular candidate screening and property optimization in drug discovery.

## Quick Start

1. **Initialize project**:
   ```bash
   python init_project.py
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download data**:
   ```bash
   python scripts/download_data.py --source chembl --target "EGFR kinase" --limit 30000
   ```

4. **Run preprocessing**:
   ```bash
   python scripts/run_preprocessing.py
   ```

## Project Structure

- `src/` - Source code modules
- `notebooks/` - Jupyter notebooks for exploration
- `scripts/` - Executable scripts
- `data/` - Dataset storage
- `results/` - Models, figures, and reports
- `config/` - Configuration files

## Current Phase

**Phase 1**: Classical ML Pipeline Development

## License

MIT License
"""
    
    readme_file = Path('README.md')
    if not readme_file.exists():
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        print("✓ README.md created!")
    else:
        print("⚠ README.md already exists, skipping...")


def main():
    """Main initialization function"""
    
    print("\n" + "=" * 70)
    print("QRUCIBLE - PROJECT INITIALIZATION")
    print("=" * 70 + "\n")
    
    # Create all directories
    create_directory_structure()
    
    # Create Python package files
    create_init_files()
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Create .gitignore
    create_gitignore()
    
    # Create README
    create_readme()
    
    print("\n" + "=" * 70)
    print("✅ PROJECT INITIALIZATION COMPLETE!")
    print("=" * 70)
    
    print("\n📝 Next Steps:")
    print("  1. Review/edit config/config.yaml")
    print("  2. Run: python scripts/download_data.py --source chembl --target 'EGFR kinase' --limit 30000")
    print("  3. Or run: python scripts/run_preprocessing.py")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()