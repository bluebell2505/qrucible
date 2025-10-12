# Qrucible

**Hybrid Quantum-Classical Machine Learning Framework for Molecular Candidate Optimization in Drug Discovery**

## Overview

Qrucible is a cutting-edge framework that combines classical machine learning with quantum computing to accelerate early-stage drug discovery. By leveraging quantum computation's parallelism alongside classical ML's scalability, the system enables faster and more accurate molecular candidate screening and property optimization.

## Features

- **Classical ML Screening**: Rapid screening using Random Forest and Gradient Boosting
- **Quantum Optimization**: VQE and QAOA for molecular energy calculations
- **Active Learning**: Intelligent feedback loop between quantum and classical models
- **Explainability**: SHAP and LIME for interpretable predictions
- **Multi-Objective Ranking**: Optimized candidate selection based on multiple criteria

## Project Structure

```
qrucible/
├── src/              # Source code
├── notebooks/        # Jupyter notebooks
├── data/            # Datasets (raw, processed, interim)
├── results/         # Models, figures, reports
├── scripts/         # Executable scripts
├── tests/           # Unit tests
└── config/          # Configuration files
```

## Quick Start

1. **Activate environment**:
   ```bash
   conda activate qrucible
   ```

2. **Download data**:
   ```bash
   python scripts/download_data.py
   ```

3. **Preprocess**:
   ```bash
   python src/data/preprocessor.py
   ```

4. **Train models**:
   ```bash
   python scripts/train_classical.py
   ```

## Development Phases

- **Phase 1**: Classical ML Pipeline (Current)
- **Phase 2**: Quantum Integration
- **Phase 3**: Active Learning & Advanced Features
- **Phase 4**: Validation & Scalability

## Requirements

- See `requirements.txt` for full list


**Hybrid Quantum-Classical Machine Learning Framework for Drug Discovery**

<!-- [![Phase 1](https://img.shields.io/badge/Phase%201-Complete-brightgreen)]()
[![Test R²](https://img.shields.io/badge/Test%20R²-0.64-blue)]()
[![Models](https://img.shields.io/badge/Models-4-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]() -->

<!-- ## 🎯 Project Overview

Qrucible combines classical machine learning with quantum computing to accelerate molecular candidate screening and property optimization in early-stage drug discovery. The framework leverages quantum computation's parallelism alongside classical ML's scalability for enhanced molecular discovery. -->

## 🏆 Key Achievements (Phase 1)

- ✅ **70% Performance Improvement** using molecular fingerprints
- ✅ **Test R² = 0.64** (exceeded target of 0.60)
- ✅ **9,224 compounds** processed from ChEMBL (EGFR kinase)
- ✅ **72.2% drug-like** molecules (Lipinski compliant)
- ✅ **4 trained models** with comprehensive benchmarking

## 📊 Model Performance

| Model | Test R² | RMSE | MAE | Pearson r |
|-------|---------|------|-----|-----------|
| RF (Descriptors) | 0.376 | 0.961 | 0.753 | 0.615 |
| XGBoost (Descriptors) | 0.387 | 0.952 | 0.750 | 0.623 |
| **RF (Fingerprints)** | **0.640** | **0.728** | **0.553** | **0.804** |
| **XGBoost (Fingerprints)** | **0.614** | **0.754** | **0.580** | **0.789** |
| **Ensemble** | **0.637** | **0.731** | **0.559** | **0.804** |

**Best Model:** Random Forest + Morgan Fingerprints

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
conda or venv
```

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd qrucible

# Create environment
conda create -n qrucible_env python=3.10 -y
conda activate qrucible_env

# Install dependencies
pip install -r requirements.txt
```

### Usage

**1. Download Data:**
```bash
python scripts/download_data.py --source chembl --target "EGFR kinase" --limit 30000
```

**2. Preprocess:**
```bash
python scripts/run_preprocessing.py
```

**3. Train Models:**
```bash
# Train with descriptors
python scripts/train_classical.py

# Train with fingerprints (better performance)
python scripts/train_with_fingerprints.py
```

**4. Generate Report:**
```bash
python scripts/generate_final_report.py
```

## 🏗️ Project Structure

```
qrucible/
├── src/                    # Source code
│   ├── data/              # Data loading & preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models (classical & quantum)
│   ├── explainability/    # SHAP, LIME
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
├── notebooks/             # Jupyter notebooks
├── data/                  # Datasets
├── results/               # Models, figures, reports
└── config/                # Configuration files
```

## 🔬 Features

### Phase 1 (Complete)
- ✅ Data acquisition from ChEMBL, BindingDB, ZINC
- ✅ Molecular preprocessing & standardization
- ✅ 2D molecular descriptors (RDKit)
- ✅ Morgan & MACCS fingerprints
- ✅ Random Forest & XGBoost models
- ✅ Ensemble learning
- ✅ Model explainability (feature importance)
- ✅ Drug-likeness filtering (Lipinski)

### Phase 2 (In Progress)
- 🔄 VQE for ground-state energy
- 🔄 QAOA for molecular optimization
- 🔄 Quantum kernels
- 🔄 Hybrid quantum-classical workflow
- 🔄 Active learning loop

### Phase 3 (Planned)
- ⏳ Federated quantum learning
- ⏳ Quantum generative models
- ⏳ Web deployment (Streamlit)

## 📈 Results

### Data Statistics
- **Source:** ChEMBL EGFR Kinase inhibitors
- **Downloaded:** 16,449 compounds
- **Cleaned:** 9,224 compounds
- **Activity Range:** pIC50 5.00 - 17.30
- **Mean Activity:** pIC50 7.20 ± 1.22

### Model Files
- `results/models/random_forest.pkl`
- `results/models/xgboost.pkl`
- `results/models/random_forest_fingerprints.pkl`
- `results/models/xgboost_fingerprints.pkl`

### Visualizations
- Prediction plots: `results/figures/`
- Model comparison: `results/figures/model_comparison.png`

## 🛠️ Tech Stack

**Classical ML:**
- scikit-learn
- XGBoost
- PyTorch (optional)

**Cheminformatics:**
- RDKit
- Morgan fingerprints
- MACCS keys

**Quantum (Phase 2):**
- Qiskit
- Qiskit Nature
- PennyLane

**Visualization:**
- Matplotlib
- Seaborn
- Plotly

## 📚 Documentation

- [Phase 1 Report](PHASE1_REPORT.md)
- [Setup Guide](GETTING_STARTED.md)
- [Notebook Guide](VSCode_Notebook_Setup.md)
- [Next Steps](NEXT_STEPS.md)

## 🎓 Key Insights

**Most Important Features:**
1. Molecular Weight (25%)
2. TPSA - Topological Polar Surface Area (21%)
3. LogP - Lipophilicity (20%)
4. Rotatable Bonds (9%)
5. Hydrogen Donors (9%)

**Fingerprints > Descriptors:**
- Fingerprints capture molecular structure better
- 70% improvement in prediction accuracy
- Essential for similarity-based predictions

## 🔮 Roadmap

- [x] Phase 1: Classical ML Pipeline (Complete)
- [ ] Phase 2: Quantum Integration (In Progress)
- [ ] Phase 3: Advanced Features & Deployment
- [ ] Phase 4: Validation & Scaling

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Open an issue
- Submit a pull request
- Contact: vdixit2505@gmail.com

## 📄 License

MIT License

## 🙏 Acknowledgments

- ChEMBL Database for molecular data
- RDKit for cheminformatics toolkit
- IBM Qiskit for quantum computing framework

## 📞 Support

For issues or questions:
1. Check documentation in `docs/`
2. Review `GETTING_STARTED.md`
3. Open an issue on GitHub

---

**Status:** Phase 1 Complete ✅ | Phase 2 In Progress 🔄

**Last Updated:** 2025-10-12

**Performance:** Test R² = 0.64 | Pearson r = 0.804

## License

MIT License

## Contact

For questions or collaboration, please open an issue.

