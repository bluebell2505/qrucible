# 🧪 Qrucible — Quantum‑Validated Molecular Screening Platform

**Qrucible** is a real‑world–aligned, decision‑support system for **early‑stage drug discovery**.  
It combines **classical machine learning (QSAR/SAR)** with **physics‑grounded quantum feasibility checks** to reduce false positives before wet‑lab synthesis.

> **Core idea:**  
> _Use ML to screen fast, then use quantum methods as a mandatory feasibility gate — not as a novelty._

---

## 🔍 Why Qrucible Exists:

In real pharmaceutical pipelines:

- ML/QSAR models generate **many false positives**
    
- Chemically implausible molecules survive virtual screening
    
- Wet‑lab synthesis and assays are **expensive and slow**
    
- Physics‑based validation is often skipped due to cost
    

**Qrucible solves this gap** by inserting a **quantum feasibility gate** between ML screening and wet‑lab experiments.

---

## 🎯 What Qrucible Does:

- Accepts **researcher‑provided molecular libraries**
    
- Performs **classical ML screening** for target properties
    
- Applies **medicinal chemistry rule filters**
    
- Uses **quantum algorithms to reject physically implausible candidates**
    
- Outputs a **short, synthesis‑ready molecule list**

---

## 🧠 High‑Level Pipeline (Version 4)

```
Researcher Molecules
        ↓
Classical ML Screening (QSAR / SAR)
        ↓
Chemical Feasibility Filters
        ↓
Quantum Feasibility Gate (YES / NO)
        ↓
Shortlist for Wet‑Lab Synthesis

```
---

## 🧬 Detailed Workflow

### **STEP 0 — Researcher Input**

- Existing molecules (hits, analogs, fragments)
    
- Target property objectives (e.g., pIC50)
    
- Optional target context
    

📌 _Matches real pharma workflows_

---

### **STEP 1 — Classical ML Screening**

- Molecular descriptors + fingerprints (RDKit)
    
- Models: Random Forest, XGBoost (optionally GNNs)
    
- Outputs:
    
    - Predicted activity
        
    - Prediction uncertainty
        
    - Ranking
        

📌 _High‑throughput, noisy by design — just like industry ML_

---

### **STEP 2 — Chemical Feasibility Filters**

- Lipinski / Veber rules
    
- Toxic or reactive substructures
    
- Extreme phys‑chem removal
    
- Simple synthetic accessibility heuristics
    

📌 _Medicinal chemistry sanity check_

---

### **STEP 3 — Quantum Feasibility Gate (Core Contribution)**

Quantum methods are used **only for decision‑critical validation**, not ranking.

**Quantum checks include:**

- Electronic stability
    
- Ground‑state energy consistency
    
- Conformational feasibility
    
- Detection of unphysical energy landscapes
    

**Decision logic:**

- ❌ Reject if physically implausible
    
- ✅ Pass if feasible
    

📌 _This step is non‑removable and justifies quantum usage_

---

### **STEP 4 — Final Output**

- 10–30 shortlisted molecules
    
- For each candidate:
    
    - ML score + uncertainty
        
    - Quantum pass/fail
        
    - Rejection rationale (if any)
        

📌 _Direct input to wet‑lab synthesis_

---

## ⚛️ Role of Quantum Computing:

- Physics‑based feasibility validation
    
- False‑positive rejection
    
- Pre‑synthesis decision making

---

## 🧰 Tech Stack

### **Classical ML**

- Python 3.10
    
- RDKit (conda‑forge)
    
- NumPy, Pandas
    
- Scikit‑learn
    
- XGBoost
    
- Joblib
    

### **Quantum**

- Qiskit / PennyLane
    
- Quantum simulators (NISQ‑compatible)
    
- VQE / QAOA (small systems only)
    

### **Data**

- ChEMBL (primary source)
    
- Curated target‑specific datasets
    
- Descriptor + fingerprint representations

---

## 🚀 Intended Use Case

Qrucible is designed for:

- **Hit‑to‑lead optimization**
    
- **SAR refinement**
    
- **Reducing wet‑lab failures**
    
- **Academic + industry‑aligned research**
    

It is **not** intended as a fully autonomous drug discovery system.

---

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
5. **Start Project**:
   ```shell
   streamlit run app.py
   ```
## Requirements

- See `requirements.txt` for full list

---

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

---

**Status:** Version 3 completed ✅ | Version 4 Phase 1 In Progress 🔄

%% **Last Updated:** 2025-10-12 %%

%% **Performance:** Test R² = 0.64 | Pearson r = 0.804 %%

---

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Open an issue
- Submit a pull request
- Contact: vdixit2505@gmail.com | rajhansdhruv@gmail.com

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


