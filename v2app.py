"""
Qrucible - Complete Drug Discovery Pipeline
With scientist inputs, candidate selection, and quantum readiness
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Draw, AllChem
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Qrucible - Drug Discovery AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .scientist-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .candidate-card {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .quantum-ready {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_data' not in st.session_state:
    st.session_state.session_data = {
        'target_defined': False,
        'candidates_selected': False,
        'top_candidates': None
    }

def calculate_descriptors(smiles):
    """Calculate comprehensive molecular descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Crippen.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_h_donors': Descriptors.NumHDonors(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'n_heavy_atoms': mol.GetNumHeavyAtoms(),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
    }
    
    # Lipinski violations
    violations = 0
    if descriptors['mol_weight'] > 500: violations += 1
    if descriptors['logp'] > 5: violations += 1
    if descriptors['num_h_donors'] > 5: violations += 1
    if descriptors['num_h_acceptors'] > 10: violations += 1
    descriptors['lipinski_violations'] = violations
    
    # Synthetic Accessibility (simplified approximation)
    descriptors['sa_score'] = min(10, descriptors['num_rotatable_bonds'] * 0.5 + 
                                   descriptors['num_aromatic_rings'] * 0.3 + 2)
    
    return descriptors

def check_constraints(descriptors, constraints):
    """Check if molecule meets scientist's constraints"""
    passes = True
    violations = []
    
    # Molecular weight
    if not (constraints['mw_min'] <= descriptors['mol_weight'] <= constraints['mw_max']):
        passes = False
        violations.append(f"MW: {descriptors['mol_weight']:.1f} (required: {constraints['mw_min']}-{constraints['mw_max']})")
    
    # LogP
    if descriptors['logp'] > constraints['logp_max']:
        passes = False
        violations.append(f"LogP: {descriptors['logp']:.2f} (required: ≤{constraints['logp_max']})")
    
    # H-bond donors
    if descriptors['num_h_donors'] > constraints['h_donors_max']:
        passes = False
        violations.append(f"H-donors: {descriptors['num_h_donors']} (required: ≤{constraints['h_donors_max']})")
    
    # H-bond acceptors
    if descriptors['num_h_acceptors'] > constraints['h_acceptors_max']:
        passes = False
        violations.append(f"H-acceptors: {descriptors['num_h_acceptors']} (required: ≤{constraints['h_acceptors_max']})")
    
    # SA score
    if descriptors['sa_score'] > constraints['sa_score_max']:
        passes = False
        violations.append(f"SA score: {descriptors['sa_score']:.1f} (required: ≤{constraints['sa_score_max']})")
    
    return passes, violations

def predict_activity_with_uncertainty(descriptors):
    """Predict activity with uncertainty estimation"""
    # Simplified prediction for demo
    # In production, load actual model
    base_pred = 7.5 - 0.001 * descriptors['mol_weight'] + 0.2 * descriptors['logp']
    
    # Simulate uncertainty based on descriptor values
    uncertainty = 0.3 + 0.05 * abs(descriptors['logp'] - 2.5)
    
    return base_pred, uncertainty

def calculate_multi_objective_score(pred_activity, descriptors, constraints, weights):
    """Calculate multi-objective optimization score"""
    scores = {}
    
    # Activity score (normalized)
    scores['activity'] = (pred_activity - 5.0) / (10.0 - 5.0)  # Normalize to 0-1
    
    # Drug-likeness score
    scores['druglike'] = (5 - descriptors['lipinski_violations']) / 5
    
    # Synthetic accessibility score (inverted - lower is better)
    scores['synthetic'] = (10 - descriptors['sa_score']) / 10
    
    # Constraint satisfaction score
    passes, violations = check_constraints(descriptors, constraints)
    scores['constraints'] = 1.0 if passes else 0.5
    
    # Weighted total
    total_score = (
        weights['activity'] * scores['activity'] +
        weights['druglike'] * scores['druglike'] +
        weights['synthetic'] * scores['synthetic'] +
        weights['constraints'] * scores['constraints']
    )
    
    return total_score, scores

def mol_to_image(smiles, size=(300, 300)):
    """Convert SMILES to image"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

def main():
    # Header
    st.markdown('<p class="main-header">🔬 Qrucible</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid Quantum-Classical Drug Discovery Platform</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Session Configuration
    with st.sidebar:
        st.header("🎯 Session Configuration")
        
        with st.expander("📋 Define Target & Objective", expanded=not st.session_state.session_data['target_defined']):
            # Target specification
            st.subheader("1. Target Specification")
            target_type = st.selectbox(
                "Target Type",
                ["Protein Kinase", "GPCR", "Ion Channel", "Enzyme", "Other"]
            )
            
            pdb_id = st.text_input(
                "PDB ID or File",
                value="2SRC",
                help="Enter PDB ID (e.g., 2SRC) or upload file"
            )
            
            binding_pocket = st.text_area(
                "Binding Pocket Coordinates (optional)",
                placeholder="x: 10.5, y: -3.2, z: 15.8",
                height=60
            )
            
            catalytic_residues = st.text_input(
                "Catalytic Residues (optional)",
                placeholder="e.g., ASP166, LYS168, GLU144"
            )
            
            # Session objective
            st.subheader("2. Session Objective")
            objective = st.text_area(
                "Research Objective",
                placeholder="e.g., Identify potent and selective EGFR inhibitors with improved oral bioavailability",
                height=80
            )
            
            if st.button("✓ Confirm Target & Objective"):
                st.session_state.session_data['target_defined'] = True
                st.session_state.session_data['target_type'] = target_type
                st.session_state.session_data['pdb_id'] = pdb_id
                st.session_state.session_data['objective'] = objective
                st.success("Target and objective saved!")
        
        # Constraints
        with st.expander("⚙️ Set Constraints", expanded=st.session_state.session_data['target_defined']):
            st.subheader("3. Molecular Constraints")
            
            col1, col2 = st.columns(2)
            with col1:
                mw_min = st.number_input("MW Min (Da)", value=200, step=50)
                mw_max = st.number_input("MW Max (Da)", value=500, step=50)
            
            with col2:
                logp_max = st.number_input("LogP Max", value=5.0, step=0.5)
                h_donors_max = st.number_input("H-Donors Max", value=5, step=1)
            
            h_acceptors_max = st.number_input("H-Acceptors Max", value=10, step=1)
            
            solubility_req = st.select_slider(
                "Solubility Requirement",
                options=["Low", "Medium", "High"],
                value="Medium"
            )
            
            toxicity_check = st.checkbox("Enable Toxicity Screening", value=True)
            
            sa_score_max = st.slider(
                "Max SA Score (Synthetic Accessibility)",
                min_value=1.0, max_value=10.0, value=6.0, step=0.5
            )
            
            constraints = {
                'mw_min': mw_min,
                'mw_max': mw_max,
                'logp_max': logp_max,
                'h_donors_max': h_donors_max,
                'h_acceptors_max': h_acceptors_max,
                'sa_score_max': sa_score_max,
                'solubility': solubility_req,
                'toxicity_check': toxicity_check
            }
            
            st.session_state.session_data['constraints'] = constraints
        
        # Optimization weights
        with st.expander("⚖️ Optimization Weights"):
            st.subheader("4. Multi-Objective Weights")
            activity_weight = st.slider("Activity", 0.0, 1.0, 0.4, 0.05)
            druglike_weight = st.slider("Drug-likeness", 0.0, 1.0, 0.3, 0.05)
            synthetic_weight = st.slider("Synthetic Ease", 0.0, 1.0, 0.2, 0.05)
            constraint_weight = st.slider("Constraint Match", 0.0, 1.0, 0.1, 0.05)
            
            total = activity_weight + druglike_weight + synthetic_weight + constraint_weight
            if abs(total - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum to {total:.2f}, not 1.0")
            
            weights = {
                'activity': activity_weight,
                'druglike': druglike_weight,
                'synthetic': synthetic_weight,
                'constraints': constraint_weight
            }
            
            st.session_state.session_data['weights'] = weights
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Screen Candidates", 
        "🏆 Top 10 Selection",
        "⚛️ Quantum Optimization"
    ])
    
    # Tab 1: Screen Candidates
    with tab1:
        st.header("Molecular Candidate Screening")
        
        if not st.session_state.session_data.get('target_defined'):
            st.warning("⚠️ Please define target and objective in the sidebar first")
        else:
            st.success(f"🎯 Target: {st.session_state.session_data.get('target_type')} ({st.session_state.session_data.get('pdb_id')})")
            st.info(f"📋 Objective: {st.session_state.session_data.get('objective')}")
            
            # Input method selection
            input_method = st.radio(
                "Input Method",
                ["Batch Upload (CSV)", "Database Search", "Manual Entry"],
                horizontal=True
            )
            
            candidates_df = None
            
            if input_method == "Batch Upload (CSV)":
                uploaded_file = st.file_uploader(
                    "Upload CSV with SMILES column",
                    type=['csv'],
                    help="CSV must contain 'SMILES' or 'smiles' column"
                )
                
                if uploaded_file:
                    candidates_df = pd.read_csv(uploaded_file)
                    smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
                    st.write(f"📊 Loaded {len(candidates_df)} candidates")
            
            elif input_method == "Manual Entry":
                smiles_input = st.text_area(
                    "Enter SMILES (one per line)",
                    height=200,
                    placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
                )
                
                if smiles_input:
                    smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                    candidates_df = pd.DataFrame({'smiles': smiles_list})
            
            # Process candidates
            if candidates_df is not None and st.button("🚀 Screen Candidates", type="primary"):
                st.subheader("📊 Screening Results")
                
                smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
                constraints = st.session_state.session_data.get('constraints', {})
                weights = st.session_state.session_data.get('weights', {
                    'activity': 0.4, 'druglike': 0.3, 'synthetic': 0.2, 'constraints': 0.1
                })
                
                progress_bar = st.progress(0)
                results = []
                
                for idx, smiles in enumerate(candidates_df[smiles_col]):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        desc = calculate_descriptors(smiles)
                        if desc:
                            pred_activity, uncertainty = predict_activity_with_uncertainty(desc)
                            total_score, subscores = calculate_multi_objective_score(
                                pred_activity, desc, constraints, weights
                            )
                            passes, violations = check_constraints(desc, constraints)
                            
                            results.append({
                                'SMILES': smiles,
                                'Predicted_pIC50': pred_activity,
                                'Uncertainty': uncertainty,
                                'Total_Score': total_score,
                                'Activity_Score': subscores['activity'],
                                'DrugLike_Score': subscores['druglike'],
                                'Synthetic_Score': subscores['synthetic'],
                                'Passes_Constraints': passes,
                                'MW': desc['mol_weight'],
                                'LogP': desc['logp'],
                                'TPSA': desc['tpsa'],
                                'SA_Score': desc['sa_score'],
                                'Lipinski_Violations': desc['lipinski_violations']
                            })
                    
                    progress_bar.progress((idx + 1) / len(candidates_df))
                
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Total_Score', ascending=False)
                
                st.success(f"✓ Screened {len(results_df)} valid candidates")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Candidates", len(results_df))
                with col2:
                    passing = results_df['Passes_Constraints'].sum()
                    st.metric("Pass Constraints", f"{passing} ({passing/len(results_df)*100:.1f}%)")
                with col3:
                    st.metric("Mean pIC50", f"{results_df['Predicted_pIC50'].mean():.2f}")
                with col4:
                    st.metric("Mean Score", f"{results_df['Total_Score'].mean():.3f}")
                
                # Display results
                st.dataframe(
                    results_df.style.background_gradient(subset=['Total_Score'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Save to session
                st.session_state.session_data['all_candidates'] = results_df
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download All Results",
                    csv,
                    "screening_results.csv",
                    "text/csv"
                )
    
    # Tab 2: Top 10 Selection
    with tab2:
        st.header("🏆 Top 10 Candidate Selection")
        
        if 'all_candidates' not in st.session_state.session_data:
            st.info("👈 Please screen candidates in the 'Screen Candidates' tab first")
        else:
            results_df = st.session_state.session_data['all_candidates']
            
            # Select top 10
            top_10 = results_df.head(10).copy()
            
            st.success(f"Selected top 10 candidates from {len(results_df)} screened molecules")
            
            # Display top 10 with visualization
            for idx, row in top_10.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="candidate-card">
                        <h4>Candidate #{idx+1} | Score: {row['Total_Score']:.3f}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    with col1:
                        img = mol_to_image(row['SMILES'], size=(200, 200))
                        if img:
                            st.image(img, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Properties:**")
                        st.write(f"pIC50: {row['Predicted_pIC50']:.2f} ± {row['Uncertainty']:.2f}")
                        st.write(f"MW: {row['MW']:.1f} Da")
                        st.write(f"LogP: {row['LogP']:.2f}")
                        st.write(f"TPSA: {row['TPSA']:.1f}")
                        st.write(f"SA Score: {row['SA_Score']:.1f}")
                    
                    with col3:
                        st.markdown("**Scores:**")
                        st.progress(row['Activity_Score'], f"Activity: {row['Activity_Score']:.2f}")
                        st.progress(row['DrugLike_Score'], f"Drug-like: {row['DrugLike_Score']:.2f}")
                        st.progress(row['Synthetic_Score'], f"Synthetic: {row['Synthetic_Score']:.2f}")
                        
                        if row['Passes_Constraints']:
                            st.success("✓ Passes all constraints")
                        else:
                            st.warning("⚠ Some constraints not met")
            
            # Save for quantum
            if st.button("✓ Confirm Top 10 → Send to Quantum Optimization", type="primary"):
                st.session_state.session_data['top_candidates'] = top_10
                st.session_state.session_data['candidates_selected'] = True
                st.success("✓ Top 10 candidates confirmed and ready for quantum optimization!")
                st.balloons()
    
    # Tab 3: Quantum Optimization
    with tab3:
        st.header("⚛️ Quantum Optimization Module")
        
        if not st.session_state.session_data.get('candidates_selected'):
            st.info("👈 Please select top 10 candidates first")
        else:
            st.markdown("""
            <div class="quantum-ready">
                🎉 READY FOR QUANTUM OPTIMIZATION 🎉
            </div>
            """, unsafe_allow_html=True)
            
            top_10 = st.session_state.session_data['top_candidates']
            
            st.write(f"\n**{len(top_10)} candidates ready for quantum refinement:**")
            st.dataframe(top_10[['SMILES', 'Predicted_pIC50', 'Total_Score']])
            
            st.subheader("Quantum Optimization Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                quantum_backend = st.selectbox(
                    "Quantum Backend",
                    ["Simulator", "IBM Quantum (Real Hardware)"]
                )
                
                algorithm = st.selectbox(
                    "Algorithm",
                    ["VQE (Energy Calculation)", "QAOA (Optimization)", "Quantum Kernel"]
                )
            
            with col2:
                max_iterations = st.number_input("Max Iterations", value=1000, step=100)
                convergence_threshold = st.number_input(
                    "Convergence Threshold",
                    value=1e-6,
                    format="%.2e"
                )
            
            if st.button("🚀 Start Quantum Optimization", type="primary"):
                st.info("🔄 Quantum optimization will be implemented in Phase 2...")
                st.code("""
# Quantum optimization pseudo-code:
for candidate in top_10:
    # 1. Generate molecular Hamiltonian
    hamiltonian = generate_hamiltonian(candidate)
    
    # 2. Run VQE
    vqe_result = run_vqe(hamiltonian, backend=quantum_backend)
    
    # 3. Calculate refined energy
    refined_energy = vqe_result.eigenvalue
    
    # 4. Update prediction with quantum correction
    final_prediction = ml_prediction + quantum_correction
                """)
                
                st.success("Phase 2 quantum module will be activated after deployment!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Qrucible v1.0 | Phase 1: Classical ML ✓ | Phase 2: Quantum Integration 🔄</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()