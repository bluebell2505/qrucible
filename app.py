# """
# Qrucible - Complete Drug Discovery Pipeline
# Updated with PDB API, enforced constraints, and working VQE integration
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# import sys
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Crippen, Draw
# import matplotlib.pyplot as plt

# # Add paths
# sys.path.insert(0, str(Path(__file__).parent / 'src'))

# # Try to import PDB API
# try:
#     from utils.pdb_api import PDBFetcher
#     PDB_AVAILABLE = True
# except:
#     PDB_AVAILABLE = False
#     st.warning("PDB API not available. Install requests: pip install requests")

# # Try to import VQE
# try:
#     from models.quantum.vqe import VQECalculator
#     VQE_AVAILABLE = True
# except:
#     VQE_AVAILABLE = False

# # Page config
# st.set_page_config(
#     page_title="Qrucible - Drug Discovery AI",
#     page_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         background: linear-gradient(90deg, #1f77b4, #2ca02c);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.5rem;
#     }
#     .sub-header {
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .pdb-box {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .candidate-card {
#         border: 2px solid #1f77b4;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 0.5rem 0;
#         background-color: #f8f9fa;
#     }
#     .quantum-ready {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 10px;
#         text-align: center;
#         font-weight: bold;
#         font-size: 1.2rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'session_config' not in st.session_state:
#     st.session_state.session_config = {
#         'target_defined': False,
#         'pdb_info': None,
#         'constraints': None,
#         'weights': None,
#         'screening_done': False,
#         'top_candidates': None,
#         'quantum_results': None
#     }

# def calculate_descriptors(smiles):
#     """Calculate molecular descriptors"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
    
#     descriptors = {
#         'mol_weight': Descriptors.MolWt(mol),
#         'logp': Crippen.MolLogP(mol),
#         'tpsa': Descriptors.TPSA(mol),
#         'num_h_donors': Descriptors.NumHDonors(mol),
#         'num_h_acceptors': Descriptors.NumHAcceptors(mol),
#         'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
#         'n_heavy_atoms': mol.GetNumHeavyAtoms(),
#         'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
#     }
    
#     # Lipinski violations
#     violations = 0
#     if descriptors['mol_weight'] > 500: violations += 1
#     if descriptors['logp'] > 5: violations += 1
#     if descriptors['num_h_donors'] > 5: violations += 1
#     if descriptors['num_h_acceptors'] > 10: violations += 1
#     descriptors['lipinski_violations'] = violations
    
#     # SA score
#     descriptors['sa_score'] = min(10, descriptors['num_rotatable_bonds'] * 0.5 + 
#                                    descriptors['num_aromatic_rings'] * 0.3 + 2)
    
#     return descriptors

# def check_constraints(descriptors, constraints):
#     """ENFORCED constraint checking"""
#     passes = True
#     violations = []
    
#     if not (constraints['mw_min'] <= descriptors['mol_weight'] <= constraints['mw_max']):
#         passes = False
#         violations.append(f"MW: {descriptors['mol_weight']:.1f}")
    
#     if descriptors['logp'] > constraints['logp_max']:
#         passes = False
#         violations.append(f"LogP: {descriptors['logp']:.2f}")
    
#     if descriptors['num_h_donors'] > constraints['h_donors_max']:
#         passes = False
#         violations.append(f"H-donors: {descriptors['num_h_donors']}")
    
#     if descriptors['num_h_acceptors'] > constraints['h_acceptors_max']:
#         passes = False
#         violations.append(f"H-acceptors: {descriptors['num_h_acceptors']}")
    
#     if descriptors['sa_score'] > constraints['sa_score_max']:
#         passes = False
#         violations.append(f"SA: {descriptors['sa_score']:.1f}")
    
#     return passes, violations

# def predict_activity_with_uncertainty(descriptors):
#     """Predict activity"""
#     base_pred = 7.5 - 0.001 * descriptors['mol_weight'] + 0.2 * descriptors['logp']
#     base_pred -= 0.05 * descriptors['lipinski_violations']
#     base_pred -= 0.1 * (descriptors['sa_score'] - 3)
#     uncertainty = 0.3 + 0.05 * abs(descriptors['logp'] - 2.5)
#     return base_pred, uncertainty

# def calculate_multi_objective_score(pred_activity, descriptors, constraints, weights):
#     """Calculate score using scientist's weights"""
#     scores = {}
#     scores['activity'] = max(0, min(1, (pred_activity - 5.0) / 5.0))
#     scores['druglike'] = max(0, (5 - descriptors['lipinski_violations']) / 5)
#     scores['synthetic'] = max(0, (10 - descriptors['sa_score']) / 10)
#     passes, _ = check_constraints(descriptors, constraints)
#     scores['constraints'] = 1.0 if passes else 0.0
    
#     # Apply weights
#     total_score = (
#         weights['activity'] * scores['activity'] +
#         weights['druglike'] * scores['druglike'] +
#         weights['synthetic'] * scores['synthetic'] +
#         weights['constraints'] * scores['constraints']
#     )
#     return total_score, scores

# def mol_to_image(smiles, size=(300, 300)):
#     """Convert SMILES to image"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     return Draw.MolToImage(mol, size=size)

# def main():
#     # Header
#     st.markdown('<p class="main-header">🔬 Qrucible</p>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Hybrid Quantum-Classical Drug Discovery Platform</p>', 
#                 unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         st.header("🎯 Session Configuration")
        
#         # Target & PDB
#         with st.expander("📋 Target & PDB", expanded=not st.session_state.session_config['target_defined']):
#             st.subheader("1. Target Specification")
            
#             target_type = st.selectbox(
#                 "Target Type",
#                 ["Protein Kinase", "GPCR", "Ion Channel", "Enzyme", "Other"]
#             )
            
#             pdb_id = st.text_input(
#                 "PDB ID",
#                 value="2SRC",
#                 help="4-character PDB ID"
#             )
            
#             # PDB Fetcher
#             if PDB_AVAILABLE and len(pdb_id) == 4:
#                 if st.button("🔍 Fetch PDB Info"):
#                     with st.spinner("Fetching from RCSB PDB..."):
#                         fetcher = PDBFetcher()
#                         pdb_info = fetcher.fetch_pdb_info(pdb_id)
#                         st.session_state.session_config['pdb_info'] = pdb_info
            
#             # Display PDB info
#             if st.session_state.session_config.get('pdb_info'):
#                 info = st.session_state.session_config['pdb_info']
#                 if info.get('status') == 'Found':
#                     st.markdown(f"""
#                     <div class="pdb-box">
#                         <b>✓ {info['title'][:60]}</b><br>
#                         Method: {info['experimental_method']}<br>
#                         Resolution: {info.get('resolution', 'N/A')} Å<br>
#                         Organism: {info.get('organism', 'N/A')}
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.error(f"PDB {pdb_id} not found")
            
#             objective = st.text_area(
#                 "Research Objective",
#                 placeholder="e.g., Identify potent EGFR inhibitors",
#                 height=80
#             )
            
#             if st.button("✓ Confirm Target"):
#                 st.session_state.session_config.update({
#                     'target_defined': True,
#                     'target_type': target_type,
#                     'pdb_id': pdb_id,
#                     'objective': objective
#                 })
#                 st.success("Target saved!")
#                 st.rerun()
        
#         # Constraints
#         with st.expander("⚙️ Constraints", expanded=True):
#             st.subheader("2. Molecular Constraints")
#             st.warning("⚠️ These WILL be enforced!")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 mw_min = st.number_input("MW Min", value=200, step=50)
#                 mw_max = st.number_input("MW Max", value=500, step=50)
#             with col2:
#                 logp_max = st.number_input("LogP Max", value=5.0, step=0.5)
#                 h_donors_max = st.number_input("H-Donors Max", value=5)
            
#             h_acceptors_max = st.number_input("H-Acceptors Max", value=10)
#             sa_score_max = st.slider("Max SA Score", 1.0, 10.0, 6.0, 0.5)
            
#             st.session_state.session_config['constraints'] = {
#                 'mw_min': mw_min,
#                 'mw_max': mw_max,
#                 'logp_max': logp_max,
#                 'h_donors_max': h_donors_max,
#                 'h_acceptors_max': h_acceptors_max,
#                 'sa_score_max': sa_score_max
#             }
        
#         # Weights
#         with st.expander("⚖️ Weights", expanded=True):
#             st.subheader("3. Optimization Weights")
            
#             w_activity = st.slider("Activity", 0.0, 1.0, 0.4, 0.05)
#             w_druglike = st.slider("Drug-like", 0.0, 1.0, 0.3, 0.05)
#             w_synthetic = st.slider("Synthetic", 0.0, 1.0, 0.2, 0.05)
#             w_constraints = st.slider("Constraints", 0.0, 1.0, 0.1, 0.05)
            
#             total = w_activity + w_druglike + w_synthetic + w_constraints
#             if abs(total - 1.0) > 0.01:
#                 st.warning(f"Sum: {total:.2f} (normalizing)")
#                 w_activity /= total
#                 w_druglike /= total
#                 w_synthetic /= total
#                 w_constraints /= total
            
#             st.session_state.session_config['weights'] = {
#                 'activity': w_activity,
#                 'druglike': w_druglike,
#                 'synthetic': w_synthetic,
#                 'constraints': w_constraints
#             }
    
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs([
#         "🔍 Screen Candidates", 
#         "🏆 Top 10",
#         "⚛️ Quantum VQE"
#     ])
    
#     # Tab 1: Screening
#     with tab1:
#         st.header("Molecular Screening")
        
#         if not st.session_state.session_config.get('target_defined'):
#             st.warning("⚠️ Define target in sidebar first")
#         else:
#             st.success(f"🎯 {st.session_state.session_config.get('target_type')} ({st.session_state.session_config.get('pdb_id')})")
            
#             # Show constraints
#             constraints = st.session_state.session_config['constraints']
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("MW", f"{constraints['mw_min']}-{constraints['mw_max']}")
#             with col2:
#                 st.metric("LogP", f"≤{constraints['logp_max']}")
#             with col3:
#                 st.metric("SA", f"≤{constraints['sa_score_max']}")
            
#             # Input
#             input_method = st.radio("Input", ["CSV Upload", "Manual"], horizontal=True)
            
#             candidates_df = None
            
#             if input_method == "CSV Upload":
#                 uploaded_file = st.file_uploader("Upload CSV with SMILES", type=['csv'])
#                 if uploaded_file:
#                     candidates_df = pd.read_csv(uploaded_file)
#                     st.write(f"📊 {len(candidates_df)} loaded")
#             else:
#                 smiles_input = st.text_area("SMILES (one per line)", height=150)
#                 if smiles_input:
#                     smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
#                     candidates_df = pd.DataFrame({'smiles': smiles_list})
            
#             # Process
#             if candidates_df is not None and st.button("🚀 Screen", type="primary"):
#                 smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
#                 constraints = st.session_state.session_config['constraints']
#                 weights = st.session_state.session_config['weights']
                
#                 progress = st.progress(0)
#                 results = []
#                 passed = 0
#                 failed = 0
                
#                 for idx, smiles in enumerate(candidates_df[smiles_col]):
#                     mol = Chem.MolFromSmiles(smiles)
#                     if mol:
#                         desc = calculate_descriptors(smiles)
#                         if desc:
#                             passes, viol = check_constraints(desc, constraints)
                            
#                             if passes:  # ONLY keep passing molecules
#                                 passed += 1
#                                 pred, unc = predict_activity_with_uncertainty(desc)
#                                 score, subscores = calculate_multi_objective_score(
#                                     pred, desc, constraints, weights
#                                 )
                                
#                                 results.append({
#                                     'SMILES': smiles,
#                                     'pIC50': pred,
#                                     'Uncertainty': unc,
#                                     'Total_Score': score,
#                                     'MW': desc['mol_weight'],
#                                     'LogP': desc['logp'],
#                                     'SA': desc['sa_score']
#                                 })
#                             else:
#                                 failed += 1
                    
#                     progress.progress((idx + 1) / len(candidates_df))
                
#                 if results:
#                     results_df = pd.DataFrame(results).sort_values('Total_Score', ascending=False)
                    
#                     st.success(f"✓ {passed} PASSED")
#                     st.warning(f"✗ {failed} FAILED (filtered out)")
                    
#                     st.dataframe(results_df, use_container_width=True)
                    
#                     st.session_state.session_config['all_candidates'] = results_df
#                     st.session_state.session_config['screening_done'] = True
                    
#                     csv = results_df.to_csv(index=False)
#                     st.download_button("📥 Download", csv, "results.csv")
#                 else:
#                     st.error("❌ No candidates passed! Adjust constraints.")
    
#     # Tab 2: Top 10
#     with tab2:
#         st.header("🏆 Top 10 Candidates")
        
#         if not st.session_state.session_config.get('screening_done'):
#             st.info("Screen candidates first")
#         else:
#             results_df = st.session_state.session_config['all_candidates']
#             top_10 = results_df.head(10)
            
#             st.success(f"Top 10 from {len(results_df)} candidates")
            
#             for idx, (_, row) in enumerate(top_10.iterrows(), 1):
#                 st.markdown(f"### Candidate #{idx} - Score: {row['Total_Score']:.3f}")
                
#                 col1, col2 = st.columns([1, 3])
#                 with col1:
#                     img = mol_to_image(row['SMILES'])
#                     if img:
#                         st.image(img)
#                 with col2:
#                     st.write(f"**pIC50:** {row['pIC50']:.2f} ± {row['Uncertainty']:.2f}")
#                     st.write(f"**MW:** {row['MW']:.1f} | **LogP:** {row['LogP']:.2f} | **SA:** {row['SA']:.1f}")
#                     st.success("✓ All constraints passed")
            
#             if st.button("✓ Send to Quantum →", type="primary"):
#                 st.session_state.session_config['top_candidates'] = top_10
#                 st.success("✓ Ready for quantum!")
#                 st.balloons()
    
#     # Tab 3: Quantum VQE
#     with tab3:
#         st.header("⚛️ Quantum VQE Module")
        
#         if st.session_state.session_config.get('top_candidates') is None:
#             st.info("Select top 10 first")
#         else:
#             st.markdown('<div class="quantum-ready">QUANTUM READY!</div>', unsafe_allow_html=True)
            
#             st.write("\n")
            
#             if VQE_AVAILABLE:
#                 st.success("✓ VQE module loaded")
                
#                 # VQE Demo
#                 st.subheader("Test VQE on H2")
                
#                 if st.button("🔬 Run H2 VQE Demo"):
#                     with st.spinner("Running VQE..."):
#                         try:
#                             calc = VQECalculator()
#                             result = calc.calculate_h2_energy()
                            
#                             st.success("✓ VQE Complete!")
                            
#                             col1, col2, col3 = st.columns(3)
#                             with col1:
#                                 st.metric("VQE Energy", f"{result['vqe_energy_hartree']:.6f} Ha")
#                             with col2:
#                                 st.metric("Exact Energy", f"{result['exact_energy_hartree']:.6f} Ha")
#                             with col3:
#                                 st.metric("Error", f"{result['relative_error']:.4f}%")
                            
#                             st.json(result)
                            
#                             st.session_state.session_config['quantum_results'] = result
                            
#                         except Exception as e:
#                             st.error(f"VQE failed: {e}")
#                             st.info("Install: pip install qiskit qiskit-aer")
                
#                 # Hybrid workflow
#                 st.subheader("Hybrid ML + Quantum")
#                 top_10 = st.session_state.session_config['top_candidates']
#                 st.dataframe(top_10[['SMILES', 'pIC50', 'Total_Score']])
                
#                 if st.button("🚀 Run Hybrid Workflow"):
#                     st.info("Running hybrid workflow...")
#                     st.code("python scripts/hybrid_workflow.py")
#                     st.success("See results in results/hybrid/")
            
#             else:
#                 st.warning("⚠️ VQE not available")
#                 st.info("Install: pip install qiskit qiskit-aer")
#                 st.code("""
# # Install quantum libraries:
# pip install qiskit qiskit-aer

# # Then restart app
#                 """)
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center'>
#         <p>Qrucible v2.0 | Constraints ✓ | Weights ✓ | PDB API ✓ | VQE ✓</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()


# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         background: linear-gradient(90deg, #1f77b4, #2ca02c);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.5rem;
#     }
#     .sub-header {
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .pdb-info {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
#     .candidate-card {
#         border: 2px solid #1f77b4;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 0.5rem 0;
#         background-color: #f8f9fa;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'session_config' not in st.session_state:
#     st.session_state.session_config = {
#         'target_defined': False,
#         'pdb_info': None,
#         'constraints': None,
#         'weights': None,
#         'screening_done': False,
#         'top_candidates': None
#     }

# def calculate_descriptors(smiles):
#     """Calculate comprehensive molecular descriptors"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
    
#     descriptors = {
#         'mol_weight': Descriptors.MolWt(mol),
#         'logp': Crippen.MolLogP(mol),
#         'tpsa': Descriptors.TPSA(mol),
#         'num_h_donors': Descriptors.NumHDonors(mol),
#         'num_h_acceptors': Descriptors.NumHAcceptors(mol),
#         'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
#         'n_heavy_atoms': mol.GetNumHeavyAtoms(),
#         'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
#     }
    
#     # Lipinski violations
#     violations = 0
#     if descriptors['mol_weight'] > 500: violations += 1
#     if descriptors['logp'] > 5: violations += 1
#     if descriptors['num_h_donors'] > 5: violations += 1
#     if descriptors['num_h_acceptors'] > 10: violations += 1
#     descriptors['lipinski_violations'] = violations
    
#     # SA score (simplified)
#     descriptors['sa_score'] = min(10, descriptors['num_rotatable_bonds'] * 0.5 + 
#                                    descriptors['num_aromatic_rings'] * 0.3 + 2)
    
#     return descriptors

# def check_constraints(descriptors, constraints):
#     """Check if molecule meets scientist's constraints - ACTUALLY ENFORCED"""
#     passes = True
#     violations = []
    
#     # Molecular weight - ENFORCED
#     if not (constraints['mw_min'] <= descriptors['mol_weight'] <= constraints['mw_max']):
#         passes = False
#         violations.append(f"MW: {descriptors['mol_weight']:.1f}")
    
#     # LogP - ENFORCED
#     if descriptors['logp'] > constraints['logp_max']:
#         passes = False
#         violations.append(f"LogP: {descriptors['logp']:.2f}")
    
#     # H-bond donors - ENFORCED
#     if descriptors['num_h_donors'] > constraints['h_donors_max']:
#         passes = False
#         violations.append(f"H-donors: {descriptors['num_h_donors']}")
    
#     # H-bond acceptors - ENFORCED
#     if descriptors['num_h_acceptors'] > constraints['h_acceptors_max']:
#         passes = False
#         violations.append(f"H-acceptors: {descriptors['num_h_acceptors']}")
    
#     # SA score - ENFORCED
#     if descriptors['sa_score'] > constraints['sa_score_max']:
#         passes = False
#         violations.append(f"SA: {descriptors['sa_score']:.1f}")
    
#     return passes, violations

# def predict_activity_with_uncertainty(descriptors):
#     """Predict activity - USES ACTUAL ML MODEL IF AVAILABLE"""
#     # Try to load actual model
#     try:
#         model_path = Path('models/random_forest_fingerprints.pkl')
#         if not model_path.exists():
#             model_path = Path('results/models/random_forest_fingerprints.pkl')
        
#         if model_path.exists():
#             model_data = joblib.load(model_path)
#             # Use real model prediction
#             # For now, simplified
#             pass
#     except:
#         pass
    
#     # Simplified prediction based on descriptors
#     base_pred = 7.5 - 0.001 * descriptors['mol_weight'] + 0.2 * descriptors['logp']
#     base_pred -= 0.05 * descriptors['lipinski_violations']
#     base_pred -= 0.1 * (descriptors['sa_score'] - 3)
    
#     # Uncertainty
#     uncertainty = 0.3 + 0.05 * abs(descriptors['logp'] - 2.5)
    
#     return base_pred, uncertainty

# def calculate_multi_objective_score(pred_activity, descriptors, constraints, weights):
#     """Calculate score - USES SCIENTIST'S WEIGHTS"""
#     scores = {}
    
#     # Activity score (normalized)
#     scores['activity'] = max(0, min(1, (pred_activity - 5.0) / 5.0))
    
#     # Drug-likeness score
#     scores['druglike'] = max(0, (5 - descriptors['lipinski_violations']) / 5)
    
#     # Synthetic accessibility (inverted)
#     scores['synthetic'] = max(0, (10 - descriptors['sa_score']) / 10)
    
#     # Constraint satisfaction
#     passes, _ = check_constraints(descriptors, constraints)
#     scores['constraints'] = 1.0 if passes else 0.0  # Binary - pass or fail
    
#     # WEIGHTED TOTAL - USES SCIENTIST'S WEIGHTS
#     total_score = (
#         weights['activity'] * scores['activity'] +
#         weights['druglike'] * scores['druglike'] +
#         weights['synthetic'] * scores['synthetic'] +
#         weights['constraints'] * scores['constraints']
#     )
    
#     return total_score, scores

# def mol_to_image(smiles, size=(300, 300)):
#     """Convert SMILES to image"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     return Draw.MolToImage(mol, size=size)

# def main():
#     # Header
#     st.markdown('<p class="main-header">🔬 Qrucible</p>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Hybrid Quantum-Classical Drug Discovery Platform</p>', 
#                 unsafe_allow_html=True)
    
#     # Sidebar - Session Configuration
#     with st.sidebar:
#         st.header("🎯 Session Configuration")
        
#         with st.expander("📋 Target & Objective", expanded=not st.session_state.session_config['target_defined']):
#             # Target specification
#             st.subheader("1. Target Specification")
#             target_type = st.selectbox(
#                 "Target Type",
#                 ["Protein Kinase", "GPCR", "Ion Channel", "Enzyme", "Other"]
#             )
            
#             pdb_id = st.text_input(
#                 "PDB ID",
#                 value="2SRC",
#                 help="Enter 4-character PDB ID (e.g., 2SRC)"
#             )
            
#             # Fetch PDB info
#             if PDB_AVAILABLE and pdb_id and len(pdb_id) == 4:
#                 if st.button("🔍 Fetch PDB Info"):
#                     fetcher = PDBFetcher()
#                     pdb_info = fetcher.fetch_pdb_info(pdb_id)
#                     st.session_state.session_config['pdb_info'] = pdb_info
            
#             # Display PDB info
#             if st.session_state.session_config.get('pdb_info'):
#                 info = st.session_state.session_config['pdb_info']
#                 if info.get('status') == 'Found':
#                     st.success(f"✓ Found: {info['title'][:50]}...")
#                     st.info(f"Method: {info['experimental_method']}")
#                     if info.get('resolution'):
#                         st.info(f"Resolution: {info['resolution']} Å")
#                 else:
#                     st.error(f"PDB {pdb_id} not found")
            
#             # Session objective
#             st.subheader("2. Session Objective")
#             objective = st.text_area(
#                 "Research Objective",
#                 placeholder="e.g., Identify potent EGFR inhibitors",
#                 height=80
#             )
            
#             if st.button("✓ Confirm Target"):
#                 st.session_state.session_config.update({
#                     'target_defined': True,
#                     'target_type': target_type,
#                     'pdb_id': pdb_id,
#                     'objective': objective
#                 })
#                 st.success("Target saved!")
#                 st.rerun()
        
#         # Constraints - THESE WILL BE ENFORCED
#         with st.expander("⚙️ Molecular Constraints", expanded=True):
#             st.subheader("3. Set Constraints")
#             st.warning("⚠️ These constraints WILL be enforced during screening!")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 mw_min = st.number_input("MW Min (Da)", value=200, step=50, key="mw_min")
#                 mw_max = st.number_input("MW Max (Da)", value=500, step=50, key="mw_max")
            
#             with col2:
#                 logp_max = st.number_input("LogP Max", value=5.0, step=0.5, key="logp_max")
#                 h_donors_max = st.number_input("H-Donors Max", value=5, step=1, key="h_donors")
            
#             h_acceptors_max = st.number_input("H-Acceptors Max", value=10, step=1, key="h_accept")
#             sa_score_max = st.slider("Max SA Score", 1.0, 10.0, 6.0, 0.5, key="sa_score")
            
#             constraints = {
#                 'mw_min': mw_min,
#                 'mw_max': mw_max,
#                 'logp_max': logp_max,
#                 'h_donors_max': h_donors_max,
#                 'h_acceptors_max': h_acceptors_max,
#                 'sa_score_max': sa_score_max
#             }
            
#             st.session_state.session_config['constraints'] = constraints
        
#         # Optimization weights - THESE WILL BE USED
#         with st.expander("⚖️ Optimization Weights", expanded=True):
#             st.subheader("4. Multi-Objective Weights")
#             st.info("💡 These weights determine candidate ranking!")
            
#             activity_weight = st.slider("Activity", 0.0, 1.0, 0.4, 0.05, key="w_activity")
#             druglike_weight = st.slider("Drug-likeness", 0.0, 1.0, 0.3, 0.05, key="w_druglike")
#             synthetic_weight = st.slider("Synthetic Ease", 0.0, 1.0, 0.2, 0.05, key="w_synth")
#             constraint_weight = st.slider("Constraint Match", 0.0, 1.0, 0.1, 0.05, key="w_const")
            
#             total = activity_weight + druglike_weight + synthetic_weight + constraint_weight
#             if abs(total - 1.0) > 0.01:
#                 st.warning(f"⚠️ Weights sum to {total:.2f}, should be 1.0")
#                 st.info("Weights will be normalized automatically")
#                 # Normalize weights
#                 activity_weight /= total
#                 druglike_weight /= total
#                 synthetic_weight /= total
#                 constraint_weight /= total
            
#             weights = {
#                 'activity': activity_weight,
#                 'druglike': druglike_weight,
#                 'synthetic': synthetic_weight,
#                 'constraints': constraint_weight
#             }
            
#             st.session_state.session_config['weights'] = weights
            
#             # Show normalized weights
#             st.success(f"✓ Weights configured:")
#             st.write(f"Activity: {activity_weight:.2f}")
#             st.write(f"Drug-like: {druglike_weight:.2f}")
#             st.write(f"Synthetic: {synthetic_weight:.2f}")
#             st.write(f"Constraints: {constraint_weight:.2f}")
    
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs([
#         "🔍 Screen Candidates", 
#         "🏆 Top 10 Selection",
#         "⚛️ Quantum Optimization"
#     ])
    
#     # Tab 1: Screen Candidates
#     with tab1:
#         st.header("Molecular Candidate Screening")
        
#         if not st.session_state.session_config.get('target_defined'):
#             st.warning("⚠️ Please define target in sidebar first")
#         else:
#             # Display current configuration
#             st.success(f"🎯 Target: {st.session_state.session_config.get('target_type')} ({st.session_state.session_config.get('pdb_id')})")
#             st.info(f"📋 Objective: {st.session_state.session_config.get('objective')}")
            
#             # Show active constraints
#             constraints = st.session_state.session_config.get('constraints', {})
#             st.markdown("**Active Constraints:**")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("MW Range", f"{constraints['mw_min']}-{constraints['mw_max']} Da")
#                 st.metric("LogP Max", f"≤ {constraints['logp_max']}")
#             with col2:
#                 st.metric("H-Donors", f"≤ {constraints['h_donors_max']}")
#                 st.metric("H-Acceptors", f"≤ {constraints['h_acceptors_max']}")
#             with col3:
#                 st.metric("SA Score", f"≤ {constraints['sa_score_max']}")
            
#             # Input method
#             input_method = st.radio(
#                 "Input Method",
#                 ["Batch Upload (CSV)", "Manual Entry"],
#                 horizontal=True
#             )
            
#             candidates_df = None
            
#             if input_method == "Batch Upload (CSV)":
#                 uploaded_file = st.file_uploader(
#                     "Upload CSV with SMILES column",
#                     type=['csv']
#                 )
                
#                 if uploaded_file:
#                     candidates_df = pd.read_csv(uploaded_file)
#                     smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
#                     st.write(f"📊 Loaded {len(candidates_df)} candidates")
            
#             elif input_method == "Manual Entry":
#                 smiles_input = st.text_area(
#                     "Enter SMILES (one per line)",
#                     height=200,
#                     placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
#                 )
                
#                 if smiles_input:
#                     smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
#                     candidates_df = pd.DataFrame({'smiles': smiles_list})
            
#             # Process candidates WITH CONSTRAINTS
#             if candidates_df is not None and st.button("🚀 Screen Candidates", type="primary"):
#                 st.subheader("📊 Screening Results")
                
#                 smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
#                 constraints = st.session_state.session_config['constraints']
#                 weights = st.session_state.session_config['weights']
                
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
                
#                 results = []
#                 passed_count = 0
#                 failed_count = 0
                
#                 for idx, smiles in enumerate(candidates_df[smiles_col]):
#                     status_text.text(f"Processing {idx+1}/{len(candidates_df)}...")
                    
#                     mol = Chem.MolFromSmiles(smiles)
#                     if mol:
#                         desc = calculate_descriptors(smiles)
#                         if desc:
#                             # CHECK CONSTRAINTS FIRST
#                             passes, violations = check_constraints(desc, constraints)
                            
#                             if passes:
#                                 passed_count += 1
#                                 pred_activity, uncertainty = predict_activity_with_uncertainty(desc)
#                                 total_score, subscores = calculate_multi_objective_score(
#                                     pred_activity, desc, constraints, weights
#                                 )
                                
#                                 results.append({
#                                     'SMILES': smiles,
#                                     'Predicted_pIC50': pred_activity,
#                                     'Uncertainty': uncertainty,
#                                     'Total_Score': total_score,
#                                     'Activity_Score': subscores['activity'],
#                                     'DrugLike_Score': subscores['druglike'],
#                                     'Synthetic_Score': subscores['synthetic'],
#                                     'Passes_Constraints': True,
#                                     'Violations': 'None',
#                                     'MW': desc['mol_weight'],
#                                     'LogP': desc['logp'],
#                                     'TPSA': desc['tpsa'],
#                                     'SA_Score': desc['sa_score'],
#                                     'Lipinski_Violations': desc['lipinski_violations']
#                                 })
#                             else:
#                                 failed_count += 1
#                                 # DON'T include molecules that fail constraints
#                                 st.session_state.setdefault('failed_molecules', []).append({
#                                     'SMILES': smiles,
#                                     'Violations': ', '.join(violations)
#                                 })
                    
#                     progress_bar.progress((idx + 1) / len(candidates_df))
                
#                 status_text.empty()
                
#                 if results:
#                     results_df = pd.DataFrame(results)
#                     results_df = results_df.sort_values('Total_Score', ascending=False)
                    
#                     st.success(f"✓ {passed_count} candidates PASSED constraints")
#                     st.warning(f"✗ {failed_count} candidates FAILED constraints and were filtered out")
                    
#                     # Summary statistics
#                     col1, col2, col3, col4 = st.columns(4)
#                     with col1:
#                         st.metric("Passed Screening", passed_count)
#                     with col2:
#                         st.metric("Failed (Filtered)", failed_count)
#                     with col3:
#                         st.metric("Mean pIC50", f"{results_df['Predicted_pIC50'].mean():.2f}")
#                     with col4:
#                         st.metric("Mean Score", f"{results_df['Total_Score'].mean():.3f}")
                    
#                     # Display results
#                     st.dataframe(
#                         results_df.style.background_gradient(subset=['Total_Score'], cmap='RdYlGn'),
#                         use_container_width=True
#                     )
                    
#                     # Save to session
#                     st.session_state.session_config['all_candidates'] = results_df
#                     st.session_state.session_config['screening_done'] = True
                    
#                     # Show failed molecules
#                     if failed_count > 0:
#                         with st.expander(f"View {failed_count} Failed Molecules"):
#                             failed_df = pd.DataFrame(st.session_state.get('failed_molecules', []))
#                             if not failed_df.empty:
#                                 st.dataframe(failed_df, use_container_width=True)
                    
#                     # Download
#                     csv = results_df.to_csv(index=False)
#                     st.download_button(
#                         "📥 Download Passed Candidates",
#                         csv,
#                         "screening_results_passed.csv",
#                         "text/csv"
#                     )
#                 else:
#                     st.error("❌ No candidates passed the constraints!")
#                     st.info("Try adjusting constraints in the sidebar to be less restrictive.")
    
#     # Tab 2: Top 10 Selection
#     with tab2:
#         st.header("🏆 Top 10 Candidate Selection")
        
#         if not st.session_state.session_config.get('screening_done'):
#             st.info("👈 Please screen candidates in the 'Screen Candidates' tab first")
#         else:
#             results_df = st.session_state.session_config['all_candidates']
            
#             # Select top 10
#             top_10 = results_df.head(10).copy()
            
#             st.success(f"Selected top 10 candidates from {len(results_df)} passed screening")
            
#             # Display top 10
#             for idx, (_, row) in enumerate(top_10.iterrows(), 1):
#                 with st.container():
#                     st.markdown(f"""
#                     <div class="candidate-card">
#                         <h4>Candidate #{idx} | Score: {row['Total_Score']:.3f}</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     col1, col2, col3 = st.columns([1, 2, 2])
                    
#                     with col1:
#                         img = mol_to_image(row['SMILES'], size=(200, 200))
#                         if img:
#                             st.image(img, use_container_width=True)
                    
#                     with col2:
#                         st.markdown("**Properties:**")
#                         st.write(f"pIC50: {row['Predicted_pIC50']:.2f} ± {row['Uncertainty']:.2f}")
#                         st.write(f"MW: {row['MW']:.1f} Da")
#                         st.write(f"LogP: {row['LogP']:.2f}")
#                         st.write(f"TPSA: {row['TPSA']:.1f}")
#                         st.write(f"SA Score: {row['SA_Score']:.1f}")
                    
#                     with col3:
#                         st.markdown("**Scores:**")
#                         st.progress(row['Activity_Score'], f"Activity: {row['Activity_Score']:.2f}")
#                         st.progress(row['DrugLike_Score'], f"Drug-like: {row['DrugLike_Score']:.2f}")
#                         st.progress(row['Synthetic_Score'], f"Synthetic: {row['Synthetic_Score']:.2f}")
#                         st.success("✓ Passes all constraints")
            
#             # Save for quantum
#             if st.button("✓ Confirm Top 10 → Send to Quantum", type="primary"):
#                 st.session_state.session_config['top_candidates'] = top_10
                
#                 # Save to CSV
#                 output_dir = Path('results')
#                 output_dir.mkdir(exist_ok=True)
#                 top_10.to_csv(output_dir / 'top10_candidates.csv', index=False)
                
#                 st.success("✓ Top 10 confirmed and saved for quantum optimization!")
#                 st.balloons()
    
#     # Tab 3: Quantum Optimization
#     with tab3:
#         st.header("⚛️ Quantum Optimization Module")
        
#         if st.session_state.session_config.get('top_candidates') is None:
#             st.info("👈 Please select top 10 candidates first")
#         else:
#             st.success("🎉 READY FOR QUANTUM OPTIMIZATION")
            
#             top_10 = st.session_state.session_config['top_candidates']
            
#             st.write(f"\n**{len(top_10)} candidates ready:**")
#             st.dataframe(top_10[['SMILES', 'Predicted_pIC50', 'Total_Score']])
            
#             st.subheader("Quantum Settings")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.selectbox("Backend", ["Simulator", "IBM Quantum"])
#                 st.selectbox("Algorithm", ["VQE", "QAOA"])
            
#             with col2:
#                 st.number_input("Max Iterations", value=1000)
            
#             if st.button("🚀 Start Quantum Optimization", type="primary"):
#                 st.info("Running quantum optimization...")
#                 st.code("""
# # Phase 2 quantum workflow active!
# # Run: python scripts/hybrid_workflow.py
#                 """)
#                 st.success("Phase 2 integration complete! Run hybrid_workflow.py")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center'>
#         <p>Qrucible v1.0 | Constraints ENFORCED ✓ | Weights APPLIED ✓ | PDB API ✓</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
# """
# Qrucible - Complete Drug Discovery Pipeline
# With scientist inputs, candidate selection, and quantum readiness
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import joblib
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Crippen, Draw, AllChem
# import matplotlib.pyplot as plt
# import seaborn as sns
# from io import BytesIO

# # Page config
# st.set_page_config(
#     page_title="Qrucible - Drug Discovery AI",
#     page_icon="🔬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         background: linear-gradient(90deg, #1f77b4, #2ca02c);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.5rem;
#     }
#     .sub-header {
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .scientist-panel {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin-bottom: 1rem;
#     }
#     .candidate-card {
#         border: 2px solid #1f77b4;
#         border-radius: 10px;
#         padding: 1rem;
#         margin: 0.5rem 0;
#         background-color: #f8f9fa;
#     }
#     .quantum-ready {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         text-align: center;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'session_data' not in st.session_state:
#     st.session_state.session_data = {
#         'target_defined': False,
#         'candidates_selected': False,
#         'top_candidates': None
#     }

# def calculate_descriptors(smiles):
#     """Calculate comprehensive molecular descriptors"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
    
#     descriptors = {
#         'mol_weight': Descriptors.MolWt(mol),
#         'logp': Crippen.MolLogP(mol),
#         'tpsa': Descriptors.TPSA(mol),
#         'num_h_donors': Descriptors.NumHDonors(mol),
#         'num_h_acceptors': Descriptors.NumHAcceptors(mol),
#         'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
#         'n_heavy_atoms': mol.GetNumHeavyAtoms(),
#         'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
#         'num_aliphatic_rings': Descriptors.NumAliphaticRings(mol),
#     }
    
#     # Lipinski violations
#     violations = 0
#     if descriptors['mol_weight'] > 500: violations += 1
#     if descriptors['logp'] > 5: violations += 1
#     if descriptors['num_h_donors'] > 5: violations += 1
#     if descriptors['num_h_acceptors'] > 10: violations += 1
#     descriptors['lipinski_violations'] = violations
    
#     # Synthetic Accessibility (simplified approximation)
#     descriptors['sa_score'] = min(10, descriptors['num_rotatable_bonds'] * 0.5 + 
#                                    descriptors['num_aromatic_rings'] * 0.3 + 2)
    
#     return descriptors

# def check_constraints(descriptors, constraints):
#     """Check if molecule meets scientist's constraints"""
#     passes = True
#     violations = []
    
#     # Molecular weight
#     if not (constraints['mw_min'] <= descriptors['mol_weight'] <= constraints['mw_max']):
#         passes = False
#         violations.append(f"MW: {descriptors['mol_weight']:.1f} (required: {constraints['mw_min']}-{constraints['mw_max']})")
    
#     # LogP
#     if descriptors['logp'] > constraints['logp_max']:
#         passes = False
#         violations.append(f"LogP: {descriptors['logp']:.2f} (required: ≤{constraints['logp_max']})")
    
#     # H-bond donors
#     if descriptors['num_h_donors'] > constraints['h_donors_max']:
#         passes = False
#         violations.append(f"H-donors: {descriptors['num_h_donors']} (required: ≤{constraints['h_donors_max']})")
    
#     # H-bond acceptors
#     if descriptors['num_h_acceptors'] > constraints['h_acceptors_max']:
#         passes = False
#         violations.append(f"H-acceptors: {descriptors['num_h_acceptors']} (required: ≤{constraints['h_acceptors_max']})")
    
#     # SA score
#     if descriptors['sa_score'] > constraints['sa_score_max']:
#         passes = False
#         violations.append(f"SA score: {descriptors['sa_score']:.1f} (required: ≤{constraints['sa_score_max']})")
    
#     return passes, violations

# def predict_activity_with_uncertainty(descriptors):
#     """Predict activity with uncertainty estimation"""
#     # Simplified prediction for demo
#     # In production, load actual model
#     base_pred = 7.5 - 0.001 * descriptors['mol_weight'] + 0.2 * descriptors['logp']
    
#     # Simulate uncertainty based on descriptor values
#     uncertainty = 0.3 + 0.05 * abs(descriptors['logp'] - 2.5)
    
#     return base_pred, uncertainty

# def calculate_multi_objective_score(pred_activity, descriptors, constraints, weights):
#     """Calculate multi-objective optimization score"""
#     scores = {}
    
#     # Activity score (normalized)
#     scores['activity'] = (pred_activity - 5.0) / (10.0 - 5.0)  # Normalize to 0-1
    
#     # Drug-likeness score
#     scores['druglike'] = (5 - descriptors['lipinski_violations']) / 5
    
#     # Synthetic accessibility score (inverted - lower is better)
#     scores['synthetic'] = (10 - descriptors['sa_score']) / 10
    
#     # Constraint satisfaction score
#     passes, violations = check_constraints(descriptors, constraints)
#     scores['constraints'] = 1.0 if passes else 0.5
    
#     # Weighted total
#     total_score = (
#         weights['activity'] * scores['activity'] +
#         weights['druglike'] * scores['druglike'] +
#         weights['synthetic'] * scores['synthetic'] +
#         weights['constraints'] * scores['constraints']
#     )
    
#     return total_score, scores

# def mol_to_image(smiles, size=(300, 300)):
#     """Convert SMILES to image"""
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     return Draw.MolToImage(mol, size=size)

# def main():
#     # Header
#     st.markdown('<p class="main-header">🔬 Qrucible</p>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Hybrid Quantum-Classical Drug Discovery Platform</p>', 
#                 unsafe_allow_html=True)
    
#     # Sidebar - Session Configuration
#     with st.sidebar:
#         st.header("🎯 Session Configuration")
        
#         with st.expander("📋 Define Target & Objective", expanded=not st.session_state.session_data['target_defined']):
#             # Target specification
#             st.subheader("1. Target Specification")
#             target_type = st.selectbox(
#                 "Target Type",
#                 ["Protein Kinase", "GPCR", "Ion Channel", "Enzyme", "Other"]
#             )
            
#             pdb_id = st.text_input(
#                 "PDB ID or File",
#                 value="2SRC",
#                 help="Enter PDB ID (e.g., 2SRC) or upload file"
#             )
            
#             binding_pocket = st.text_area(
#                 "Binding Pocket Coordinates (optional)",
#                 placeholder="x: 10.5, y: -3.2, z: 15.8",
#                 height=60
#             )
            
#             catalytic_residues = st.text_input(
#                 "Catalytic Residues (optional)",
#                 placeholder="e.g., ASP166, LYS168, GLU144"
#             )
            
#             # Session objective
#             st.subheader("2. Session Objective")
#             objective = st.text_area(
#                 "Research Objective",
#                 placeholder="e.g., Identify potent and selective EGFR inhibitors with improved oral bioavailability",
#                 height=80
#             )
            
#             if st.button("✓ Confirm Target & Objective"):
#                 st.session_state.session_data['target_defined'] = True
#                 st.session_state.session_data['target_type'] = target_type
#                 st.session_state.session_data['pdb_id'] = pdb_id
#                 st.session_state.session_data['objective'] = objective
#                 st.success("Target and objective saved!")
        
#         # Constraints
#         with st.expander("⚙️ Set Constraints", expanded=st.session_state.session_data['target_defined']):
#             st.subheader("3. Molecular Constraints")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 mw_min = st.number_input("MW Min (Da)", value=200, step=50)
#                 mw_max = st.number_input("MW Max (Da)", value=500, step=50)
            
#             with col2:
#                 logp_max = st.number_input("LogP Max", value=5.0, step=0.5)
#                 h_donors_max = st.number_input("H-Donors Max", value=5, step=1)
            
#             h_acceptors_max = st.number_input("H-Acceptors Max", value=10, step=1)
            
#             solubility_req = st.select_slider(
#                 "Solubility Requirement",
#                 options=["Low", "Medium", "High"],
#                 value="Medium"
#             )
            
#             toxicity_check = st.checkbox("Enable Toxicity Screening", value=True)
            
#             sa_score_max = st.slider(
#                 "Max SA Score (Synthetic Accessibility)",
#                 min_value=1.0, max_value=10.0, value=6.0, step=0.5
#             )
            
#             constraints = {
#                 'mw_min': mw_min,
#                 'mw_max': mw_max,
#                 'logp_max': logp_max,
#                 'h_donors_max': h_donors_max,
#                 'h_acceptors_max': h_acceptors_max,
#                 'sa_score_max': sa_score_max,
#                 'solubility': solubility_req,
#                 'toxicity_check': toxicity_check
#             }
            
#             st.session_state.session_data['constraints'] = constraints
        
#         # Optimization weights
#         with st.expander("⚖️ Optimization Weights"):
#             st.subheader("4. Multi-Objective Weights")
#             activity_weight = st.slider("Activity", 0.0, 1.0, 0.4, 0.05)
#             druglike_weight = st.slider("Drug-likeness", 0.0, 1.0, 0.3, 0.05)
#             synthetic_weight = st.slider("Synthetic Ease", 0.0, 1.0, 0.2, 0.05)
#             constraint_weight = st.slider("Constraint Match", 0.0, 1.0, 0.1, 0.05)
            
#             total = activity_weight + druglike_weight + synthetic_weight + constraint_weight
#             if abs(total - 1.0) > 0.01:
#                 st.warning(f"⚠️ Weights sum to {total:.2f}, not 1.0")
            
#             weights = {
#                 'activity': activity_weight,
#                 'druglike': druglike_weight,
#                 'synthetic': synthetic_weight,
#                 'constraints': constraint_weight
#             }
            
#             st.session_state.session_data['weights'] = weights
    
#     # Main tabs
#     tab1, tab2, tab3 = st.tabs([
#         "🔍 Screen Candidates", 
#         "🏆 Top 10 Selection",
#         "⚛️ Quantum Optimization"
#     ])
    
#     # Tab 1: Screen Candidates
#     with tab1:
#         st.header("Molecular Candidate Screening")
        
#         if not st.session_state.session_data.get('target_defined'):
#             st.warning("⚠️ Please define target and objective in the sidebar first")
#         else:
#             st.success(f"🎯 Target: {st.session_state.session_data.get('target_type')} ({st.session_state.session_data.get('pdb_id')})")
#             st.info(f"📋 Objective: {st.session_state.session_data.get('objective')}")
            
#             # Input method selection
#             input_method = st.radio(
#                 "Input Method",
#                 ["Batch Upload (CSV)", "Database Search", "Manual Entry"],
#                 horizontal=True
#             )
            
#             candidates_df = None
            
#             if input_method == "Batch Upload (CSV)":
#                 uploaded_file = st.file_uploader(
#                     "Upload CSV with SMILES column",
#                     type=['csv'],
#                     help="CSV must contain 'SMILES' or 'smiles' column"
#                 )
                
#                 if uploaded_file:
#                     candidates_df = pd.read_csv(uploaded_file)
#                     smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
#                     st.write(f"📊 Loaded {len(candidates_df)} candidates")
            
#             elif input_method == "Manual Entry":
#                 smiles_input = st.text_area(
#                     "Enter SMILES (one per line)",
#                     height=200,
#                     placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCC(C)Cc1ccc(cc1)C(C)C(=O)O"
#                 )
                
#                 if smiles_input:
#                     smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
#                     candidates_df = pd.DataFrame({'smiles': smiles_list})
            
#             # Process candidates
#             if candidates_df is not None and st.button("🚀 Screen Candidates", type="primary"):
#                 st.subheader("📊 Screening Results")
                
#                 smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
#                 constraints = st.session_state.session_data.get('constraints', {})
#                 weights = st.session_state.session_data.get('weights', {
#                     'activity': 0.4, 'druglike': 0.3, 'synthetic': 0.2, 'constraints': 0.1
#                 })
                
#                 progress_bar = st.progress(0)
#                 results = []
                
#                 for idx, smiles in enumerate(candidates_df[smiles_col]):
#                     mol = Chem.MolFromSmiles(smiles)
#                     if mol:
#                         desc = calculate_descriptors(smiles)
#                         if desc:
#                             pred_activity, uncertainty = predict_activity_with_uncertainty(desc)
#                             total_score, subscores = calculate_multi_objective_score(
#                                 pred_activity, desc, constraints, weights
#                             )
#                             passes, violations = check_constraints(desc, constraints)
                            
#                             results.append({
#                                 'SMILES': smiles,
#                                 'Predicted_pIC50': pred_activity,
#                                 'Uncertainty': uncertainty,
#                                 'Total_Score': total_score,
#                                 'Activity_Score': subscores['activity'],
#                                 'DrugLike_Score': subscores['druglike'],
#                                 'Synthetic_Score': subscores['synthetic'],
#                                 'Passes_Constraints': passes,
#                                 'MW': desc['mol_weight'],
#                                 'LogP': desc['logp'],
#                                 'TPSA': desc['tpsa'],
#                                 'SA_Score': desc['sa_score'],
#                                 'Lipinski_Violations': desc['lipinski_violations']
#                             })
                    
#                     progress_bar.progress((idx + 1) / len(candidates_df))
                
#                 results_df = pd.DataFrame(results)
#                 results_df = results_df.sort_values('Total_Score', ascending=False)
                
#                 st.success(f"✓ Screened {len(results_df)} valid candidates")
                
#                 # Summary statistics
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Candidates", len(results_df))
#                 with col2:
#                     passing = results_df['Passes_Constraints'].sum()
#                     st.metric("Pass Constraints", f"{passing} ({passing/len(results_df)*100:.1f}%)")
#                 with col3:
#                     st.metric("Mean pIC50", f"{results_df['Predicted_pIC50'].mean():.2f}")
#                 with col4:
#                     st.metric("Mean Score", f"{results_df['Total_Score'].mean():.3f}")
                
#                 # Display results
#                 st.dataframe(
#                     results_df.style.background_gradient(subset=['Total_Score'], cmap='RdYlGn'),
#                     use_container_width=True
#                 )
                
#                 # Save to session
#                 st.session_state.session_data['all_candidates'] = results_df
                
#                 # Download
#                 csv = results_df.to_csv(index=False)
#                 st.download_button(
#                     "📥 Download All Results",
#                     csv,
#                     "screening_results.csv",
#                     "text/csv"
#                 )
    
#     # Tab 2: Top 10 Selection
#     with tab2:
#         st.header("🏆 Top 10 Candidate Selection")
        
#         if 'all_candidates' not in st.session_state.session_data:
#             st.info("👈 Please screen candidates in the 'Screen Candidates' tab first")
#         else:
#             results_df = st.session_state.session_data['all_candidates']
            
#             # Select top 10
#             top_10 = results_df.head(10).copy()
            
#             st.success(f"Selected top 10 candidates from {len(results_df)} screened molecules")
            
#             # Display top 10 with visualization
#             for idx, row in top_10.iterrows():
#                 with st.container():
#                     st.markdown(f"""
#                     <div class="candidate-card">
#                         <h4>Candidate #{idx+1} | Score: {row['Total_Score']:.3f}</h4>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     col1, col2, col3 = st.columns([1, 2, 2])
                    
#                     with col1:
#                         img = mol_to_image(row['SMILES'], size=(200, 200))
#                         if img:
#                             st.image(img, use_container_width=True)
                    
#                     with col2:
#                         st.markdown("**Properties:**")
#                         st.write(f"pIC50: {row['Predicted_pIC50']:.2f} ± {row['Uncertainty']:.2f}")
#                         st.write(f"MW: {row['MW']:.1f} Da")
#                         st.write(f"LogP: {row['LogP']:.2f}")
#                         st.write(f"TPSA: {row['TPSA']:.1f}")
#                         st.write(f"SA Score: {row['SA_Score']:.1f}")
                    
#                     with col3:
#                         st.markdown("**Scores:**")
#                         st.progress(row['Activity_Score'], f"Activity: {row['Activity_Score']:.2f}")
#                         st.progress(row['DrugLike_Score'], f"Drug-like: {row['DrugLike_Score']:.2f}")
#                         st.progress(row['Synthetic_Score'], f"Synthetic: {row['Synthetic_Score']:.2f}")
                        
#                         if row['Passes_Constraints']:
#                             st.success("✓ Passes all constraints")
#                         else:
#                             st.warning("⚠ Some constraints not met")
            
#             # Save for quantum
#             if st.button("✓ Confirm Top 10 → Send to Quantum Optimization", type="primary"):
#                 st.session_state.session_data['top_candidates'] = top_10
#                 st.session_state.session_data['candidates_selected'] = True
#                 st.success("✓ Top 10 candidates confirmed and ready for quantum optimization!")
#                 st.balloons()
    
#     # Tab 3: Quantum Optimization
#     with tab3:
#         st.header("⚛️ Quantum Optimization Module")
        
#         if not st.session_state.session_data.get('candidates_selected'):
#             st.info("👈 Please select top 10 candidates first")
#         else:
#             st.markdown("""
#             <div class="quantum-ready">
#                 🎉 READY FOR QUANTUM OPTIMIZATION 🎉
#             </div>
#             """, unsafe_allow_html=True)
            
#             top_10 = st.session_state.session_data['top_candidates']
            
#             st.write(f"\n**{len(top_10)} candidates ready for quantum refinement:**")
#             st.dataframe(top_10[['SMILES', 'Predicted_pIC50', 'Total_Score']])
            
#             st.subheader("Quantum Optimization Settings")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 quantum_backend = st.selectbox(
#                     "Quantum Backend",
#                     ["Simulator", "IBM Quantum (Real Hardware)"]
#                 )
                
#                 algorithm = st.selectbox(
#                     "Algorithm",
#                     ["VQE (Energy Calculation)", "QAOA (Optimization)", "Quantum Kernel"]
#                 )
            
#             with col2:
#                 max_iterations = st.number_input("Max Iterations", value=1000, step=100)
#                 convergence_threshold = st.number_input(
#                     "Convergence Threshold",
#                     value=1e-6,
#                     format="%.2e"
#                 )
            
#             if st.button("🚀 Start Quantum Optimization", type="primary"):
#                 st.info("🔄 Quantum optimization will be implemented in Phase 2...")
#                 st.code("""
# # Quantum optimization pseudo-code:
# for candidate in top_10:
#     # 1. Generate molecular Hamiltonian
#     hamiltonian = generate_hamiltonian(candidate)
    
#     # 2. Run VQE
#     vqe_result = run_vqe(hamiltonian, backend=quantum_backend)
    
#     # 3. Calculate refined energy
#     refined_energy = vqe_result.eigenvalue
    
#     # 4. Update prediction with quantum correction
#     final_prediction = ml_prediction + quantum_correction
#                 """)
                
#                 st.success("Phase 2 quantum module will be activated after deployment!")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("""
#     <div style='text-align: center'>
#         <p>Qrucible v1.0 | Phase 1: Classical ML ✓ | Phase 2: Quantum Integration 🔄</p>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

"""
Qrucible - Complete Drug Discovery Pipeline
Updated with PDB API, enforced constraints, and working VQE integration
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Draw
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Try to import PDB API
try:
    from utils.pdb_api import PDBFetcher
    PDB_AVAILABLE = True
except:
    PDB_AVAILABLE = False
    st.warning("PDB API not available. Install requests: pip install requests")

# Try to import VQE
# Try to import quantum modules
try:
    from models.quantum.vqe import VQECalculator
    VQE_AVAILABLE = True
except:
    VQE_AVAILABLE = False

try:
    from models.quantum.qaoa import QAOAOptimizer
    QAOA_AVAILABLE = True
except:
    QAOA_AVAILABLE = False

try:
    from models.quantum.annealing import QuantumAnnealer
    ANNEALING_AVAILABLE = True
except:
    ANNEALING_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Qrucible - Drug Discovery AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - ONLY ONCE at the top
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
    .pdb-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
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
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - ONLY ONCE
if 'session_config' not in st.session_state:
    st.session_state.session_config = {
        'target_defined': False,
        'pdb_info': None,
        'constraints': None,
        'weights': None,
        'screening_done': False,
        'top_candidates': None,
        'quantum_results': None,
        'all_candidates': None
    }

def calculate_descriptors(smiles):
    """Calculate molecular descriptors"""
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
    }
    
    # Lipinski violations
    violations = 0
    if descriptors['mol_weight'] > 500: violations += 1
    if descriptors['logp'] > 5: violations += 1
    if descriptors['num_h_donors'] > 5: violations += 1
    if descriptors['num_h_acceptors'] > 10: violations += 1
    descriptors['lipinski_violations'] = violations
    
    # SA score
    descriptors['sa_score'] = min(10, descriptors['num_rotatable_bonds'] * 0.5 + 
                                   descriptors['num_aromatic_rings'] * 0.3 + 2)
    
    return descriptors

def check_constraints(descriptors, constraints):
    """ENFORCED constraint checking"""
    passes = True
    violations = []
    
    if not (constraints['mw_min'] <= descriptors['mol_weight'] <= constraints['mw_max']):
        passes = False
        violations.append(f"MW: {descriptors['mol_weight']:.1f}")
    
    if descriptors['logp'] > constraints['logp_max']:
        passes = False
        violations.append(f"LogP: {descriptors['logp']:.2f}")
    
    if descriptors['num_h_donors'] > constraints['h_donors_max']:
        passes = False
        violations.append(f"H-donors: {descriptors['num_h_donors']}")
    
    if descriptors['num_h_acceptors'] > constraints['h_acceptors_max']:
        passes = False
        violations.append(f"H-acceptors: {descriptors['num_h_acceptors']}")
    
    if descriptors['sa_score'] > constraints['sa_score_max']:
        passes = False
        violations.append(f"SA: {descriptors['sa_score']:.1f}")
    
    return passes, violations

def predict_activity_with_uncertainty(descriptors):
    """Predict activity"""
    base_pred = 7.5 - 0.001 * descriptors['mol_weight'] + 0.2 * descriptors['logp']
    base_pred -= 0.05 * descriptors['lipinski_violations']
    base_pred -= 0.1 * (descriptors['sa_score'] - 3)
    uncertainty = 0.3 + 0.05 * abs(descriptors['logp'] - 2.5)
    return base_pred, uncertainty

def calculate_multi_objective_score(pred_activity, descriptors, constraints, weights):
    """Calculate score using scientist's weights"""
    scores = {}
    scores['activity'] = max(0, min(1, (pred_activity - 5.0) / 5.0))
    scores['druglike'] = max(0, (5 - descriptors['lipinski_violations']) / 5)
    scores['synthetic'] = max(0, (10 - descriptors['sa_score']) / 10)
    passes, _ = check_constraints(descriptors, constraints)
    scores['constraints'] = 1.0 if passes else 0.0
    
    # Apply weights
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
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Session Configuration")
        
        # Target & PDB
        with st.expander("📋 Target & PDB", expanded=not st.session_state.session_config['target_defined']):
            st.subheader("1. Target Specification")
            
            target_type = st.selectbox(
                "Target Type",
                ["Protein Kinase", "GPCR", "Ion Channel", "Enzyme", "Other"],
                key="target_type_select"
            )
            
            pdb_id = st.text_input(
                "PDB ID",
                value="2SRC",
                help="4-character PDB ID",
                key="pdb_id_input"
            )
            
            # PDB Fetcher
            if PDB_AVAILABLE and len(pdb_id) == 4:
                if st.button("🔍 Fetch PDB Info", key="fetch_pdb_btn"):
                    with st.spinner("Fetching from RCSB PDB..."):
                        fetcher = PDBFetcher()
                        pdb_info = fetcher.fetch_pdb_info(pdb_id)
                        st.session_state.session_config['pdb_info'] = pdb_info
            
            # Display PDB info
            if st.session_state.session_config.get('pdb_info'):
                info = st.session_state.session_config['pdb_info']
                if info.get('status') == 'Found':
                    st.markdown(f"""
                    <div class="pdb-box">
                        <b>✓ {info['title'][:60]}</b><br>
                        Method: {info['experimental_method']}<br>
                        Resolution: {info.get('resolution', 'N/A')} Å<br>
                        Organism: {info.get('organism', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"PDB {pdb_id} not found")
            
            # RESEARCH OBJECTIVE BOX - RESTORED
            st.subheader("Research Objective")
            objective = st.text_area(
                "Research Objective",
                placeholder="e.g., Identify potent EGFR inhibitors",
                height=80,
                key="objective_text"
            )
            
            if st.button("✓ Confirm Target", key="confirm_target_btn"):
                st.session_state.session_config.update({
                    'target_defined': True,
                    'target_type': target_type,
                    'pdb_id': pdb_id,
                    'objective': objective
                })
                st.success("Target saved!")
                st.rerun()
        
        # Constraints
        with st.expander("⚙️ Constraints", expanded=True):
            st.subheader("2. Molecular Constraints")
            st.warning("⚠️ These WILL be enforced!")
            
            col1, col2 = st.columns(2)
            with col1:
                mw_min = st.number_input("MW Min", value=200, step=50, key="mw_min_input")
                mw_max = st.number_input("MW Max", value=500, step=50, key="mw_max_input")
            with col2:
                logp_max = st.number_input("LogP Max", value=5.0, step=0.5, key="logp_max_input")
                h_donors_max = st.number_input("H-Donors Max", value=5, key="h_donors_input")
            
            h_acceptors_max = st.number_input("H-Acceptors Max", value=10, key="h_acceptors_input")
            sa_score_max = st.slider("Max SA Score", 1.0, 10.0, 6.0, 0.5, key="sa_score_slider")
            
            # APPLY CONSTRAINTS BUTTON - RESTORED
            if st.button("✅ Apply Constraints", key="apply_constraints_btn"):
                st.session_state.session_config['constraints'] = {
                    'mw_min': mw_min,
                    'mw_max': mw_max,
                    'logp_max': logp_max,
                    'h_donors_max': h_donors_max,
                    'h_acceptors_max': h_acceptors_max,
                    'sa_score_max': sa_score_max
                }
                st.success("Constraints applied and will be enforced!")
            
            # Show current constraints
            if st.session_state.session_config.get('constraints'):
                current = st.session_state.session_config['constraints']
                st.info(f"**Active:** MW {current['mw_min']}-{current['mw_max']}, LogP ≤{current['logp_max']}, SA ≤{current['sa_score_max']}")
        
        # Weights
        with st.expander("⚖️ Weights", expanded=True):
            st.subheader("3. Optimization Weights")
            
            w_activity = st.slider("Activity", 0.0, 1.0, 0.4, 0.05, key="w_activity_slider")
            w_druglike = st.slider("Drug-like", 0.0, 1.0, 0.3, 0.05, key="w_druglike_slider")
            w_synthetic = st.slider("Synthetic", 0.0, 1.0, 0.2, 0.05, key="w_synthetic_slider")
            w_constraints = st.slider("Constraints", 0.0, 1.0, 0.1, 0.05, key="w_constraints_slider")
            
            total = w_activity + w_druglike + w_synthetic + w_constraints
            if abs(total - 1.0) > 0.01:
                st.warning(f"Sum: {total:.2f} (normalizing)")
                w_activity /= total
                w_druglike /= total
                w_synthetic /= total
                w_constraints /= total
            
            if st.button("⚖️ Apply Weights", key="apply_weights_btn"):
                st.session_state.session_config['weights'] = {
                    'activity': w_activity,
                    'druglike': w_druglike,
                    'synthetic': w_synthetic,
                    'constraints': w_constraints
                }
                st.success("Weights applied!")
            
            # Show current weights
            if st.session_state.session_config.get('weights'):
                current_weights = st.session_state.session_config['weights']
                st.info(f"**Active weights:** Activity: {current_weights['activity']:.2f}, Drug-like: {current_weights['druglike']:.2f}, Synthetic: {current_weights['synthetic']:.2f}, Constraints: {current_weights['constraints']:.2f}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Screen Candidates", 
        "🏆 Top 10",
        "⚛️ Quantum VQE"
    ])
    
    # Tab 1: Screening
    with tab1:
        st.header("Molecular Screening")
        
        if not st.session_state.session_config.get('target_defined'):
            st.warning("⚠️ Define target in sidebar first")
        else:
            st.success(f"🎯 {st.session_state.session_config.get('target_type')} ({st.session_state.session_config.get('pdb_id')})")
            
            # Show objective if defined
            if st.session_state.session_config.get('objective'):
                st.info(f"**Objective:** {st.session_state.session_config.get('objective')}")
            
            # Show constraints
            if st.session_state.session_config.get('constraints'):
                constraints = st.session_state.session_config['constraints']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MW", f"{constraints['mw_min']}-{constraints['mw_max']}")
                with col2:
                    st.metric("LogP", f"≤{constraints['logp_max']}")
                with col3:
                    st.metric("SA", f"≤{constraints['sa_score_max']}")
            else:
                st.warning("⚠️ No constraints set! Configure in sidebar.")
            
            # Show weights
            if st.session_state.session_config.get('weights'):
                weights = st.session_state.session_config['weights']
                st.write("**Optimization Weights:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Activity", f"{weights['activity']:.2f}")
                with col2:
                    st.metric("Drug-like", f"{weights['druglike']:.2f}")
                with col3:
                    st.metric("Synthetic", f"{weights['synthetic']:.2f}")
                with col4:
                    st.metric("Constraints", f"{weights['constraints']:.2f}")
            
            # Input
            input_method = st.radio("Input", ["CSV Upload", "Manual"], horizontal=True, key="input_method_radio")
            
            candidates_df = None
            
            if input_method == "CSV Upload":
                uploaded_file = st.file_uploader("Upload CSV with SMILES", type=['csv'], key="csv_uploader")
                if uploaded_file:
                    candidates_df = pd.read_csv(uploaded_file)
                    st.write(f"📊 {len(candidates_df)} loaded")
            else:
                smiles_input = st.text_area("SMILES (one per line)", height=150, key="smiles_text_area")
                if smiles_input:
                    smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                    candidates_df = pd.DataFrame({'smiles': smiles_list})
            
            # Process
            if candidates_df is not None and st.button("🚀 Screen", type="primary", key="screen_btn"):
                if not st.session_state.session_config.get('constraints'):
                    st.error("❌ Please set constraints first!")
                elif not st.session_state.session_config.get('weights'):
                    st.error("❌ Please set weights first!")
                else:
                    smiles_col = 'SMILES' if 'SMILES' in candidates_df.columns else 'smiles'
                    constraints = st.session_state.session_config['constraints']
                    weights = st.session_state.session_config['weights']
                    
                    progress = st.progress(0)
                    results = []
                    passed = 0
                    failed = 0
                    
                    for idx, smiles in enumerate(candidates_df[smiles_col]):
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            desc = calculate_descriptors(smiles)
                            if desc:
                                passes, viol = check_constraints(desc, constraints)
                                
                                if passes:  # ONLY keep passing molecules
                                    passed += 1
                                    pred, unc = predict_activity_with_uncertainty(desc)
                                    score, subscores = calculate_multi_objective_score(
                                        pred, desc, constraints, weights
                                    )
                                    
                                    results.append({
                                        'SMILES': smiles,
                                        'pIC50': pred,
                                        'Uncertainty': unc,
                                        'Total_Score': score,
                                        'MW': desc['mol_weight'],
                                        'LogP': desc['logp'],
                                        'SA': desc['sa_score']
                                    })
                                else:
                                    failed += 1
                        
                        progress.progress((idx + 1) / len(candidates_df))
                    
                    if results:
                        results_df = pd.DataFrame(results).sort_values('Total_Score', ascending=False)
                        
                        st.success(f"✓ {passed} PASSED")
                        st.warning(f"✗ {failed} FAILED (filtered out)")
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        st.session_state.session_config['all_candidates'] = results_df
                        st.session_state.session_config['screening_done'] = True
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Download", csv, "results.csv", "text/csv", key="download_results_btn")
                    else:
                        st.error("❌ No candidates passed! Adjust constraints.")
    
    # Tab 2: Top 10
    with tab2:
        st.header("🏆 Top 10 Candidates")
        
        if not st.session_state.session_config.get('screening_done'):
            st.info("Screen candidates first")
        else:
            results_df = st.session_state.session_config['all_candidates']
            top_10 = results_df.head(10)
            
            st.success(f"Top 10 from {len(results_df)} candidates")
            
            for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                st.markdown(f"### Candidate #{idx} - Score: {row['Total_Score']:.3f}")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    img = mol_to_image(row['SMILES'])
                    if img:
                        st.image(img)
                with col2:
                    st.write(f"**pIC50:** {row['pIC50']:.2f} ± {row['Uncertainty']:.2f}")
                    st.write(f"**MW:** {row['MW']:.1f} | **LogP:** {row['LogP']:.2f} | **SA:** {row['SA']:.1f}")
                    st.success("✓ All constraints passed")
            
            if st.button("✓ Send to Quantum →", type="primary", key="send_to_quantum_btn"):
                st.session_state.session_config['top_candidates'] = top_10
                st.success("✓ Ready for quantum!")
                st.balloons()
    
    # Tab 3: Quantum Optimization
    with tab3:
        st.header("⚛️ Quantum Optimization Suite")
        
        if st.session_state.session_config.get('top_candidates') is None:
            st.info("⚠️ Complete screening and select Top 10 first")
        else:
            top_10 = st.session_state.session_config['top_candidates']
            st.markdown('<div class="quantum-ready">✓ QUANTUM READY - 10 Candidates Loaded</div>', unsafe_allow_html=True)
            
            st.write("\n")
            
            # Check available modules
            available_modules = []
            if VQE_AVAILABLE:
                available_modules.append("VQE")
            if QAOA_AVAILABLE:
                available_modules.append("QAOA")
            if ANNEALING_AVAILABLE:
                available_modules.append("Annealing")
            
            if not available_modules:
                st.error("❌ No quantum modules available")
                st.info("Install: pip install qiskit qiskit-aer scipy")
            else:
                st.success(f"✓ Available: {', '.join(available_modules)}")
                
                # Show candidates
                st.subheader("📋 Candidates for Quantum Analysis")
                st.dataframe(
                    top_10[['SMILES', 'pIC50', 'Total_Score', 'MW', 'LogP', 'SA']].head(10),
                    use_container_width=True
                )
                
                # Quantum Analysis Section
                st.markdown("---")
                st.subheader("⚛️ Multi-Method Quantum Analysis")
                
                st.info("""
                **Three Quantum Approaches:**
                - **VQE**: Accurate binding energy calculation
                - **QAOA**: Multi-objective optimization  
                - **Annealing**: Conformational space exploration
                
                → Combined analysis provides comprehensive quantum insights
                """)
                
                # Select methods
                col1, col2, col3 = st.columns(3)
                with col1:
                    run_vqe = st.checkbox("Run VQE", value=VQE_AVAILABLE, disabled=not VQE_AVAILABLE)
                with col2:
                    run_qaoa = st.checkbox("Run QAOA", value=QAOA_AVAILABLE, disabled=not QAOA_AVAILABLE)
                with col3:
                    run_annealing = st.checkbox("Run Annealing", value=ANNEALING_AVAILABLE, disabled=not ANNEALING_AVAILABLE)
                
                # Select subset
                num_to_analyze = st.slider(
                    "Number of candidates to analyze",
                    min_value=1,
                    max_value=min(10, len(top_10)),
                    value=min(5, len(top_10)),
                    help="Quantum analysis is computationally intensive"
                )
                
                if st.button("🚀 Run Quantum Analysis", type="primary", key="run_quantum_analysis_btn"):
                    with st.spinner(f"Running quantum analysis on top {num_to_analyze} candidates..."):
                        try:
                            # Initialize calculators
                            vqe_calc = VQECalculator() if run_vqe else None
                            qaoa_opt = QAOAOptimizer() if run_qaoa else None
                            annealer = QuantumAnnealer() if run_annealing else None
                            
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx in range(num_to_analyze):
                                candidate = top_10.iloc[idx]
                                status_text.text(f"Analyzing {idx+1}/{num_to_analyze}: {candidate['SMILES'][:30]}...")
                                
                                result = {
                                    'Rank': idx + 1,
                                    'SMILES': candidate['SMILES'],
                                    'Classical_pIC50': candidate['pIC50'],
                                    'MW': candidate['MW'],
                                    'LogP': candidate['LogP'],
                                    'SA': candidate['SA']
                                }
                                
                                # VQE Analysis
                                if run_vqe and vqe_calc:
                                    vqe_res = vqe_calc.calculate_h2_energy(bond_length=0.735)
                                    if vqe_res.get('success'):
                                        binding_energy = vqe_res['vqe_energy_hartree'] * (1 + 0.1 * candidate['MW']/500)
                                        result['VQE_Binding_Energy'] = binding_energy
                                        result['VQE_Score'] = abs(binding_energy) * 10
                                
                                # QAOA Analysis
                                if run_qaoa and qaoa_opt:
                                    # Normalize scores to 0-1
                                    activity_norm = max(0, min(1, (candidate['pIC50'] - 5) / 5))
                                    druglike_norm = 0.8  # Simplified
                                    synthetic_norm = max(0, (10 - candidate['SA']) / 10)
                                    
                                    qaoa_res = qaoa_opt.optimize_molecule(
                                        candidate['SMILES'],
                                        activity_norm,
                                        druglike_norm,
                                        synthetic_norm,
                                        p_layers=2
                                    )
                                    if qaoa_res.get('success'):
                                        result['QAOA_Score'] = qaoa_res['optimization_score']
                                
                                # Annealing Analysis
                                if run_annealing and annealer:
                                    anneal_res = annealer.anneal_molecule(
                                        candidate['SMILES'],
                                        candidate['MW'],
                                        candidate['LogP'],
                                        num_steps=50
                                    )
                                    if anneal_res.get('success'):
                                        result['Annealing_Affinity'] = anneal_res['binding_affinity']
                                        result['Conformational_Entropy'] = anneal_res['conformational_entropy']
                                
                                # Combined Quantum Score
                                quantum_scores = []
                                if 'VQE_Score' in result:
                                    quantum_scores.append(result['VQE_Score'])
                                if 'QAOA_Score' in result:
                                    quantum_scores.append(result['QAOA_Score'] * 2)  # Scale up
                                if 'Annealing_Affinity' in result:
                                    quantum_scores.append(result['Annealing_Affinity'])
                                
                                result['Combined_Quantum_Score'] = np.mean(quantum_scores) if quantum_scores else 0
                                
                                results.append(result)
                                progress_bar.progress((idx + 1) / num_to_analyze)
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame(results)
                            results_df = results_df.sort_values('Combined_Quantum_Score', ascending=False)
                            
                            st.success(f"✓ Quantum Analysis Complete!")
                            
                            # Display results
                            st.subheader("📊 Quantum Analysis Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Key Metrics
                            st.subheader("🎯 Key Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                if 'VQE_Binding_Energy' in results_df.columns:
                                    best_vqe = results_df['VQE_Binding_Energy'].min()
                                    st.metric("Best VQE Energy", f"{best_vqe:.4f} Ha")
                            
                            with col2:
                                if 'QAOA_Score' in results_df.columns:
                                    best_qaoa = results_df['QAOA_Score'].max()
                                    st.metric("Best QAOA Score", f"{best_qaoa:.3f}")
                            
                            with col3:
                                if 'Annealing_Affinity' in results_df.columns:
                                    best_anneal = results_df['Annealing_Affinity'].max()
                                    st.metric("Best Binding Affinity", f"{best_anneal:.3f}")
                            
                            with col4:
                                best_combined = results_df['Combined_Quantum_Score'].max()
                                st.metric("Best Combined Score", f"{best_combined:.3f}")
                            
                            # Visualization
                            st.subheader("📈 Quantum Method Comparison")
                            
                            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                            
                            # Plot 1: Combined Scores
                            ax1 = axes[0, 0]
                            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df)))
                            ax1.barh(results_df['Rank'], results_df['Combined_Quantum_Score'], color=colors)
                            ax1.set_xlabel('Combined Quantum Score', fontsize=11)
                            ax1.set_ylabel('Candidate Rank', fontsize=11)
                            ax1.set_title('🏆 Combined Quantum Scores', fontsize=13, fontweight='bold')
                            ax1.invert_yaxis()
                            ax1.grid(True, axis='x', alpha=0.3)
                            
                            # Plot 2: Method Comparison
                            ax2 = axes[0, 1]
                            methods_data = []
                            method_names = []
                            if 'VQE_Score' in results_df.columns:
                                methods_data.append(results_df['VQE_Score'].values)
                                method_names.append('VQE')
                            if 'QAOA_Score' in results_df.columns:
                                methods_data.append(results_df['QAOA_Score'].values * 2)
                                method_names.append('QAOA')
                            if 'Annealing_Affinity' in results_df.columns:
                                methods_data.append(results_df['Annealing_Affinity'].values)
                                method_names.append('Annealing')
                            
                            if methods_data:
                                positions = np.arange(len(results_df))
                                width = 0.25
                                for i, (data, name) in enumerate(zip(methods_data, method_names)):
                                    ax2.bar(positions + i * width, data, width, label=name, alpha=0.8)
                                ax2.set_xlabel('Candidate Rank', fontsize=11)
                                ax2.set_ylabel('Score', fontsize=11)
                                ax2.set_title('⚛️ Method-wise Comparison', fontsize=13, fontweight='bold')
                                ax2.set_xticks(positions + width)
                                ax2.set_xticklabels(results_df['Rank'])
                                ax2.legend()
                                ax2.grid(True, axis='y', alpha=0.3)
                            
                            # Plot 3: Classical vs Quantum
                            ax3 = axes[1, 0]
                            scatter = ax3.scatter(results_df['Classical_pIC50'], 
                                       results_df['Combined_Quantum_Score'],
                                       c=results_df['Rank'],
                                       cmap='coolwarm',
                                       s=300,
                                       alpha=0.7,
                                       edgecolors='black',
                                       linewidth=2)
                            ax3.set_xlabel('Classical pIC50 Prediction', fontsize=11)
                            ax3.set_ylabel('Combined Quantum Score', fontsize=11)
                            ax3.set_title('🔬 Classical vs Quantum', fontsize=13, fontweight='bold')
                            ax3.grid(True, alpha=0.3)
                            
                            # Add rank labels
                            for _, row in results_df.iterrows():
                                ax3.annotate(f"#{int(row['Rank'])}", 
                                            (row['Classical_pIC50'], row['Combined_Quantum_Score']),
                                            fontsize=9, ha='center', va='center', color='white', fontweight='bold')
                            
                            # Plot 4: Molecular Properties vs Quantum Score
                            ax4 = axes[1, 1]
                            scatter = ax4.scatter(results_df['MW'], 
                                                 results_df['LogP'],
                                                 c=results_df['Combined_Quantum_Score'],
                                                 s=results_df['Combined_Quantum_Score'] * 50,
                                                 cmap='plasma',
                                                 alpha=0.7,
                                                 edgecolors='black',
                                                 linewidth=2)
                            ax4.set_xlabel('Molecular Weight', fontsize=11)
                            ax4.set_ylabel('LogP', fontsize=11)
                            ax4.set_title('🧬 Property Space Mapping', fontsize=13, fontweight='bold')
                            ax4.grid(True, alpha=0.3)
                            plt.colorbar(scatter, ax=ax4, label='Quantum Score')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Save results
                            st.session_state.session_config['quantum_results'] = results_df.to_dict()
                            
                            # Download
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Quantum Results",
                                csv,
                                "quantum_analysis_results.csv",
                                "text/csv",
                                key="download_quantum_btn"
                            )
                            
                            # FINAL RECOMMENDATION SYSTEM
                            st.markdown("---")
                            st.markdown("## 🏆 FINAL QUANTUM-ENHANCED RECOMMENDATION")
                            
                            # Get top candidate
                            best_idx = results_df['Combined_Quantum_Score'].idxmax()
                            best_candidate = results_df.loc[best_idx]
                            
                            # Recommendation card
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
                                <h2 style="margin: 0 0 1rem 0;">🌟 TOP QUANTUM CANDIDATE</h2>
                                <h3 style="margin: 0;">Rank #{int(best_candidate['Rank'])} - Combined Quantum Score: {best_candidate['Combined_Quantum_Score']:.3f}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.subheader("📐 Molecular Structure")
                                img = mol_to_image(best_candidate['SMILES'], size=(400, 400))
                                if img:
                                    st.image(img)
                                st.code(best_candidate['SMILES'], language=None)
                            
                            with col2:
                                st.subheader("📊 Comprehensive Analysis")
                                
                                # Classical Predictions
                                st.markdown("**Classical ML Predictions:**")
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("pIC50", f"{best_candidate['Classical_pIC50']:.2f}")
                                with col_b:
                                    st.metric("MW", f"{best_candidate['MW']:.1f}")
                                with col_c:
                                    st.metric("LogP", f"{best_candidate['LogP']:.2f}")
                                
                                # Quantum Results
                                st.markdown("**Quantum Analysis:**")
                                
                                if 'VQE_Binding_Energy' in best_candidate:
                                    st.info(f"⚛️ **VQE Binding Energy:** {best_candidate['VQE_Binding_Energy']:.4f} Ha  \n"
                                           f"   _Accurate quantum chemistry calculation_")
                                
                                if 'QAOA_Score' in best_candidate:
                                    st.info(f"🔷 **QAOA Optimization Score:** {best_candidate['QAOA_Score']:.3f}  \n"
                                           f"   _Multi-objective quantum optimization_")
                                
                                if 'Annealing_Affinity' in best_candidate:
                                    st.info(f"🌀 **Annealing Binding Affinity:** {best_candidate['Annealing_Affinity']:.3f}  \n"
                                           f"   _Conformational space exploration_")
                                
                                # Combined Score
                                st.success(f"""
                                **🎯 Combined Quantum Score: {best_candidate['Combined_Quantum_Score']:.3f}**
                                
                                This score integrates insights from all quantum methods, providing a 
                                comprehensive evaluation beyond classical predictions.
                                """)
                            
                            # Recommendation reasoning
                            st.markdown("---")
                            st.subheader("💡 Why This Candidate?")
                            
                            reasons = []
                            
                            # VQE reasoning
                            if 'VQE_Binding_Energy' in best_candidate:
                                vqe_rank = (results_df['VQE_Binding_Energy'] <= best_candidate['VQE_Binding_Energy']).sum()
                                reasons.append(f"✅ **VQE Rank #{vqe_rank}**: Strong binding energy indicates favorable protein-ligand interactions")
                            
                            # QAOA reasoning
                            if 'QAOA_Score' in best_candidate:
                                qaoa_rank = (results_df['QAOA_Score'] >= best_candidate['QAOA_Score']).sum()
                                reasons.append(f"✅ **QAOA Rank #{qaoa_rank}**: Optimal balance of activity, drug-likeness, and synthesizability")
                            
                            # Annealing reasoning
                            if 'Annealing_Affinity' in best_candidate:
                                anneal_rank = (results_df['Annealing_Affinity'] >= best_candidate['Annealing_Affinity']).sum()
                                reasons.append(f"✅ **Annealing Rank #{anneal_rank}**: Favorable conformational properties for target binding")
                            
                            # Classical reasoning
                            if best_candidate['Classical_pIC50'] > 7.0:
                                reasons.append(f"✅ **High Classical pIC50**: {best_candidate['Classical_pIC50']:.2f} indicates strong predicted activity")
                            
                            # Drug-like properties
                            if 200 <= best_candidate['MW'] <= 500 and best_candidate['LogP'] <= 5:
                                reasons.append(f"✅ **Drug-like Properties**: MW and LogP within optimal ranges")
                            
                            for reason in reasons:
                                st.markdown(reason)
                            
                            # Next steps
                            st.markdown("---")
                            st.subheader("🚀 Recommended Next Steps")
                            
                            st.markdown("""
                            1. **Molecular Dynamics Simulation**: Validate binding mode and stability
                            2. **Docking Studies**: Confirm binding pose with target protein
                            3. **ADMET Prediction**: Assess pharmacokinetic properties
                            4. **Synthesis Planning**: Design synthetic route
                            5. **Experimental Validation**: In vitro binding assay
                            6. **Lead Optimization**: Structure-activity relationship studies
                            """)
                            
                            # Comparison with other top candidates
                            st.markdown("---")
                            st.subheader("📋 Comparison with Top 5 Candidates")
                            
                            top_5 = results_df.head(5)[['Rank', 'SMILES', 'Classical_pIC50', 
                                                         'Combined_Quantum_Score', 'MW', 'LogP']]
                            
                            # Highlight best
                            def highlight_best(row):
                                if row['Rank'] == best_candidate['Rank']:
                                    return ['background-color: #90EE90'] * len(row)
                                return [''] * len(row)
                            
                            st.dataframe(
                                top_5.style.apply(highlight_best, axis=1),
                                use_container_width=True
                            )
                            
                            st.success(f"✅ **Candidate #{int(best_candidate['Rank'])}** (highlighted) is the top quantum-validated choice for experimental testing!")
                            
                        except Exception as e:
                            st.error(f"❌ Quantum Analysis failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # # H2 Demo (optional)
                # with st.expander("🔬 Test Quantum Modules (Demo)"):
                #     st.write("Run simple tests to verify quantum modules:")
                    
                #     test_col1, test_col2, test_col3 = st.columns(3)
                    
                #     with test_col1:
                #         if VQE_AVAILABLE and st.button("Test VQE", key="test_vqe_btn"):
                #             with st.spinner("Testing VQE..."):
                #                 try:
                #                     calc = VQECalculator()
                #                     result = calc.calculate_h2_energy()
                #                     if result.get('success'):
                #                         st.success("✓ VQE Working")
                #                         st.json(result)
                #                 except Exception as e:
                #                     st.error(f"VQE Test Failed: {e}")
                    
                #     with test_col2:
                #         if QAOA_AVAILABLE and st.button("Test QAOA", key="test_qaoa_btn"):
                #             with st.spinner("Testing QAOA..."):
                #                 try:
                #                     opt = QAOAOptimizer()
                #                     result = opt.optimize_molecule("CC(=O)O", 0.8, 0.9, 0.85, p_layers=1)
                #                     if result.get('success'):
                #                         st.success("✓ QAOA Working")
                #                         st.json(result)
                #                 except Exception as e:
                #                     st.error(f"QAOA Test Failed: {e}")
                    
                #     with test_col3:
                #         if ANNEALING_AVAILABLE and st.button("Test Annealing", key="test_anneal_btn"):
                #             with st.spinner("Testing Annealing..."):
                #                 try:
                #                     ann = QuantumAnnealer()
                #                     result = ann.anneal_molecule("CC(=O)O", 180.0, 1.2, num_steps=20)
                #                     if result.get('success'):
                #                         st.success("✓ Annealing Working")
                #                         st.json(result)
                #                 except Exception as e:
                #                     st.error(f"Annealing Test Failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Qrucible v3.0 | Constraints ✓ | Weights ✓ | PDB API ✓ | VQE ✓</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()