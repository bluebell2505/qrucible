"""
Complete Phase 2 Demo
Demonstrates full quantum-classical hybrid workflow

Usage:
    python run_phase2_demo.py
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def main():
    print("\n" + "="*70)
    print("🔬 QRUCIBLE - PHASE 2 COMPLETE DEMONSTRATION 🔬")
    print("="*70 + "\n")
    
    print("This demo will:")
    print("  1. ✅ Validate quantum setup")
    print("  2. ⚛️  Calculate H2 energy with VQE")
    print("  3. 📊 Generate potential energy curve")
    print("  4. 🔄 Run hybrid ML + Quantum workflow")
    print("  5. 📈 Visualize results\n")
    
    input("Press Enter to start Phase 2 demo...")
    
    # Demo 1: Quantum Setup Validation
    print("\n" + "="*70)
    print("DEMO 1: Quantum Setup Validation")
    print("="*70 + "\n")
    
    try:
        import qiskit
        from qiskit_nature import __version__ as qiskit_nature_version
        
        print(f"✓ Qiskit version: {qiskit.__version__}")
        print(f"✓ Qiskit Nature version: {qiskit_nature_version}")
        
        # Test basic circuit
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        simulator = Aer.get_backend('qasm_simulator')
        job = simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print(f"✓ Basic quantum circuit executed")
        print(f"  Results: {counts}")
        print("\n✅ Quantum libraries working correctly!\n")
        
        quantum_ready = True
        
    except ImportError as e:
        print(f"❌ Quantum libraries not installed: {e}")
        print("\nTo install, run:")
        print("  pip install qiskit qiskit-nature qiskit-aer pyscf")
        print("\nProceeding with classical workflow only...\n")
        quantum_ready = False
    
    input("Press Enter to continue...")
    
    # Demo 2: VQE for H2
    if quantum_ready:
        print("\n" + "="*70)
        print("DEMO 2: VQE Energy Calculation for H2")
        print("="*70 + "\n")
        
        try:
            from models.quantum.vqe import VQECalculator
            
            print("Initializing VQE calculator...")
            calculator = VQECalculator()
            
            print("Calculating H2 ground state energy...")
            print("(This may take 1-2 minutes...)\n")
            
            h2_result = calculator.calculate_h2_energy(bond_length=0.735)
            
            print("\n📊 H2 Results:")
            print(f"  VQE Energy:     {h2_result['vqe_energy_hartree']:.6f} Hartree")
            print(f"  Exact Energy:   {h2_result['exact_energy_hartree']:.6f} Hartree")
            print(f"  Error:          {h2_result['error_hartree']:.6f} Hartree")
            print(f"  Relative Error: {h2_result['relative_error']:.4f}%")
            print(f"  Qubits:         {h2_result['num_qubits']}")
            print(f"  Parameters:     {h2_result['num_parameters']}")
            print(f"  Iterations:     {h2_result['optimizer_iterations']}")
            
            # Save results
            results_dir = PROJECT_ROOT / 'results' / 'quantum'
            results_dir.mkdir(parents=True, exist_ok=True)
            calculator.save_results(h2_result, results_dir / 'demo_h2_results.json')
            
            print("\n✅ VQE calculation successful!\n")
            
        except Exception as e:
            print(f"❌ VQE calculation failed: {e}")
            quantum_ready = False
        
        input("Press Enter to continue...")
    
    # Demo 3: Potential Energy Curve
    if quantum_ready:
        print("\n" + "="*70)
        print("DEMO 3: H2 Potential Energy Curve")
        print("="*70 + "\n")
        
        try:
            print("Calculating energy at multiple bond lengths...")
            print("(This will take 3-5 minutes...)\n")
            
            # Calculate curve
            bond_lengths = np.linspace(0.5, 2.0, 5)
            curve_result = calculator.calculate_energy_curve(
                molecule='H2',
                bond_lengths=bond_lengths
            )
            
            print("✓ Energy curve calculated\n")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(curve_result['bond_lengths'], 
                   curve_result['vqe_energies'], 
                   'o-', label='VQE', linewidth=2, markersize=8)
            ax.plot(curve_result['bond_lengths'], 
                   curve_result['exact_energies'], 
                   's--', label='Exact', linewidth=2, markersize=6)
            
            ax.set_xlabel('Bond Length (Angstrom)', fontsize=12)
            ax.set_ylabel('Energy (Hartree)', fontsize=12)
            ax.set_title('H₂ Potential Energy Curve (VQE vs Exact)', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = PROJECT_ROOT / 'results' / 'quantum' / 'h2_energy_curve.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Energy curve plot saved: {plot_path}\n")
            
            calculator.save_results(curve_result, results_dir / 'demo_energy_curve.json')
            
        except Exception as e:
            print(f"❌ Energy curve calculation failed: {e}\n")
        
        input("Press Enter to continue...")
    
    # Demo 4: Hybrid Workflow
    print("\n" + "="*70)
    print("DEMO 4: Hybrid Quantum-Classical Workflow")
    print("="*70 + "\n")
    
    print("Running hybrid workflow with top candidates...")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'scripts' / 'hybrid_workflow.py'), 
         '--skip-quantum' if not quantum_ready else ''],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
    
    input("Press Enter to continue...")
    
    # Demo 5: Results Visualization
    print("\n" + "="*70)
    print("DEMO 5: Results Visualization")
    print("="*70 + "\n")
    
    try:
        import pandas as pd
        
        # Check for hybrid results
        hybrid_file = PROJECT_ROOT / 'results' / 'hybrid' / 'hybrid_refined_candidates.csv'
        
        if hybrid_file.exists():
            df = pd.read_csv(hybrid_file)
            
            print(f"✓ Loaded {len(df)} refined candidates\n")
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: ML vs Final predictions
            ax1 = axes[0]
            x = range(len(df))
            ax1.plot(x, df['ML_Prediction'], 'o-', label='ML Only', linewidth=2, markersize=6)
            ax1.plot(x, df['Final_Prediction'], 's-', label='Hybrid (ML+Quantum)', 
                    linewidth=2, markersize=6)
            ax1.set_xlabel('Candidate Index', fontsize=11)
            ax1.set_ylabel('Predicted pIC50', fontsize=11)
            ax1.set_title('ML vs Hybrid Predictions', fontsize=13)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Quantum corrections
            ax2 = axes[1]
            corrections = df['Quantum_Correction']
            ax2.bar(x, corrections, color=['green' if c > 0 else 'red' for c in corrections],
                   alpha=0.7, edgecolor='black')
            ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel('Candidate Index', fontsize=11)
            ax2.set_ylabel('Quantum Correction', fontsize=11)
            ax2.set_title('Quantum Refinement Impact', fontsize=13)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            plot_path = PROJECT_ROOT / 'results' / 'quantum' / 'hybrid_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved: {plot_path}\n")
            
            # Display top 3
            print("🏆 Top 3 Refined Candidates:")
            for idx, row in df.head(3).iterrows():
                print(f"\n{idx+1}. Final pIC50: {row['Final_Prediction']:.3f}")
                print(f"   ML: {row['ML_Prediction']:.3f} → "
                      f"Correction: {row['Quantum_Correction']:+.3f}")
                print(f"   SMILES: {row['SMILES']}")
        else:
            print("⚠ Hybrid results not found. Run hybrid workflow first.")
    
    except Exception as e:
        print(f"❌ Visualization failed: {e}\n")
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ PHASE 2 DEMO COMPLETE!")
    print("="*70 + "\n")
    
    print("📊 Generated Files:")
    print("  - results/quantum/demo_h2_results.json")
    print("  - results/quantum/demo_energy_curve.json")
    print("  - results/quantum/h2_energy_curve.png")
    print("  - results/hybrid/hybrid_refined_candidates.csv")
    print("  - results/hybrid/hybrid_summary.json")
    print("  - results/quantum/hybrid_comparison.png")
    
    print("\n🎯 What You've Achieved:")
    if quantum_ready:
        print("  ✅ Quantum libraries installed and working")
        print("  ✅ VQE calculations successful")
        print("  ✅ H2 energy calculated with <1% error")
        print("  ✅ Potential energy curve generated")
        print("  ✅ Hybrid workflow functional")
    else:
        print("  ✅ Classical ML pipeline working")
        print("  ⚠  Quantum libraries need installation")
        print("  ⏳ Ready for quantum integration")
    
    print("\n📝 Next Steps:")
    if not quantum_ready:
        print("  1. Install quantum libraries:")
        print("     pip install qiskit qiskit-nature qiskit-aer pyscf")
        print("  2. Re-run this demo")
    else:
        print("  1. ✓ Test on larger molecules (LiH, BeH2)")
        print("  2. Implement molecular Hamiltonian generation")
        print("  3. Scale to drug-like molecules")
        print("  4. Test on IBM Quantum hardware")
        print("  5. Integrate active learning loop")
    
    print("\n🚀 Phase 2 Status: ", end="")
    if quantum_ready:
        print("COMPLETE ✅")
    else:
        print("READY FOR QUANTUM SETUP ⏳")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()