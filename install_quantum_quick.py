"""
Quick Quantum Installation (WITHOUT PySCF)
Fast setup for time-constrained situations

Usage:
    python install_quantum_quick.py
"""

import subprocess
import sys

def install_package(package):
    """Install package with pip"""
    print(f"\nInstalling {package}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            package, "--quiet"
        ])
        print(f"✓ {package}")
        return True
    except Exception as e:
        print(f"✗ {package}: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("QRUCIBLE - QUICK QUANTUM SETUP (NO PySCF)")
    print("="*70 + "\n")
    
    print("Installing quantum libraries (2-3 minutes)...")
    print("Note: Skipping PySCF for faster installation\n")
    
    # Essential packages only
    packages = [
        "qiskit",
        "qiskit-aer",
        "requests",  # For PDB API
    ]
    
    success_count = 0
    
    for pkg in packages:
        if install_package(pkg):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"INSTALLATION COMPLETE: {success_count}/{len(packages)} packages")
    print("="*70)
    
    # Test imports
    print("\nTesting imports...")
    
    try:
        import qiskit
        print(f"✓ Qiskit {qiskit.__version__}")
    except:
        print("✗ Qiskit")
    
    try:
        from qiskit_aer import Aer
        print("✓ Qiskit Aer")
    except:
        print("✗ Qiskit Aer")
    
    try:
        import requests
        print("✓ Requests (for PDB API)")
    except:
        print("✗ Requests")
    
    # Test quantum circuit
    print("\nTesting quantum circuit...")
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        simulator = Aer.get_backend('qasm_simulator')
        job = simulator.run(qc, shots=100)
        result = job.result()
        
        print("✓ Quantum circuit executed successfully!")
        print(f"  Result: {result.get_counts()}")
    except Exception as e:
        print(f"✗ Quantum test failed: {e}")
    
    print("\n" + "="*70)
    print("READY FOR PHASE 2!")
    print("="*70)
    print("\nNext steps:")
    print("  1. python run_phase2_demo.py")
    print("  2. python scripts/hybrid_workflow.py")
    print("  3. streamlit run app.py")
    print("\nNote: Using simplified VQE (no PySCF molecular driver)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()