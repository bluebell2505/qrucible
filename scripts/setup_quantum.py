"""
Phase 2 Quantum Setup Script
Installs and verifies quantum computing libraries

Usage:
    python scripts/setup_quantum.py
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("\n" + "="*70)
    print("QRUCIBLE - PHASE 2 QUANTUM SETUP")
    print("="*70 + "\n")
    
    # Packages to install
    packages = [
        "qiskit>=1.0.0",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "qiskit-nature",
        "pyscf",
        "pennylane",
        "pennylane-qiskit"
    ]
    
    print("Installing quantum computing packages...")
    print("This may take 5-10 minutes...\n")
    
    for package in packages:
        try:
            install_package(package)
            print(f"✓ {package} installed\n")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}\n")
    
    # Verify installations
    print("\n" + "="*70)
    print("VERIFYING INSTALLATIONS")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test Qiskit
    try:
        import qiskit
        print(f"✓ Qiskit version: {qiskit.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"✗ Qiskit import failed: {e}")
        tests_failed += 1
    
    # Test Qiskit Aer
    try:
        from qiskit_aer import Aer
        print(f"✓ Qiskit Aer available")
        tests_passed += 1
    except ImportError:
        print(f"✗ Qiskit Aer not available")
        tests_failed += 1
    
    # Test Qiskit Nature
    try:
        import qiskit_nature
        print(f"✓ Qiskit Nature version: {qiskit_nature.__version__}")
        tests_passed += 1
    except ImportError:
        print(f"✗ Qiskit Nature not available")
        tests_failed += 1
    
    # Test PennyLane
    try:
        import pennylane as qml
        print(f"✓ PennyLane version: {qml.__version__}")
        tests_passed += 1
    except ImportError:
        print(f"✗ PennyLane not available")
        tests_failed += 1
    
    # Test PySCF
    try:
        import pyscf
        print(f"✓ PySCF version: {pyscf.__version__}")
        tests_passed += 1
    except ImportError:
        print(f"✗ PySCF not available (optional)")
    
    # Run basic quantum circuit
    print("\n" + "="*70)
    print("TESTING BASIC QUANTUM CIRCUIT")
    print("="*70 + "\n")
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        
        # Create simple circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Run simulation
        simulator = Aer.get_backend('qasm_simulator')
        job = simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        print("✓ Basic quantum circuit executed successfully")
        print(f"  Results: {counts}")
        tests_passed += 1
        
    except Exception as e:
        print(f"✗ Quantum circuit test failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    print(f"\nTests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✓ Quantum setup complete!")
        print("\nNext steps:")
        print("  1. Run VQE demo: python scripts/quantum_vqe_demo.py")
        print("  2. Test H2 molecule: python scripts/test_h2_molecule.py")
        print("  3. Integrate with ML pipeline")
    else:
        print("\n⚠ Some tests failed. Please check error messages above.")
        print("\nTry manual installation:")
        print("  pip install qiskit qiskit-nature qiskit-aer")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()