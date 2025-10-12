"""
Quantum VQE Demo for H2 Molecule
Demonstrates quantum energy calculation using VQE

Usage:
    python scripts/quantum_vqe_demo.py
"""

import numpy as np
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("QRUCIBLE - PHASE 2: VQE FOR H2 MOLECULE")
    print("="*70 + "\n")
    
    try:
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
        from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
        from qiskit.primitives import Estimator
        from qiskit.algorithms.optimizers import SLSQP
        from qiskit.algorithms.minimum_eigensolvers import VQE
        from qiskit_aer import Aer
        
        print("✓ All quantum libraries imported successfully\n")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease run: python scripts/setup_quantum.py")
        return
    
    # Define H2 molecule
    print("Step 1: Defining H2 molecule...")
    print("  Bond length: 0.735 Angstrom")
    print("  Basis set: sto-3g\n")
    
    driver = PySCFDriver(
        atom='H 0 0 0; H 0 0 0.735',
        basis='sto-3g',
        charge=0,
        spin=0
    )
    
    # Get problem
    print("Step 2: Generating electronic structure problem...")
    problem = driver.run()
    hamiltonian = problem.hamiltonian.second_q_op()
    
    print(f"  Number of qubits: {problem.num_spatial_orbitals * 2}")
    print(f"  Number of particles: {problem.num_particles}\n")
    
    # Map to qubits
    print("Step 3: Mapping to qubits (Jordan-Wigner)...")
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(hamiltonian)
    
    print(f"  Qubit operator terms: {len(qubit_op)}\n")
    
    # Initial state
    print("Step 4: Preparing initial state (Hartree-Fock)...")
    num_qubits = problem.num_spatial_orbitals * 2
    num_particles = problem.num_particles
    
    init_state = HartreeFock(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )
    
    print(f"  Initial state prepared\n")
    
    # Ansatz
    print("Step 5: Creating ansatz circuit (UCCSD)...")
    ansatz = UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=init_state
    )
    
    print(f"  Ansatz parameters: {ansatz.num_parameters}\n")
    
    # VQE
    print("Step 6: Running VQE optimization...")
    print("  Optimizer: SLSQP")
    print("  Backend: Aer Simulator")
    print("  This may take 1-2 minutes...\n")
    
    optimizer = SLSQP(maxiter=100)
    estimator = Estimator()
    
    vqe = VQE(estimator, ansatz, optimizer)
    
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    # Results
    print("\n" + "="*70)
    print("VQE RESULTS")
    print("="*70 + "\n")
    
    vqe_energy = result.eigenvalue.real
    print(f"VQE Ground State Energy:     {vqe_energy:.6f} Hartree")
    
    # Compare with exact (FCI)
    try:
        from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
        numpy_solver = NumPyMinimumEigensolver()
        exact_result = numpy_solver.compute_minimum_eigenvalue(qubit_op)
        exact_energy = exact_result.eigenvalue.real
        
        print(f"Exact Ground State Energy:   {exact_energy:.6f} Hartree")
        
        error = abs(vqe_energy - exact_energy)
        print(f"\nAbsolute Error:              {error:.6f} Hartree")
        print(f"Relative Error:              {error/abs(exact_energy)*100:.4f}%")
        
        # Convert to more familiar units
        hartree_to_ev = 27.2114
        print(f"\nVQE Energy:                  {vqe_energy * hartree_to_ev:.4f} eV")
        print(f"Exact Energy:                {exact_energy * hartree_to_ev:.4f} eV")
        
    except Exception as e:
        print(f"\n⚠ Could not compute exact energy: {e}")
    
    print(f"\nOptimizer iterations:        {result.optimizer_result.nfev}")
    print(f"Optimal parameters:          {len(result.optimal_parameters)} values")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")
    
    results_dir = Path('results/quantum')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'molecule': 'H2',
        'bond_length': 0.735,
        'vqe_energy_hartree': float(vqe_energy),
        'vqe_energy_ev': float(vqe_energy * 27.2114),
        'num_qubits': num_qubits,
        'num_parameters': ansatz.num_parameters,
        'optimizer_iterations': result.optimizer_result.nfev
    }
    
    import json
    with open(results_dir / 'h2_vqe_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✓ Results saved to: {results_dir / 'h2_vqe_results.json'}")
    
    print("\n" + "="*70)
    print("VQE DEMO COMPLETE!")
    print("="*70)
    print("\n✓ Successfully calculated H2 ground state energy")
    print("✓ Quantum circuit executed on simulator")
    print("✓ Results saved for comparison")
    
    print("\nNext steps:")
    print("  1. Try different molecules (LiH, BeH2)")
    print("  2. Test on real quantum hardware")
    print("  3. Integrate with ML pipeline")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()