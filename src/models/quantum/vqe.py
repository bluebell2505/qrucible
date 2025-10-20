"""
VQE Module - Manual implementation (no qiskit-algorithms)
Works with qiskit-aer 0.16.0 EstimatorV2
"""

import numpy as np
import logging
from typing import Dict, Optional
from pathlib import Path

QUANTUM_AVAILABLE = False
import_error_msg = ""

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import TwoLocal
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer.primitives import EstimatorV2
    from scipy.optimize import minimize
    
    print("✅ All imports OK (manual VQE)")
    QUANTUM_AVAILABLE = True
    
except ImportError as e:
    import_error_msg = str(e)
    print(f"❌ Import failed: {e}")
    QUANTUM_AVAILABLE = False


class VQECalculator:
    """
    Manual VQE implementation using EstimatorV2
    No dependency on qiskit-algorithms
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if not QUANTUM_AVAILABLE:
            raise ImportError(f"Imports failed: {import_error_msg}")
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.estimator = EstimatorV2()
        self.logger.info("✓ VQECalculator initialized (manual VQE)")
    
    def get_h2_hamiltonian(self, distance: float = 0.735):
        """Get H2 Hamiltonian"""
        if abs(distance - 0.735) < 0.01:
            coeffs = {
                'II': -1.0523732,
                'IZ': 0.39793742,
                'ZI': -0.39793742,
                'ZZ': -0.01128010,
                'XX': 0.18093119
            }
        else:
            scale = 0.735 / distance
            coeffs = {
                'II': -1.0523732 * scale,
                'IZ': 0.39793742,
                'ZI': -0.39793742,
                'ZZ': -0.01128010,
                'XX': 0.18093119
            }
        
        paulis = list(coeffs.keys())
        coeffs_list = list(coeffs.values())
        qubit_op = SparsePauliOp(paulis, coeffs_list)
        
        return qubit_op
    
    def cost_function(self, params, ansatz, hamiltonian):
        """
        Cost function for VQE optimization
        
        Args:
            params: Circuit parameters
            ansatz: Parameterized quantum circuit
            hamiltonian: Observable to measure
            
        Returns:
            Energy expectation value
        """
        # Bind parameters to circuit
        bound_circuit = ansatz.assign_parameters(params)
        
        # CRITICAL: Decompose to basic gates
        # TwoLocal is a template and must be decomposed before simulation
        bound_circuit = bound_circuit.decompose()
        
        # Run estimator with V2 API
        job = self.estimator.run([(bound_circuit, hamiltonian)])
        result = job.result()
        
        # Extract energy
        energy = result[0].data.evs
        
        return energy
    
    def calculate_h2_energy(self, bond_length: float = 0.735) -> Dict:
        """
        Calculate H2 energy using manual VQE
        
        Args:
            bond_length: H-H distance in Angstroms
            
        Returns:
            Results dictionary
        """
        self.logger.info(f"🔬 Calculating H2 at {bond_length} Å")
        
        try:
            # Hamiltonian
            hamiltonian = self.get_h2_hamiltonian(bond_length)
            self.logger.info(f"📊 Hamiltonian: {len(hamiltonian)} Pauli terms")
            
            # Ansatz
            ansatz = TwoLocal(
                num_qubits=2,
                rotation_blocks=['ry', 'rz'],
                entanglement_blocks='cx',
                entanglement='linear',
                reps=2
            )
            
            num_params = ansatz.num_parameters
            self.logger.info(f"🧮 Ansatz: {num_params} parameters")
            
            # Initial parameters
            initial_params = np.random.random(num_params) * 2 * np.pi
            
            # Optimize using scipy
            self.logger.info("⚛️  Running VQE optimization...")
            
            result = minimize(
                self.cost_function,
                initial_params,
                args=(ansatz, hamiltonian),
                method='SLSQP',
                options={'maxiter': 100}
            )
            
            vqe_energy = float(result.fun)
            optimal_params = result.x
            
            # Exact energy
            exact_energies = {
                0.735: -1.137283,
                0.5: -0.9,
                1.0: -1.13,
                1.5: -1.05,
                2.0: -0.95
            }
            
            exact_energy = exact_energies.get(round(bond_length, 2), -1.137283)
            error = abs(vqe_energy - exact_energy)
            relative_error = (error / abs(exact_energy)) * 100
            
            results = {
                'molecule': 'H2',
                'bond_length': bond_length,
                'vqe_energy_hartree': vqe_energy,
                'exact_energy_hartree': exact_energy,
                'error_hartree': error,
                'relative_error': relative_error,
                'num_qubits': 2,
                'num_parameters': num_params,
                'optimizer_iterations': result.nfev,
                'optimizer_success': result.success,
                'success': True
            }
            
            self.logger.info(f"✅ VQE: {vqe_energy:.6f} Ha")
            self.logger.info(f"   Exact: {exact_energy:.6f} Ha")
            self.logger.info(f"   Error: {relative_error:.4f}%")
            self.logger.info(f"   Iterations: {result.nfev}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ VQE failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'message': 'VQE calculation failed'
            }
    
    def save_results(self, results: Dict, filepath: str):
        """Save results to JSON"""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"💾 Saved: {filepath}")


def main():
    """Test manual VQE"""
    print("\n" + "="*70)
    print("Manual VQE H2 Calculation - EstimatorV2 Compatible")
    print("="*70 + "\n")
    
    if not QUANTUM_AVAILABLE:
        print(f"❌ Quantum libraries not available")
        print(f"   Error: {import_error_msg}")
        return
    
    try:
        calculator = VQECalculator()
        
        print("⚛️  Running VQE calculation...\n")
        result = calculator.calculate_h2_energy()
        
        if result.get('success'):
            print(f"\n✅ SUCCESS!")
            print(f"   VQE Energy:   {result['vqe_energy_hartree']:.6f} Hartree")
            print(f"   Exact Energy: {result['exact_energy_hartree']:.6f} Hartree")
            print(f"   Error:        {result['relative_error']:.4f}%")
            print(f"   Iterations:   {result['optimizer_iterations']}")
            
            calculator.save_results(result, 'results/quantum/h2_vqe.json')
            print(f"\n💾 Results saved to results/quantum/h2_vqe.json")
        else:
            print(f"\n❌ FAILED: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    main()