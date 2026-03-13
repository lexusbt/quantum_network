"""
IQM Quantum Hardware QAOA Solver
Updated for iqm-client[qiskit] API with scipy optimizer
"""
import numpy as np
from qiskit import transpile
from scipy.optimize import minimize
from iqm.qiskit_iqm import IQMProvider  # ✅ Only this import
from typing import Dict, Optional
import logging
import time
import pickle
from pathlib import Path

from src.qaoa_optimization.circuit_builder import QAOACircuitBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IQMQAOASolver:
    """
    QAOA solver using IQM Resonance quantum hardware
    Supports IQM Sirius (20 qubits)
    """
    
    def __init__(
        self,
        iqm_server_url: str = "https://resonance.meetiqm.com/",
        quantum_computer: str = "sirius",
        token: Optional[str] = None,
        shots: int = 1024,
        max_iterations: int = 30
    ):
        """
        Initialize IQM QAOA solver
        
        Args:
            iqm_server_url: IQM Resonance server URL (default: https://resonance.meetiqm.com/)
            quantum_computer: Quantum computer name (default: sirius)
            token: IQM authentication token (will prompt if not provided)
            shots: Number of measurement shots
            max_iterations: Maximum optimization iterations
        """
        self.iqm_server_url = iqm_server_url
        self.quantum_computer = quantum_computer
        self.shots = shots
        self.max_iterations = max_iterations
        
        # Get token if not provided
        if token is None:
            token = input("Enter your IQM Resonance token: ").strip()
        
        # Connect to IQM
        logger.info(f"Connecting to IQM: {iqm_server_url} ({quantum_computer})")
        
        try:
            # Create IQM provider (per IQM sample notebook)
            self.provider = IQMProvider(
                iqm_server_url,
                quantum_computer=quantum_computer,
                token=token
            )
            self.backend = self.provider.get_backend()
            
            # Get backend properties
            logger.info(f"✓ Connected to IQM backend")
            logger.info(f"  Backend: {self.backend.name}")
            
            # IQM Sirius has 16 qubits (from IQM documentation)
            # Backend object doesn't expose num_qubits property
            self.max_qubits = 16
            logger.info(f"  Qubits: {self.max_qubits}")
            
        except Exception as e:
            logger.error(f"Failed to connect to IQM: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Failed to connect to IQM: {e}")
            raise
        
        self.iteration_count = 0
        self.cost_history = []
    
    def solve(
        self,
        Q: np.ndarray,
        p: int = 1,
        initial_params: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Solve QUBO using IQM quantum hardware
        
        Args:
            Q: QUBO matrix (must be <= max_qubits)
            p: Number of QAOA layers
            initial_params: Initial parameters
            verbose: Print progress
            
        Returns:
            Results dictionary
        """
        start_time = time.time()
        n_qubits = Q.shape[0]
        
        if n_qubits > self.max_qubits:
            raise ValueError(
                f"Problem size ({n_qubits} qubits) exceeds "
                f"IQM backend capacity ({self.max_qubits} qubits)"
            )
        
        if verbose:
            logger.info(f"Solving QUBO on IQM: {n_qubits} qubits, p={p} layers")
        
        # Build QAOA circuit
        builder = QAOACircuitBuilder(n_qubits)
        qc, gamma_params, beta_params = builder.build_qaoa_circuit(Q, p)
        
        # Initial parameters
        if initial_params is None:
            initial_params = np.concatenate([
                np.random.uniform(0, 2*np.pi, p),  # gamma
                np.random.uniform(0, np.pi, p)      # beta
            ])
        
        self.iteration_count = 0
        self.cost_history = []
        
        def cost_function(params):
            """Cost function using IQM hardware"""
            self.iteration_count += 1
            
            # Bind parameters
            gamma_vals = params[:p]
            beta_vals = params[p:]
            
            param_dict = {}
            for i in range(p):
                param_dict[gamma_params[i]] = gamma_vals[i]
                param_dict[beta_params[i]] = beta_vals[i]
            
            bound_circuit = qc.assign_parameters(param_dict)
            
            # Transpile for IQM backend
            transpiled = transpile(
                bound_circuit,
                backend=self.backend,
                optimization_level=3
            )
            
            if verbose:
                logger.info(
                    f"Iteration {self.iteration_count}: "
                    f"Submitting to IQM (depth={transpiled.depth()}, "
                    f"gates={len(transpiled)})"
                )
            
            # Run on IQM hardware
            job = self.backend.run(transpiled, shots=self.shots, use_timeslot=False)
            
            # Wait for result
            result = job.result()
            counts = result.get_counts()
            
            # Calculate expectation value
            expectation = 0.0
            total_shots = sum(counts.values())
            
            for bitstring, count in counts.items():
                x = np.array([int(bit) for bit in bitstring[::-1]])
                cost = x @ Q @ x
                prob = count / total_shots
                expectation += prob * cost
            
            self.cost_history.append(expectation)
            
            if verbose:
                logger.info(f"  Result: cost = {expectation:.4f}")
            
            return expectation
        
        # Run optimization using scipy
        if verbose:
            logger.info("Starting optimization on IQM quantum hardware...")
            logger.info(f"Max iterations: {self.max_iterations}")

        result = minimize(  # ✅ Use scipy.optimize.minimize
            fun=cost_function,
            x0=initial_params,
            method='COBYLA',
            options={'maxiter': self.max_iterations}
        )
        
        # Get best solution from final run
        best_bitstring, best_cost = self._get_best_solution(
            Q, result.x, p, qc, gamma_params, beta_params
        )
        
        elapsed_time = time.time() - start_time
        
        results = {
            'optimal_params': result.x,
            'optimal_cost': result.fun,
            'best_bitstring': best_bitstring,
            'best_cost': best_cost,
            'iterations': self.iteration_count,
            'cost_history': self.cost_history,
            'elapsed_time': elapsed_time,
            'p_layers': p,
            'n_qubits': n_qubits,
            'backend': 'IQM_Resonance',
            'shots': self.shots,
            'converged': result.success if hasattr(result, 'success') else True
        }
        
        if verbose:
            logger.info(f"Optimization complete in {elapsed_time:.2f}s")
            logger.info(f"Best cost: {best_cost:.4f}")
        
        return results
    
    def _get_best_solution(self, Q, params, p, qc, gamma_params, beta_params):
        """Run final circuit and extract best bitstring"""
        # Bind parameters
        gamma_vals = params[:p]
        beta_vals = params[p:]
        
        param_dict = {}
        for i in range(p):
            param_dict[gamma_params[i]] = gamma_vals[i]
            param_dict[beta_params[i]] = beta_vals[i]
        
        bound_circuit = qc.assign_parameters(param_dict)
        
        # Transpile
        transpiled = transpile(
            bound_circuit,
            backend=self.backend,
            optimization_level=3
        )
        
        # Run with more shots for final measurement
        job = self.backend.run(transpiled, shots=self.shots * 2, use_timeslot=False)
        result = job.result()
        counts = result.get_counts()
        
        # Find best bitstring
        best_bitstring = None
        best_cost = float('inf')
        
        for bitstring, count in counts.items():
            x = np.array([int(bit) for bit in bitstring[::-1]])
            cost = x @ Q @ x
            
            if cost < best_cost:
                best_cost = cost
                best_bitstring = bitstring
        
        return best_bitstring, best_cost
    
    def solve_instance(
        self,
        instance_path: str,
        p: int = 1,
        initial_params: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """Solve from instance file"""
        with open(instance_path, 'rb') as f:
            instance = pickle.load(f)
        
        metadata = instance['metadata']
        Q = instance['qubo_matrix']
        
        if verbose:
            logger.info(f"Loaded instance: {metadata['instance_id']}")
            logger.info(f"  Dataset: {metadata['dataset_name']}")
            logger.info(f"  Nodes: {metadata['n_nodes']}, Qubits: {metadata['n_qubits']}")
            logger.info(f"  Classical optimal: {metadata['classical_optimal_length']} hops")
        
        result = self.solve(Q, p, initial_params, verbose)
        
        # Add instance metadata
        result['instance_id'] = metadata['instance_id']
        result['dataset_name'] = metadata['dataset_name']
        result['classical_optimal'] = metadata['classical_optimal_length']
        if metadata['classical_optimal_length'] > 0:
            result['approximation_ratio'] = abs(result['best_cost']) / metadata['classical_optimal_length']
        else:
            result['approximation_ratio'] = None
        
        return result


def main():
    """Test IQM QAOA solver"""
    print("="*70)
    print("IQM RESONANCE QAOA SOLVER TEST")
    print("="*70)
    
    # Get IQM credentials
    print("\nEnter your IQM Resonance server URL:")
    print("Example: https://cocos.resonance.meetiqm.com")
    iqm_url = input("URL: ").strip()
    
    if not iqm_url:
        print("\n✗ No URL provided.")
        print("\nTo get your IQM URL:")
        print("1. Log in to https://resonance.meetiqm.com/")
        print("2. Navigate to your quantum computer")
        print("3. Copy the server URL from settings")
        return
    
    # Simple 4-qubit test
    Q = np.array([
        [-2,  1,  1,  0],
        [ 1, -2,  1,  1],
        [ 1,  1, -2,  1],
        [ 0,  1,  1, -2]
    ])
    
    print("\nTest QUBO (4 qubits):")
    print(Q)
    print("\nConnecting to IQM Resonance...")
    
    try:
        solver = IQMQAOASolver(
            iqm_server_url=iqm_url,
            shots=1024,
            max_iterations=10
        )
        
        print("\n✓ Connected! Running QAOA on IQM quantum hardware...")
        
        results = solver.solve(Q, p=1, verbose=True)
        
        print("\n" + "="*70)
        print("RESULTS FROM IQM QUANTUM HARDWARE")
        print("="*70)
        print(f"Best bitstring: {results['best_bitstring']}")
        print(f"Best cost: {results['best_cost']:.4f}")
        print(f"Optimal cost: {results['optimal_cost']:.4f}")
        print(f"Iterations: {results['iterations']}")
        print(f"Time: {results['elapsed_time']:.2f}s")
        
        print("\n✓ Success! IQM QAOA solver working!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("1. Check your IQM Resonance URL is correct")
        print("2. Verify you have access to the quantum computer")
        print("3. Check your network connection")


if __name__ == "__main__":
    main()