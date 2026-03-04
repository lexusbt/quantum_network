"""
QAOA Solver for Network Routing Optimization
Main solver class that runs variational quantum optimization
"""

import numpy as np
from qiskit import transpile
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
from qiskit_aer import AerSimulator
from typing import Dict, List, Tuple, Optional
import time
import logging
import pickle
from pathlib import Path

from src.qaoa_optimization.circuit_builder import QAOACircuitBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAOASolver:
    """
    QAOA solver for QUBO routing problems
    
    Implements variational quantum optimization to find
    approximate solutions to network routing problems
    """
    
    def __init__(
        self,
        backend: str = "qasm_simulator",
        optimizer: str = "COBYLA",
        shots: int = 1024,
        max_iterations: int = 100
    ):
        """
        Initialize QAOA solver
        
        Args:
            backend: Quantum backend ('qasm_simulator', 'statevector_simulator', or IBM device)
            optimizer: Classical optimizer ('COBYLA', 'SPSA', 'L_BFGS_B')
            shots: Number of measurement shots
            max_iterations: Maximum optimization iterations
        """
        self.backend_name = backend
        self.shots = shots
        self.max_iterations = max_iterations
        
        # Set up backend
        if backend == "qasm_simulator":
            self.backend = AerSimulator(method='automatic')
        elif backend == "statevector_simulator":
            self.backend = AerSimulator(method='statevector')
        else:
            # For real devices, this would connect to IBM Quantum
            raise ValueError(f"Backend {backend} not yet implemented")
        
        # Set up optimizer
        self.optimizer = self._get_optimizer(optimizer)
        self.optimizer_name = optimizer
        
        # Tracking
        self.iteration_count = 0
        self.cost_history = []
        
        logger.info(f"Initialized QAOA solver: {optimizer} optimizer, {backend} backend, {shots} shots")
    
    def _get_optimizer(self, name: str):
        """Get classical optimizer instance"""
        if name == "COBYLA":
            return COBYLA(maxiter=self.max_iterations, disp=False)
        elif name == "SPSA":
            return SPSA(maxiter=self.max_iterations)
        elif name == "L_BFGS_B":
            return L_BFGS_B(maxiter=self.max_iterations)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def solve(
        self,
        Q: np.ndarray,
        p: int = 1,
        initial_params: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Solve QUBO problem using QAOA
        
        Args:
            Q: QUBO matrix (n × n)
            p: Number of QAOA layers
            initial_params: Initial parameter values (if None, random)
            verbose: Print progress
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        n_qubits = Q.shape[0]
    
        if verbose:
            logger.info(f"Solving QUBO: {n_qubits} qubits, p={p} layers")
        
        # Build circuit
        builder = QAOACircuitBuilder(n_qubits)
        qc, gamma_params, beta_params = builder.build_qaoa_circuit(Q, p)
        
        # Initial parameters
        if initial_params is None:
            # Random initialization in reasonable range
            initial_params = np.random.uniform(0, 2*np.pi, 2*p)
        
        # Reset tracking
        self.iteration_count = 0
        self.cost_history = []
        
        # Define cost function
        def cost_function(params):
            """Evaluate QAOA cost for given parameters"""
            self.iteration_count += 1
            
            # Split params into gamma and beta
            gamma_vals = params[:p]
            beta_vals = params[p:]
            
            # Bind parameters to circuit
            param_dict = {}
            for i in range(p):
                param_dict[gamma_params[i]] = gamma_vals[i]
                param_dict[beta_params[i]] = beta_vals[i]
            
            bound_circuit = qc.assign_parameters(param_dict)
            
            # Transpile for simulator (no coupling map constraints)
            transpiled = transpile(
                bound_circuit,
                basis_gates=['u1', 'u2', 'u3', 'cx'],
                optimization_level=1
            )
            
            # Run circuit
            job = self.backend.run(transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Calculate expectation value of QUBO
            expectation = self._calculate_expectation(counts, Q)
            
            self.cost_history.append(expectation)
            
            if verbose and self.iteration_count % 10 == 0:
                logger.info(f"Iteration {self.iteration_count}: cost = {expectation:.4f}")
            
            return expectation
        
        # Run optimization
        if verbose:
            logger.info(f"Starting optimization with {self.optimizer_name}...")
        
        result = self.optimizer.minimize(
            fun=cost_function,
            x0=initial_params
        )
        
        # Get final solution
        optimal_params = result.x
        optimal_cost = result.fun
        
        # Extract best bitstring from final run
        best_bitstring, best_cost = self._get_best_solution(Q, optimal_params, p, qc, gamma_params, beta_params)
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        results = {
            'optimal_params': optimal_params,
            'optimal_cost': optimal_cost,
            'best_bitstring': best_bitstring,
            'best_cost': best_cost,
            'iterations': self.iteration_count,
            'cost_history': self.cost_history,
            'elapsed_time': elapsed_time,
            'p_layers': p,
            'n_qubits': n_qubits,
            'optimizer': self.optimizer_name,
            'shots': self.shots,
            'converged': result.success if hasattr(result, 'success') else True
        }
        
        if verbose:
            logger.info(f"Optimization complete in {elapsed_time:.2f}s")
            logger.info(f"Final cost: {optimal_cost:.4f}")
            logger.info(f"Best solution cost: {best_cost:.4f}")
        
        return results
    
    def _calculate_expectation(self, counts: Dict[str, int], Q: np.ndarray) -> float:
        """Calculate expectation value of QUBO from measurement counts"""
        total_shots = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Convert bitstring to binary vector
            x = np.array([int(bit) for bit in bitstring[::-1]])  # Reverse for Qiskit ordering
            
            # Calculate QUBO cost: x^T Q x
            cost = x @ Q @ x
            
            # Add weighted contribution
            expectation += (count / total_shots) * cost
        
        return expectation
    
    def _get_best_solution(
        self, 
        Q: np.ndarray, 
        params: np.ndarray, 
        p: int,
        qc, 
        gamma_params, 
        beta_params
    ) -> Tuple[str, float]:
        """Run final circuit and get best measured bitstring"""
        # Bind parameters
        gamma_vals = params[:p]
        beta_vals = params[p:]
        
        param_dict = {}
        for i in range(p):
            param_dict[gamma_params[i]] = gamma_vals[i]
            param_dict[beta_params[i]] = beta_vals[i]
        
        bound_circuit = qc.assign_parameters(param_dict)
        
        # Transpile without coupling map constraints
        transpiled = transpile(
            bound_circuit,
            basis_gates=['u1', 'u2', 'u3', 'cx'],
            optimization_level=1
        )
        
        # Run with more shots for final measurement
        job = self.backend.run(transpiled, shots=self.shots * 2)
        result = job.result()
        counts = result.get_counts()
        
        # Find bitstring with lowest cost
        best_bitstring = None
        best_cost = float('inf')
        
        for bitstring in counts.keys():
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
        verbose: bool = True
    ) -> Dict:
        """
        Solve a routing instance from file
        
        Args:
            instance_path: Path to instance .pkl file
            p: Number of QAOA layers
            verbose: Print progress
            
        Returns:
            Results dictionary with instance metadata
        """
        # Load instance
        with open(instance_path, 'rb') as f:
            instance = pickle.load(f)
        
        Q = instance['qubo_matrix']
        metadata = instance['metadata']
        
        if verbose:
            logger.info(f"Loaded instance: {metadata['instance_id']}")
            logger.info(f"  Dataset: {metadata['dataset_name']}")
            logger.info(f"  Nodes: {metadata['n_nodes']}, Qubits: {metadata['n_qubits']}")
            logger.info(f"  Classical optimal: {metadata['classical_optimal_length']} hops")
        
        # Solve
        results = self.solve(Q, p, verbose=verbose)
        
        # Add instance metadata
        results['instance_id'] = metadata['instance_id']
        results['dataset_name'] = metadata['dataset_name']
        results['classical_optimal'] = metadata['classical_optimal_length']
        
        # Calculate approximation ratio
        if metadata['classical_optimal_length'] > 0:
            results['approximation_ratio'] = results['best_cost'] / metadata['classical_optimal_length']
        else:
            results['approximation_ratio'] = None
        
        return results
    
    def batch_solve(
        self,
        instance_paths: List[str],
        p: int = 1,
        save_results: bool = True,
        output_dir: str = "results/qaoa"
    ) -> List[Dict]:
        """
        Solve multiple instances
        
        Args:
            instance_paths: List of paths to instance files
            p: Number of QAOA layers
            save_results: Save individual results
            output_dir: Directory to save results
            
        Returns:
            List of results dictionaries
        """
        results_list = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Batch solving {len(instance_paths)} instances with p={p}")
        
        for i, instance_path in enumerate(instance_paths, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Instance {i}/{len(instance_paths)}")
            logger.info(f"{'='*70}")
            
            try:
                results = self.solve_instance(instance_path, p, verbose=True)
                results_list.append(results)
                
                # Save individual result
                if save_results:
                    result_file = output_path / f"{results['instance_id']}_p{p}.pkl"
                    with open(result_file, 'wb') as f:
                        pickle.dump(results, f)
                
            except Exception as e:
                logger.error(f"Failed to solve {instance_path}: {e}")
                continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Batch complete: {len(results_list)}/{len(instance_paths)} successful")
        logger.info(f"{'='*70}")
        
        return results_list


def main():
    """Example usage"""
    # Create a simple test QUBO
    Q = np.array([
        [-2,  1,  1,  0],
        [ 1, -2,  1,  1],
        [ 1,  1, -2,  1],
        [ 0,  1,  1, -2]
    ])
    
    print("\nTest QUBO Matrix:")
    print(Q)
    
    # Solve with QAOA
    solver = QAOASolver(
        backend="qasm_simulator",
        optimizer="COBYLA",
        shots=1024,
        max_iterations=50
    )
    
    results = solver.solve(Q, p=2, verbose=True)
    
    print("\n" + "="*70)
    print("QAOA RESULTS")
    print("="*70)
    print(f"Optimal parameters: {results['optimal_params']}")
    print(f"Optimal cost: {results['optimal_cost']:.4f}")
    print(f"Best bitstring: {results['best_bitstring']}")
    print(f"Best cost: {results['best_cost']:.4f}")
    print(f"Iterations: {results['iterations']}")
    print(f"Time: {results['elapsed_time']:.2f}s")


if __name__ == "__main__":
    main()