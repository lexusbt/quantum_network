"""
Quantum Circuit Builder for QAOA
Constructs parameterized quantum circuits for QUBO optimization
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAOACircuitBuilder:
    """
    Build QAOA circuits for QUBO problems
    
    Circuit structure:
    1. Initial state: Equal superposition (Hadamard on all qubits)
    2. Problem Hamiltonian: Cost function encoding
    3. Mixer Hamiltonian: X rotations for quantum tunneling
    4. Measurement: Computational basis
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize circuit builder
        
        Args:
            n_qubits: Number of qubits needed for the problem
        """
        self.n_qubits = n_qubits
        logger.info(f"Initialized QAOA circuit builder for {n_qubits} qubits")
    
    def build_qaoa_circuit(
        self, 
        Q: np.ndarray, 
        p: int = 1,
        insert_barriers: bool = True,
        parameterized: bool = True
    ) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        """
        Build a p-layer QAOA circuit for QUBO matrix Q
        
        Args:
            Q: QUBO matrix (n_qubits × n_qubits)
            p: Number of QAOA layers (circuit depth)
            insert_barriers: Add barriers for visualization
            parameterized: Use parameters (True) or zeros (False)
            
        Returns:
            (circuit, gamma_params, beta_params)
        """
        if Q.shape != (self.n_qubits, self.n_qubits):
            raise ValueError(f"QUBO matrix shape {Q.shape} doesn't match n_qubits {self.n_qubits}")
        
        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Create parameter vectors
        if parameterized:
            gamma = ParameterVector('γ', p)  # Problem Hamiltonian angles
            beta = ParameterVector('β', p)   # Mixer Hamiltonian angles
        else:
            gamma = [0] * p
            beta = [0] * p
        
        # Initial state: Equal superposition
        qc.h(range(self.n_qubits))
        
        if insert_barriers:
            qc.barrier()
        
        # Build p layers
        for layer in range(p):
            # Problem Hamiltonian: U_P(γ)
            self._apply_problem_hamiltonian(qc, Q, gamma[layer])
            
            if insert_barriers:
                qc.barrier()
            
            # Mixer Hamiltonian: U_M(β)
            self._apply_mixer_hamiltonian(qc, beta[layer])
            
            if insert_barriers:
                qc.barrier()
        
        # Measurement
        qc.measure_all()
        
        logger.info(f"Built QAOA circuit: {p} layers, {self.n_qubits} qubits, depth={qc.depth()}")
        
        return qc, gamma, beta
    
    def _apply_problem_hamiltonian(self, qc: QuantumCircuit, Q: np.ndarray, gamma):
        """
        Apply problem Hamiltonian U_P(γ) = e^(-iγH_P)
        
        For QUBO: H_P = Σ Q_ij Z_i Z_j + Σ Q_ii Z_i
        """
        # Two-qubit terms: Q_ij Z_i Z_j (i ≠ j)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if Q[i, j] != 0:
                    # ZZ interaction: e^(-iγ Q_ij Z_i Z_j)
                    # Implemented as: CNOT, RZ, CNOT
                    qc.cx(i, j)
                    qc.rz(2 * gamma * Q[i, j], j)
                    qc.cx(i, j)
        
        # Single-qubit terms: Q_ii Z_i
        for i in range(self.n_qubits):
            if Q[i, i] != 0:
                # Z rotation: e^(-iγ Q_ii Z_i)
                qc.rz(2 * gamma * Q[i, i], i)
    
    def _apply_mixer_hamiltonian(self, qc: QuantumCircuit, beta):
        """
        Apply mixer Hamiltonian U_M(β) = e^(-iβH_M)
        
        Standard mixer: H_M = Σ X_i
        Implemented as RX rotations on all qubits
        """
        for i in range(self.n_qubits):
            qc.rx(2 * beta, i)
    
    def build_statevector_circuit(
        self,
        Q: np.ndarray,
        p: int = 1,
        gamma_vals: List[float] = None,
        beta_vals: List[float] = None
    ) -> QuantumCircuit:
        """
        Build QAOA circuit with bound parameters for statevector simulation
        
        Args:
            Q: QUBO matrix
            p: Number of layers
            gamma_vals: Values for gamma parameters
            beta_vals: Values for beta parameters
            
        Returns:
            Quantum circuit without measurements
        """
        if gamma_vals is None:
            gamma_vals = [0.5] * p
        if beta_vals is None:
            beta_vals = [0.5] * p
        
        if len(gamma_vals) != p or len(beta_vals) != p:
            raise ValueError(f"Parameter lengths must match p={p}")
        
        # Create circuit without measurements
        qc = QuantumCircuit(self.n_qubits)
        
        # Initial state
        qc.h(range(self.n_qubits))
        qc.barrier()
        
        # Apply layers with specific parameter values
        for layer in range(p):
            self._apply_problem_hamiltonian(qc, Q, gamma_vals[layer])
            qc.barrier()
            self._apply_mixer_hamiltonian(qc, beta_vals[layer])
            qc.barrier()
        
        return qc
    
    def estimate_circuit_resources(self, Q: np.ndarray, p: int) -> dict:
        """
        Estimate quantum circuit resources
        
        Returns:
            Dictionary with gate counts, depth, etc.
        """
        # Build circuit to analyze
        qc, _, _ = self.build_qaoa_circuit(Q, p, insert_barriers=False)
        
        # Count gates
        gate_counts = qc.count_ops()
        
        # Calculate theoretical depth (without transpilation)
        # Each layer has: ZZ gates + single-qubit rotations
        n_nonzero = np.count_nonzero(Q - np.diag(np.diag(Q))) // 2  # Upper triangle
        depth_per_layer = n_nonzero * 3 + self.n_qubits  # CNOT-RZ-CNOT + RX
        
        resources = {
            'n_qubits': self.n_qubits,
            'n_layers': p,
            'circuit_depth': qc.depth(),
            'theoretical_depth': depth_per_layer * p + self.n_qubits,  # +H layer
            'gate_counts': gate_counts,
            'total_gates': sum(gate_counts.values()),
            'two_qubit_gates': gate_counts.get('cx', 0),
            'single_qubit_gates': sum(v for k, v in gate_counts.items() if k != 'cx'),
        }
        
        return resources


def main():
    """Example usage"""
    # Create a simple 4-qubit QUBO problem
    n_qubits = 4
    Q = np.array([
        [-1,  1,  0,  0],
        [ 1, -1,  1,  0],
        [ 0,  1, -1,  1],
        [ 0,  0,  1, -1]
    ])
    
    # Build circuit
    builder = QAOACircuitBuilder(n_qubits)
    qc, gamma, beta = builder.build_qaoa_circuit(Q, p=2)
    
    print("\nQAOA Circuit:")
    print(f"Qubits: {qc.num_qubits}")
    print(f"Depth: {qc.depth()}")
    print(f"Parameters: {len(gamma)} gamma, {len(beta)} beta")
    print(f"\nCircuit:\n{qc}")
    
    # Estimate resources
    resources = builder.estimate_circuit_resources(Q, p=2)
    print("\nCircuit Resources:")
    for key, value in resources.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()