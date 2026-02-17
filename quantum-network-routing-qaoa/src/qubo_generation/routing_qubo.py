"""
Routing QUBO Generation Module
Generates QUBO formulations for shortest path routing problems
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoutingQUBO:
    """
    Generate QUBO formulations for network routing problems
    
    Encodes shortest path routing as a QUBO optimization problem
    Variables: x_{v,t} = 1 if node v is visited at time step t
    
    Based on: Machine Learning-Enhanced QAOA for Network Routing
    """
    
    def __init__(
        self, 
        penalty_A: float = 0.47,
        penalty_B: float = 1.0,
        penalty_C: float = 1.0
    ):
        """
        Initialize QUBO generator with penalty weights
        
        Args:
            penalty_A: One-node-per-timestep constraint (optimal ≈ 0.47)
            penalty_B: Valid edge transitions constraint
            penalty_C: Boundary conditions constraint
        """
        self.penalty_A = penalty_A
        self.penalty_B = penalty_B
        self.penalty_C = penalty_C
    
    def generate_qubo(
        self,
        G: nx.Graph,
        source: int,
        dest: int,
        K: int,
        edge_weights: str = "uniform"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate QUBO matrix for routing problem
        
        Args:
            G: NetworkX graph
            source: Source node
            dest: Destination node
            K: Maximum path length (number of hops)
            edge_weights: Weight scheme ('uniform', 'random', 'betweenness')
            
        Returns:
            (Q, metadata) where Q is QUBO matrix and metadata contains problem info
        """
        n_nodes = G.number_of_nodes()
        n_vars = n_nodes * (K + 1)  # Variables x_{v,t} for v in V, t in [0,K]
        
        # Initialize QUBO matrix
        Q = np.zeros((n_vars, n_vars))
        
        # Get edge weights
        edge_weight_dict = self._get_edge_weights(G, edge_weights)
        
        # Add Hamiltonian components
        Q += self._add_cost_hamiltonian(G, K, edge_weight_dict)
        Q += self._add_one_node_constraint(n_nodes, K, self.penalty_A)
        Q += self._add_valid_edge_constraint(G, K, self.penalty_B)
        Q += self._add_boundary_conditions(n_nodes, source, dest, K, self.penalty_C)
        
        # Create variable mapping
        var_mapping = self._create_variable_mapping(n_nodes, K)
        
        # Metadata
        metadata = {
            'n_nodes': n_nodes,
            'n_edges': G.number_of_edges(),
            'source': source,
            'dest': dest,
            'K': K,
            'n_qubits': n_vars,
            'penalty_A': self.penalty_A,
            'penalty_B': self.penalty_B,
            'penalty_C': self.penalty_C,
            'edge_weights': edge_weights,
            'var_mapping': var_mapping
        }
        
        logger.info(f"Generated QUBO: {n_vars} variables ({n_nodes} nodes × {K+1} timesteps)")
        
        return Q, metadata
    
    def _get_edge_weights(self, G: nx.Graph, scheme: str) -> Dict[Tuple[int, int], float]:
        """Get edge weights based on scheme"""
        edge_weights = {}
        
        if scheme == "uniform":
            for u, v in G.edges():
                edge_weights[(u, v)] = 1.0
                edge_weights[(v, u)] = 1.0
        
        elif scheme == "random":
            for u, v in G.edges():
                weight = np.random.uniform(0.5, 2.0)
                edge_weights[(u, v)] = weight
                edge_weights[(v, u)] = weight
        
        elif scheme == "betweenness":
            # Weight by edge betweenness centrality
            betweenness = nx.edge_betweenness_centrality(G)
            max_bet = max(betweenness.values()) if betweenness else 1.0
            
            for u, v in G.edges():
                # Invert so higher betweenness = lower cost
                bet = betweenness[(u, v)]
                weight = 1.0 + (1.0 - bet / max_bet)
                edge_weights[(u, v)] = weight
                edge_weights[(v, u)] = weight
        
        else:
            raise ValueError(f"Unknown edge weight scheme: {scheme}")
        
        return edge_weights
    
    def _add_cost_hamiltonian(
        self, 
        G: nx.Graph, 
        K: int, 
        edge_weights: Dict[Tuple[int, int], float]
    ) -> np.ndarray:
        """
        Add path cost minimization term
        H_cost = Σ_{t=0}^{K-1} Σ_{(u,v)∈E} w_{uv} x_{u,t} x_{v,t+1}
        """
        n_nodes = G.number_of_nodes()
        n_vars = n_nodes * (K + 1)
        H = np.zeros((n_vars, n_vars))
        
        for t in range(K):
            for u, v in G.edges():
                weight = edge_weights.get((u, v), 1.0)
                
                idx_u_t = u * (K + 1) + t
                idx_v_t1 = v * (K + 1) + (t + 1)
                
                # Add to QUBO (quadratic term)
                H[idx_u_t, idx_v_t1] += weight
                H[idx_v_t1, idx_u_t] += weight  # Symmetric
        
        return H
    
    def _add_one_node_constraint(self, n_nodes: int, K: int, penalty: float) -> np.ndarray:
        """
        Add constraint: exactly one node visited per timestep
        H_one = A * Σ_{t=0}^K (1 - Σ_{v∈V} x_{v,t})²
        """
        n_vars = n_nodes * (K + 1)
        H = np.zeros((n_vars, n_vars))
        
        for t in range(K + 1):
            # Expand (1 - Σ_v x_{v,t})² = 1 - 2Σ_v x_{v,t} + (Σ_v x_{v,t})²
            
            # Linear terms: -2Σ_v x_{v,t}
            for v in range(n_nodes):
                idx = v * (K + 1) + t
                H[idx, idx] += penalty * (-2.0)
            
            # Quadratic terms: (Σ_v x_{v,t})² = Σ_v x_{v,t}² + 2Σ_{v<u} x_{v,t}x_{u,t}
            for v in range(n_nodes):
                idx_v = v * (K + 1) + t
                H[idx_v, idx_v] += penalty * 1.0  # x_v²
                
                for u in range(v + 1, n_nodes):
                    idx_u = u * (K + 1) + t
                    H[idx_v, idx_u] += penalty * 2.0
                    H[idx_u, idx_v] += penalty * 2.0
            
            # Constant term (not needed in QUBO matrix)
        
        return H
    
    def _add_valid_edge_constraint(self, G: nx.Graph, K: int, penalty: float) -> np.ndarray:
        """
        Add constraint: transitions must follow existing edges
        H_edge = B * Σ_{t=0}^{K-1} Σ_{(u,v)∉E} x_{u,t} x_{v,t+1}
        """
        n_nodes = G.number_of_nodes()
        n_vars = n_nodes * (K + 1)
        H = np.zeros((n_vars, n_vars))
        
        # Get set of edges
        edges = set(G.edges())
        edges.update((v, u) for u, v in edges)  # Add reverse edges
        
        for t in range(K):
            for u in range(n_nodes):
                for v in range(n_nodes):
                    if u != v and (u, v) not in edges:
                        # Invalid transition
                        idx_u_t = u * (K + 1) + t
                        idx_v_t1 = v * (K + 1) + (t + 1)
                        
                        H[idx_u_t, idx_v_t1] += penalty
                        H[idx_v_t1, idx_u_t] += penalty
        
        return H
    
    def _add_boundary_conditions(
        self, 
        n_nodes: int, 
        source: int, 
        dest: int, 
        K: int, 
        penalty: float
    ) -> np.ndarray:
        """
        Add boundary conditions: start at source, end at destination
        H_boundary = C * [(1 - x_{s,0})² + (1 - x_{d,K})²]
        """
        n_vars = n_nodes * (K + 1)
        H = np.zeros((n_vars, n_vars))
        
        # Start at source: (1 - x_{s,0})² = 1 - 2x_{s,0} + x_{s,0}²
        idx_source = source * (K + 1) + 0
        H[idx_source, idx_source] += penalty * (-2.0 + 1.0)  # -2 + 1 from expansion
        
        # End at destination: (1 - x_{d,K})²
        idx_dest = dest * (K + 1) + K
        H[idx_dest, idx_dest] += penalty * (-2.0 + 1.0)
        
        return H
    
    def _create_variable_mapping(self, n_nodes: int, K: int) -> Dict[int, Tuple[int, int]]:
        """
        Create mapping from qubit index to (node, timestep)
        """
        mapping = {}
        idx = 0
        for v in range(n_nodes):
            for t in range(K + 1):
                mapping[idx] = (v, t)
                idx += 1
        return mapping
    
    def decode_solution(
        self, 
        bitstring: str, 
        metadata: Dict
    ) -> Tuple[List[int], bool]:
        """
        Decode a bitstring solution to a path
        
        Args:
            bitstring: Binary string solution
            metadata: Problem metadata with variable mapping
            
        Returns:
            (path, is_valid) where path is list of nodes and is_valid is bool
        """
        var_mapping = metadata['var_mapping']
        K = metadata['K']
        n_nodes = metadata['n_nodes']
        
        # Convert bitstring to variable assignments
        assignments = {}
        for idx, bit in enumerate(bitstring):
            if bit == '1':
                v, t = var_mapping[idx]
                if t not in assignments:
                    assignments[t] = []
                assignments[t].append(v)
        
        # Extract path
        path = []
        is_valid = True
        
        for t in range(K + 1):
            if t not in assignments:
                is_valid = False
                break
            if len(assignments[t]) != 1:
                is_valid = False
                break
            path.append(assignments[t][0])
        
        # Verify path validity
        if is_valid:
            # Check source and dest
            if path[0] != metadata['source'] or path[-1] != metadata['dest']:
                is_valid = False
        
        return path, is_valid
    
    def compute_path_cost(self, G: nx.Graph, path: List[int]) -> float:
        """Compute total cost of a path"""
        cost = 0.0
        for i in range(len(path) - 1):
            if G.has_edge(path[i], path[i+1]):
                cost += 1.0  # Assuming unit weights
            else:
                return float('inf')  # Invalid path
        return cost


def main():
    """Example usage"""
    from src.data_processing.graph_preprocessor import GraphPreprocessor
    
    # Load a small graph
    preprocessor = GraphPreprocessor()
    G = preprocessor.load_snap_graph("ca-GrQc")
    G = preprocessor.preprocess_graph(G, "ca-GrQc")
    
    # Sample small subgraph
    subgraph = preprocessor.sample_subgraph(G, size=10, method="random_walk", seed=42)
    
    # Generate QUBO
    qubo_gen = RoutingQUBO(penalty_A=0.47, penalty_B=1.0, penalty_C=1.0)
    source = 0
    dest = 5
    K = 4
    
    Q, metadata = qubo_gen.generate_qubo(subgraph, source, dest, K, edge_weights="uniform")
    
    print("\nQUBO Problem:")
    print("=" * 50)
    print(f"Nodes: {metadata['n_nodes']}")
    print(f"Edges: {metadata['n_edges']}")
    print(f"Source → Dest: {source} → {dest}")
    print(f"Max path length K: {K}")
    print(f"Number of qubits: {metadata['n_qubits']}")
    print(f"QUBO matrix shape: {Q.shape}")
    print(f"Non-zero entries: {np.count_nonzero(Q)}")
    
    # Verify classical solution exists
    try:
        shortest_path = nx.shortest_path(subgraph, source, dest)
        print(f"\nClassical shortest path: {shortest_path}")
        print(f"Classical path length: {len(shortest_path) - 1} hops")
    except nx.NetworkXNoPath:
        print("\nNo path exists between source and destination!")


if __name__ == "__main__":
    main()