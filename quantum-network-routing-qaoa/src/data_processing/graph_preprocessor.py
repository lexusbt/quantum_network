"""
Graph Preprocessing Module
Loads and preprocesses SNAP network datasets for quantum routing optimization
"""

import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphPreprocessor:
    """
    Preprocesses SNAP datasets for quantum network routing
    - Loads edge lists from SNAP format
    - Extracts largest connected component
    - Validates graph properties
    - Generates subgraphs for NISQ compatibility
    """
    
    def __init__(self, config_path: str = "configs/datasets.yaml"):
        """Initialize preprocessor with configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_dir = Path("data/raw")
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self) -> dict:
        """Load dataset configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_snap_graph(self, dataset_name: str) -> nx.Graph:
        """
        Load a SNAP dataset from edge list format
        
        Args:
            dataset_name: Name of dataset (e.g., 'ca-GrQc')
            
        Returns:
            NetworkX Graph object
        """
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.config['datasets'][dataset_name]
        filepath = self.data_dir / dataset_info['filename']
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}\n"
                f"Please download datasets first (see data/README.md)"
            )
        
        logger.info(f"Loading {dataset_name} from {filepath}")
        
        # Read edge list, skip comment lines
        G = nx.Graph()
        
        with open(filepath, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                # Parse edge
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        source = int(parts[0])
                        target = int(parts[1])
                        G.add_edge(source, target)
                    except ValueError:
                        continue
        
        logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def preprocess_graph(self, G: nx.Graph, dataset_name: str) -> nx.Graph:
        """
        Preprocess graph for routing problems
        - Remove self-loops
        - Extract largest connected component
        - Relabel nodes to 0..N-1
        - Add basic metadata
        
        Args:
            G: Input graph
            dataset_name: Name for metadata
            
        Returns:
            Preprocessed graph
        """
        logger.info(f"Preprocessing {dataset_name}...")
        
        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Get largest connected component
        if not nx.is_connected(G):
            logger.info("Graph not connected, extracting largest component")
            components = list(nx.connected_components(G))
            largest_cc = max(components, key=len)
            G = G.subgraph(largest_cc).copy()
        
        # Relabel nodes to sequential integers starting from 0
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        
        # Add metadata
        G.graph['name'] = dataset_name
        G.graph['n_nodes'] = G.number_of_nodes()
        G.graph['n_edges'] = G.number_of_edges()
        
        logger.info(f"Preprocessed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def sample_subgraph(
        self, 
        G: nx.Graph, 
        size: int, 
        method: str = "random_walk",
        seed: Optional[int] = None
    ) -> nx.Graph:
        
        """
        Sample a connected subgraph of specified size
        
        Args:
            G: Input graph
            size: Number of nodes in subgraph
            method: Sampling method ('random_walk', 'high_centrality', 'bfs')
            seed: Random seed for reproducibility
            
        Returns:
            Connected subgraph
        """
        if seed is not None:
            np.random.seed(seed)
        
        if size > G.number_of_nodes():
            raise ValueError(f"Subgraph size {size} exceeds graph size {G.number_of_nodes()}")
        
        if method == "random_walk":
            subgraph = self._sample_random_walk(G, size)
        elif method == "high_centrality":
            subgraph = self._sample_high_centrality(G, size)
        elif method == "bfs":
            subgraph = self._sample_bfs(G, size)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Ensure connectivity
        if not nx.is_connected(subgraph):
            # Extract largest component and try again
            components = list(nx.connected_components(subgraph))
            largest = max(components, key=len)
            if len(largest) < size * 0.8:  # If we lost too many nodes
                return self.sample_subgraph(G, size, method, seed)
            subgraph = subgraph.subgraph(largest).copy()
        
        # Relabel to 0..N-1
        subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0)
        
        print(f"    ✓ Sampled successfully") 
        return subgraph
    
    def _sample_random_walk(self, G: nx.Graph, size: int) -> nx.Graph:
        """Sample using random walk"""
        nodes = set()
        current = np.random.choice(list(G.nodes()))
        nodes.add(current)
        
        while len(nodes) < size:
            neighbors = list(G.neighbors(current))
            if neighbors:
                current = np.random.choice(neighbors)
                nodes.add(current)
            else:
                # Stuck, add random node
                remaining = set(G.nodes()) - nodes
                if remaining:
                    current = np.random.choice(list(remaining))
                    nodes.add(current)
        
        return G.subgraph(nodes).copy()
    
    def _sample_high_centrality(self, G: nx.Graph, size: int) -> nx.Graph:
        """Sample nodes with highest betweenness centrality"""
        centrality = nx.betweenness_centrality(G)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sorted_nodes[:size]]
        
        subgraph = G.subgraph(top_nodes).copy()
        
        # If not connected, add bridging nodes
        if not nx.is_connected(subgraph):
            components = list(nx.connected_components(subgraph))
            while len(components) > 1:
                # Find shortest path between components
                c1_node = list(components[0])[0]
                c2_node = list(components[1])[0]
                
                try:
                    path = nx.shortest_path(G, c1_node, c2_node)
                    top_nodes.extend(path)
                    top_nodes = list(set(top_nodes))[:size]
                    subgraph = G.subgraph(top_nodes).copy()
                    components = list(nx.connected_components(subgraph))
                except nx.NetworkXNoPath:
                    break
        
        return subgraph
    
    def _sample_bfs(self, G: nx.Graph, size: int) -> nx.Graph:
        """Sample using BFS from random starting node"""
        start_node = np.random.choice(list(G.nodes()))
        nodes = []
        
        for node in nx.bfs_tree(G, start_node):
            nodes.append(node)
            if len(nodes) >= size:
                break
        
        return G.subgraph(nodes[:size]).copy()
    
    def generate_subgraphs(
        self, 
        dataset_name: str, 
        sizes: List[int], 
        samples_per_size: int = 10
    ) -> Dict[int, List[nx.Graph]]:
        """
        Generate multiple subgraphs of different sizes from a dataset
        
        Args:
            dataset_name: Name of dataset
            sizes: List of subgraph sizes to generate
            samples_per_size: Number of samples per size
            
        Returns:
            Dictionary mapping size -> list of subgraphs
        """
        # Load and preprocess full graph
        G = self.load_snap_graph(dataset_name)
        G = self.preprocess_graph(G, dataset_name)
        
        subgraphs = {}
        
        for size in sizes:
            logger.info(f"Generating {samples_per_size} subgraphs of size {size}")
            subgraphs[size] = []
            
            methods = ['random_walk', 'high_centrality', 'bfs']
            
            for i in range(samples_per_size):
                method = methods[i % len(methods)]
                try:
                    subgraph = self.sample_subgraph(G, size, method, seed=42+i)
                    subgraph.graph['source_dataset'] = dataset_name
                    subgraph.graph['size'] = size
                    subgraph.graph['sample_id'] = i
                    subgraph.graph['method'] = method
                    subgraphs[size].append(subgraph)
                except Exception as e:
                    logger.warning(f"Failed to generate subgraph {i} of size {size}: {e}")
        
        return subgraphs
    
    def save_graph(self, G: nx.Graph, filename: str, format: str = "graphml"):
        """Save graph to file"""
        filepath = self.output_dir / filename
        
        if format == "graphml":
            nx.write_graphml(G, filepath)
        elif format == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(G, f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved graph to {filepath}")
    
    def load_graph(self, filename: str, format: str = "graphml") -> nx.Graph:
        """Load graph from file"""
        filepath = self.output_dir / filename
        
        if format == "graphml":
            return nx.read_graphml(filepath)
        elif format == "pickle":
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")


def main():
    """Example usage"""
    preprocessor = GraphPreprocessor()
    
    # Process a single dataset
    dataset_name = "ca-GrQc"
    G = preprocessor.load_snap_graph(dataset_name)
    G = preprocessor.preprocess_graph(G, dataset_name)
    
    # Generate subgraphs
    subgraphs = preprocessor.generate_subgraphs(
        dataset_name, 
        sizes=[10, 15, 20],
        samples_per_size=5
    )
    
    # Save processed graph
    preprocessor.save_graph(G, f"{dataset_name}_processed.graphml")
    
    print(f"\nGenerated subgraphs:")
    for size, graphs in subgraphs.items():
        print(f"  Size {size}: {len(graphs)} subgraphs")


if __name__ == "__main__":
    main()