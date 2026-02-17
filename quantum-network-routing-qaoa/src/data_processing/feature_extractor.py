"""
Feature Extraction Module
Extracts topology features from network graphs for ML optimization
"""

import networkx as nx
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract network topology features for ML-enhanced QAOA
    Features are used to predict optimal QAOA parameters
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = [
            # Global features
            'n_nodes',
            'n_edges',
            'avg_degree',
            'density',
            'diameter',
            'avg_shortest_path',
            'radius',
            'clustering_coefficient',
            'transitivity',
            
            # Centrality statistics
            'avg_betweenness_centrality',
            'max_betweenness_centrality',
            'avg_closeness_centrality',
            'avg_degree_centrality',
            
            # Degree distribution
            'degree_variance',
            'degree_skewness',
            'max_degree',
            'min_degree',
            
            # Connectivity
            'edge_connectivity',
            'node_connectivity',
            'algebraic_connectivity',
        ]
    
    def extract_features(self, G: nx.Graph) -> Dict[str, float]:
        """
        Extract all features from a graph
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        # Basic features
        features.update(self._extract_basic_features(G))
        
        # Path features (expensive for large graphs)
        if G.number_of_nodes() <= 1000:
            features.update(self._extract_path_features(G))
        else:
            # Use approximations for large graphs
            features.update(self._extract_path_features_approx(G))
        
        # Centrality features
        features.update(self._extract_centrality_features(G))
        
        # Degree distribution features
        features.update(self._extract_degree_features(G))
        
        # Connectivity features
        features.update(self._extract_connectivity_features(G))
        
        return features
    
    def _extract_basic_features(self, G: nx.Graph) -> Dict[str, float]:
        """Extract basic graph statistics"""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        features = {
            'n_nodes': float(n_nodes),
            'n_edges': float(n_edges),
            'density': nx.density(G),
            'clustering_coefficient': nx.average_clustering(G),
            'transitivity': nx.transitivity(G),
        }
        
        # Average degree
        degrees = [d for _, d in G.degree()]
        features['avg_degree'] = np.mean(degrees)
        
        return features
    
    def _extract_path_features(self, G: nx.Graph) -> Dict[str, float]:
        """Extract shortest path features (exact)"""
        try:
            diameter = nx.diameter(G)
            radius = nx.radius(G)
            avg_shortest_path = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            # Graph not connected
            diameter = -1
            radius = -1
            avg_shortest_path = -1
        
        return {
            'diameter': float(diameter),
            'radius': float(radius),
            'avg_shortest_path': float(avg_shortest_path),
        }
    
    def _extract_path_features_approx(self, G: nx.Graph) -> Dict[str, float]:
        """Extract shortest path features (approximate for large graphs)"""
        # Sample nodes for approximation
        sample_size = min(100, G.number_of_nodes())
        nodes = np.random.choice(list(G.nodes()), sample_size, replace=False)
        
        path_lengths = []
        for source in nodes:
            lengths = nx.single_source_shortest_path_length(G, source)
            path_lengths.extend(lengths.values())
        
        return {
            'diameter': float(np.max(path_lengths)) if path_lengths else -1,
            'radius': float(np.min(path_lengths)) if path_lengths else -1,
            'avg_shortest_path': float(np.mean(path_lengths)) if path_lengths else -1,
        }
    
    def _extract_centrality_features(self, G: nx.Graph) -> Dict[str, float]:
        """Extract centrality statistics"""
        # Betweenness centrality (expensive, so sample for large graphs)
        if G.number_of_nodes() <= 100:
            betweenness = nx.betweenness_centrality(G)
        else:
            k = min(100, G.number_of_nodes())
            betweenness = nx.betweenness_centrality(G, k=k)
        
        # Closeness centrality
        closeness = nx.closeness_centrality(G)
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        
        return {
            'avg_betweenness_centrality': float(np.mean(list(betweenness.values()))),
            'max_betweenness_centrality': float(np.max(list(betweenness.values()))),
            'avg_closeness_centrality': float(np.mean(list(closeness.values()))),
            'avg_degree_centrality': float(np.mean(list(degree_centrality.values()))),
        }
    
    def _extract_degree_features(self, G: nx.Graph) -> Dict[str, float]:
        """Extract degree distribution features"""
        degrees = [d for _, d in G.degree()]
        
        return {
            'degree_variance': float(np.var(degrees)),
            'degree_skewness': float(self._skewness(degrees)),
            'max_degree': float(np.max(degrees)),
            'min_degree': float(np.min(degrees)),
        }
    
    def _extract_connectivity_features(self, G: nx.Graph) -> Dict[str, float]:
        """Extract connectivity features"""
        # Edge connectivity (expensive for large graphs)
        if G.number_of_nodes() <= 100:
            edge_conn = nx.edge_connectivity(G)
            node_conn = nx.node_connectivity(G)
        else:
            edge_conn = -1
            node_conn = -1
        
        # Algebraic connectivity (Fiedler value)
        try:
            alg_conn = nx.algebraic_connectivity(G)
        except:
            alg_conn = -1
        
        return {
            'edge_connectivity': float(edge_conn),
            'node_connectivity': float(node_conn),
            'algebraic_connectivity': float(alg_conn),
        }
    
    @staticmethod
    def _skewness(data: List[float]) -> float:
        """Calculate skewness of distribution"""
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def extract_routing_features(
        self, 
        G: nx.Graph, 
        source: int, 
        dest: int, 
        K: int
    ) -> Dict[str, float]:
        """
        Extract features specific to a routing problem instance
        
        Args:
            G: Graph
            source: Source node
            dest: Destination node
            K: Maximum path length
            
        Returns:
            Dictionary of routing-specific features
        """
        features = {}
        
        # Path length constraint
        features['path_length_k'] = float(K)
        
        # Shortest path length
        try:
            shortest = nx.shortest_path_length(G, source, dest)
            features['shortest_path_length'] = float(shortest)
            features['k_over_shortest'] = float(K / shortest) if shortest > 0 else -1
        except nx.NetworkXNoPath:
            features['shortest_path_length'] = -1
            features['k_over_shortest'] = -1
        
        # Source/dest centrality
        degree_cent = nx.degree_centrality(G)
        features['source_degree'] = float(G.degree(source))
        features['dest_degree'] = float(G.degree(dest))
        features['source_centrality'] = float(degree_cent[source])
        features['dest_centrality'] = float(degree_cent[dest])
        
        # Distance between source and dest
        try:
            features['hop_distance'] = float(nx.shortest_path_length(G, source, dest))
        except:
            features['hop_distance'] = -1
        
        # Number of alternative paths
        try:
            all_paths = list(nx.all_simple_paths(G, source, dest, cutoff=K))
            features['num_alternative_paths'] = float(len(all_paths))
        except:
            features['num_alternative_paths'] = -1
        
        return features
    
    def create_feature_vector(
        self, 
        graph_features: Dict[str, float], 
        routing_features: Dict[str, float]
    ) -> np.ndarray:
        """
        Combine graph and routing features into a single vector
        
        Args:
            graph_features: Graph topology features
            routing_features: Routing problem features
            
        Returns:
            Feature vector as numpy array
        """
        all_features = {**graph_features, **routing_features}
        
        # Sort by feature name for consistency
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        return feature_vector


def main():
    """Example usage"""
    from src.data_processing.graph_preprocessor import GraphPreprocessor
    
    # Load a graph
    preprocessor = GraphPreprocessor()
    G = preprocessor.load_snap_graph("ca-GrQc")
    G = preprocessor.preprocess_graph(G, "ca-GrQc")
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_features(G)
    
    print("\nExtracted Graph Features:")
    print("=" * 50)
    for name, value in sorted(features.items()):
        print(f"{name:30s}: {value:10.4f}")
    
    # Extract routing-specific features
    source = 0
    dest = 10
    K = 5
    routing_features = extractor.extract_routing_features(G, source, dest, K)
    
    print("\nRouting Problem Features:")
    print("=" * 50)
    for name, value in sorted(routing_features.items()):
        print(f"{name:30s}: {value:10.4f}")


if __name__ == "__main__":
    main()