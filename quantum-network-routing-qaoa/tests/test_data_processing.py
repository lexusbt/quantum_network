"""
Unit tests for data processing modules
"""

import pytest
import networkx as nx
import numpy as np
from src.data_processing.graph_preprocessor import GraphPreprocessor
from src.data_processing.feature_extractor import FeatureExtractor


def test_graph_preprocessor_init():
    """Test preprocessor initialization"""
    preprocessor = GraphPreprocessor()
    assert preprocessor.config is not None
    assert 'datasets' in preprocessor.config


def test_load_snap_graph():
    """Test loading SNAP graph"""
    preprocessor = GraphPreprocessor()
    G = preprocessor.load_snap_graph("ca-GrQc")
    
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0


def test_preprocess_graph():
    """Test graph preprocessing"""
    preprocessor = GraphPreprocessor()
    G = preprocessor.load_snap_graph("ca-GrQc")
    G_processed = preprocessor.preprocess_graph(G, "ca-GrQc")
    
    assert nx.is_connected(G_processed)
    assert 'name' in G_processed.graph
    assert G_processed.graph['name'] == "ca-GrQc"


def test_sample_subgraph():
    """Test subgraph sampling"""
    preprocessor = GraphPreprocessor()
    G = preprocessor.load_snap_graph("ca-GrQc")
    G = preprocessor.preprocess_graph(G, "ca-GrQc")
    
    subgraph = preprocessor.sample_subgraph(G, size=10, seed=42)
    
    assert subgraph.number_of_nodes() == 10
    assert nx.is_connected(subgraph)


def test_feature_extractor():
    """Test feature extraction"""
    preprocessor = GraphPreprocessor()
    G = preprocessor.load_snap_graph("ca-GrQc")
    G = preprocessor.preprocess_graph(G, "ca-GrQc")
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(G)
    
    assert 'n_nodes' in features
    assert 'n_edges' in features
    assert 'avg_degree' in features
    assert features['n_nodes'] == G.number_of_nodes()


def test_routing_features():
    """Test routing-specific features"""
    preprocessor = GraphPreprocessor()
    G = preprocessor.load_snap_graph("ca-GrQc")
    G = preprocessor.preprocess_graph(G, "ca-GrQc")
    subgraph = preprocessor.sample_subgraph(G, size=15, seed=42)
    
    extractor = FeatureExtractor()
    features = extractor.extract_routing_features(subgraph, source=0, dest=10, K=5)
    
    assert 'path_length_k' in features
    assert 'source_degree' in features
    assert features['path_length_k'] == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])