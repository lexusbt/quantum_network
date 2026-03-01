"""
Instance Generator Module
Generates 200 routing problem instances for QAOA optimization
"""

import numpy as np
import networkx as nx
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import yaml
import pandas as pd
import logging

from src.data_processing.graph_preprocessor import GraphPreprocessor
from src.data_processing.feature_extractor import FeatureExtractor
from src.qubo_generation.routing_qubo import RoutingQUBO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstanceGenerator:
    """
    Generate routing problem instances for ML-enhanced QAOA
    Creates 200 diverse instances across different topologies and problem sizes
    """
    
    def __init__(self, config_path: str = "configs/datasets.yaml"):
        """Initialize instance generator"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.preprocessor = GraphPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        # Get QUBO parameters from config
        qubo_params = self.config['qubo_params']['penalty_weights']
        self.qubo_generator = RoutingQUBO(
            penalty_A=qubo_params['A'],
            penalty_B=qubo_params['B'],
            penalty_C=qubo_params['C']
        )
        
        # Output directories
        self.instances_dir = Path("instances")
        self.train_dir = self.instances_dir / "train"
        self.val_dir = self.instances_dir / "validation"
        self.test_dir = self.instances_dir / "test"
        
        for dir in [self.train_dir, self.val_dir, self.test_dir]:
            dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_all_instances(self, n_instances: int = 200, seed: int = 42) -> List[Dict]:
        """
        Generate all routing problem instances
        
        Args:
            n_instances: Total number of instances to generate
            seed: Random seed
            
        Returns:
            List of instance metadata dictionaries
        """
        np.random.seed(seed)
        
        logger.info(f"Generating {n_instances} routing problem instances")
        logger.info("=" * 70)
        
        # Calculate distribution across datasets
        datasets = [name for name, info in self.config['datasets'].items() 
                   if info['enabled']]
        instances_per_dataset = n_instances // len(datasets)
        
        all_instances = []
        instance_id = 0
        
        for dataset_name in datasets:
            logger.info(f"\nProcessing dataset: {dataset_name}")
            dataset_info = self.config['datasets'][dataset_name]
            
            # Load and preprocess graph
            G = self.preprocessor.load_snap_graph(dataset_name)
            G = self.preprocessor.preprocess_graph(G, dataset_name)
            
            # Generate subgraphs
            subgraph_sizes = dataset_info['subgraph_sizes']
            n_per_size = instances_per_dataset // len(subgraph_sizes)
            
            for size in subgraph_sizes:
                logger.info(f"  Generating {n_per_size} instances of size {size}")
                
                for i in tqdm(range(n_per_size), desc=f"Size {size}"):
                    try:
                        instance = self._generate_single_instance(
                            G, dataset_name, size, instance_id, seed + instance_id
                        )
                        all_instances.append(instance)
                        instance_id += 1
                    except Exception as e:
                        logger.warning(f"Failed to generate instance {instance_id}: {e}")
        
        # Ensure we have exactly n_instances
        while len(all_instances) < n_instances:
            dataset_name = np.random.choice(datasets)
            G = self.preprocessor.load_snap_graph(dataset_name)
            G = self.preprocessor.preprocess_graph(G, dataset_name)
            size = np.random.choice(self.config['datasets'][dataset_name]['subgraph_sizes'])
            
            instance = self._generate_single_instance(
                G, dataset_name, size, instance_id, seed + instance_id
            )
            all_instances.append(instance)
            instance_id += 1
        
        all_instances = all_instances[:n_instances]
        
        logger.info(f"\n✓ Generated {len(all_instances)} instances")
        
        # Split into train/val/test
        self._split_and_save_instances(all_instances)
        
        # Create master index
        self._create_master_index(all_instances)
        
        return all_instances
    
    def _generate_single_instance(
        self,
        G: nx.Graph,
        dataset_name: str,
        size: int,
        instance_id: int,
        seed: int
    ) -> Dict:
        """Generate a single routing problem instance"""
        np.random.seed(seed)
        
        # Sample subgraph
        subgraph = self.preprocessor.sample_subgraph(G, size, seed=seed)
        
        # Select source and destination
        source, dest, K = self._select_source_dest_k(subgraph)
        
        # Generate QUBO
        Q, qubo_metadata = self.qubo_generator.generate_qubo(
            subgraph, source, dest, K, edge_weights="uniform"
        )
        
        # Extract features
        graph_features = self.feature_extractor.extract_features(subgraph)
        routing_features = self.feature_extractor.extract_routing_features(
            subgraph, source, dest, K
        )
        
        # Compute classical baseline
        classical_solution = self._compute_classical_baseline(subgraph, source, dest)
        
        # Create instance metadata
        instance = {
            'instance_id': f"instance_{instance_id:04d}",
            'dataset_name': dataset_name,
            'n_nodes': subgraph.number_of_nodes(),
            'n_edges': subgraph.number_of_edges(),
            'source': source,
            'dest': dest,
            'K': K,
            'n_qubits': Q.shape[0],
            'graph_features': graph_features,
            'routing_features': routing_features,
            'classical_optimal_length': classical_solution['length'],
            'classical_optimal_path': classical_solution['path'],
            'qubo_metadata': qubo_metadata,
            'seed': seed
        }
        
        # Save instance files
        self._save_instance(instance, subgraph, Q)
        
        return instance
    
    def _select_source_dest_k(self, G: nx.Graph) -> Tuple[int, int, int]:
        """Select source, destination, and path length K"""
        min_distance = self.config['instance_generation']['min_hop_distance']
        path_length_config = self.config['instance_generation']['path_lengths']
        
        # Select K from distribution
        K_options = []
        for k_str, count in path_length_config.items():
            k_val = int(k_str[1:])  # Extract number from 'K3', 'K4', etc.
            K_options.extend([k_val] * count)
        K = np.random.choice(K_options)
        
        # Select source and dest with sufficient distance
        max_attempts = 100
        for _ in range(max_attempts):
            source = np.random.choice(list(G.nodes()))
            dest = np.random.choice(list(G.nodes()))
            
            if source == dest:
                continue
            
            try:
                distance = nx.shortest_path_length(G, source, dest)
                if min_distance <= distance <= K:
                    return source, dest, K
            except nx.NetworkXNoPath:
                continue
        
        # Fallback: just pick any two connected nodes
        source = np.random.choice(list(G.nodes()))
        neighbors_at_distance = {}
        
        for node in G.nodes():
            if node != source:
                try:
                    dist = nx.shortest_path_length(G, source, node)
                    if dist not in neighbors_at_distance:
                        neighbors_at_distance[dist] = []
                    neighbors_at_distance[dist].append(node)
                except:
                    pass
        
        if neighbors_at_distance:
            available_dists = sorted([d for d in neighbors_at_distance.keys() if d <= K])
            if available_dists:
                chosen_dist = np.random.choice(available_dists)
                dest = np.random.choice(neighbors_at_distance[chosen_dist])
                return source, dest, K
        
        # Last resort
        dest = np.random.choice([n for n in G.nodes() if n != source])
        return source, dest, K
    
    def _compute_classical_baseline(
        self, 
        G: nx.Graph, 
        source: int, 
        dest: int
    ) -> Dict:
        """Compute classical optimal solution using Dijkstra"""
        try:
            path = nx.shortest_path(G, source, dest)
            length = len(path) - 1
            return {'path': path, 'length': length, 'exists': True}
        except nx.NetworkXNoPath:
            return {'path': [], 'length': -1, 'exists': False}
    
    def _save_instance(self, instance: Dict, subgraph: nx.Graph, Q: np.ndarray):
        """Save instance to multiple formats"""
        instance_id = instance['instance_id']
        
        # Save to temporary location (will be moved during split)
        temp_dir = self.instances_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Save full instance (pickle)
        full_instance = {
            'metadata': instance,
            'graph': subgraph,
            'qubo_matrix': Q
        }
        with open(temp_dir / f"{instance_id}.pkl", 'wb') as f:
            pickle.dump(full_instance, f)
        
        # Save QUBO matrix (numpy)
        np.save(temp_dir / f"{instance_id}_qubo.npy", Q)
        
        # Save metadata (JSON) - FIX: Convert numpy types to Python types
        metadata_json = {}
        
        # Convert top-level metadata
        for k, v in instance.items():
            if k not in ['graph_features', 'routing_features', 'qubo_metadata']:
                if isinstance(v, np.integer):
                    metadata_json[k] = int(v)
                elif isinstance(v, np.floating):
                    metadata_json[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metadata_json[k] = v.tolist()
                elif isinstance(v, list):
                    # Handle lists that might contain numpy types
                    metadata_json[k] = [int(x) if isinstance(x, np.integer) else 
                                    float(x) if isinstance(x, np.floating) else x 
                                    for x in v]
                else:
                    metadata_json[k] = v
        
        # Convert nested dictionaries (features)
        if 'graph_features' in instance:
            metadata_json['graph_features'] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in instance['graph_features'].items()
            }
        
        if 'routing_features' in instance:
            metadata_json['routing_features'] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in instance['routing_features'].items()
            }
        
        with open(temp_dir / f"{instance_id}_metadata.json", 'w') as f:
            json.dump(metadata_json, f, indent=2)
    
    def _split_and_save_instances(self, instances: List[Dict]):
        """Split instances into train/val/test sets"""
        splits = self.config['instance_generation']
        train_ratio = splits['train_split']
        val_ratio = splits['val_split']
        
        n_total = len(instances)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Shuffle instances
        indices = np.random.permutation(n_total)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        temp_dir = self.instances_dir / "temp"
        
        # Move files to appropriate directories
        for idx in train_indices:
            instance_id = instances[idx]['instance_id']
            self._move_instance_files(temp_dir, self.train_dir, instance_id)
            instances[idx]['split'] = 'train'
        
        for idx in val_indices:
            instance_id = instances[idx]['instance_id']
            self._move_instance_files(temp_dir, self.val_dir, instance_id)
            instances[idx]['split'] = 'validation'
        
        for idx in test_indices:
            instance_id = instances[idx]['instance_id']
            self._move_instance_files(temp_dir, self.test_dir, instance_id)
            instances[idx]['split'] = 'test'
        
        # Remove temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        logger.info(f"\nSplit: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
    def _move_instance_files(self, src_dir: Path, dest_dir: Path, instance_id: str):
        """Move instance files from source to destination directory"""
        import shutil
        for ext in ['.pkl', '_qubo.npy', '_metadata.json']:
            src_file = src_dir / f"{instance_id}{ext}"
            if src_file.exists():
                shutil.move(str(src_file), str(dest_dir / f"{instance_id}{ext}"))
    
    def _create_master_index(self, instances: List[Dict]):
        """Create master index CSV"""
        # Flatten instances for DataFrame
        records = []
        for inst in instances:
            record = {
                'instance_id': inst['instance_id'],
                'split': inst['split'],
                'dataset_name': inst['dataset_name'],
                'n_nodes': inst['n_nodes'],
                'n_edges': inst['n_edges'],
                'source': inst['source'],
                'dest': inst['dest'],
                'K': inst['K'],
                'n_qubits': inst['n_qubits'],
                'classical_optimal_length': inst['classical_optimal_length'],
            }
            # Add some key features
            record['avg_degree'] = inst['graph_features'].get('avg_degree', -1)
            record['diameter'] = inst['graph_features'].get('diameter', -1)
            record['shortest_path'] = inst['routing_features'].get('shortest_path_length', -1)
            
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(self.instances_dir / "master_index.csv", index=False)
        
        logger.info(f"\n✓ Created master index: {self.instances_dir / 'master_index.csv'}")
        
        # Print summary statistics
        print("\n" + "=" * 70)
        print("INSTANCE GENERATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal instances: {len(instances)}")
        print(f"\nSplit distribution:")
        print(df['split'].value_counts())
        print(f"\nDataset distribution:")
        print(df['dataset_name'].value_counts())
        print(f"\nQubit count statistics:")
        print(df['n_qubits'].describe())
        print(f"\nPath length (K) distribution:")
        print(df['K'].value_counts().sort_index())


def main():
    """Generate all 200 instances"""
    generator = InstanceGenerator()
    instances = generator.generate_all_instances(n_instances=200, seed=42)
    
    print("\n✓ Instance generation complete!")
    print(f"Instances saved to: {generator.instances_dir}")


if __name__ == "__main__":
    main()