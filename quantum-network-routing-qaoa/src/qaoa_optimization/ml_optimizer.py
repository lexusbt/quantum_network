"""
XGBoost ML Optimizer for QAOA Parameter Prediction
Predicts optimal QAOA parameters from graph topology features
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLParameterPredictor:
    """
    Machine learning model to predict optimal QAOA parameters
    
    Uses XGBoost regression to learn the mapping:
    Graph features + Problem features → Optimal (γ, β) parameters
    
    This enables fast parameter initialization, reducing optimization time
    """
    
    def __init__(self, p_layers: int = 1):
        """
        Initialize ML predictor
        
        Args:
            p_layers: Number of QAOA layers to predict parameters for
        """
        self.p_layers = p_layers
        self.n_params = 2 * p_layers  # γ and β for each layer
        
        # Separate model for each parameter
        self.models = {
            f'gamma_{i}': None for i in range(p_layers)
        }
        self.models.update({
            f'beta_{i}': None for i in range(p_layers)
        })
        
        self.feature_names = None
        self.scaler_params = None
        
        logger.info(f"Initialized ML predictor for p={p_layers} layers ({self.n_params} parameters)")
    
    def prepare_training_data(
        self,
        instance_dir: str = "instances/train",
        results_dir: str = "results/qaoa"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data from instances and QAOA results
        
        Args:
            instance_dir: Directory with instance .pkl files
            results_dir: Directory with QAOA result .pkl files
            
        Returns:
            (features_df, targets_df)
        """
        logger.info(f"Loading training data from {instance_dir} and {results_dir}")
        
        instance_path = Path(instance_dir)
        results_path = Path(results_dir)
        
        features_list = []
        targets_list = []
        
        # Load all instances that have corresponding QAOA results
        instance_files = list(instance_path.glob("instance_*.pkl"))
        
        for instance_file in instance_files:
            instance_id = instance_file.stem
            result_file = results_path / f"{instance_id}_p{self.p_layers}.pkl"
            
            if not result_file.exists():
                continue
            
            # Load instance
            with open(instance_file, 'rb') as f:
                instance = pickle.load(f)
            
            # Load QAOA result
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            
            # Extract features
            features = self._extract_features(instance['metadata'])
            
            # Extract target parameters
            optimal_params = result['optimal_params']
            targets = {
                f'gamma_{i}': optimal_params[i] for i in range(self.p_layers)
            }
            targets.update({
                f'beta_{i}': optimal_params[self.p_layers + i] for i in range(self.p_layers)
            })
            
            features_list.append(features)
            targets_list.append(targets)
        
        features_df = pd.DataFrame(features_list)
        targets_df = pd.DataFrame(targets_list)
        
        self.feature_names = features_df.columns.tolist()
        
        logger.info(f"Prepared {len(features_df)} training samples")
        logger.info(f"Features: {len(self.feature_names)}")
        
        return features_df, targets_df
    
    def _extract_features(self, metadata: Dict) -> Dict:
        """Extract feature vector from instance metadata"""
        features = {}
        
        # Graph features
        if 'graph_features' in metadata:
            for k, v in metadata['graph_features'].items():
                features[f'graph_{k}'] = v
        
        # Routing problem features
        if 'routing_features' in metadata:
            for k, v in metadata['routing_features'].items():
                features[f'routing_{k}'] = v
        
        # Problem size features
        features['n_nodes'] = metadata.get('n_nodes', 0)
        features['n_edges'] = metadata.get('n_edges', 0)
        features['n_qubits'] = metadata.get('n_qubits', 0)
        features['path_length_k'] = metadata.get('K', 0)
        
        return features
    
    def train(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train XGBoost models for each parameter
        
        Args:
            features_df: Feature matrix
            targets_df: Target parameters
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Training XGBoost models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        metrics = {}
        
        # Train separate model for each parameter
        for param_name in self.models.keys():
            logger.info(f"Training model for {param_name}...")
            
            # XGBoost configuration
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=random_state,
                n_jobs=-1
            )
            
            # Train
            model.fit(
                X_train, y_train[param_name],
                eval_set=[(X_test, y_test[param_name])],
                verbose=False
            )
            
            self.models[param_name] = model
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            metrics[param_name] = {
                'train_mae': mean_absolute_error(y_train[param_name], y_pred_train),
                'test_mae': mean_absolute_error(y_test[param_name], y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train[param_name], y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test[param_name], y_pred_test)),
                'test_r2': r2_score(y_test[param_name], y_pred_test)
            }
            
            logger.info(f"  Test MAE: {metrics[param_name]['test_mae']:.4f}")
            logger.info(f"  Test R²: {metrics[param_name]['test_r2']:.4f}")
        
        # Overall metrics
        all_predictions = np.column_stack([
            self.models[p].predict(X_test) for p in self.models.keys()
        ])
        all_targets = y_test.values
        
        metrics['overall'] = {
            'test_mae': mean_absolute_error(all_targets, all_predictions),
            'test_rmse': np.sqrt(mean_squared_error(all_targets, all_predictions))
        }
        
        logger.info(f"\nOverall Test MAE: {metrics['overall']['test_mae']:.4f}")
        
        return metrics
    
    def predict(self, features: Dict) -> np.ndarray:
        """
        Predict optimal QAOA parameters for a new instance
        
        Args:
            features: Feature dictionary for the instance
            
        Returns:
            Array of predicted parameters [γ₁, γ₂, ..., β₁, β₂, ...]
        """
        if self.feature_names is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert features to DataFrame with correct column order
        feature_df = pd.DataFrame([features])[self.feature_names]
        
        # Predict each parameter
        predictions = []
        
        # Gamma parameters
        for i in range(self.p_layers):
            pred = self.models[f'gamma_{i}'].predict(feature_df)[0]
            predictions.append(pred)
        
        # Beta parameters
        for i in range(self.p_layers):
            pred = self.models[f'beta_{i}'].predict(feature_df)[0]
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_from_instance(self, instance_path: str) -> np.ndarray:
        """Predict parameters directly from instance file"""
        with open(instance_path, 'rb') as f:
            instance = pickle.load(f)
        
        features = self._extract_features(instance['metadata'])
        return self.predict(features)
    
    def save(self, filepath: str):
        """Save trained models"""
        save_data = {
            'models': self.models,
            'feature_names': self.feature_names,
            'p_layers': self.p_layers,
            'scaler_params': self.scaler_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.feature_names = save_data['feature_names']
        self.p_layers = save_data['p_layers']
        self.scaler_params = save_data.get('scaler_params')
        
        logger.info(f"Loaded model from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance across all models"""
        importance_data = []
        
        for param_name, model in self.models.items():
            importances = model.feature_importances_
            for feature, importance in zip(self.feature_names, importances):
                importance_data.append({
                    'parameter': param_name,
                    'feature': feature,
                    'importance': importance
                })
        
        df = pd.DataFrame(importance_data)
        
        # Average importance across parameters
        avg_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        return avg_importance


def main():
    """Example usage"""
    print("="*70)
    print("ML PARAMETER PREDICTOR - TRAINING EXAMPLE")
    print("="*70)
    
    # Note: This requires QAOA results to exist
    predictor = MLParameterPredictor(p_layers=1)
    
    # Check if training data exists
    if not Path("results/qaoa").exists():
        print("\n⚠️  No QAOA results found. Run QAOA solver first:")
        print("  python -m scripts.test_qaoa_solver")
        return
    
    try:
        # Prepare data
        features_df, targets_df = predictor.prepare_training_data()
        
        if len(features_df) == 0:
            print("\n⚠️  No training data found. Generate QAOA results first.")
            return
        
        # Train
        metrics = predictor.train(features_df, targets_df)
        
        # Show results
        print("\n" + "="*70)
        print("TRAINING RESULTS")
        print("="*70)
        print(f"\nOverall MAE: {metrics['overall']['test_mae']:.4f}")
        print(f"Overall RMSE: {metrics['overall']['test_rmse']:.4f}")
        
        print("\nPer-parameter metrics:")
        for param, m in metrics.items():
            if param != 'overall':
                print(f"  {param}: MAE={m['test_mae']:.4f}, R²={m['test_r2']:.4f}")
        
        # Save model
        Path("models").mkdir(exist_ok=True)
        predictor.save("models/ml_predictor_p1.pkl")
        
        print("\n✓ Model saved to models/ml_predictor_p1.pkl")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()