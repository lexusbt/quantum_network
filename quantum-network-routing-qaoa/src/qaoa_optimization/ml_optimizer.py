"""
ML Parameter Predictor for QAOA
Trains XGBoost model to predict optimal parameters from problem features
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLParameterPredictor:
    """Predict optimal QAOA parameters from problem features"""
    
    def __init__(self, p_layers: int = 1):
        self.p_layers = p_layers
        self.n_params = 2 * p_layers
        
        # Separate model for gamma and beta
        self.gamma_model = None
        self.beta_model = None
        self.feature_names = None
        
        logger.info(f"Initialized ML predictor for p={p_layers} (2 parameters)")
    
    def prepare_training_data(
        self,
        results_dir: str = "results/qaoa_synthetic"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training data from QAOA results"""
        
        logger.info(f"Loading training data from {results_dir}")
        
        results_path = Path(results_dir)
        result_files = list(results_path.glob("*_p1.pkl"))
        
        features_list = []
        targets_list = []
        
        for result_file in result_files:
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            
            # Extract features from result metadata
            features = {
                'n_qubits': metadata['n_qubits'],
                'K': metadata['K'],
                'classical_optimal': metadata.get('classical_optimal_length', 3),
                # Add these if available in metadata:
                'avg_degree': metadata.get('avg_degree', 0),
                'density': metadata.get('density', 0),
                'clustering': metadata.get('clustering_coefficient', 0)
            }
            
            # Extract target parameters
            optimal_params = result['optimal_params']
            targets = {
                'gamma': optimal_params[0],
                'beta': optimal_params[1]
            }
            
            features_list.append(features)
            targets_list.append(targets)
        
        features_df = pd.DataFrame(features_list)
        targets_df = pd.DataFrame(targets_list)
        
        self.feature_names = features_df.columns.tolist()
        
        logger.info(f"Loaded {len(features_df)} training samples")
        logger.info(f"Features: {self.feature_names}")
        
        return features_df, targets_df
    
    def train(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """Train XGBoost models for gamma and beta"""
        
        logger.info("Training XGBoost models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_df,
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train gamma model
        logger.info("Training gamma model...")
        self.gamma_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
        self.gamma_model.fit(X_train, y_train['gamma'])
        
        # Train beta model
        logger.info("Training beta model...")
        self.beta_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
        self.beta_model.fit(X_train, y_train['beta'])
        
        # Evaluate
        gamma_pred_test = self.gamma_model.predict(X_test)
        beta_pred_test = self.beta_model.predict(X_test)
        
        metrics = {
            'gamma': {
                'test_mae': mean_absolute_error(y_test['gamma'], gamma_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test['gamma'], gamma_pred_test)),
                'test_r2': r2_score(y_test['gamma'], gamma_pred_test)
            },
            'beta': {
                'test_mae': mean_absolute_error(y_test['beta'], beta_pred_test),
                'test_rmse': np.sqrt(mean_squared_error(y_test['beta'], beta_pred_test)),
                'test_r2': r2_score(y_test['beta'], beta_pred_test)
            }
        }
        
        logger.info(f"Gamma - MAE: {metrics['gamma']['test_mae']:.4f}, R²: {metrics['gamma']['test_r2']:.4f}")
        logger.info(f"Beta - MAE: {metrics['beta']['test_mae']:.4f}, R²: {metrics['beta']['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, features: Dict) -> np.ndarray:
        """Predict optimal parameters for new instance"""
        
        if self.gamma_model is None or self.beta_model is None:
            raise ValueError("Model not trained")
        
        # Convert to DataFrame with correct feature order
        feature_df = pd.DataFrame([features])[self.feature_names]
        
        gamma = self.gamma_model.predict(feature_df)[0]
        beta = self.beta_model.predict(feature_df)[0]
        
        return np.array([gamma, beta])
    
    def predict_from_instance(self, instance_path: str) -> np.ndarray:
        """Predict parameters from instance file"""
        
        with open(instance_path, 'rb') as f:
            instance = pickle.load(f)
        
        metadata = instance['metadata']
        
        features = {
            'n_qubits': metadata['n_qubits'],
            'K': metadata['K'],
            'classical_optimal': metadata.get('classical_optimal_length', 3)
        }
        
        return self.predict(features)
    
    def save(self, filepath: str):
        """Save trained models"""
        
        save_data = {
            'gamma_model': self.gamma_model,
            'beta_model': self.beta_model,
            'feature_names': self.feature_names,
            'p_layers': self.p_layers
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load trained models"""
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.gamma_model = save_data['gamma_model']
        self.beta_model = save_data['beta_model']
        self.feature_names = save_data['feature_names']
        self.p_layers = save_data['p_layers']
        
        logger.info(f"Loaded model from {filepath}")


def main():
    """Train ML parameter predictor"""
    
    print("="*70)
    print("ML PARAMETER PREDICTOR - TRAINING")
    print("="*70)
    
    # Check if synthetic results exist
    if not Path("results/qaoa_synthetic").exists():
        print("\n⚠️  No synthetic QAOA results found.")
        print("Run first: python -m scripts.generate_synthetic_qaoa_results")
        return
    
    # Initialize predictor
    predictor = MLParameterPredictor(p_layers=1)
    
    # Load data
    features_df, targets_df = predictor.prepare_training_data()
    
    if len(features_df) == 0:
        print("\n⚠️  No training data found.")
        return
    
    # Train
    metrics = predictor.train(features_df, targets_df)
    
    # Display results
    print("\n" + "="*70)
    print("TRAINING RESULTS")
    print("="*70)
    
    print("\nGamma prediction:")
    print(f"  MAE: {metrics['gamma']['test_mae']:.4f}")
    print(f"  RMSE: {metrics['gamma']['test_rmse']:.4f}")
    print(f"  R²: {metrics['gamma']['test_r2']:.4f}")
    
    print("\nBeta prediction:")
    print(f"  MAE: {metrics['beta']['test_mae']:.4f}")
    print(f"  RMSE: {metrics['beta']['test_rmse']:.4f}")
    print(f"  R²: {metrics['beta']['test_r2']:.4f}")
    
    # Save model
    Path("models").mkdir(exist_ok=True)
    predictor.save("models/ml_predictor_p1.pkl")
    
    print("\n✓ Model saved to: models/ml_predictor_p1.pkl")
    
    # Test prediction
    print("\n" + "="*70)
    print("TEST PREDICTION")
    print("="*70)
    
    test_features = {
        'n_qubits': 16,
        'K': 3,
        'classical_optimal': 2
    }
    
    predicted_params = predictor.predict(test_features)
    
    print(f"Test problem: {test_features}")
    print(f"Predicted parameters:")
    print(f"  Gamma: {predicted_params[0]:.4f}")
    print(f"  Beta: {predicted_params[1]:.4f}")
    
    print("\n✓ ML training complete!")
    print("\nNext steps:")
    print("1. Test more instances on IQM: python -m scripts.test_iqm_batch")
    print("2. Compare ML vs optimization: python -m scripts.compare_ml_vs_qaoa")


if __name__ == "__main__":
    main()
