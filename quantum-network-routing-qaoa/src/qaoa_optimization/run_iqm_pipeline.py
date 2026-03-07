"""
run_iqm_pipeline.py
===================
Main entry point for the IQM hardware integration pipeline.

This script demonstrates the full capstone workflow end-to-end:

  Phase 1 (Hardware Validation):
    - Load your 200 routing instances
    - Filter to hardware-suitable ones (≤ 20 qubits for IQM Sirius)
    - Run ~20 instances on IQM hardware
    - Compare with AER simulation results
    - Save correlation metrics for capstone report

  Phase 3 (ML Inference Validation):
    - Load trained XGBoost model
    - Run test set on IQM hardware with two strategies:
        A) XGBoost warm-start
        B) Random initialization (baseline)
    - Compute speedup % and solution quality metrics
    - Save results for capstone report

Usage:
    # Development (no API key, full simulation)
    python run_iqm_pipeline.py --mode simulator

    # Pre-hardware testing with IQM noise model
    python run_iqm_pipeline.py --mode mock --token YOUR_TOKEN

    # Real IQM hardware
    python run_iqm_pipeline.py --mode hardware --token YOUR_TOKEN --phase 1
    python run_iqm_pipeline.py --mode hardware --token YOUR_TOKEN --phase 3

Environment variable alternative:
    export IQM_TOKEN=your_token
    python run_iqm_pipeline.py --mode hardware --phase 3
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Add your project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iqm_backend import BackendMode
from iqm_qaoa_runner import IQMHardwareValidator, IQMInferenceValidator
from iqm_transpiler import filter_instances_for_hardware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/iqm_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="IQM Hardware Pipeline for QAOA Routing Capstone")
    parser.add_argument(
        "--mode",
        choices=["simulator", "mock", "hardware"],
        default="simulator",
        help="Execution mode: simulator (AER), mock (IQM noise model), hardware (real IQM)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 3],
        default=1,
        help="Pipeline phase: 1 (hardware validation) or 3 (ML inference validation)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="IQM Resonance API token (or set IQM_TOKEN env variable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="sirius",
        help="IQM device name (default: sirius)",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=20,
        help="Number of instances to run on hardware (Phase 1)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots per circuit",
    )
    parser.add_argument(
        "--instances-path",
        type=str,
        default="instances/routing_instances.pkl",
        help="Path to your routing instances pickle file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/xgboost_qaoa_params.pkl",
        help="Path to trained XGBoost model (Phase 3)",
    )
    return parser.parse_args()


def load_instances(path: str) -> list:
    """Loads your 200 routing problem instances."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"Instances file not found at {path}. Using synthetic demo instances.")
        return _create_demo_instances(n=30)

    with open(path, "rb") as f:
        instances = pickle.load(f)

    logger.info(f"Loaded {len(instances)} instances from {path}")
    return instances


def load_xgb_model(path: str):
    """Loads your trained XGBoost model."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"XGBoost model not found at {path}. Using mock model for demo.")
        return MockXGBModel()

    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded XGBoost model from {path}")
    return model


def load_qaoa_solver():
    """Loads your existing QAOASolver from the capstone codebase."""
    try:
        from qaoa_optimization.qaoa_solver import QAOASolver
        return QAOASolver()
    except ImportError:
        logger.warning("QAOASolver not found in path. Using MockQAOASolver for demo.")
        return MockQAOASolver()


def run_phase1(args, instances, qaoa_solver):
    """Phase 1: Hardware validation of Aer simulation results."""
    logger.info("=" * 60)
    logger.info("PHASE 1: IQM Hardware Validation")
    logger.info("=" * 60)

    use_mock = args.mode == "mock"
    validator = IQMHardwareValidator(
        api_token=args.token,
        device=args.device,
        shots=args.shots,
        use_mock=use_mock,
    )

    results = validator.validate_subset(
        instances=instances,
        qaoa_solver=qaoa_solver,
        n_instances=args.n_instances,
        p_layers=1,  # Use p=1 for hardware validation (keeps circuit depth manageable)
    )

    output_path = validator.save_results(results)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 Summary:")
    summary = results["summary"]
    logger.info(f"  Instances run:           {summary.get('n_successful', 0)}")
    logger.info(f"  Mean HW approx ratio:    {summary.get('mean_hw_approximation_ratio', 'N/A')}")
    logger.info(f"  AER-HW correlation:      {summary.get('aer_hardware_correlation', 'N/A')}")
    logger.info(f"  Results saved to:        {output_path}")
    logger.info("=" * 60)

    return results


def run_phase3(args, instances, xgb_model, qaoa_solver):
    """Phase 3: ML inference validation on IQM hardware."""
    logger.info("=" * 60)
    logger.info("PHASE 3: ML Parameter Inference Validation on IQM")
    logger.info("=" * 60)

    use_mock = args.mode == "mock"
    validator = IQMInferenceValidator(
        api_token=args.token,
        device=args.device,
        shots=args.shots,
        use_mock=use_mock,
    )

    # Use test split (last 30 instances by convention from your 140/30/30 split)
    test_instances = instances[-30:]

    results = validator.compare_strategies(
        test_instances=test_instances,
        xgb_model=xgb_model,
        qaoa_solver=qaoa_solver,
        p_layers=1,
        max_iter=30,
    )

    output_path = validator.save_results(results)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 Summary:")
    summary = results["summary"]
    logger.info(f"  Instances compared:      {summary.get('n_successful', 0)}")
    logger.info(f"  Mean speedup:            {summary.get('mean_speedup_pct', 'N/A')}%")
    logger.info(f"  Speedup range:           {summary.get('min_speedup_pct', 'N/A')}% — {summary.get('max_speedup_pct', 'N/A')}%")
    logger.info(f"  Mean quality gap:        {summary.get('mean_quality_gap', 'N/A')}")
    logger.info(f"  Results saved to:        {output_path}")
    logger.info("=" * 60)

    return results


def main():
    args = parse_args()
    Path("logs").mkdir(exist_ok=True)

    logger.info(f"IQM Pipeline starting — mode: {args.mode}, phase: {args.phase}")

    # Load resources
    instances = load_instances(args.instances_path)
    qaoa_solver = load_qaoa_solver()

    if args.phase == 1:
        run_phase1(args, instances, qaoa_solver)
    elif args.phase == 3:
        xgb_model = load_xgb_model(args.model_path)
        run_phase3(args, instances, xgb_model, qaoa_solver)


# ---------------------------------------------------------------------------
# Demo/mock classes for testing without full capstone codebase
# ---------------------------------------------------------------------------

class MockQAOASolver:
    """Minimal mock solver for testing IQM integration standalone."""

    def build_circuit(self, qubo_matrix: np.ndarray, p: int = 1):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        n = qubo_matrix.shape[0]
        gamma = ParameterVector("γ", p)
        beta = ParameterVector("β", p)

        qc = QuantumCircuit(n)
        qc.h(range(n))  # Initial superposition

        for layer in range(p):
            # Cost layer — RZZ for each QUBO edge
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(qubo_matrix[i, j]) > 1e-8:
                        weight = float(qubo_matrix[i, j])
                        qc.rzz(2 * gamma[layer] * weight, i, j)

            # Mixer layer
            for i in range(n):
                qc.rx(2 * beta[layer], i)

        qc.measure_all()
        initial_params = np.random.uniform(0, np.pi, 2 * p)
        return qc, initial_params


class MockXGBModel:
    """Returns random parameter predictions for demo purposes."""

    def predict(self, X):
        return np.random.uniform(0, np.pi, (X.shape[0], 2))


def _create_demo_instances(n: int = 30) -> list:
    """Creates synthetic routing instances for standalone testing."""
    instances = []
    for i in range(n):
        size = np.random.randint(4, 12)  # Small instances that fit on IQM Sirius
        qubo = np.random.randn(size, size) * 0.5
        qubo = (qubo + qubo.T) / 2  # Symmetrize

        instances.append({
            "id": f"demo_{i:03d}",
            "n_qubits": size,
            "qubo_matrix": qubo,
            "features": {
                "n_nodes": size,
                "n_edges": size * (size - 1) // 2,
                "density": 0.6,
                "avg_degree": size - 1,
                "clustering": 0.4,
            },
            "aer_approximation_ratio": np.random.uniform(0.7, 0.98),
            "classical_optimal_cost": float(np.random.uniform(-5, -1)),
            "dataset": "demo",
        })
    return instances


if __name__ == "__main__":
    main()
