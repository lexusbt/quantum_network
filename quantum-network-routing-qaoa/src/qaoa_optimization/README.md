# IQM Hardware Integration
## ML-Enhanced QAOA Network Routing — Capstone Addendum

This module integrates IQM Resonance quantum hardware into the capstone pipeline. It sits alongside the existing Qiskit/AerSimulator codebase with **zero changes required to your existing QAOA solver or instance generator**.

---

## Files

| File | Purpose |
|---|---|
| `iqm_backend.py` | Backend factory — returns `(backend, sampler)` for Aer, IQM mock, or IQM hardware |
| `iqm_transpiler.py` | IQM gate translation, qubit count checks, resource estimation |
| `iqm_qaoa_runner.py` | Phase 1 + Phase 3 orchestration classes |
| `run_iqm_pipeline.py` | Main CLI entry point |

---

## Installation

```bash
# Add to your existing environment
conda activate quantum-routing

pip install "iqm-client[qiskit]>=33.0.1"
```

That's the only new dependency. Your existing `qiskit`, `qiskit-aer`, `scipy`, `numpy` stack is unchanged.

---

## Quick Start

### Test without an API token (uses AerSimulator)
```bash
python run_iqm_pipeline.py --mode simulator --phase 1
```

### Test with IQM noise model (requires token, no hardware quota used)
```bash
export IQM_TOKEN=your_resonance_token
python run_iqm_pipeline.py --mode mock --phase 1
```

### Run on real IQM Sirius hardware
```bash
# Phase 1: validate 20 instances
python run_iqm_pipeline.py --mode hardware --phase 1 --n-instances 20

# Phase 3: compare XGBoost warm-start vs random init
python run_iqm_pipeline.py --mode hardware --phase 3
```

---

## Qubit Budget

IQM Sirius has **20 qubits**. Your QUBO formulation uses 1 qubit per edge variable.

This means instances must have ≤ 20 edges. From your SNAP datasets:

| Dataset | Full size | Suitable subgraph size |
|---|---|---|
| ca-GrQc | 5,242 nodes | 8–15 edge subgraphs ✅ |
| email-Enron | 36,692 nodes | 8–15 edge subgraphs ✅ |
| p2p-Gnutella08 | 6,301 nodes | 8–15 edge subgraphs ✅ |

The `filter_instances_for_hardware()` function handles this automatically.

---

## Capstone Report Metrics

After running Phase 3, your `results/iqm_phase3/` directory will contain JSON files with:

- **`mean_speedup_pct`** — average % reduction in circuit evaluations (target: 70–90%)
- **`mean_quality_gap`** — approximation ratio: XGBoost warm-start vs. random init
- **`aer_hardware_correlation`** — Pearson r between simulation and hardware results (Phase 1)

These feed directly into Table 3 and Figure 5 of your capstone paper.

---

## Hardware Strategy

```
Training (200 instances)        Validation
─────────────────────────────   ─────────────────────────────
Aer simulation (all 200)    →   IQM hardware (20–30 instances)
Fast, free, no quota used       Confirms simulation-hardware gap
                                Anchors your hardware claims
```

Recommended run order:
1. `--mode mock` first — validate transpilation, check for errors, zero quota cost
2. `--mode hardware --phase 1` — 20 instances, ~30–60 min of hardware time
3. `--mode hardware --phase 3` — test set, ~45–90 min of hardware time
