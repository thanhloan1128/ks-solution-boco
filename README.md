# ks-solution-boco

Source code, instances, and comparison scripts for computing Kalai-Smorodinsky (KS) solutions of the Bi-Objective Knapsack Problem (`bokp/`) and the Bi-Objective Spanning Tree Problem (`bostp/`).

## Requirements

- Python 3.11+ for `bokp/`, Python 3.10+ for `bostp/`.
- Gurobi solver via `gurobipy`: version `13.0.3` for `bokp/` (required), version `>=11.0` for `bostp/` (optional, used by Phase II and the comparison scripts). You need a valid Gurobi license (a free academic license from https://www.gurobi.com works) to run any Gurobi-based step.
- `networkx>=3.2` for `bostp/`.
- `pytest>=8.0` to run the tests (optional).

## How to run

Knapsack (BOKP), one small instance (n=50):
```bash
cd bokp
python -m pip install -e '.[dev]'
python -m bokp_ks --ns 50 --output results_demo --instances-per-n 1
```
Input: reuses the saved instance `results_30_instances_20261017/instances/bokp_n0050_i001.json`
(generated with base seed `20261017` if not already present).
Result: `results_demo/instance_results/bokp_n0050_i001.json`, field `final_ks_points`.

Spanning tree (BOSTP), one small instance (n=80, p=0.3):
```bash
cd bostp
python -m pip install -e '.[dev]'
python -m bostp.cli full --master-seed 12345 --sizes 80 --probabilities 0.3 --instances-per-pair 1 --output results_demo
```
Input: generates instance `bostp_n0080_p03_i001` (master seed `12345`, same as
`results_corrected_stage1/`) under `results_demo/<timestamp>/data/instances/`.
Result: `results_demo/<timestamp>/checkpoints/bostp_n0080_p03_i001.json`, field `final_ks_points`.
