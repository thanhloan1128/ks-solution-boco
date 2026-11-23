# KS Solutions for Bi-Objective Combinatorial Optimization (ks-solution-boco)

Source code used to compute Kalai–Smorodinsky (KS) solutions for two
bi-objective combinatorial optimization problems:

- **BOKP** — Bi-Objective Knapsack Problem (`bokp/`)
- **BOSTP** — Bi-Objective Spanning Tree Problem (`bostp/`)

Each package implements the two-phase KS procedure (Phase I: discrete search
for the supporting segment / KS point; Phase II: refinement around the KS
point), the comparison methods used in the paper (Proportional Fairness and
Maxmin for BOKP; Network-Flow and Maxmin for BOSTP), and the exact scripts
that produced the paper's tables.

## Repository layout

```
bokp/                               Bi-Objective Knapsack Problem
├── src/bokp_ks/                    Library: instance generation, Phase I/II, PF, Maxmin
├── scripts/                        Scripts that reproduce the paper's experiments/tables
├── tests/                          Unit tests
├── results_archive_20261017_original/   1 instance per n, n=50..1000 (pilot run)
├── results_30_instances_20261017/  30 instances per n, n=50..1000 (main KS run)
├── pf_results_20261017_final2/     Proportional Fairness comparison results
└── pf_maxmin_ks_comparison_20261017/  Maxmin results + final PF/MM/KS comparison tables

bostp/                              Bi-Objective Spanning Tree Problem
├── src/bostp/                      Library: instance generation, Phase I/II, comparison runner
├── tests/                          Unit tests
├── results_corrected_stage1/                 n in {80, 100}, master seed 12345
├── results_corrected_stage2_seed20261101/    n in {120, 140}, master seed 20261101
├── results_corrected_stage3_seed20261102/    n in {160, 180}, master seed 20261102
├── results_corrected_combined/     Final paper table combining the three stages
└── results_nf_minmax/              Network-Flow/Maxmin comparison results
```

## Software and versions

| Package | Python   | Required                       | Optional                                |
|---------|----------|---------------------------------|------------------------------------------|
| bokp    | >= 3.11  | `gurobipy==13.0.3`              | `pytest>=8.0` (dev/tests)                 |
| bostp   | >= 3.10  | `networkx>=3.2`                 | `gurobipy>=11.0` (Phase II), `pytest>=8.0` |

`gurobipy` requires a valid Gurobi license (e.g. a free academic license from
https://www.gurobi.com). All experiments were run with `gurobi_seed=271828`
(BOKP) and thread count 1 (BOSTP), as recorded in each run's metadata file.

## Input data

Instances are generated deterministically from a seeded RNG, and the exact
instances used in the paper are included so results can be reproduced or
inspected without regenerating anything:

- **BOKP**: `bokp/results_30_instances_20261017/instances/` (30 instances per
  n, n = 50..1000, step 50) and `bokp/results_archive_20261017_original/`
  (1 instance per n). Base seed `20261017`; per-instance seed is
  `BASE_SEED + n * 1009 + (instance_index - 1)`. Generation ranges: item
  weight in [1, 20], profit `p` in [1, 30], profit `q` in [10, 40].
- **BOSTP**: `bostp/results_corrected_stage*/.../data/instances/`. Connected
  Erdős–Rényi graphs with n in {80, 100, 120, 140, 160, 180}, edge
  probability p in {0.3, 0.5, 0.7}, 2 instances per (n, p) pair, edge cost
  range [20, 120] and edge time range [1, 100]. Master seeds: `12345`
  (n=80,100), `20261101` (n=120,140), `20261102` (n=160,180).

To regenerate instances from scratch, use `InstanceGenerator` in each
package's `instance_generator.py` with the seeds above; the generator is
deterministic given the same seed.

## Example run

### BOKP

```bash
cd bokp
python -m pip install -e '.[dev]'
python -m pytest
python -m bokp_ks --output results_demo --instances-per-n 1
```

This solves one instance for every n in {50, 100, ..., 1000} and writes
Phase I/II results under `results_demo/`. The scripts that reproduce the
paper's full tables are run from `bokp/`:

```bash
python scripts/run_30_experiment.py        # KS results for 30 instances per n
python scripts/run_pf_experiment.py        # Proportional Fairness comparison
python scripts/run_maxmin_comparison.py    # Maxmin + final PF/MM/KS comparison
```

### BOSTP

```bash
cd bostp
python -m pip install -e '.[dev]'
python -m pytest
python -m bostp.cli smoke --output results_demo
```

`smoke` solves a single small instance as a sanity check. The full
experiment and the Network-Flow/Maxmin comparison used in the paper are:

```bash
python -m bostp.cli full --master-seed 12345 --output results_demo
python -m bostp.comparison_runner --stage 1 --output results_nf_minmax_demo
```

## License

MIT License — see [LICENSE](LICENSE).
