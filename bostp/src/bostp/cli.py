from __future__ import annotations

import argparse
from pathlib import Path
from .config import ExperimentConfig
from .experiment_runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible BOSTP KS experiments")
    parser.add_argument("mode", choices=("smoke", "full"))
    parser.add_argument("--master-seed", type=int)
    parser.add_argument("--output", default="results")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--instances-from", type=Path,
                        help="start a separate run from saved instance data without legacy checkpoints")
    parser.add_argument("--overall-budget-seconds", type=float)
    parser.add_argument("--instance-budget-seconds", type=float)
    parser.add_argument("--phase2-gap-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    path = ExperimentRunner(ExperimentConfig(), args.master_seed).run(
        Path(args.output), args.mode == "full", args.resume,
        args.overall_budget_seconds, args.instance_budget_seconds, args.instances_from,
        args.phase2_gap_tolerance,
    )
    print(path)


if __name__ == "__main__":
    main()