from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BOKP KS experiment")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--instances-per-n", type=int, default=1)
    parser.add_argument(
        "--ns", type=int, nargs="+", default=list(range(50, 1001, 50)),
        help="item counts to solve, e.g. '--ns 50' for a single quick instance (default: 50..1000 step 50)",
    )
    args = parser.parse_args()
    metadata = ExperimentRunner(args.output).run(args.ns, args.instances_per_n)
    print(f"Finished {metadata['instance_count']} instances in {metadata['total_wall_seconds']:.3f}s")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()