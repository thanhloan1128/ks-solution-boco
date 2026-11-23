from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import importlib.util
import hashlib
import json
import networkx
from time import monotonic
from .config import ExperimentConfig
from .instance_generator import InstanceGenerator
from .instance import BOSTPInstance
from .ks_solver import KSSolver
from .phase_two import PhaseTwoSearch
from .result_writer import ResultWriter


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, master_seed=None):
        self.config, self.master_seed = config, master_seed

    def run(self, output: Path, full: bool = True, resume: Path | None = None,
            overall_budget_seconds: float | None = None,
            instance_budget_seconds: float | None = None,
            instances_from: Path | None = None,
            phase2_gap_tolerance: float = PhaseTwoSearch.optimality_gap_tolerance) -> Path:
        if resume is not None and instances_from is not None:
            raise ValueError("use either --resume or --instances-from, not both")
        writer = ResultWriter(output, resume)
        logger = logging.getLogger(f"bostp.run.{writer.directory.name}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(writer.directory / "run.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        try:
            return self._run(
                writer, full, resume, overall_budget_seconds, instance_budget_seconds, logger,
                instances_from, phase2_gap_tolerance,
            )
        finally:
            logger.removeHandler(handler)
            handler.close()

    def _run(self, writer, full, resume, overall_budget_seconds,
             instance_budget_seconds, logger, instances_from=None,
             phase2_gap_tolerance=PhaseTwoSearch.optimality_gap_tolerance):
        if not 0 <= phase2_gap_tolerance < 1:
            raise ValueError("relative MIP gap tolerance must be in [0, 1)")
        if instances_from is not None:
            source_metadata = json.loads((instances_from / "run_metadata.json").read_text())
            if source_metadata["config"] != self.config.as_dict():
                raise ValueError("source instance configuration does not match")
            instances = sorted(
                (BOSTPInstance.load(path)
                 for path in (instances_from / "data" / "instances").glob("*.json")),
                key=lambda instance: (instance.n, instance.p, instance.instance_id),
            )
            expected_ids = source_metadata.get("retained_instance_ids")
            if not instances or (expected_ids is not None
                                 and {item.instance_id for item in instances} != set(expected_ids)):
                raise ValueError("source instance files do not match the saved manifest")
            if expected_ids is None and len(instances) != source_metadata["accepted_instances"]:
                raise ValueError("source instance count does not match generation metadata")
            for instance in instances:
                instance.save(writer.directory / "data" / "instances" / f"{instance.instance_id}.json")
            metadata = self._metadata(writer, instances, source_metadata["master_seed"])
            metadata.update({
                "dataset_origin": "existing saved instances; no regeneration",
                "source_run": str(instances_from.resolve()),
                "source_generation_metadata": source_metadata,
                "retained_instance_ids": [instance.instance_id for instance in instances],
                "retained_instance_count": len(instances),
            })
            writer.write_metadata(metadata)
        elif resume is None:
            generator = InstanceGenerator(self.config, self.master_seed)
            instances_dir = writer.directory / "data" / "instances"
            if full:
                instances = generator.generate_all(instances_dir)
            else:
                instances = generator.generate_pair(8, 0.5, 2)
                for instance in instances:
                    instance.save(instances_dir / f"{instance.instance_id}.json")
            metadata = self._metadata(writer, instances, generator.master_seed)
            writer.write_metadata(metadata)
        else:
            metadata = json.loads((writer.directory / "run_metadata.json").read_text())
            if metadata.get("phase2_formulation") not in {
                PhaseTwoSearch.formulation, "mtz-static-v1-gap1e-5",
            }:
                raise ValueError(
                    "saved run uses a different or unverified Phase II formulation; "
                    "create a separate run using the saved instances, not legacy checkpoints"
                )
            if metadata["config"] != self.config.as_dict():
                raise ValueError("resume configuration does not match the saved run")
            instances = sorted(
                (BOSTPInstance.load(path) for path in (writer.directory / "data" / "instances").glob("*.json")),
                key=lambda instance: (instance.n, instance.p, instance.instance_id),
            )
            retained_ids = metadata.get("retained_instance_ids")
            if retained_ids is not None:
                if len(retained_ids) != metadata.get("retained_instance_count"):
                    raise ValueError("retained instance manifest does not match its count")
                if {instance.instance_id for instance in instances} != set(retained_ids):
                    raise ValueError("saved instances do not match the retained instance manifest")
            elif len(instances) != metadata["accepted_instances"]:
                raise ValueError("saved instance count does not match run metadata")

        if overall_budget_seconds is not None and overall_budget_seconds <= 0:
            raise ValueError("overall budget must be positive")
        if instance_budget_seconds is not None and instance_budget_seconds <= 0:
            raise ValueError("per-instance budget must be positive")
        overall_deadline = (
            monotonic() + overall_budget_seconds if overall_budget_seconds is not None else None
        )
        started_at = datetime.now(timezone.utc)
        metadata["phase2_formulation"] = PhaseTwoSearch.formulation
        metadata["phase2"] = (
            "Gurobi static MTZ formulation; relative MIP gap "
            f"{phase2_gap_tolerance:g} and absolute gap 0"
        )
        metadata["phase2_gap_tolerance"] = phase2_gap_tolerance
        metadata.setdefault("phase2_gap_tolerance_runs", []).append({
            "started_at": started_at.isoformat(),
            "relative_mip_gap_tolerance": phase2_gap_tolerance,
            "source_sha256": self._source_fingerprint(),
        })
        metadata.setdefault("budget_windows", []).append({
            "started_at": started_at.isoformat(),
            "deadline_at": (
                (started_at + timedelta(seconds=overall_budget_seconds)).isoformat()
                if overall_budget_seconds is not None else None
            ),
            "overall_budget_seconds": overall_budget_seconds,
            "instance_budget_seconds": instance_budget_seconds,
            "phase2_gap_tolerance": phase2_gap_tolerance,
        })
        metadata["run_status"] = "IN_PROGRESS"
        writer.write_metadata(metadata)

        completed = writer.load_checkpoints()
        if any(result.get("phase2_formulation") not in {
            PhaseTwoSearch.formulation, "mtz-static-v1-gap1e-5",
        }
               for result in completed.values()):
            raise ValueError("checkpoint formulation does not match the current Phase II solver")
        instance_ids = {instance.instance_id for instance in instances}
        if not set(completed).issubset(instance_ids):
            raise ValueError("checkpoint contains an instance outside this run")
        logger.info(
            "Run policy: overall budget=%s seconds; shared instance budget=%s seconds",
            overall_budget_seconds, instance_budget_seconds,
        )
        if instance_budget_seconds is not None:
            metadata["checkpointed_instances"] = len(completed)
            metadata["exact_completed_instances"] = sum(
                item.get("status") == "COMPLETED" for item in completed.values()
            )
            writer.write(instances, [
                completed.get(item.instance_id, {
                    "instance_id": item.instance_id, "status": "NOT_STARTED",
                }) for item in instances
            ], metadata)

        target = next((instance for instance in instances if instance.n == 180 and instance.p == 0.7), None)
        ordered = list(instances)
        if full:
            coverage = []
            for n in self.config.sizes:
                for p in self.config.probabilities:
                    representative = next(
                        (instance for instance in instances
                         if instance.n == n and instance.p == p
                         and instance.instance_id.endswith("_i001")),
                        None,
                    )
                    if representative is not None:
                        coverage.append(representative)
            priority = ([target] if target is not None else []) + [
                instance for instance in coverage if instance is not target
            ]
            ordered = priority + [instance for instance in ordered if instance not in priority]

        if full and target is not None and target.instance_id in completed:
            if completed[target.instance_id].get("status") != "COMPLETED":
                del completed[target.instance_id]

        for instance in ordered:
            previous = completed.get(instance.instance_id, {})
            if previous.get("status") == "COMPLETED":
                continue
            if (instance_budget_seconds is not None
                    and previous.get("budget_exhausted")
                    and previous.get("instance_budget_seconds", 0) >= instance_budget_seconds):
                logger.info("Deferring previously timed-out instance %s", instance.instance_id)
                continue
            if overall_deadline is not None and monotonic() >= overall_deadline:
                logger.info("Overall budget exhausted; remaining instances are resumable")
                break
            instance_deadline = None
            if instance_budget_seconds is not None:
                instance_deadline = monotonic() + instance_budget_seconds
            if overall_deadline is not None:
                instance_deadline = (
                    overall_deadline if instance_deadline is None
                    else min(instance_deadline, overall_deadline)
                )
            instance_started = monotonic()
            logger.info("Starting instance %s", instance.instance_id)
            writer.write_run_state({
                "current_instance": instance.instance_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })
            solver = KSSolver(
                self.config.phase2_time_limit, self.config.gurobi_threads,
                phase2_gap_tolerance,
            )
            result = solver.solve(instance, deadline=instance_deadline)
            result["phase2_formulation"] = PhaseTwoSearch.formulation
            result["phase2_gap_tolerance"] = phase2_gap_tolerance
            result["instance_elapsed_seconds"] = monotonic() - instance_started
            result["instance_budget_seconds"] = instance_budget_seconds
            result["budget_exhausted"] = (
                result.get("status") != "COMPLETED"
                and instance_deadline is not None and monotonic() >= instance_deadline
            )
            writer.write_checkpoint(result)
            completed[instance.instance_id] = result
            writer.write_run_state({"current_instance": None})
            logger.info("Finished instance %s with status %s", instance.instance_id, result.get("status"))
            if instance_budget_seconds is not None:
                report_results = [
                    completed.get(item.instance_id, {
                        "instance_id": item.instance_id, "status": "NOT_STARTED",
                    }) for item in instances
                ]
                metadata["checkpointed_instances"] = len(completed)
                metadata["exact_completed_instances"] = sum(
                    item.get("status") == "COMPLETED" for item in completed.values()
                )
                writer.write(instances, report_results, metadata)
            if (full and resume is None and instances_from is None and instance is target
                    and result.get("status") != "COMPLETED"):
                metadata["run_status"] = "STOPPED_TEST_INSTANCE_INCOMPLETE"
                results = [completed[item.instance_id] for item in instances if item.instance_id in completed]
                writer.write([item for item in instances if item.instance_id in completed], results, metadata)
                raise RuntimeError(
                    f"test instance {instance.instance_id} did not complete; remaining instances were not run"
                )
            if overall_deadline is not None and monotonic() >= overall_deadline:
                logger.info("Overall budget exhausted after %s", instance.instance_id)
                break

        results = [completed[instance.instance_id] for instance in instances if instance.instance_id in completed]
        exact_count = sum(result.get("status") == "COMPLETED" for result in results)
        all_exact = len(results) == len(instances) and exact_count == len(instances)
        metadata["run_status"] = (
            "COMPLETED" if all_exact else
            "INCOMPLETE_RESULTS" if len(results) == len(instances) else
            "BUDGET_EXHAUSTED" if overall_deadline is not None and monotonic() >= overall_deadline else
            "IN_PROGRESS"
        )
        metadata["checkpointed_instances"] = len(results)
        metadata["exact_completed_instances"] = exact_count
        if all_exact:
            group_counts = {}
            for instance, result in zip(instances, results):
                if result.get("status") == "COMPLETED":
                    key = (instance.n, instance.p)
                    group_counts[key] = group_counts.get(key, 0) + 1
            expected_pairs = {
                (n, p) for n in self.config.sizes for p in self.config.probabilities
            }
            expected_count = (
                metadata["retained_instance_count"] // len(expected_pairs)
                if metadata.get("retained_instance_ids") is not None
                else self.config.instances_per_pair
            )
            if full and (
                set(group_counts) != expected_pairs
                or any(count != expected_count for count in group_counts.values())
            ):
                raise ValueError("final table requires all selected instances certified in each configuration")
            writer.write(instances, results, metadata)
        elif instance_budget_seconds is not None:
            report_results = [
                completed.get(item.instance_id, {
                    "instance_id": item.instance_id, "status": "NOT_STARTED",
                }) for item in instances
            ]
            writer.write(instances, report_results, metadata)
        else:
            writer.write_metadata(metadata)
        return writer.directory

    def _metadata(self, writer, instances, master_seed):
        return {
            "run_id": writer.directory.name,
            "master_seed": master_seed,
            "dataset_origin": "new locally generated dataset; original instance files and seed metadata were unavailable",
            "seed_provenance": "fixed master seed supplied for this run",
            "config": self.config.as_dict(),
            "accepted_instances": len(instances),
            "graph_generation": "networkx.gnp_random_graph; disconnected candidates rejected",
            "rejected_candidates_by_pair": self._rejection_metadata(instances),
            "solver_versions": {"networkx": networkx.__version__, "gurobipy": self._gurobi_version()},
            "source_sha256": self._source_fingerprint(),
            "phase2": (
                "Gurobi static MTZ formulation; relative MIP gap "
                f"{PhaseTwoSearch.optimality_gap_tolerance:g} and absolute gap 0"
            ),
            "phase2_formulation": PhaseTwoSearch.formulation,
            "phase2_gap_tolerance": PhaseTwoSearch.optimality_gap_tolerance,
            "run_status": "IN_PROGRESS",
        }

    @staticmethod
    def _gurobi_version():
        if importlib.util.find_spec("gurobipy") is None:
            return None
        import gurobipy
        return ".".join(str(value) for value in gurobipy.gurobi.version())

    @staticmethod
    def _source_fingerprint():
        digest = hashlib.sha256()
        root = Path(__file__).parent
        for path in sorted(root.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _rejection_metadata(instances):
        result = {}
        for instance in instances:
            key = f"n{instance.n}_p{instance.p:.1f}"
            group = result.setdefault(key, {"accepted_instances": [], "accepted_seeds": [],
                                            "rejected_candidates": 0, "rejected_seeds": []})
            group["accepted_instances"].append(instance.instance_id)
            group["accepted_seeds"].append(instance.seed)
            group["rejected_candidates"] += instance.rejected_candidates
            group["rejected_seeds"].extend(instance.rejected_seeds or [])
        return result