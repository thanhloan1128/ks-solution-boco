import json
from pathlib import Path
from fractions import Fraction
from dataclasses import replace
from bostp.config import ExperimentConfig
from bostp.instance_generator import InstanceGenerator
from bostp.reference_points import ReferencePointCalculator
from bostp.spanning_tree_model import MSTSolver
from bostp.ks_solver import KSSolver
from bostp.phase_two import PhaseTwoSearch
from bostp.spanning_tree_model import TreeSolution
from bostp.result_writer import ResultWriter
from bostp.experiment_runner import ExperimentRunner


def test_generation_is_reproducible(tmp_path: Path):
    config = ExperimentConfig(instances_per_pair=2)
    first = InstanceGenerator(config, 17).generate_pair(8, 0.5)
    second = InstanceGenerator(config, 17).generate_pair(8, 0.5)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]

def test_result_checkpoints_can_be_loaded(tmp_path: Path):
    writer = ResultWriter(tmp_path)
    result = {"instance_id": "fixture_1", "status": "COMPLETED"}
    writer.write_checkpoint(result)
    assert writer.load_checkpoints() == {"fixture_1": result}
    resumed_writer = ResultWriter(tmp_path, writer.directory)
    assert resumed_writer.load_checkpoints() == {"fixture_1": result}

def test_instance_timings_distinguish_certified_timeout_and_interrupted(tmp_path):
    import csv

    writer = ResultWriter(tmp_path)
    instances = InstanceGenerator(ExperimentConfig(), 17).generate_pair(8, 0.5, 2)
    results = [
        {"instance_id": instances[0].instance_id, "status": "COMPLETED", "total_seconds": 90},
        {"instance_id": instances[1].instance_id, "status": "INCOMPLETE",
         "instance_elapsed_seconds": 1200, "total_seconds": 1199, "budget_exhausted": True},
    ]
    writer.write_instance_timings(instances, results)
    rows = list(csv.DictReader((writer.directory / "instance_timings.csv").open()))
    assert rows[0]["elapsed_minutes"] == "1.5"
    assert rows[0]["status"] == "COMPLETED"
    assert rows[1]["elapsed_minutes"] == "20.0"
    assert rows[1]["status"] == "TIME_LIMIT"
    assert "1.50 & Certified & 20.00 & Time limit" in (
        writer.directory / "instance_timings.tex"
    ).read_text()

    results[1].update({
        "interrupted_elapsed_lower_bound_seconds": 7200,
        "deferral_reason": "Long retry interrupted; saved 300-second attempt is older.",
    })
    writer.write_instance_timings(instances, results)
    rows = list(csv.DictReader((writer.directory / "instance_timings.csv").open()))
    assert rows[1]["elapsed_seconds"] == ""
    assert rows[1]["status"] == "INTERRUPTED"
    assert "$>120.00$ & Interrupted" in (
        writer.directory / "instance_timings.tex"
    ).read_text()

def test_full_run_prioritizes_configuration_coverage_and_logs(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(sizes=(8, 10), probabilities=(0.3, 0.7), instances_per_pair=2)
    solved = []

    class FakeSolver:
        def __init__(self, *args):
            pass

        def solve(self, instance, deadline=None):
            solved.append(instance.instance_id)
            return {
                "instance_id": instance.instance_id,
                "status": "COMPLETED",
                "reference_seconds": 0,
                "phase1_seconds": 0,
                "phase2_P_seconds": 0,
                "phase2_Q_seconds": 0,
                "total_seconds": 0,
                "phase1": {"signed_rho": [], "midpoint_iterations": 0, "final_solves": 0},
                "phase2": {
                    "P": {"midpoint_iterations": 0, "signed_rho": None},
                    "Q": {"midpoint_iterations": 0, "signed_rho": None},
                },
            }

    monkeypatch.setattr("bostp.experiment_runner.KSSolver", FakeSolver)
    run = ExperimentRunner(config, 19).run(
        tmp_path / "results", full=True,
        overall_budget_seconds=30, instance_budget_seconds=5,
    )

    assert solved == [
        "bostp_n0008_p03_i001", "bostp_n0008_p07_i001",
        "bostp_n0010_p03_i001", "bostp_n0010_p07_i001",
        "bostp_n0008_p03_i002", "bostp_n0008_p07_i002",
        "bostp_n0010_p03_i002", "bostp_n0010_p07_i002",
    ]
    assert len(list((run / "data" / "instances").glob("*.json"))) == 8
    assert len(list((run / "checkpoints").glob("*.json"))) == 8
    assert (run / "run.log").is_file()

def test_resume_uses_retained_subset_and_requires_two_exact_results(tmp_path: Path, monkeypatch):
    config = ExperimentConfig(sizes=(8,), probabilities=(0.5,), instances_per_pair=3)
    runner = ExperimentRunner(config, 23)
    writer = ResultWriter(tmp_path / "results")
    generated = InstanceGenerator(config, 23).generate_pair(8, 0.5, 3)
    instances_dir = writer.directory / "data" / "instances"
    for instance in generated[:2]:
        instance.save(instances_dir / f"{instance.instance_id}.json")
    metadata = runner._metadata(writer, generated, 23)
    metadata["phase2_formulation"] = "mtz-static-v1-gap1e-5"
    metadata["retained_instance_ids"] = [instance.instance_id for instance in generated[:2]]
    metadata["retained_instance_count"] = 2
    writer.write_metadata(metadata)

    def completed_result(instance_id):
        return {
            "instance_id": instance_id,
            "phase2_formulation": "mtz-static-v1-gap1e-5",
            "status": "COMPLETED",
            "reference_seconds": 0.1,
            "phase1_seconds": 0.1,
            "phase2_P_seconds": 0.1,
            "phase2_Q_seconds": 0.1,
            "total_seconds": 0.3,
            "phase1": {"signed_rho": [], "midpoint_iterations": 0, "final_solves": 0},
            "phase2": {
                "P": {"midpoint_iterations": 0, "signed_rho": None},
                "Q": {"midpoint_iterations": 0, "signed_rho": None},
            },
        }

    writer.write_checkpoint(completed_result(generated[0].instance_id))

    class FakeSolver:
        def __init__(self, *args):
            pass

        def solve(self, instance, deadline=None):
            return completed_result(instance.instance_id)

    monkeypatch.setattr("bostp.experiment_runner.KSSolver", FakeSolver)
    run = runner.run(
        tmp_path / "results", full=True, resume=writer.directory,
        phase2_gap_tolerance=5e-4,
    )

    final_metadata = json.loads((run / "run_metadata.json").read_text())
    assert final_metadata["accepted_instances"] == 3
    assert final_metadata["retained_instance_count"] == 2
    assert final_metadata["run_status"] == "COMPLETED"
    assert final_metadata["phase2_gap_tolerance"] == 5e-4
    assert final_metadata["phase2_formulation"] == PhaseTwoSearch.formulation
    assert (run / "ks_table.tex").is_file()
    assert len(list((run / "checkpoints").glob("*.json"))) == 2
    before = (run / "run_metadata.json").read_bytes()

    def forbidden_generation(*args):
        raise AssertionError("existing dataset must not be regenerated")

    monkeypatch.setattr(InstanceGenerator, "generate_all", forbidden_generation)
    separate = runner.run(tmp_path / "lazy_results", instances_from=run)
    imported_metadata = json.loads((separate / "run_metadata.json").read_text())
    assert imported_metadata["source_generation_metadata"] == final_metadata
    assert imported_metadata["phase2_formulation"] == PhaseTwoSearch.formulation
    assert (run / "run_metadata.json").read_bytes() == before
    for path in (run / "data" / "instances").glob("*.json"):
        assert path.read_bytes() == (separate / "data" / "instances" / path.name).read_bytes()


def test_imported_full_run_continues_if_test_configuration_times_out(tmp_path, monkeypatch):
    config = ExperimentConfig(sizes=(180,), probabilities=(0.7,), instances_per_pair=2)
    source_writer = ResultWriter(tmp_path / "source")
    generated = InstanceGenerator(config, 29).generate_pair(180, 0.7, 2)
    for instance in generated:
        instance.save(source_writer.directory / "data" / "instances" / f"{instance.instance_id}.json")
    source_metadata = ExperimentRunner(config, 29)._metadata(source_writer, generated, 29)
    source_metadata["retained_instance_ids"] = [instance.instance_id for instance in generated]
    source_metadata["retained_instance_count"] = 2
    source_writer.write_metadata(source_metadata)
    solved = []

    class FakeSolver:
        def __init__(self, *args):
            pass

        def solve(self, instance, deadline=None):
            solved.append(instance.instance_id)
            return {"instance_id": instance.instance_id, "status": "INCOMPLETE"}

    monkeypatch.setattr("bostp.experiment_runner.KSSolver", FakeSolver)
    run = ExperimentRunner(config).run(
        tmp_path / "imported", full=True, instances_from=source_writer.directory,
        instance_budget_seconds=10,
    )
    assert solved == [instance.instance_id for instance in generated]
    results = json.loads((run / "solutions.json").read_text())
    assert len(results) == 2
    assert all(result["status"] == "INCOMPLETE" for result in results)


def test_reference_points_are_feasible():
    instance = InstanceGenerator(ExperimentConfig(), 4).generate_pair(8, 0.7, 1)[0]
    reference = ReferencePointCalculator().calculate(instance)
    solver = MSTSolver(instance)
    solver.verify(reference.x1); solver.verify(reference.x2)
    assert reference.f1_u <= reference.f1_n
    assert reference.f2_u <= reference.f2_n


def test_legacy_scf_resume_is_rejected_without_rewriting_metadata(tmp_path):
    import pytest

    writer = ResultWriter(tmp_path)
    original = {"phase2": "Gurobi SCF", "run_status": "INCOMPLETE_RESULTS"}
    writer.write_metadata(original)
    with pytest.raises(ValueError, match="different or unverified"):
        ExperimentRunner(ExperimentConfig(), 23).run(tmp_path, resume=writer.directory)
    assert json.loads((writer.directory / "run_metadata.json").read_text()) == original


def test_budgeted_pass_continues_reports_and_defers_timeouts(tmp_path, monkeypatch):
    config = ExperimentConfig(sizes=(8,), probabilities=(0.5,), instances_per_pair=2)
    clock = [0.0]
    calls = []

    class TimedOutSolver:
        def __init__(self, *args):
            pass

        def solve(self, instance, deadline=None):
            assert deadline == clock[0] + 1200
            calls.append(instance.instance_id)
            clock[0] += 1201
            return {"instance_id": instance.instance_id, "status": "INCOMPLETE"}

    monkeypatch.setattr("bostp.experiment_runner.monotonic", lambda: clock[0])
    monkeypatch.setattr("bostp.experiment_runner.KSSolver", TimedOutSolver)
    runner = ExperimentRunner(config, 23)
    run = runner.run(tmp_path, instance_budget_seconds=1200)
    assert len(calls) == 2
    results = json.loads((run / "solutions.json").read_text())
    assert all(result["budget_exhausted"] for result in results)
    assert all(result["instance_elapsed_seconds"] == 1201 for result in results)
    assert (run / "ks_table.tex").read_text().startswith("% PARTIAL:")
    runner.run(tmp_path, resume=run, instance_budget_seconds=1200)
    assert len(calls) == 2


def test_expired_budget_does_not_optimize_model(monkeypatch):
    instance = InstanceGenerator(ExperimentConfig(), 11).generate_pair(8, 0.5, 1)[0]
    reference = ReferencePointCalculator().calculate(instance)
    search = PhaseTwoSearch(instance, reference, deadline=1200)

    class Model:
        disposed = False

        def dispose(self):
            self.disposed = True

        def optimize(self):
            raise AssertionError("expired instance must not optimize another subproblem")

    model = Model()
    monkeypatch.setattr(search, "_build_model", lambda *args: (model, {}, [], {}, {}))
    monkeypatch.setattr("bostp.phase_two.monotonic", lambda: 1201)
    import pytest
    with pytest.raises(RuntimeError, match="shared instance budget exhausted"):
        search._solve_subproblem("P", 1)
    assert model.disposed


def test_phase_two_model_directions_and_trace():
    instance = InstanceGenerator(ExperimentConfig(), 11).generate_pair(8, 0.5, 1)[0]
    reference = ReferencePointCalculator().calculate(instance)
    search = PhaseTwoSearch(instance, reference)
    positive = reference.x1
    negative = reference.x2
    p_candidate = replace(positive, f1=reference.f1_n - 3, f2=reference.f2_n - 8)
    q_candidate = replace(negative, f1=reference.f1_n - 7, f2=reference.f2_n - 2)

    seen = []
    def fake_solve(orientation, epsilon):
        assert 0 <= epsilon <= (reference.q_max if orientation == "P" else reference.p_max)
        seen.append((orientation, epsilon))
        if orientation == "P":
            return replace(p_candidate, f2=reference.f2_n - epsilon)
        return replace(q_candidate, f1=reference.f1_n - epsilon)

    search._solve_subproblem = fake_solve
    p_result, q_result = search.run([positive, negative])
    assert p_result.status == "OPTIMAL"
    assert q_result.status == "OPTIMAL"
    assert p_result.solution.f1 == reference.f1_n - 3
    assert 0 <= reference.f2_n - p_result.solution.f2 <= reference.q_max
    assert q_result.solution.f2 == reference.f2_n - 2
    assert 0 <= reference.f1_n - q_result.solution.f1 <= reference.p_max
    assert q_result.solution.f2 == reference.f2_n - 2
    assert {orientation for orientation, _ in seen} == {"P", "Q"}


def test_phase_two_accepts_configured_relative_gap():
    instance = InstanceGenerator(ExperimentConfig(), 11).generate_pair(8, 0.5, 1)[0]
    reference = ReferencePointCalculator().calculate(instance)
    search = PhaseTwoSearch(instance, reference, optimality_gap_tolerance=5e-4)
    model, *_ = search._build_model("P", 1)
    try:
        assert model.Params.MIPGap == 5e-4
    finally:
        model.dispose()


def test_phase_two_failure_is_not_a_zero_solution():
    instance = InstanceGenerator(ExperimentConfig(), 11).generate_pair(8, 0.5, 1)[0]
    reference = ReferencePointCalculator().calculate(instance)
    search = PhaseTwoSearch(instance, reference)

    def failed_solve(orientation, epsilon):
        raise RuntimeError("license failure")

    search._solve_subproblem = failed_solve
    result = search._search("P", reference.x1, reference.x2)
    assert result.status == "FAILED"
    assert result.solution is None
    assert result.message == "license failure"


def test_ks_selection_uses_exact_k_values():
    instance = InstanceGenerator(ExperimentConfig(), 11).generate_pair(8, 0.5, 1)[0]
    result = KSSolver().solve(instance)
    if result["status"] == "COMPLETED" and result["phase1"]["outcome"] == "C_SN":
        selection = result["ks_selection"]
        assert selection["K_minus"] is not None
        assert selection["K_plus"] is not None
        assert len(result["final_ks_points"]) == (2 if selection["K_minus"] == selection["K_plus"] else 1)
    else:
        assert result["final_ks_points"] == []