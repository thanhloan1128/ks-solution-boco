from __future__ import annotations

import csv, json
from datetime import datetime, timezone
from fractions import Fraction
import os
from pathlib import Path


class ResultWriter:
    def __init__(self, root: Path, resume_directory: Path | None = None):
        if resume_directory is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.directory = root / stamp
            self.directory.mkdir(parents=True, exist_ok=False)
        else:
            self.directory = resume_directory
            if not self.directory.is_dir():
                raise FileNotFoundError(f"run directory does not exist: {self.directory}")
        (self.directory / "solutions").mkdir(exist_ok=resume_directory is not None)

    def write_checkpoint(self, result):
        checkpoints = self.directory / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        path = checkpoints / f"{result['instance_id']}.json"
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w") as handle:
            json.dump(result, handle, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def load_checkpoints(self):
        checkpoints = self.directory / "checkpoints"
        if not checkpoints.exists():
            return {}
        results = {}
        for path in sorted(checkpoints.glob("*.json")):
            result = json.loads(path.read_text())
            instance_id = result["instance_id"]
            if path.stem != instance_id or instance_id in results:
                raise ValueError(f"invalid or duplicate checkpoint: {path}")
            results[instance_id] = result
        return results

    def write_run_state(self, value):
        self._write_json(self.directory / "run_state.json", value)

    def write_metadata(self, metadata):
        self._write_json(self.directory / "run_metadata.json", metadata)

    def write(self, instances, results, metadata):
        self.write_instance_timings(instances, results)
        self.write_metadata(metadata)
        (self.directory / "solutions.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
        fields = sorted({key for result in results for key in result})
        with (self.directory / "per_instance_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(results)
        groups: dict[tuple[int, float], list[dict]] = {}
        for instance, result in zip(instances, results): groups.setdefault((instance.n, instance.p), []).append(result)
        average_rows = []
        for (n, p), group in sorted(groups.items()):
            successful = [result for result in group if result.get("status") == "COMPLETED"]
            row = {"n": n, "p": p, "instance_count": len(successful), "failed_or_degenerate": len(group) - len(successful)}
            for field in ("reference_seconds", "phase1_seconds", "phase2_P_seconds", "phase2_Q_seconds", "total_seconds"):
                row[f"mean_{field}"] = sum(result.get(field, 0) for result in successful) / len(successful) if successful else ""
            metrics = {
                "phase1_iterations": lambda result: result.get("phase1", {}).get("midpoint_iterations"),
                "phase1_final_solves": lambda result: result.get("phase1", {}).get("final_solves"),
                "phase2_P_iterations": lambda result: result.get("phase2", {}).get("P", {}).get("midpoint_iterations"),
                "phase2_Q_iterations": lambda result: result.get("phase2", {}).get("Q", {}).get("midpoint_iterations"),
                "phase2_iterations": lambda result: result.get("phase2", {}).get("P", {}).get("midpoint_iterations", 0) + result.get("phase2", {}).get("Q", {}).get("midpoint_iterations", 0),
                "phase1_rho_minus": lambda result: [abs(float(Fraction(value))) for value in result.get("phase1", {}).get("signed_rho", []) if Fraction(value) < 0],
                "phase1_rho_plus": lambda result: [abs(float(Fraction(value))) for value in result.get("phase1", {}).get("signed_rho", []) if Fraction(value) > 0],
                "phase2_rho_minus": lambda result: [abs(float(Fraction(str(result["phase2"]["Q"]["signed_rho"]))))] if result.get("phase2", {}).get("Q", {}).get("signed_rho") is not None else [],
                "phase2_rho_plus": lambda result: [abs(float(Fraction(str(result["phase2"]["P"]["signed_rho"]))))] if result.get("phase2", {}).get("P", {}).get("signed_rho") is not None else [],
            }
            for field, getter in metrics.items():
                values = [item for result in successful for item in ([getter(result)] if isinstance(getter(result), (int, float)) else getter(result)) if item is not None]
                row[f"mean_{field}"] = sum(values) / len(values) if values else ""
            average_rows.append(row)
        with (self.directory / "average_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(average_rows[0]) if average_rows else ["n", "p"]); writer.writeheader(); writer.writerows(average_rows)
        with (self.directory / "ks_table.tex").open("w") as handle:
            if any(result.get("status") != "COMPLETED" for result in results):
                handle.write(
                    "% PARTIAL: certified results only; N_complete is the averaging sample size.\n"
                    "% Rows with fewer than two completions are not two-instance averages.\n"
                )
            handle.write("% Generated by bostp-ks\n\\begin{tabular}{rrr rrrrrrrr}\n\\toprule\n$n$ & $p$ & $N_{\\mathrm{complete}}$ & $\\rho^-_{\\mathrm I}$ & $\\rho^+_{\\mathrm I}$ & $\\mathrm{Time}_{\\mathrm I}$ & $\\mathrm{Iter}_{\\mathrm I}$ & $\\rho^-_{\\mathrm{II}}$ & $\\rho^+_{\\mathrm{II}}$ & $\\mathrm{Time}_{\\mathrm{II}}$ & $\\mathrm{Iter}_{\\mathrm{II}}$ \\\\\n\\midrule\n")
            for row in average_rows:
                phase2_time = (row.get("mean_phase2_P_seconds", "") + row.get("mean_phase2_Q_seconds", "")
                               if row.get("mean_phase2_P_seconds", "") != "" and row.get("mean_phase2_Q_seconds", "") != "" else "")
                values = (row.get("mean_phase1_rho_minus", ""), row.get("mean_phase1_rho_plus", ""), row.get("mean_phase1_seconds", ""), row.get("mean_phase1_iterations", ""), row.get("mean_phase2_rho_minus", ""), row.get("mean_phase2_rho_plus", ""), phase2_time, row.get("mean_phase2_iterations", ""))
                formatted = ["" if value == "" else f"{value:.4f}" if index in (0, 1, 4, 5) else f"{value:.2f}" for index, value in enumerate(values)]
                handle.write(f"{row['n']} & {row['p']:.1f} & {row['instance_count']} & " + " & ".join(formatted) + " \\\\\n")
            handle.write("\\bottomrule\n\\end{tabular}\n")

    def write_instance_timings(self, instances, results):
        rows = []
        for instance, result in zip(instances, results):
            seconds = result.get("instance_elapsed_seconds", result.get("total_seconds"))
            status = result.get("status", "NOT_STARTED")
            note = "Saved attempt only; not cumulative retry time."
            lower_bound = result.get("interrupted_elapsed_lower_bound_seconds")
            if lower_bound is not None:
                seconds = None
                status = "INTERRUPTED"
                note = result["deferral_reason"]
            elif result.get("deferral_reason"):
                seconds = None
                status = "INTERRUPTED"
                note = result["deferral_reason"]
            elif result.get("budget_exhausted"):
                status = "TIME_LIMIT"
            rows.append({
                "n": instance.n, "p": instance.p,
                "instance_id": instance.instance_id,
                "instance_index": int(instance.instance_id.rsplit("_i", 1)[1]),
                "status": status, "elapsed_seconds": seconds,
                "elapsed_minutes": seconds / 60 if seconds is not None else None,
                "elapsed_lower_bound_seconds": lower_bound,
                "note": note,
            })
        rows.sort(key=lambda row: (row["n"], row["p"], row["instance_index"]))
        fields = [
            "n", "p", "instance_id", "instance_index", "status",
            "elapsed_seconds", "elapsed_minutes", "elapsed_lower_bound_seconds", "note",
        ]
        with (self.directory / "instance_timings.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        groups = {}
        for row in rows:
            groups.setdefault((row["n"], row["p"]), {})[row["instance_index"]] = row
        labels = {
            "COMPLETED": "Certified", "TIME_LIMIT": "Time limit",
            "INTERRUPTED": "Interrupted", "INCOMPLETE": "Incomplete",
            "NOT_STARTED": "Not started",
        }
        with (self.directory / "instance_timings.tex").open("w") as handle:
            handle.write(
                "% Minutes per saved attempt, not cumulative retry time. "
                "A time limit is not a certified solve time.\n"
                "\\begin{tabular}{rr rlr l}\n\\toprule\n"
                "$n$ & $p$ & $t_1$ (min) & Status 1 & $t_2$ (min) & Status 2 \\\\\n"
                "\\midrule\n"
            )
            for (n, p), group in sorted(groups.items()):
                values = []
                for index in (1, 2):
                    row = group.get(index)
                    if row is None:
                        values.extend(("--", "Not started"))
                        continue
                    if row["elapsed_lower_bound_seconds"] is not None:
                        duration = f"$>{row['elapsed_lower_bound_seconds'] / 60:.2f}$"
                    elif row["elapsed_minutes"] is not None:
                        duration = f"{row['elapsed_minutes']:.2f}"
                    else:
                        duration = "--"
                    values.extend((duration, labels.get(row["status"], "Uncertified")))
                handle.write(f"{n} & {p:.1f} & " + " & ".join(values) + " \\\\\n")
            handle.write("\\bottomrule\n\\end{tabular}\n")

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, indent=2, default=str) + "\n")