#!/usr/bin/env python3
"""Verification harness for the FP8 MoE agent-kernel example.

The script supports two modes:

1. Metadata mode: read workload JSONL rows containing `agent_latency_us`,
   `baseline_latency_us`, and `passed`.
2. Runner mode: execute commands supplied with `--agent-cmd` and
   `--baseline-cmd`. Commands must print JSON to stdout.

Command templates may use: {uuid}, {T}, {split}, {workload_json}.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_WORKLOAD_FIELDS = ("uuid", "T")
DEFAULT_ATOL = 1.0
DEFAULT_RTOL = 0.3
DEFAULT_REQUIRED_MATCHED_RATIO = 0.9


def load_workloads(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"workload file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            missing = [key for key in REQUIRED_WORKLOAD_FIELDS if key not in row]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing required fields: {missing}")
            row.setdefault("split", "dev")
            rows.append(row)
    if not rows:
        raise ValueError(f"no workloads found in {path}")
    return rows


def select_workloads(rows: list[dict[str, Any]], use_all: bool, uuids: list[str]) -> list[dict[str, Any]]:
    selected = rows if use_all else [row for row in rows if row.get("split", "dev") == "dev"]
    if uuids:
        wanted = set(uuids)
        selected = [row for row in selected if str(row["uuid"]) in wanted]
    if not selected:
        raise ValueError("workload selection is empty")
    return selected


def format_template(template: str, workload: dict[str, Any]) -> list[str]:
    values = {k: str(v) for k, v in workload.items()}
    values["workload_json"] = json.dumps(workload, sort_keys=True)
    rendered = template.format(**values)
    return shlex.split(rendered)


def run_json_command(template: str, workload: dict[str, Any]) -> dict[str, Any]:
    cmd = format_template(template, workload)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return {
            "passed": False,
            "latency_us": math.nan,
            "error": f"command failed with exit code {proc.returncode}",
            "stderr": proc.stderr.strip(),
            "cmd": cmd,
        }
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "latency_us": math.nan,
            "error": f"command did not print JSON: {exc}",
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "cmd": cmd,
        }
    result.setdefault("cmd", cmd)
    return result


def metadata_result(workload: dict[str, Any], latency_key: str) -> dict[str, Any]:
    latency = workload.get(latency_key)
    if latency is None:
        return {
            "passed": bool(workload.get("passed", True)),
            "latency_us": math.nan,
            "notes": f"metadata field {latency_key} missing",
        }
    return {
        "passed": bool(workload.get("passed", True)),
        "latency_us": float(latency),
        "notes": workload.get("notes", ""),
    }


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return statistics.fmean(finite) if finite else math.nan


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    workloads = select_workloads(load_workloads(args.workloads), args.all, args.uuid)
    results: list[dict[str, Any]] = []

    for workload in workloads:
        if args.agent_cmd:
            agent = run_json_command(args.agent_cmd, workload)
        else:
            agent = metadata_result(workload, "agent_latency_us")

        baseline = None
        if args.baseline:
            if args.baseline_cmd:
                baseline = run_json_command(args.baseline_cmd, workload)
            else:
                baseline = metadata_result(workload, "baseline_latency_us")

        agent_latency = float(agent.get("latency_us", math.nan))
        baseline_latency = float(baseline.get("latency_us", math.nan)) if baseline else math.nan
        speedup = baseline_latency / agent_latency if (
            math.isfinite(agent_latency)
            and math.isfinite(baseline_latency)
            and agent_latency > 0
        ) else math.nan

        passed = bool(agent.get("passed", False))
        if baseline is not None:
            passed = passed and bool(baseline.get("passed", True))

        results.append({
            "uuid": workload["uuid"],
            "T": workload["T"],
            "split": workload.get("split", "dev"),
            "passed": passed,
            "agent_latency_us": agent_latency,
            "baseline_latency_us": baseline_latency if baseline else None,
            "speedup_vs_baseline": speedup if baseline else None,
            "agent": agent,
            "baseline": baseline,
        })

    agent_mean = mean([row["agent_latency_us"] for row in results])
    baseline_values = [
        row["baseline_latency_us"]
        for row in results
        if row["baseline_latency_us"] is not None
    ]
    baseline_mean = mean([float(v) for v in baseline_values]) if baseline_values else math.nan
    speedup_mean = baseline_mean / agent_mean if (
        math.isfinite(agent_mean) and math.isfinite(baseline_mean) and agent_mean > 0
    ) else math.nan

    return {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "workload_file": str(args.workloads),
        "correctness": {
            "atol": args.atol,
            "rtol": args.rtol,
            "required_matched_ratio": args.required_matched_ratio,
            "reference": "PyTorch definition reference",
        },
        "speed_baseline": "flashinfer_wrapper_9sdjf3" if args.baseline else None,
        "workload_count": len(results),
        "passed_count": sum(1 for row in results if row["passed"]),
        "all_passed": all(row["passed"] for row in results),
        "agent_mean_us": agent_mean,
        "baseline_mean_us": baseline_mean if args.baseline else None,
        "mean_speedup_vs_baseline": speedup_mean if args.baseline else None,
        "results": results,
    }


def print_summary(summary: dict[str, Any]) -> None:
    total = summary["workload_count"]
    passed = summary["passed_count"]
    correctness = summary["correctness"]
    print(f"{passed}/{total} passed")
    print(
        "correctness:  "
        f"atol={correctness['atol']} "
        f"rtol={correctness['rtol']} "
        f"required_matched_ratio={correctness['required_matched_ratio']}"
    )
    print(f"agent mean:    {summary['agent_mean_us']:.4f} us")
    if summary["baseline_mean_us"] is not None:
        print(f"baseline mean: {summary['baseline_mean_us']:.4f} us")
        print(f"speedup:       {summary['mean_speedup_vs_baseline']:.3f}x")
    print()
    for row in summary["results"]:
        parts = [
            f"uuid={row['uuid']}",
            f"T={row['T']}",
            f"passed={row['passed']}",
            f"agent={row['agent_latency_us']:.4f}us",
        ]
        if row["baseline_latency_us"] is not None:
            parts.append(f"baseline={row['baseline_latency_us']:.4f}us")
            parts.append(f"speedup={row['speedup_vs_baseline']:.3f}x")
        print("  " + " ".join(parts))


def append_benchmark(summary: dict[str, Any], path: Path, solution_name: str, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "timestamp",
                "commit",
                "solution_name",
                "workload_set",
                "batch_size",
                "latency_us",
                "baseline_latency_us",
                "speedup_vs_baseline",
                "passed_correctness",
                "notes",
            ])
        writer.writerow([
            summary["timestamp"],
            commit,
            solution_name,
            "selected_mean",
            "various",
            f"{summary['agent_mean_us']:.6f}",
            "" if summary["baseline_mean_us"] is None else f"{summary['baseline_mean_us']:.6f}",
            "" if summary["mean_speedup_vs_baseline"] is None else f"{summary['mean_speedup_vs_baseline']:.6f}",
            str(summary["all_passed"]).lower(),
            f"{summary['passed_count']}/{summary['workload_count']} workloads passed",
        ])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, default=Path("data/workloads/fp8_moe.jsonl"))
    parser.add_argument("--all", action="store_true", help="use all workloads instead of split=dev")
    parser.add_argument("--uuid", action="append", default=[], help="select a workload UUID; repeatable")
    parser.add_argument("--baseline", action="store_true", help="run or read baseline timing")
    parser.add_argument("--agent-cmd", help="command template that prints agent result JSON")
    parser.add_argument("--baseline-cmd", help="command template that prints baseline result JSON")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--required-matched-ratio", type=float, default=DEFAULT_REQUIRED_MATCHED_RATIO)
    parser.add_argument("--dump-json", type=Path, help="write full summary JSON")
    parser.add_argument("--append-benchmark", type=Path, help="append mean row to benchmark CSV")
    parser.add_argument("--solution-name", default="agent_candidate")
    parser.add_argument("--commit", default=os.environ.get("GIT_COMMIT", "local"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        summary = run_verification(args)
    except Exception as exc:
        print(f"verify.py: error: {exc}", file=sys.stderr)
        return 2

    print_summary(summary)

    if args.dump_json:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.append_benchmark:
        append_benchmark(summary, args.append_benchmark, args.solution_name, args.commit)

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
