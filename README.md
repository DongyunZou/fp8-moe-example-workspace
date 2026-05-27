# FP8 MoE Agent Kernel Example

This repository is a reproducible template for running an agent-driven CUDA
kernel optimization experiment. It mirrors the protocol used in the FP8 MoE
workspace:

- `AGENTS.md` tells the agent how to work.
- `docs/TASK.md` defines the target, constraints, and acceptance criteria.
- `docs/HANDOFF.md` lets a new agent continue from the current state.
- `benchmark.csv` records promoted performance results.
- `solutions.jsonl` records the candidate DAG.
- `profile/<run_name>/REPORT.md` stores NCU-backed diagnoses.

The repository intentionally starts as a scaffold. Bring your benchmark runner,
workload traces, and kernel implementation, then let the agent iterate under the
same rules.

The FP8 MoE definition and PyTorch numerical reference are vendored under
`references/flashinfer-trace/`; see `docs/REFERENCE.md`. The PyTorch reference is
only the correctness oracle. Speed is measured against the FlashInfer baseline
`flashinfer_wrapper_9sdjf3`.

## 1. Environment

Required for a real FP8 MoE run:

- NVIDIA B200 / SM100
- CUDA 13.2
- Nsight Compute (`ncu`)
- Python 3.10+
- `uv` or a normal Python virtual environment
- A benchmark runner that can execute the agent solution and the baseline

Create an environment:

```bash
cd ~/workspace/projects/kda-examples/fp8-moe-example
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

If your real benchmark uses `uv`, add a `pyproject.toml` and run:

```bash
uv sync
```

## 2. Download Workloads

Set `DATA_DIR` to the local dataset directory, then download the FlashInfer
contest dataset from Hugging Face:

```bash
export DATA_DIR="${DATA_DIR:-$PWD/data}"
mkdir -p "$DATA_DIR"
hf download flashinfer-ai/mlsys26-contest --repo-type=dataset --local-dir "$DATA_DIR"
```

Expected layout:

```text
data/
├── definitions/
└── workloads/
```

For a dry run without private data, use the included example metadata:

```bash
python3 verify.py --workloads examples/sample_workloads.jsonl
```

## 3. Run Verification

Correctness acceptance for this MoE target:

```text
atol = 1
rtol = 0.3
required_matched_ratio = 0.9
```

Metadata-only smoke test:

```bash
python3 verify.py --workloads examples/sample_workloads.jsonl
```

Run with a real agent benchmark command:

```bash
python3 verify.py \
  --workloads data/workloads/fp8_moe.jsonl \
  --agent-cmd 'python real_runner.py --solution outputs/packed/solution.json --uuid {uuid}' \
  --dump-json runs/agent_dev.json
```

Run agent and baseline:

```bash
python3 verify.py \
  --all --baseline \
  --workloads data/workloads/fp8_moe.jsonl \
  --agent-cmd 'python real_runner.py --solution outputs/packed/solution.json --uuid {uuid}' \
  --baseline-cmd 'python real_runner.py --baseline flashinfer_wrapper_9sdjf3 --uuid {uuid}' \
  --dump-json runs/all_with_baseline.json
```

Runner commands must print JSON to stdout. Minimal accepted schema:

```json
{"passed": true, "latency_us": 123.4}
```

Optional fields such as `matched_ratio`, `max_abs_diff`, and `notes` are copied
into the report.

## 4. Start Research With Codex

Ask the agent:

```text
Read this repository. Use KernelWiki. Build a research plan for optimizing the
FP8 MoE kernel on B200. Use docs/TASK.md as the source of truth. Write the plan
to docs/draft.md. Do not edit solution code yet.
```

The expected output is a component-level plan: routing, dispatch, GEMM1,
SwiGLU/quant, GEMM2, combine, and profiling targets.

## 5. Start The Kernel Optimization Loop

Ask the agent:

```text
Create a goal to improve FP8 MoE latency against the configured baseline.
Follow AGENTS.md. Use KernelWiki for Blackwell kernel references and
ncu-report-skill for profiling. Implement one candidate at a time, run
verify.py, profile evidence-backed bottlenecks, record promoted results in
benchmark.csv and solutions.jsonl, and update docs/HANDOFF.md.
```

Promotion rule:

- correctness passes;
- latency is compared against the baseline;
- regressions are either fixed or shape-gated;
- benchmark and candidate metadata are recorded;
- any major design claim has a profile report under `profile/`.

## 6. Profile A Bottleneck

Ask the agent:

```text
Profile the slowest current kernel on workload <uuid>. Use ncu-report-skill.
Create a new profile/<run_name>/ directory and write REPORT.md with metrics,
diagnosis, and ranked next steps.
```

The profile directory is gitignored; commit only the summarized report if it is
needed for handoff.
