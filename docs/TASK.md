# Task

## Context

This repository is a template for an agent-driven FP8 MoE CUDA kernel
optimization experiment.

Target benchmark:

```text
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
```

Typical model shape:

- `hidden = 7168`
- `intermediate = 2048`
- local experts = `32`
- top-k = `8`
- FP8 E4M3 activations and weights
- DeepSeek-style block scales

## Correctness

Use the benchmark's MoE tolerance. If reproducing the original experiment, use:

- `atol = 1`
- `rtol = 0.3`
- `required_matched_ratio = 0.9`

The Python or framework reference is the numerical oracle. Speed is evaluated
against the configured baseline implementation.

## Current Objective

Bootstrap the repository so another agent can:

1. download or attach workloads;
2. run `verify.py`;
3. research Blackwell kernel directions with KernelWiki;
4. profile with ncu-report-skill;
5. implement and promote candidates through a documented loop.

## Constraints

- CUDA C++ / CuTe / CUTLASS solution path.
- Target B200 / SM100.
- Use NCU/NSYS evidence for performance claims.
- Compare with the configured baseline.
- Keep failed attempts documented when they teach a design constraint.
- Do not default-enable unstable experimental paths.

## Acceptance Criteria

- [ ] Workload metadata can be validated with `verify.py`.
- [ ] Real benchmark commands can be plugged into `verify.py`.
- [ ] A baseline run and agent run can be dumped to JSON.
- [ ] Promoted results are recorded in `benchmark.csv`.
- [ ] Candidate lineage is recorded in `solutions.jsonl`.
- [ ] Profiling reports go under `profile/<run_name>/REPORT.md`.
- [ ] `docs/HANDOFF.md` is enough for a new agent to resume.

## Suggested First Optimization Goal

After wiring a real kernel and benchmark runner, start with:

```text
Identify the slowest FP8 MoE phase on dev workloads, profile it with NCU, and
replace the highest-impact bottleneck with a CUDA/CUTLASS implementation.
```

