# Agent Instructions

## Objective

Build and optimize a CUDA/CUTLASS FP8 MoE kernel for NVIDIA B200 / SM100.

The experiment is successful only when the agent can show:

- correctness on the configured dev workloads;
- measured speed against the configured baseline;
- a reproducible candidate history in `benchmark.csv` and `solutions.jsonl`;
- profile-backed explanations for major design decisions.

## Hard Constraints

- Use CUDA C++ / CuTe / CUTLASS for the solution path.
- Do not use Triton or pure Python kernels unless the user explicitly changes
  the experiment.
- Target NVIDIA B200 / SM100 / CUDA 13.2.
- Compare speed against the benchmark baseline, not against a Python reference.
- Do not enable an experimental path by default unless it passes correctness
  and improves the target workload.
- Do not regress large workloads while fixing small workloads unless the path is
  shape-gated and documented.
- The solution must be self-contained inside this repository.
- Any third-party source code used by the solution path must be vendored into
  this repository. Do not depend on local checkouts, generated caches, system
  include paths, or uncommitted files outside the repo.
- Do not add a dependency on a Python package, shared library, header-only
  library, cubin, or generated source unless the exact source or binary artifact
  needed for reproduction is included under this repository and documented.
- Preserve unrelated worktree changes.
- Record performance-relevant promoted candidates.

## Self-Contained Code Rule

The repository must be enough to build and run the submitted kernel on a fresh
machine with the documented CUDA/Python/toolchain versions and the workload
data. A future agent or user should not need files from `/tmp`, `~/.cache`,
another private checkout, or a previous experiment workspace.

Allowed:

- CUDA, CUDA toolkit headers, driver/runtime libraries, and Nsight tools from
  the documented system installation.
- Standard Python library code used by local scripts.
- Third-party source copied into this repository under a clear directory such as
  `third_party/`, with license/provenance notes.
- Generated code or binary artifacts committed under this repository when they
  are required to reproduce the solution.

Not allowed:

- `#include` paths that point outside this repository except CUDA/toolchain
  headers.
- Runtime loading from `~/.cache`, `/tmp`, another project checkout, or an
  agent-local build directory.
- Depending on `pip install <package>` for solution logic unless the repository
  vendors the source or the package is used only for non-solution developer
  tooling and is documented as such.
- Copying code from prohibited submissions or sources whose license does not
  allow redistribution.

## Required Skills

Use `KernelWiki` when researching Blackwell/Hopper kernel design:

- SM100 tensor core and TMEM patterns;
- CUTLASS grouped GEMM;
- FlashInfer, TensorRT-LLM, DeepGEMM, Sonic MoE, vLLM, SGLang references;
- MoE scheduling, load imbalance, CTA wave count, and fused epilogues.

Use `ncu-report-skill` when diagnosing performance:

- create one new `profile/<run_name>/` directory per profiling run;
- collect overview and source-counter reports when possible;
- write `profile/<run_name>/REPORT.md`;
- diagnose from measured metrics before changing code.

Profiling rule:

```text
Profile -> diagnose -> plan -> edit -> verify -> record
```

## Standard Workflow

1. Read `docs/TASK.md`, `docs/HANDOFF.md`, `benchmark.csv`, and
   `solutions.jsonl`.
2. Reproduce the current state:

   ```bash
   python3 verify.py --workloads examples/sample_workloads.jsonl
   ```

   For real data, use the command documented in `README.md`.

3. If the next step is unclear, use KernelWiki and write a short research note
   under `docs/`.
4. If performance is being diagnosed, use ncu-report-skill and write a profile
   report.
5. Implement one focused candidate.
6. Run `verify.py`.
7. Promote only if correctness passes and performance improves or the failure
   produces useful documented evidence.
8. Append promoted results to `benchmark.csv`.
9. Append candidate metadata to `solutions.jsonl`.
10. Update `docs/HANDOFF.md`.

## Goal Usage

For a serious optimization session, create a concrete goal such as:

```text
Improve small-T FP8 MoE latency by replacing a one-wave grouped GEMM path with
a multi-wave BMM path.
```

Keep the goal tied to acceptance criteria in `docs/TASK.md`. Mark it complete
only after the criteria are met. Mark it blocked only when the same external
blocker prevents progress across repeated attempts.

## Promotion Record

Every promoted candidate needs:

- commit or candidate id;
- parent id;
- workload set;
- correctness result;
- agent latency;
- baseline latency if available;
- speedup;
- short note explaining the change.

Use `benchmark.csv` for timing rows and `solutions.jsonl` for candidate DAG
metadata.
