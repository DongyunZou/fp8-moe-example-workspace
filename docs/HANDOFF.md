# Handoff

## Current State

This is a fresh scaffold. No real kernel has been promoted yet.

Implemented:

- agent instructions in `AGENTS.md`;
- user workflow in `README.md`;
- workload/runner verification harness in `verify.py`;
- sample workload metadata in `examples/sample_workloads.jsonl`;
- experiment ledgers: `benchmark.csv` and `solutions.jsonl`.

## Next Step

1. Attach real workload traces under `data/`.
2. Plug the benchmark runner into `verify.py` with `--agent-cmd` and
   `--baseline-cmd`.
3. Ask the agent to use KernelWiki and write `docs/draft.md`.
4. Run the first profile with ncu-report-skill.
5. Implement the first candidate kernel.

## Known Gaps

- `solution/cuda/` contains only a placeholder.
- `verify.py` does not implement benchmark-specific packing by itself.
- Real CUDA build integration should be added once the benchmark harness is
  known.

