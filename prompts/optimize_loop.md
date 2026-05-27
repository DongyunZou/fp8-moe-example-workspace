Create a goal for one concrete FP8 MoE latency improvement.

Follow `AGENTS.md`.

Use KernelWiki for design references and ncu-report-skill for profiling. Work
one candidate at a time:

1. reproduce current correctness/performance;
2. profile the bottleneck if needed;
3. implement the candidate;
4. run `verify.py`;
5. compare with baseline;
6. promote or reject;
7. update `benchmark.csv`, `solutions.jsonl`, and `docs/HANDOFF.md`.

