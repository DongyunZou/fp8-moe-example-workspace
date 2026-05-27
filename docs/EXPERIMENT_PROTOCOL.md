# Experiment Protocol

## Candidate Loop

Each candidate follows this loop:

1. State the hypothesis.
2. If the hypothesis is performance-related, profile or cite an existing
   profile report.
3. Implement a small scoped change.
4. Run correctness.
5. Run timing against baseline.
6. Promote, reject, or park.
7. Record the outcome.

## Promotion

Promote a candidate when:

- correctness passes;
- latency improves on the target workload set;
- any regressions are understood and shape-gated;
- the candidate is documented in `benchmark.csv` and `solutions.jsonl`.

## Rejection

Reject a candidate when:

- it fails correctness;
- it improves one workload by regressing a more important one;
- the profile disproves the hypothesis;
- it is correct but slower than the current default path.

Keep rejected candidates in notes if they are useful. Do not default-enable
them.

## Ledgers

`benchmark.csv` is for measurements:

```text
timestamp,commit,solution_name,workload_set,batch_size,latency_us,baseline_latency_us,speedup_vs_baseline,passed_correctness,notes
```

`solutions.jsonl` is for lineage:

```json
{"name":"candidate","parent":"previous","commit":"local","dev_pass":true,"notes":"what changed"}
```

