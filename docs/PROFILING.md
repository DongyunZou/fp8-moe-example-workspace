# Profiling

Use ncu-report-skill for performance diagnosis.

## Directory Rule

Each profile run gets a new directory:

```text
profile/<run_name>/
├── harness/
├── reports/
├── analysis/
└── REPORT.md
```

Do not reuse old profile directories.

## Minimum Report

`REPORT.md` should include:

- command lines;
- workload UUIDs;
- kernel names;
- latency measurements;
- top bottleneck metrics;
- diagnosis;
- ranked next steps.

## Common FP8 MoE Questions

Check these first:

- Is the slow phase GEMM1, SwiGLU/quant, GEMM2, routing, dispatch, or combine?
- Is the GEMM path one persistent wave or many CTA waves?
- Is the kernel limited by TMA/barriers, memory bandwidth, tensor core issue,
  occupancy, or load imbalance?
- Does a small-T path need a different schedule than a large-T path?
- Does a fused path remove enough global memory traffic to justify complexity?

