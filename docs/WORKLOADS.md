# Workloads

## Expected Layout

```text
data/
├── definitions/
│   └── moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
└── workloads/
    └── fp8_moe.jsonl
```

`verify.py` expects JSON Lines workload metadata. Required fields:

```json
{"uuid": "e05c6c03", "T": 1, "split": "dev"}
```

Optional fields:

```json
{
  "baseline_latency_us": 59.8,
  "agent_latency_us": 62.1,
  "passed": true,
  "notes": "sample metadata row"
}
```

When `--agent-cmd` or `--baseline-cmd` is provided, `verify.py` executes the
command for each selected workload and reads JSON from stdout.

## Download

Use:

```bash
export WORKLOAD_ARCHIVE_URL=<archive-url>
bash scripts/download_workloads.sh
```

If the dataset is private, document the access path here and keep the actual
data out of git.

## Definition And Reference

The benchmark definition and reference semantics are vendored in:

```text
references/flashinfer-trace/definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
references/flashinfer-trace/references/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_reference.py
```

The reference Python code is the numerical reference only. Use it to understand
the operation and validate correctness with:

```text
atol = 1
rtol = 0.3
required_matched_ratio = 0.9
```

For speed, compare against the vendored FlashInfer baseline metadata:

```text
references/flashinfer-trace/baselines/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/flashinfer_wrapper_9sdjf3.json
```
