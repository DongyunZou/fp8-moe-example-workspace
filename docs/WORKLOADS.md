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

