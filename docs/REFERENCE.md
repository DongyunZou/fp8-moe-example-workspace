# Reference And Baseline

This repository vendors the FP8 MoE benchmark definition and reference material
from the local FlashInfer trace dataset.

## Vendored Files

Definition:

```text
references/flashinfer-trace/definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
```

PyTorch numerical reference extracted from the definition's `reference` field:

```text
references/flashinfer-trace/references/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_reference.py
```

FlashInfer speed baseline metadata:

```text
references/flashinfer-trace/baselines/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/flashinfer_wrapper_9sdjf3.json
```

## Correctness Acceptance

MoE correctness uses loose tolerances:

```text
atol = 1
rtol = 0.3
required_matched_ratio = 0.9
```

The PyTorch reference is only the numerical oracle for correctness. It is not
the performance target.

## Speed Baseline

Speed must be compared against the FlashInfer baseline:

```text
flashinfer_wrapper_9sdjf3
```

Do not report speedups against the PyTorch reference. The PyTorch reference is
expected to be much slower and exists to define the operation semantics.

