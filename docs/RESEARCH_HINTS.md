# Research Hints

KernelWiki is the first index, but it is not enough by itself. A serious FP8
MoE optimization pass must also inspect the upstream implementations below.

When using any source here:

- read the actual source, not only summaries;
- record exact repo, commit, and file paths in `docs/draft.md`;
- extract design ideas, not copy code blindly;
- if code is copied into the solution path, vendor the needed source under this
  repository and document license/provenance.

## 1. DeepSeek DeepGEMM Mega MoE

Repository:

```text
https://github.com/deepseek-ai/DeepGEMM
```

Start with these paths, or search for the same filenames if upstream moves:

```text
deep_gemm/mega/__init__.py
deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
csrc/jit_kernels/heuristics/mega_moe.hpp
```

What to learn:

- Mega MoE scheduler shape: how it chooses wave/expert grouping to keep SMs
  busy under routing imbalance.
- How dispatch, Linear1, SwiGLU/quant, Linear2, and combine are staged.
- Gate/up interleaving: GEMM1 weights may be arranged as small gate/up groups,
  for example `[gate0..7, up0..7, gate8..15, up8..15, ...]`, so the epilogue
  can pair accumulators without expensive post-processing.
- Custom epilogue structure for `SiLU(gate) * up`, amax/scale generation, and
  direct FP8 activation output for GEMM2.

Caveat:

- DeepGEMM Mega MoE is not a drop-in implementation for this benchmark. It may
  target FP8 activation x FP4 weight or include distributed/communication
  assumptions. Use it for scheduler and fusion design, then adapt to this
  benchmark's FP8 x FP8 contract.

## 2. CUTLASS Blackwell Block-Scale GEMM Examples

Repository:

```text
https://github.com/NVIDIA/cutlass
```

Start with:

```text
examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_blockwise.cu
examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_blockwise.cu
examples/81_blackwell_gemm_blockwise/81_blackwell_grouped_gemm_groupwise.cu
examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu
examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped.cu
```

If a filename has changed, search the CUTLASS tree for:

```text
81_blackwell_gemm_blockwise
Sm100BlockwiseScaleConfig
KernelPtrArrayTmaWarpSpecializedBlockwise
92_blackwell_moe_gemm
```

What to learn:

- FP8 block-scale GEMM with FP32 accumulation on SM100.
- Correct `ScaleConfig` choice and SFA/SFB major modes.
- Ptr-array grouped GEMM descriptor layout for per-expert problems.
- TMA alignment, stride construction, and packed scale layout.
- Which parts of the grouped scheduler are persistent/CLC-shaped and can become
  one-wave bottlenecks for small-M MoE.
- Tile-shape tradeoffs such as wider N reducing wave count, larger M increasing
  padding tax, and 2-CTA variants changing shared-memory pressure.

## 3. SGLang FP8 MoE Implementation

Repository:

```text
https://github.com/sgl-project/sglang
```

Start with:

```text
python/sglang/srt/layers/quantization/fp8.py
python/sglang/srt/layers/quantization/fp8_utils.py
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
python/sglang/srt/layers/moe/cutlass_moe_params.py
```

Search terms:

```text
Fp8MoEMethod
w8a8_block_fp8
blockwise
DeepGEMM
cutlass_fused_experts_fp8
SGLANG_ENABLE_FLASHINFER_FP8_GEMM
SGLANG_SUPPORT_CUTLASS_BLOCK_FP8
```

What to learn:

- How serving code maps global experts to local experts under expert
  parallelism.
- How FP8 MoE weights and block scales are loaded, packed, or fused.
- Backend selection logic between DeepGEMM, CUTLASS, FlashInfer, and Triton.
- How SGLang handles blockwise scale shape constraints and model-specific
  weight layouts.
- Which paths are correctness-sensitive versus performance-only.

## 4. FlashInfer / TensorRT-LLM Baseline Parity

Vendored baseline metadata:

```text
references/flashinfer-trace/baselines/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/flashinfer_wrapper_9sdjf3.json
```

What to learn:

- The speed baseline calls `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`.
- Shape and dtype assertions define the runner contract.
- The PyTorch reference is not the speed target.

For small/mid sequence lengths, explicitly investigate whether FlashInfer or
TensorRT-LLM uses a static/many-CTA BMM-like schedule instead of CUTLASS grouped
persistent scheduling.

## Required Research Output

`docs/draft.md` must include a table with:

```text
source | exact path | relevant mechanism | why it matters | whether it is usable here
```

At minimum, cover:

- DeepGEMM Mega MoE scheduler and epilogue fusion;
- CUTLASS SM100 block-scale GEMM scale layouts and grouped descriptors;
- SGLang FP8 MoE backend/scale/EP handling;
- FlashInfer baseline contract.

