Read this repository and build a Blackwell FP8 MoE optimization research plan.

Use KernelWiki for references, but do not rely on KernelWiki alone. Start from
`docs/TASK.md`, `docs/HANDOFF.md`, `docs/RESEARCH_HINTS.md`,
`benchmark.csv`, and `solutions.jsonl`.

Mandatory upstream hints to inspect:

- DeepSeek DeepGEMM Mega MoE.
- CUTLASS Blackwell FP8 block-scale GEMM examples, especially examples under
  `81_blackwell_gemm_blockwise` and `92_blackwell_moe_gemm`.
- SGLang FP8 MoE implementation and backend selection.
- The vendored FlashInfer `flashinfer_wrapper_9sdjf3` speed baseline metadata.

Write `docs/draft.md` with:

- current implementation summary;
- likely bottlenecks;
- relevant KernelWiki pages and upstream references;
- a table of source, exact path, relevant mechanism, why it matters, and whether
  it is directly usable here;
- candidate directions;
- expected profile evidence for each direction;
- acceptance criteria.

Do not edit solution code yet.
