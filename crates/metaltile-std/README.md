# metaltile-std

MetalTile kernel standard library — benchmark metadata and type definitions.
Provides the data types shared between `#[kernel(bench(...))]`-annotated kernel
definitions and the `tile bench` CLI runner. Contains no GPU runtime code.

Each `#[kernel(bench(...))]` attribute (from `metaltile-macros`) generates an
`inventory::submit! { BenchSpec { ... } }` alongside the kernel. The bench
CLI collects all registered `BenchSpec` instances via `inventory::iter`,
then runs each kernel against its MLX reference for throughput and
correctness verification.

## Supported Operations

<details>
<summary>Kernel coverage is complete — every op in the MLX / FFAI survey is ported. Click to expand the full list.</summary>

| Operation | Status |
|---|---|
| Unary elementwise — `exp`, `log`, `sqrt`, trig/hyperbolic, `erf`, `gelu`, `silu`, `sigmoid`, `relu`, … (40+) | ✅ |
| Binary elementwise — `add`, `sub`, `mul`, `div`, `max`, `min`, `pow`, `logaddexp`, `atan2`, `remainder` | ✅ |
| Fused binary (add+mul), ternary `select`, `copy`, strided copy (2-D + N-D), `arange`, `swiglu` | ✅ |
| Reductions — all / row / column / segmented (sum / max / min / prod) | ✅ |
| `softmax`, `logsumexp` | ✅ |
| `rms_norm` (+ small-N / wide / gated / fused-residual / fused-rope / fused-qgemv variants), `layer_norm` | ✅ |
| `rope` — rotary position embedding (standard, Llama-3 banded, 2-D vision) | ✅ |
| `argmax` / `argmin`, `scan` (inclusive + exclusive prefix sum), `sort` (bitonic + multi-block merge) | ✅ |
| `random` — xorshift / key-hash | ✅ |
| GEMV — dense and masked | ✅ |
| Quantized GEMV / GEMM — `qmv` / `qvm` / `qmm`, int3–8, gather / grouped-MoE BGEMM variants | ✅ |
| Affine quantize / dequantize — int2 / 3 / 4 / 5 / 6 / 8 | ✅ |
| FP4 / FP8 quantize / dequantize (E2M1, E4M3, E5M2) | ✅ |
| SDPA — vector decode (GQA), two-pass decode, batched-Q speculative decode | ✅ |
| SDPA — Flash-Attention-2 prefill, incl. simdgroup-MMA fragments | ✅ |
| SDPA — VLM vision-tower bidirectional (SigLIP / CLIP / FastViT / PaliGemma; d=32/64/72) | ✅ |
| Tiled GEMM — `steel_gemm` fused / gather / masked / segmented / split-K | ✅ |
| Convolution — 1-D / 2-D / 3-D / general (strided, dilated, grouped) + 3×3 Winograd | ✅ |
| FFT — radix-2 Cooley–Tukey, forward + inverse | ✅ |
| Scatter / gather-indexing family — `scatter`, `gather_axis`, `gather_front`, `masked_scatter` | ✅ |
| Hadamard transform — power-of-2 (FWHT) + non-power-of-2 (M ∈ {12, 20, 28}) | ✅ |
| AURA compressed-KV codec — encode / dequant / score / value / flash-attention | ✅ |
| GatedDeltaNet + Mamba/SSM recurrence — decode, chunked prefill, tape replay | ✅ |
| MoE — router top-k, permute / unpermute, grouped quantized BGEMM | ✅ |
| NAX (Apple `mpp::tensor_ops::matmul2d`) — GEMM, attention, quantized matmul | ✅ |
| Vision / STT / TTS front-end — patch conv, patch embed, mel-spectrogram, vocoder/iSTFT | ✅ |
| Sampling — categorical inverse-CDF, top-k / top-p / min-p, temperature, repetition penalty | ✅ |

See [`specs/KERNEL_AUDIT.md`](../../specs/KERNEL_AUDIT.md) for the full per-op coverage table and [`docs/developing.md`](../../docs/developing.md) for how kernels are organised.

</details>

## Position in the pipeline

```
metaltile-macros                         metaltile-cli
  (#[kernel(bench(...))]       (tile bench collects
   generates BenchSpec)         inventory::iter::<BenchSpec>)
       │                                    │
       └────────── metaltile-std ───────────┘
                   (this crate)
                   BenchSpec · ShapeSpec · bench_types
                   runner · run_spec · stats
```

`metaltile-std` is the shared vocabulary between kernel definitions and
the bench runner. It depends on the facade, core, codegen, and runtime
crates to provide DType helpers, MSL generation utilities, and the
`inventory`-based registration mechanism.

## Quick start

Define a kernel with bench registration:

```rust,ignore
use metaltile::kernel;
use metaltile_std::bench_types::{FLOAT_DTYPES, OpBench};

#[kernel(
    bench(
        op    = "unary",
        subop = "exp",
        class = Unary,
        input = Signed,
        tol   = 1e-4,
        mlx   = "v_Exp{tn}{tn}",
        metal_file = "unary.metal",
    )
)]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}
```

This single annotation registers the kernel for benchmarking under the
`"unary"` group with sub-operation `"exp"`. The `tile bench` CLI
discovers it automatically — no manual registration needed.

## Crate contents

| Module | Purpose |
|---|---|
| `mlx` | Kernel definitions benchmarked against MLX reference, organized by category (`mlx::unary`, `mlx::binary`, `mlx::reduce`, etc.) |
| `ffai` | Beyond-MLX kernels (attention, convolution, MoE, SSM, AURA codec, sampling, etc.) |
| `probe` | Hardware probing utilities (MMA layout probe, MPP matmul smoke test) |
| `spec` | `BenchSpec`, `ShapeSpec`, `BenchDispatch`, `ShapeSpec` constants, buffer init specs |
| `bench_types` | DType helpers, `OpBench`, `OpResult`, equivalence checking, MSL generation helpers |
| `runner` | GPU dispatch runner: compile MSL, allocate buffers, run kernels, measure GPU time |
| `run_spec` | Wire a `BenchSpec` through the full compile→dispatch→measure pipeline |
| `stats` | `BenchStats` struct and throughput calculation |
| `error` | `StdError` — runner and Metal error types |

## API reference

### Op catalog

Kernels live in two submodules depending on whether they have a side-by-side
MLX reference:

**`mlx/` — MLX-compared kernels** organized by category:

| File | Kernel(s) |
|---|---|
| `mlx/unary.rs` | `mt_exp`, `mt_log`, `mt_sqrt`, `mt_rsqrt`, `mt_abs`, `mt_silu`, `mt_gelu`, `mt_relu`, `mt_sigmoid`, `mt_sin`, `mt_cos`, `mt_ceil`, `mt_floor`, `mt_recip`, `mt_neg`, `mt_sign`, `mt_round`, `mt_erf`, `mt_exp2`, `mt_log2`, `mt_square`, `mt_log1p`, `mt_softplus`, `mt_sinh`, `mt_cosh`, `mt_tan`, `mt_tanh_op`, `mt_asin`, `mt_acos`, `mt_atan`, `mt_asinh`, `mt_acosh`, `mt_atanh`, `mt_expm1`, `mt_log10`, `mt_erfinv`, fused scalar-FMA, sigmoid-mul, add-rms-norm, and more |
| `mlx/binary.rs` | `vector_add`, `mt_mul`, `mt_sub`, `mt_div`, `mt_max_elem`, `mt_min_elem`, `mt_pow`, `mt_atan2`, `mt_remainder`, `mt_logaddexp` |
| `mlx/binary_two.rs` | `mt_binary_two` (fused add + mul, two outputs) |
| `mlx/ternary.rs` | `mt_select` (ternary select) |
| `mlx/arange.rs` | `mt_arange` |
| `mlx/copy.rs` | `mt_copy` |
| `mlx/strided.rs` | Strided (non-contiguous) copy kernels |
| `mlx/reduce.rs` | `mt_all_reduce`, `mt_all_reduce_max`, `mt_all_reduce_min`, `mt_row_reduce` |
| `mlx/softmax.rs` | `mt_softmax` |
| `mlx/rms_norm.rs` | `mt_rms_norm` |
| `mlx/layer_norm.rs` | `mt_layer_norm` |
| `mlx/logsumexp.rs` | `mt_logsumexp` |
| `mlx/gemv.rs` | `mt_gemv` |
| `mlx/gemv_masked.rs` | `mt_gemv_masked` |
| `mlx/scan.rs` | `mt_scan_f32` |
| `mlx/sort.rs` | `mt_sort_f32` |
| `mlx/arg_reduce.rs` | `mt_argmax_f32` |
| `mlx/scaled_dot_product_attention.rs` | SDPA vector decode kernel |
| `mlx/sdpa_vector.rs` | Additional SDPA vector dispatch |
| `mlx/rope.rs` | `mt_rope_f16` |
| `mlx/quantized.rs` | Quantized GeMV (int4) |
| `mlx/quantized_nax.rs` | NAX-accelerated quantized matvec (M4+) |
| `mlx/quantized_nax_int8.rs` | NAX int8 quantized matvec |
| `mlx/quantized_mpp.rs` | Quantized MPP matmul |
| `mlx/quantized_mpp_int8.rs` | Int8 quantized MPP matmul |
| `mlx/quantized_mma_dynamic_m.rs` | Dynamic-M quantized MMA |
| `mlx/fp_quantized.rs` | FP4 quantize/dequantize |
| `mlx/fp_quantized_nax.rs` | NAX FP4 dequantize (M4+) |
| `mlx/fp_quantized_mma.rs` | FP quantized MMA |
| `mlx/swiglu.rs` | SwiGLU fused activation |
| `mlx/fused_gate_activation.rs` | Fused gate activation |
| `mlx/hadamard.rs` | Hadamard transform (power-of-2) |
| `mlx/hadamard_m.rs` | Non-power-of-2 Hadamard (M ∈ {12, 20, 28}) |
| `mlx/gather_axis.rs` | Gather along axis |
| `mlx/scatter_axis.rs` | Scatter along axis |
| `mlx/indexing.rs` | Indexing ops |
| `mlx/random.rs` | `mt_random_hash` |
| `mlx/sgload_smoke.rs` | SGLoad smoke test |

**`ffai/` — beyond-MLX kernels:**

| File | Kernel(s) |
|---|---|
| `ffai/arg_reduce.rs` | Arg-reduce (argmax/argmin) |
| `ffai/audio_conv1d.rs` | 1-D audio convolution |
| `ffai/aura_encode.rs` | AURA KV-cache encode |
| `ffai/aura_dequant_rotated.rs` | AURA rotated dequant |
| `ffai/aura_score.rs` | AURA attention score |
| `ffai/aura_value.rs` | AURA value aggregation |
| `ffai/aura_flash_p1.rs` | AURA flash pass 1 |
| `ffai/aura_flash_pass2.rs` | AURA flash pass 2 |
| `ffai/aura_flash_sdpa.rs` | AURA flash SDPA |
| `ffai/batched_qkv_qgemv.rs` | Batched QKV quantized GEMV |
| `ffai/conv2d.rs` / `ffai/conv2d_mma.rs` | 2-D convolution (scalar + MMA) |
| `ffai/conv3d.rs` / `ffai/conv3d_mma.rs` | 3-D convolution (scalar + MMA) |
| `ffai/dequant_gather.rs` | Gather-based dequant |
| `ffai/dequant_gemv.rs` | Dequant GEMV |
| `ffai/dequant_gemv_expert_indexed.rs` | Expert-indexed dequant GEMV |
| `ffai/gated_delta.rs` | GatedDeltaNet core |
| `ffai/gated_delta_prep.rs` | GatedDelta prep |
| `ffai/gated_delta_prep_chunk.rs` | GatedDelta chunked prep |
| `ffai/gated_delta_replay.rs` | GatedDelta tape replay |
| `ffai/gated_delta_wy.rs` | GatedDelta WY representation |
| `ffai/gated_rmsnorm.rs` | Gated RMS norm |
| `ffai/gather.rs` | Gather ops |
| `ffai/gemm.rs` | GEMM ops |
| `ffai/kv_cache.rs` | KV-cache management |
| `ffai/logits_topk.rs` | Top-K logits |
| `ffai/logits_top_p.rs` | Top-P logits |
| `ffai/logits_min_p.rs` | Min-P logits |
| `ffai/logits_processors.rs` | Logit processor pipeline |
| `ffai/mel_spectrogram.rs` | Mel spectrogram (STT/TTS) |
| `ffai/moe.rs` | Mixture of Experts |
| `ffai/moe_mpp.rs` / `ffai/moe_mpp_bm64.rs` / `ffai/moe_mpp_bm8.rs` | MoE MPP matmul |
| `ffai/moe_mpp_int8.rs` / `ffai/moe_mpp_bm64_int8.rs` / `ffai/moe_mpp_bm8_int8.rs` | MoE MPP int8 matmul |
| `ffai/patch_embed.rs` / `ffai/patch_embed_mma.rs` | Vision patch embedding |
| `ffai/rms_norm_qgemv.rs` | RMS norm + quantized GEMV |
| `ffai/rms_norm_residual.rs` | RMS norm with residual |
| `ffai/rms_norm_rope.rs` | RMS norm + RoPE fused |
| `ffai/rope_2d.rs` | 2-D vision RoPE |
| `ffai/rope_llama.rs` | Llama-3 banded RoPE |
| `ffai/rope_yarn.rs` | YaRN RoPE scaling |
| `ffai/sampling.rs` | Categorical sampling |
| `ffai/sdpa_decode.rs` | SDPA decode (GQA) |
| `ffai/sdpa_decode_2pass.rs` | SDPA two-pass decode |
| `ffai/sdpa_decode_d64.rs` / `ffai/sdpa_decode_d256.rs` / `ffai/sdpa_decode_d512.rs` | SDPA per-dim decode |
| `ffai/sdpa_decode_batched.rs` | SDPA batched decode |
| `ffai/sdpa_decode_batched_prefill.rs` | Batched prefill decode |
| `ffai/sdpa_bidirectional.rs` | VLM bidirectional SDPA |
| `ffai/sdpa_multi.rs` | Multi-head SDPA |
| `ffai/flash_quantized_sdpa.rs` | Flash quantized SDPA |
| `ffai/ssm.rs` | Mamba/SSM recurrence |
| `ffai/ssm_replay.rs` | SSM tape replay |
| `ffai/vocoder.rs` | Vocoder / iSTFT |
| `ffai/winograd_conv.rs` | 3×3 Winograd convolution |

### Benchmark spec reference

`BenchSpec` (in `spec.rs`) is the central registration type. Each
`#[kernel(bench(…))]` annotation populates these fields:

| Field | Purpose |
|---|---|
| `op` / `subop` | Group and sub-operation label (e.g. `"unary"` / `"exp"`) |
| `kernel_name` | Rust function name as `&'static str` |
| `kernel_ir` | `fn(DType) -> Kernel` — builds IR for a given dtype |
| `dtypes` | `&'static [DType]` — which dtypes to benchmark (default: `FLOAT_DTYPES`) |
| `tol` | Absolute error tolerance for correctness |
| `mlx_src` | Optional MLX reference `.metal` source (embedded via `include_str!`) |
| `mlx_pattern` | Optional MLX kernel name pattern (`{tn}` → MLX type name) |
| `shapes` | `&'static [ShapeSpec]` — input sizes, grid config, buffer layout |
| `dispatch` | `BenchDispatch::Generic` or a complex variant (`Sort`, `Scan`, `Attention`, …) |
| `kernel_mode` | Optional override for `KernelMode` (e.g. `Reduction` for dequant GEMV) |

`ShapeSpec` describes the benchmark setup:

| Field | Purpose |
|---|---|
| `n` / `b` | Benchmark element count (N) and batch size (B) |
| `check_n` / `check_b` | Correctness-check element count (smaller, for speed) |
| `mode` | `KernelMode::Elementwise` or `Reduction` |
| `tpg` | Threads per threadgroup |
| `grid` | Dispatch grid shape (`DivCeilN`, `RowsB`, `Single`, …) |
| `tensor_bufs` | `&'static [TensorBufSpec]` — buffer count, init pattern, dtype override |
| `scalar_bufs` | `&'static [ScalarBufSpec]` — scalar arguments (U32N, U64N, …) |
| `cexprs` | Constexpr bindings, e.g. `&[("n", Dim::N)]` |
| `out_elems` / `reads` | Output element count and read count (for bandwidth calculation) |
| `bytes_fn` | Bandwidth formula (e.g. `bytes_elementwise`, `bytes_row_op`) |
| `mlx_args` | Optional MLX argument layout for the reference kernel |
| `mlx_grid` / `mlx_tpg` | Optional MLX grid override |

`BenchDispatch` controls how the runner executes the kernel:

| Variant | For |
|---|---|
| `BenchDispatch::Generic` | Simple kernels — uses `ShapeSpec`-defined grid and buffers |
| `BenchDispatch::Sort { b, n, tpg }` | Sort kernels with specialized input generation |
| `BenchDispatch::Scan { shapes, tpg }` | Scan kernels with multi-shape iteration |
| `BenchDispatch::ArgReduce { n, check_n, tpg }` | Arg-reduce with index-output validation |
| `BenchDispatch::Random { n, tpg }` | Random kernels with seed management |
| `BenchDispatch::FpQuantized { n, tpg }` | FP-quantized kernels |
| `BenchDispatch::QuantizedMatVec { shapes, group_size, tpg }` | Quantized matrix-vector multiply |
| `BenchDispatch::Rope { b, h, l, d, n_per_group }` | RoPE with multi-dimensional shapes |
| `BenchDispatch::Attention { shapes, tpg }` | SDPA with (B, L, D) shape triples |
| `BenchDispatch::StridedCopy { m, n, pad }` | Strided copy with padding |

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `metaltile` | Facade — `#[kernel]`, `Tensor`, prelude items |
| `metaltile-core` | `DType`, `Kernel`, `KernelMode`, `Shape`, `ConstExpr` |
| `metaltile-codegen` | `MslGenerator` for MSL generation tests (`generate_elementwise_msl`, `generate_reduction_msl`) |
| `metaltile-runtime` | Runtime types referenced by bench infrastructure |
| `inventory` | Distributed registration — `inventory::submit!` + `inventory::collect!` |

### External

| Crate | Role |
|---|---|
| `thiserror` | Derive `Error` for `StdError` |
| `half` | `f16` / `bf16` roundtrip conversion in benchmark data preparation |
| `bytemuck` | Zero-copy byte views of benchmark data buffers |
| `rustc-hash` | `FxHashMap` for spec and runner internals |
| `objc2` / `objc2-metal` / `objc2-foundation` | Metal GPU API bindings (macOS only, cfg-gated) |

## MSRV / platform

Rust: nightly (workspace-wide, for edition 2024).
No platform gating — this crate's types compile everywhere.
Benchmark execution requires macOS + Metal, but the types and
`BenchSpec` registration compile on any host.

### Feature flags

None — the crate has no Cargo features. NAX (Apple cooperative-tensor)
kernels build by default; runtime gating happens via
`Context::chip_family()`.

## Extending

- **New MLX kernel:** Create `src/mlx/<name>.rs` with `#[kernel(bench(…))]` annotation. Add `pub mod <name>;` to `src/mlx/mod.rs`.
  The `tile bench` CLI discovers it automatically via `inventory`.

- **New FFAI kernel (no MLX comparison):** Create `src/ffai/<name>.rs` with
  `#[kernel(bench(…))]` annotation. Add `pub mod <name>;` to
  `src/ffai/mod.rs`.

- **New benchmark shape:** `src/spec.rs` — add a `ShapeSpec` constant or
  update the relevant op file's `#[kernel(bench(...))]` annotation. Common shapes
  use the constants at the top of `spec.rs` (`ELEMENTWISE_N_BENCH`,
  `ROW_REDUCE_SHAPES`, etc.).

- **New `BenchDispatch` variant:** `src/spec.rs` — add to the `BenchDispatch`
  enum. Add a match arm in `src/run_spec.rs` for the complex runner. Update
  the `#[kernel(bench(...))]` proc-macro in `metaltile-macros` if a new
  `ClassKind` variant is needed.

- **New dtype helper:** `src/bench_types.rs` — add to `dtype_label()`,
  `mlx_tname()`, `elem_bytes()`, and `dtype_tol()` / `dtype_tol_reduce()`.

- **Tests to update:** `tile bench` suite (macOS + Metal). Unit tests in
  `src/bench_types.rs`.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-macros` README](../metaltile-macros/README.md) — the `#[kernel(bench(...))]` attribute that generates `BenchSpec` registration
- [`metaltile-cli` README](../metaltile-cli/README.md) — the `tile bench` runner that consumes these specs

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).