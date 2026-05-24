# metaltile-std

MetalTile kernel standard library — benchmark metadata and type definitions.
Provides the data types shared between `#[bench_kernel]`-annotated kernel
definitions and the `tile bench` CLI runner. Contains no GPU runtime code.

Each `#[bench_kernel]` attribute (from `metaltile-macros`) generates an
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

See [`docs/KERNEL_AUDIT.md`](../../docs/KERNEL_AUDIT.md) for the full per-op coverage table and [`docs/developing.md`](../../docs/developing.md) for how kernels are organised.

</details>

## Position in the pipeline

```
metaltile-macros                         metaltile-cli
  (#[bench_kernel]             (tile bench collects
   generates BenchSpec)         inventory::iter::<BenchSpec>)
       │                                    │
       └────────── metaltile-std ───────────┘
                   (this crate)
                   BenchSpec · ShapeSpec · OpBench
                   OpResult · suite printer · term
```

`metaltile-std` is the shared vocabulary between kernel definitions and
the bench runner. It depends on the facade, core, codegen, and runtime
crates to provide DType helpers, MSL generation utilities, and the
`inventory`-based registration mechanism.

## Quick start

Define a kernel with bench registration:

```rust,ignore
use metaltile::{bench_kernel, kernel};
use metaltile_std::bench_types::{FLOAT_DTYPES, OpBench};

#[bench_kernel(
    op    = "unary",
    subop = "exp",
    class = Unary,
    input = Signed,
    tol   = 1e-4,
    mlx   = "v_Exp{tn}{tn}",
    metal_file = "unary.metal",
)]
#[kernel]
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
| `ops` | Kernel definitions registered with `#[bench_kernel]`, organized by category |
| `spec` | `BenchSpec`, `ShapeSpec`, dispatch enums, buffer init, grid computation |
| `bench_types` | DType helpers, `OpBench`, `OpResult`, equivalence checking, suite printer |
| `term` | ANSI terminal formatting — `paint_stdout`, `paint_stderr`, `Style`, `Color` |

## API reference

### Op catalog

Ops are organized by category. Each file registers one or more kernels
via `#[bench_kernel(…)]` + `#[kernel]`.

**Elementwise:** `unary.rs`, `binary.rs`, `binary_two.rs`, `ternary.rs`, `arange.rs`, `copy.rs`, `strided.rs`

| Op file | Kernel(s) |
|---|---|
| `unary.rs` | `mt_exp`, `mt_log`, `mt_sqrt`, `mt_rsqrt`, `mt_abs`, `mt_silu`, `mt_gelu`, `mt_relu`, `mt_sigmoid`, `mt_sin`, `mt_cos`, `mt_ceil`, `mt_floor`, `mt_recip`, `mt_neg`, `mt_sign`, `mt_round`, `mt_erf`, `mt_exp2`, `mt_log2`, `mt_square`, `mt_log1p`, `mt_softplus` |
| `binary.rs` | `vector_add`, `mt_mul`, `mt_sub`, `mt_div`, `mt_max_elem`, `mt_min_elem`, `mt_pow`, `mt_logaddexp` |
| `binary_two.rs` | `mt_binary_two` (fused add + mul, two outputs) |
| `ternary.rs` | `mt_select` (ternary select) |
| `arange.rs` | `mt_arange` |
| `copy.rs` | `mt_copy` |
| `strided.rs` | Strided (non-contiguous) copy kernels |

**Reductions:** `reduce.rs`, `softmax.rs`, `rms_norm.rs`, `layer_norm.rs`, `logsumexp.rs`

| Op file | Kernel(s) |
|---|---|
| `reduce.rs` | `mt_all_reduce`, `mt_all_reduce_max`, `mt_all_reduce_min`, `mt_row_reduce` |
| `softmax.rs` | `mt_softmax` |
| `rms_norm.rs` | `mt_rms_norm` |
| `layer_norm.rs` | `mt_layer_norm` |
| `logsumexp.rs` | `mt_logsumexp` |

**Matrix:** `gemv.rs`, `gemv_masked.rs`

| Op file | Kernel(s) |
|---|---|
| `gemv.rs` | `mt_gemv` |
| `gemv_masked.rs` | `mt_gemv_masked` |

**Sequence:** `scan.rs`, `sort.rs`, `arg_reduce.rs`

| Op file | Kernel(s) |
|---|---|
| `scan.rs` | `mt_scan_f32` |
| `sort.rs` | `mt_sort_f32` |
| `arg_reduce.rs` | `mt_argmax_f32` |

**Attention:** `scaled_dot_product_attention.rs`, `rope.rs`

| Op file | Kernel(s) |
|---|---|
| `scaled_dot_product_attention.rs` | SDPA vector decode kernel |
| `rope.rs` | `mt_rope_f16` |

**Quantized:** `quantized.rs`, `fp_quantized.rs`, `quantized_nax.rs`, `fp_quantized_nax.rs`

| Op file | Kernel(s) |
|---|---|
| `quantized.rs` | Quantized GeMV (int4) |
| `fp_quantized.rs` | FP4 quantize/dequantize |
| `quantized_nax.rs` | NAX-accelerated quantized matvec (M4+ runtime) |
| `fp_quantized_nax.rs` | NAX-accelerated FP4 dequantize (M4+ runtime) |

NAX kernels build by default; the runtime dispatcher gates them via `Context::chip_family()` on Apple10+ hardware. Tests use `skip_unless_apple10` to auto-skip on pre-M4 chips.

**Misc:** `random.rs`, `conv.rs`, `fft.rs`, `fence.rs`

| Op file | Kernel(s) |
|---|---|
| `random.rs` | `mt_random_hash` |
| `conv.rs` | *(stub — not yet implemented)* |
| `fft.rs` | *(stub — not yet implemented)* |
| `fence.rs` | *(stub — not yet implemented)* |

### Benchmark spec reference

`BenchSpec` (in `spec.rs`) is the central registration type. Each
`#[bench_kernel(…)]` annotation populates these fields:

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
| `metaltile` | Facade — `#[kernel]`, `#[bench_kernel]`, `Tensor`, prelude items |
| `metaltile-core` | `DType`, `Kernel`, `KernelMode`, `Shape`, `ConstExpr` |
| `metaltile-codegen` | `MslGenerator` for MSL generation tests (`generate_elementwise_msl`, `generate_reduction_msl`) |
| `metaltile-runtime` | Runtime types referenced by bench infrastructure |
| `inventory` | Distributed registration — `inventory::submit!` + `inventory::collect!` |

### External

None — all dependencies are internal workspace crates.

## MSRV / platform

Rust: nightly (workspace-wide, for edition 2024).
No platform gating — this crate is pure data types.
Benchmark execution requires macOS + Metal, but the types compile
everywhere.

### Feature flags

None — the crate has no Cargo features. NAX (Apple cooperative-tensor) kernels build by default; runtime gating happens via `Context::chip_family()`.

## Extending

- **New op file:** Create `src/ops/<name>.rs` with `#[bench_kernel(…)]` +
  `#[kernel]` annotations. Add `pub mod <name>;` to `src/ops/mod.rs`.
  The `tile bench` CLI discovers it automatically via `inventory`.

- **New benchmark shape:** `src/spec.rs` — add a `ShapeSpec` constant or
  update the relevant op file's `#[bench_kernel]` annotation. Common shapes
  use the constants at the top of `spec.rs` (`ELEMENTWISE_N_BENCH`,
  `ROW_REDUCE_SHAPES`, etc.).

- **New `BenchDispatch` variant:** `src/spec.rs` — add to the `BenchDispatch`
  enum. Add a match arm in `metaltile-cli/src/run_spec.rs` for the complex
  runner. Update the `#[bench_kernel]` proc-macro in `metaltile-macros/src/lib.rs`
  if a new `ClassKind` variant is needed.

- **New dtype helper:** `src/bench_types.rs` — add to `dtype_label()`,
  `mlx_tname()`, `elem_bytes()`, and `dtype_tol()` / `dtype_tol_reduce()`.

- **Tests to update:** `tile bench` suite (macOS + Metal). Unit tests in
  `src/bench_types.rs`.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-macros` README](../metaltile-macros/README.md) — the `#[bench_kernel]` attribute that generates `BenchSpec` registration
- [`metaltile-cli` README](../metaltile-cli/README.md) — the `tile bench` runner that consumes these specs

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
