<p align="center">
  <h1 align="center">MetalTile</h1>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

<p align="center">A Rust-embedded DSL for writing Apple Metal GPU kernels. Write tile-level algorithms in Rust, get optimized Metal Shading Language out — verified against, and frequently faster than, hand-tuned MLX.</p>

> ⚠️ Early development — APIs are not yet stable. The core DSL, codegen, and runtime work today; the autotuner and type-level shape algebra are planned.

<table>
<tr>
<th>Rust DSL — what you write</th>
<th>Metal Shading Language — what you get</th>
</tr>
<tr>
<td>

```rust
#[kernel]
pub fn mt_exp<T>(
    a: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}
```

</td>
<td>

```cpp
kernel void mt_exp(
    const device float *a [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint v_idx = tid;
    auto v1 = a[v_idx];
    auto v2 = exp(v1);
    out[v_idx] = v2;
}
```

</td>
</tr>
</table>

One generic `#[kernel]` fn becomes a monomorphised `f32` / `f16` / `bfloat16` Metal kernel — the compiler handles thread indexing, dtype lowering, and Metal idioms. Bigger kernels lean on tile-level primitives (`reduce_sum`, `strided_reduce`, `dot`); the codegen emits the simdgroup and threadgroup machinery for you.

## Why MetalTile

| Functionality | Description | Status |
|---|---|---|
| **Write kernels in Rust** | A real `#[kernel]` proc-macro — no raw MSL, no hand-written thread-position arithmetic. | ✅ |
| **Tile-level primitives** | `reduce_sum`, `strided_reduce`, `dot` — say *what* to compute; codegen emits the simdgroup + threadgroup reduction. | ✅ |
| **One source, three dtypes** | Generic `<T>` kernels lower to `f32`, `f16`, and `bfloat16` — native `bfloat` on Metal 3.1+. | ✅ |
| **Optimizing compiler** | A 14-pass pipeline — const-folding, CSE, LICM, fusion, vectorization, and more — sits between the IR and the emitted MSL. | ✅ |
| **Verified against MLX** | Every benched kernel runs side-by-side against the hand-tuned MLX Metal kernel and must match it numerically. | ✅ |
| **Frequently faster than MLX** | A meaningful slice of ops — argmax, small-N RMSNorm, quantized matmul — land 3×+ over MLX on M4 Max. | ✅ |
| **`tile` CLI** | `bench` / `build` / `inspect` / `device` / `snap` / `diff` — one binary for the whole dev loop. | ✅ |
| **Cross-hardware baselines** | Committed `tile bench` snapshots per chip; CI diffs every PR against them. | ✅ |
| **Autotuner** | Per-shape kernel tuning so no performance is left on the table. | 🚧 Planned |
| **Type-level shape algebra** | Tensor shapes checked at compile time. | 🚧 Planned |

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

See [`docs/KERNEL_AUDIT.md`](docs/KERNEL_AUDIT.md) for the full per-op coverage table and [`docs/developing.md`](docs/developing.md) for how kernels are organised.

</details>

## Benchmarks

`tile bench` dispatches every MetalTile kernel and its MLX Metal reference on identical buffers, then reports throughput and a numerical-equivalence check. Run the whole suite, or narrow with `--filter`:

```sh
tile bench                   # full suite
tile bench --filter softmax  # one op
```

A sample of what to expect: `mt_rms_norm_small` runs at **354% of MLX's hand-tuned `rms` kernel** on an Apple M4 Max (`B=1024 N=64`, f32). Full cross-hardware results live in [`baselines/`](baselines/) — committed snapshots per chip, refreshed as new hardware is benched. CI diffs every PR against the matching baseline.

See [`docs/cli.md`](docs/cli.md) for `-v` / `-vv` profiling, JSON output, and the `snap` / `diff` regression workflow.

## Documentation

Full docs live in [`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md) — toolchain, crate layout, build, first kernel.
- [Developing](docs/developing.md) — repo layout, dev loop, and the **kernel-authoring hazards** (a wrong dispatch can freeze the machine).
- [Testing](docs/testing.md) — test layers, CI, and test-infra gaps.
- [CLI](docs/cli.md) — the `tile` binary.
- [Publishing](docs/publishing.md) — the release flow.

## Contributing

Contributions — including AI-assisted ones — are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
