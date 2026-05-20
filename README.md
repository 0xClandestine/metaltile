# MetalTile

A Rust-embedded DSL for writing Apple Metal GPU kernels. Write tile-level algorithms in Rust, get optimized Metal Shading Language out.

```rust
#[kernel]
pub fn mt_rms_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let ssq = strided_reduce_dot(x, x, rs, 0, re);
    let tg_ssq = reduce_sum(ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    let n_full = n / (lsize * 4u32);
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = load(x[base]).cast::<f32>() * rms * load(w[col]).cast::<f32>();
        let n1 = load(x[base + 1u32]).cast::<f32>() * rms * load(w[col + 1u32]).cast::<f32>();
        let n2 = load(x[base + 2u32]).cast::<f32>() * rms * load(w[col + 2u32]).cast::<f32>();
        let n3 = load(x[base + 3u32]).cast::<f32>() * rms * load(w[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let ni = load(x[_i]).cast::<f32>() * rms * load(w[_i - rs]).cast::<f32>();
        store(out[_i], ni.cast::<T>());
    }
}
```

This generates ~104% of MLX's hand-tuned `rms` kernel throughput on M4 Max across f32 / f16 / bfloat16.

## Why MetalTile

- **Write once in Rust, run fast on Apple Silicon.** No raw MSL, no thread-position arithmetic.
- **Tile-level, not thread-level.** `strided_reduce`, `reduce_sum`, `dot` — express *what* to compute; the compiler handles thread mapping, vectorization, and SIMD-group reductions.
- **Verified against MLX.** Every kernel is benchmarked and numerically compared against the corresponding MLX Metal kernel.
- **All three float dtypes.** `f32`, `f16`, and `bfloat16` work identically — native `bfloat` emitted on Metal 3.1+.
- **Layered correctness.** Codegen passes and the `#[kernel]` proc-macro have unit + golden-MSL-snapshot coverage that runs on any platform; kernels are verified end-to-end by GPU correctness tests against naive CPU references and by the MLX side-by-side bench.

## Status

Early development — APIs are not yet stable. The core DSL works; an autotuner and type-level shape algebra are planned.

| Crate | Description |
|---|---|
| [`metaltile-core`](crates/metaltile-core/README.md) | IR types, `DType`, `Shape` |
| [`metaltile-macros`](crates/metaltile-macros/README.md) | `#[kernel]` proc macro |
| [`metaltile-codegen`](crates/metaltile-codegen/README.md) | MSL lowering + optimization passes |
| [`metaltile-runtime`](crates/metaltile-runtime/README.md) | Metal dispatch, PSO cache |
| [`metaltile`](crates/metaltile/README.md) | facade re-exporting all crates |
| [`metaltile-std`](crates/metaltile-std/README.md) | kernel stdlib, op files, bench types |
| [`metaltile-cli`](crates/metaltile-cli/README.md) | the `tile` CLI binary |

## Quick Start

```toml
[dependencies]
metaltile = "0.1"
```

```rust
use metaltile::prelude::*;

#[kernel]
fn vector_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) {
    let idx = program_id::<0>();
    store(c[idx], load(a[idx]) + load(b[idx]));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new()?;
    let n = 256usize;
    let a: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let c = vec![0u8; n * 4];

    let result = vector_add::launch(&ctx)
        .input("a", a)
        .input("b", b)
        .input("c", c)
        .dispatch()?;

    let out: Vec<f32> = result.outputs["c"]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("out[0] = {}", out[0]); // 1.0
    Ok(())
}
```

Full walkthrough: [`docs/getting-started.md`](docs/getting-started.md).

## Supported Operations

Operation categories benchmarked against MLX:

**Elementwise** — unary (exp, log, sqrt, sin, cos, erf, sigmoid, silu, gelu, relu, …), binary (add, mul, sub, div, max, min, pow, logaddexp), fused add+mul, ternary select, copy, arange.

**Reductions** — all-reduce sum/max/min, row-reduce sum/max/min, logsumexp, softmax, rms-norm, layer-norm.

**Matrix** — GEMV, masked GEMV, SDPA (vector decode + prefill).

**Misc** — RoPE, scan (parallel prefix sum), arg-reduce, sort (bitonic), random (xorshift32), quantized GEMV / GEMM, fp4 quantize/dequantize, strided copy.

## Benchmarks

`tile bench` runs every kernel against its MLX Metal reference on identical buffers and reports throughput + a correctness check. Selected results (M4 Max, higher = better vs MLX):

| Op | MT % of MLX |
|---|---|
| softmax f32 | ~105% |
| rms_norm f16 | ~104% |
| all_reduce f32 | ~100% |
| gemv f16 | ~100% |
| argmax f32 | **206%** |
| scan f32 | ~104% |

See [`docs/cli.md`](docs/cli.md) for the full `tile` reference.

## Architecture

```
#[kernel] fn  →  metaltile-macros (proc macro)
                          │
                    MetalTile IR  (metaltile-core)
                          │
               metaltile-codegen (optimization passes → MSL)
                          │
                  metaltile-runtime (Metal GPU dispatch)
```

Optimization passes run in order: TypeCheck → ConstFold → AlgebraicSimplify → CopyProp → CSE → LICM → IfConversion → ValueSink → TileLowering → Fusion → Unroll → Schedule → Vectorize → DeadStoreElim.

## Documentation

Full docs live in [`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md) — toolchain, build, first kernel.
- [Developing](docs/developing.md) — repo layout, dev loop, and the **kernel-authoring hazards** (a wrong dispatch can freeze the machine).
- [Testing](docs/testing.md) — test layers, CI, and test-infra gaps.
- [CLI](docs/cli.md) — the `tile` binary.
- [Publishing](docs/publishing.md) — the release flow.

## Contributing

Contributions — including AI-assisted ones — are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
