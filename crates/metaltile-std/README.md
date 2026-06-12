# metaltile-std

MetalTile's **kernel standard library** — every `#[kernel]` the project ships,
plus the host-side quantization layer they share. Each kernel is a Rust function
in the MetalTile DSL, annotated with `#[kernel]` (and optionally
`#[kernel(variants(...))]`), and carries its own GPU correctness tests
(`#[test_kernel]`) and throughput benches (`#[bench]`) in the same file. The
proc-macros register everything into the `inventory` so `tile build` / `tile
test` / `tile bench` discover it automatically — no manual registration.

This crate depends on the `metaltile` facade for the DSL macros and the harness
types; it adds no new GPU runtime of its own.

## Crate contents

| Module | Purpose |
|---|---|
| `mlx` | Kernels with an upstream MLX `.metal` counterpart they can be benched against (`mt_softmax`, `mt_copy`, `mt_gemv`, `mt_rope`, the elementwise/reduce/sort/scan families, the `steel/` tiled-GEMM/attention path, …). |
| `ffai` | Model-specific kernels with no upstream metal mirror — attention (SDPA decode/prefill/bidirectional/flash), MoE, SSM / GatedDeltaNet, RoPE variants, AURA KV codec, RMSNorm fusions, vision/STT/TTS front-ends, sampling, GGUF/DSv4 dequant. |
| `convolution` | The consolidated convolution family — `conv1d`/`conv2d`/`conv3d` (direct + depthwise + MMA + block-scaled), Winograd, the causal-streaming path, and the `steel_conv/` implicit-GEMM port. See [`docs/specs/KERNEL_CONSOLIDATION_PLAN.md`](../../docs/specs/KERNEL_CONSOLIDATION_PLAN.md) — this module is the proven exemplar for the wider reorg. |
| `quant` | The precision layer: `codec` (host element/scale encode-decode primitives — the single source of truth shared by kernels and oracles), `format` (the ~30-format block-scaled `QFormat` matrix + host packer/oracle), and `gguf` (GGUF k-quant layouts). |
| `probe` | Hardware probes (MMA-layout probe, MPP matmul smoke test). |
| `utils` | Shared host helpers — `pack_f32` / `unpack_f32` dtype round-tripping used by tests and benches. |

> The crate is mid-migration from the legacy `mlx/` + `ffai/` split to a
> family-organized `kernels/<family>/` layout — see the consolidation plan.

## Writing a kernel

A kernel file is four sections: a `//!` doc comment, the `#[kernel]` fn(s), a
`kernel_tests` module, and a `kernel_benches` module. The
[Kernel Style Guide](../../docs/STYLE_GUIDE.md) is the authority on the shape,
naming, the `variants(...)` axis, shared primitives, the CPU oracle, and the
bench. In brief:

```rust,ignore
use metaltile::kernel;

/// Multiply each element of `inp` by `alpha`.
#[kernel]
pub fn mt_scale<T>(inp: Tensor<T>, mut out: Tensor<T>, #[constexpr] alpha: f32) {
    let idx = program_id::<0>();
    store(out[idx], (load(inp[idx]).cast::<f32>() * alpha).cast::<T>());
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};
    use super::mt_scale;
    use crate::utils::{pack_f32, unpack_f32};

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3, 1e-3])]
    fn test_mt_scale(dt: DType) -> TestSetup { /* … oracle vs GPU … */ }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};
    use super::mt_scale;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mt_scale(dt: DType) -> BenchSetup { /* … */ }
}
```

`#[kernel(variants(...))]` stamps out one specialized kernel per compile-time
tuple (bit-width, head-dim, quant format, …) and is the primary tool for
collapsing the dtype/format families. The pinned MLX `.metal` sources for the
`mlx/` benches' optional `.with_reference(...)` comparators are fetched at build
time (`build.rs`).

## Dependencies

| Crate | Role |
|---|---|
| `metaltile` (facade) | the `#[kernel]` / `#[bench]` / `#[test_kernel]` macros, `Tensor`, the harness `test`/`bench` builders, and the runner. |
| `metaltile-core` | `DType`, `Kernel`, `KernelMode`, IR types. |
| `metaltile-codegen` | MSL/backend emission used by the in-source codegen tests. |
| `half`, `bytemuck`, `rustc-hash` | dtype round-tripping, byte views, fast maps. |
| `objc2-metal` (cfg-gated, macOS) | Metal bindings used by the on-device tests. |

## Platform

Rust nightly (workspace edition 2024). The kernel definitions, host
quantization layer, and inventory registration compile on any host; running the
correctness tests / benches on-device requires a supported GPU backend (Metal on
macOS by default; CUDA / HIP / Vulkan are feature-gated in `metaltile-runtime`).

## Related documentation

- [Kernel Style Guide](../../docs/STYLE_GUIDE.md) — how to write one kernel/bench/test.
- [Kernel Consolidation Plan](../../docs/specs/KERNEL_CONSOLIDATION_PLAN.md) — the `kernels/<family>/` target layout + migration roadmap.
- [`specs/KERNEL_AUDIT.md`](../../docs/specs/KERNEL_AUDIT.md) — per-op coverage table.
- [`docs/developing.md`](../../docs/developing.md) — kernel-authoring hazards.

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
