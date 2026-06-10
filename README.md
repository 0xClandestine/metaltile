<div align="center">
  <h1>MetalTile</h1>

  [![Backends][backends-badge]][backends-url]
  [![Rust][rust-badge]][rust-url]
  [![License][license-badge]][license-url]

  [backends-badge]: https://img.shields.io/badge/backends-MSL%20%C2%B7%20CUDA%20%C2%B7%20HIP%20%C2%B7%20Vulkan-black?style=flat-square
  [backends-url]: #backends
  [rust-badge]: https://img.shields.io/badge/language-Rust-orange?logo=rust&style=flat-square
  [rust-url]: https://www.rust-lang.org/
  [license-badge]: https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square
  [license-url]: LICENSE

  **[Docs](docs/)** | **[Baselines](baselines/)** | **[Contributing](CONTRIBUTING.md)**

</div>

---

A Rust-embedded DSL for writing GPU kernels once and running them everywhere. Write tile-level algorithms in Rust with `#[kernel]`, and the same kernel source lowers to **four GPU backends** — Apple Metal (MSL), NVIDIA (CUDA), AMD (HIP/ROCm), and any Vulkan-class GPU (SPIR-V) — verified against, and frequently faster than, hand-tuned MLX.

Write once, run on Apple, NVIDIA, AMD, and Vulkan-class GPUs — no per-backend kernel rewrite. metaltile is the kernel layer beneath an LLM inference engine that runs a 30B-parameter hybrid model (Mamba2 SSM + 128-expert MoE + GQA attention) resident-decode on a single Grace-Blackwell (GB10) box; the same kernels also run on Apple GPUs.

## Installation

```sh
curl -fsSL https://github.com/0xClandestine/metaltile/releases/latest/download/install.sh | sh
```

Run `tile update` at any time to upgrade to the latest release.

For contributors building from source, see [Getting Started](docs/getting-started.md).

## Getting Started

**1. Write a kernel.** Annotate a generic Rust function with `#[kernel(bench(...))]` — MetalTile generates `f32`, `f16`, and `bfloat16` variants from a single definition, lowers them to each enabled GPU backend (MSL by default; CUDA / HIP / Vulkan opt-in), and registers it against its MLX reference:

<table>
<tr>
<th>Rust DSL — what you write</th>
<th>Metal Shading Language — what you get</th>
</tr>
<tr>
<td>

```rust
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

**2. Install the CLI and run.**

```sh
cargo install --path crates/metaltile-cli
tile bench --filter mlx/gemv
```

```
tile bench · Apple M1 Max
  mlx/gemv
  Shape                                │   MT(µs) │  Ref(GB/s) │  MT(GB/s) │   MT% │  GFLOP/s │  ok
  ────────────────────────────────────────────────────────────────────────────────────────────────────
  N=16M f32                           │    192.8 │      350.1 │     348.2 │   99% │    174.1 │   ✓
  N=16M f16                           │     62.1 │      583.6 │     540.1 │   93% │    540.1 │   ✓
  N=16M bf16                          │    136.8 │      615.2 │     245.2 │   40% │    245.2 │   ✓
```

The default table adds wall-clock latency (`MT(µs)`) and compute throughput
(`GFLOP/s`, blank for memory-bound kernels); `-v` adds the roofline (`%BW` /
`%FLOP` / arithmetic intensity), occupancy/registers, and a bottleneck verdict.

Read the [docs](docs/) to learn more.

## Architecture

One `#[kernel]` DSL, four GPU backends. Your kernel lowers to a shared IR; the codegen passes optimise it once; then each backend emitter turns that IR into the target's native shader source. Two **peer hosts** consume the same kernels with no FFI between them — a Swift host (Metal/Apple, ships to the App Store) and the Rust host (`metaltile-runtime` + downstream engine crates).

![metaltile architecture](docs/architecture.png)

`#[kernel]` lowers your DSL function to IR; the codegen passes optimise it; each backend emitter then produces native shader source — MSL (`.metal`, compiled by `xcrun metal`), CUDA C++ (NVRTC → PTX at runtime), HIP C++ (hipRTC → AMDGPU code object), or SPIR-V (via shaderc). `#[bench]` / `#[test_kernel]` are optional annotations on the same function that register a setup callback the runner uses to dispatch the kernel and measure it (or diff against a CPU oracle).

### Backends

| Backend | Target GPU | Compile path | Status |
|---|---|---|---|
| **MSL** | Apple (Metal) | `.metal` → `metallib` (`xcrun metal`) | Stable — default, zero-config on macOS |
| **CUDA** | NVIDIA (sm_90 / 120 / 121, e.g. GB10) | CUDA C++ → NVRTC → PTX, runtime compile | Stable — `--features cuda` |
| **HIP** | AMD (ROCm, `gfx*`) | HIP C++ → hipRTC → AMDGPU code object | Complete · validation in progress — `--features hip` |
| **Vulkan** | Any Vulkan-class GPU | SPIR-V via shaderc → Vulkan compute | Complete · validation in progress — `--features vulkan` |

The non-Metal backends are opt-in Cargo features so the macOS Metal path stays zero-config and dependency-light. Each requires its toolchain/driver at link/run time (CUDA toolkit, ROCm, or the Vulkan SDK). HIP and Vulkan have the full kernel set implemented (codegen-complete); end-to-end model validation is in progress — they are not yet verified against a full model run. See `specs/AMD_BACKEND_SPEC.md` and `specs/VULKAN_BACKEND_SPEC.md`.

The CUDA runtime (`crates/metaltile-runtime/src/device/cuda/`) adds NVRTC runtime kernel compile, a dedicated capturable non-blocking stream, CUDA-graph capture hooks (`begin_capture` / `end_capture` / `graph_launch`), a buffer pool, pinned async host-to-device copies, and an optional `--fmad` codegen gate (`MT_FMAD=1`). See `specs/CUDA_BACKEND_SPEC.md`.

> Today `tile bench` / `tile test` dispatch through the in-process `GpuRunner` on the Metal path; moving the runner into a dedicated subprocess (for isolation and parallelism) and wiring the CLI harness across all backends (Phase 6) is planned.

## Scope & naming

metaltile began as a Metal-only (MSL) kernel/code generator. It now emits MSL, CUDA, HIP, and Vulkan (SPIR-V) from a single `#[kernel]` DSL, so the "metal" in the name understates the current scope. A rename is under discussion to better reflect the multi-backend reality — a candidate is **TileForge**, but this is not final and is open for discussion. The current name (`metaltile`) still applies everywhere until any rename is decided.

## CLI reference

| Command | What it does |
|---|---|
| `tile build` | Compile every `#[kernel]` in the workspace to MSL and (optionally) a `metallib`. |
| `tile bench` | Run every `#[bench]`, report MetalTile GB/s vs the MLX reference + correctness. |
| `tile test` | Run every `#[test_kernel]` against its CPU oracle within tolerance. |
| `tile inspect` | Dump IR / per-pass IR / MSL for one kernel. |
| `tile device` | Print GPU device info and supported feature flags. |
| `tile snap` | Save bench results as a regression baseline. |
| `tile diff` | Compare bench results to a saved baseline. |
| `tile update` | Install the latest release (or build from a PR / commit). |

See [`docs/cli.md`](docs/cli.md) for the full flag surface.

## Crates

| Crate | Role |
|---|---|
| `metaltile-core` | Core IR types and `Op` variants shared by every backend. |
| `metaltile-macros` | The `#[kernel]` / `#[bench]` / `#[test_kernel]` proc-macros. |
| `metaltile-codegen` | IR optimisation passes + the four backend emitters (`msl/`, `cuda/`, `hip/`, `spirv/`). |
| `metaltile-runtime` | Host runtime + per-backend device modules (`device/{metal,cuda,hip,vulkan}/`); CUDA/HIP/Vulkan behind the `cuda`/`hip`/`vulkan` features. |
| `metaltile-std` | Kernel standard library — bench/test metadata and shared type definitions. |
| `metaltile` | Umbrella crate re-exporting the public DSL surface. |
| `metaltile-cli` | The `tile` CLI — build, bench, test, inspect. |

The Swift host (`MetalTileSwift`, Metal/Apple, App Store) is a separate peer consumer of the same kernels and lives outside this workspace.

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## Acknowledgements

MetalTile's benchmark suite and kernel library stand on the shoulders of the MLX ecosystem. A large
portion of the `metaltile-std` kernels are ports or faithful re-implementations of kernels from the following projects:

- [**ml-explore/mlx**](https://github.com/ml-explore/mlx) — primary source for reference kernels.
- [**ekryski/mlx**](https://github.com/ekryski/mlx) (`alpha`) — FFAI extensions: gated-delta, SSM replay, AURA codec.
- [**ml-explore/mlx-lm**](https://github.com/ml-explore/mlx-lm) — reference for GatedDeltaNet step semantics.

We are grateful to the MLX team at Apple and the broader MLX community, this wouldn't have been possible without you.

See [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) for the full list of individual contributors and third-party software.

## License

<sup>
Licensed under the <a href="LICENSE">Apache License, Version 2.0</a>.
</sup>
