<!--
Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
SPDX-License-Identifier: Apache-2.0
-->
# MetalTile Architecture

How a `#[kernel]` becomes a compiled GPU shader, and how `tile bench` / `tile
test` / `tile build` run and measure it today. Companion docs:
[`TOOLCHAIN_DESIGN.md`](TOOLCHAIN_DESIGN.md) (the `#[kernel]` /
`#[kernel(variants(...))]` / `#[bench]` / `#[test_kernel]` macro surface),
[`BENCH_METRICS_SPEC.md`](BENCH_METRICS_SPEC.md) (metric definitions),
[`CONV_CONSOLIDATION_PLAN.md`](CONV_CONSOLIDATION_PLAN.md) (the `convolution/`
reorg), the backend specs ([`CUDA`](CUDA_BACKEND_SPEC.md) /
[`AMD`](AMD_BACKEND_SPEC.md) / [`VULKAN`](VULKAN_BACKEND_SPEC.md) /
[`ANE`](ANE_BACKEND_SPEC.md)), [`cli.md`](../cli.md) (command flags),
[`developing.md`](../developing.md) (kernel-authoring hazards).

> **Two things drive the current shape of the runtime:**
> 1. **The runner is a subprocess.** `tile` spawns a generated `__tile_runner`
>    binary (linked against the project's `metaltile-std`, so its kernel
>    inventory is populated) and streams results back as `ProtocolMessage`
>    JSON lines — see [Subprocess execution](#subprocess-execution).
> 2. **Codegen is multi-backend.** IR lowers through a `CodegenBackend` to
>    **Metal** (default, mature), **CUDA**, **HIP/ROCm**, or **SPIR-V/Vulkan**.
>    Non-Metal device execution is feature-gated and still being brought up —
>    see [Multi-backend codegen](#multi-backend-codegen).

## Crates

```mermaid
flowchart TD
    macros["metaltile-macros<br/>#[kernel] · variants · #[bench] · #[test_kernel]"]
    core["metaltile-core<br/>IR + wire protocol<br/>(no GPU)"]
    codegen["metaltile-codegen<br/>passes + backends<br/>(MSL · CUDA · HIP · SPIR-V)"]
    runtime["metaltile-runtime<br/>device dispatch<br/>(Metal · CUDA · HIP · Vulkan)"]
    facade["metaltile (facade)<br/>harness/ + runner/"]
    std["metaltile-std<br/>kernel stdlib<br/>(mlx · ffai · convolution · quant)"]
    cli["metaltile-cli<br/>tile binary (thin)"]

    macros --> core
    codegen --> core
    runtime --> core
    facade --> core
    facade --> codegen
    facade --> runtime
    std --> facade
    cli --> facade
    cli --> codegen
    cli --> std
    cli --> core
```

| Crate | Responsibility |
|---|---|
| `metaltile-core` | IR (`Op`, `Kernel`) + the `protocol` wire types (`ProtocolMessage`, `ProfileInfo`). Pure — no GPU, no tooling deps. |
| `metaltile-macros` | `#[kernel]` (lowers a DSL fn to IR) + `#[kernel(variants(...))]` (compile-time specialisation — stamps one kernel per tuple of int/type/float params); `#[bench]` / `#[test_kernel]` (register a setup callback via `inventory`). |
| `metaltile-codegen` | Optimization passes (const-fold, vectorize, unroll, fusion, DCE, …) + the `CodegenBackend` seam: `msl/` (Metal, default) and the `cuda/` / `hip/` / `spirv/` generators. `backend.rs` holds the `Target` enum, `TargetProfile`, and `MmaStrategy`. |
| `metaltile-runtime` | Per-backend device, buffers, PSO/module cache, dispatch + timing. `device/metal_device.rs` is the default; `device/{cuda,hip,vulkan}/` are **feature-gated** (`--features cuda\|hip\|vulkan`). |
| `metaltile` (facade) | Re-exports the above; hosts `harness/` (the kernel/bench/test **registries**) and `runner/` (the `__tile_runner` engine: `RunnerHarness`, `GpuRunner`, per-backend dispatch, arg parsing, protocol emit, profiling, device specs). |
| `metaltile-std` | The kernel standard library. Modules: `mlx/` (kernels with an upstream metal reference), `ffai/` (model-specific), `convolution/` (consolidated 1D/2D/3D/depthwise/winograd + `steel_conv/`), `quant/` (the `codec` + `format` + `gguf` precision layer), plus `probe/` and `utils`. Every `#[kernel]`/`#[bench]`/`#[test_kernel]` lives here. |
| `metaltile-cli` | The `tile` binary: config, command dispatch, result rendering. Thin — it spawns the runner subprocess rather than doing GPU work itself. |

## From source to shader

```mermaid
flowchart LR
    K["#[kernel]<br/>fn mt_exp&lt;T&gt;(..)"] --> IR["IR<br/>(Op variants)"]
    V["#[kernel(variants(...))]"] -. "stamps N kernels" .-> IR
    IR --> Passes["codegen passes<br/>const-fold · vectorize · unroll · FMA · DCE · …"]
    Passes --> Backend{CodegenBackend<br/>Target}
    Backend -->|Metal default| MSL[".metal → kernels.metallib"]
    Backend -->|Cuda| CU[".cu (NVRTC/PTX)"]
    Backend -->|Hip| HIP[".hip (hipRTC)"]
    Backend -->|Spirv| SPV[".comp → SPIR-V"]
    B["#[bench] / #[test_kernel]"] -. "inventory::submit!" .-> Reg["harness registries<br/>all_kernels / all_benches / all_tests"]
```

`#[kernel]` lowers the DSL function to IR; `#[kernel(variants(...))]` stamps out
one specialised kernel per parameter tuple before lowering. The codegen passes
optimise the IR (backend-independent), then a `CodegenBackend` emits the target
source. **Metal is the default and the only fully-executing backend today**:
MSL emission produces a `.metal` source that `xcrun metal` compiles to a
`metallib`. `#[bench]` and `#[test_kernel]` are optional annotations on the same
function that register a **setup callback** (`BenchSetup` / `TestSetup`) into an
`inventory` registry the runner iterates.

## Command dispatch

Every subcommand is a struct implementing the `TileCommand` trait
(`cmd/mod.rs`); `main.rs` parses args, builds a `Harness` (the loaded
`TileConfig`), and dispatches:

```mermaid
flowchart TD
    main["main.rs<br/>parse args"] --> cfg["ConfigLoader<br/>defaults → tile.toml → TILE_* env → CLI args"]
    cfg --> harness["Harness (owns TileConfig)"]
    harness --> cmd{TileCommand}
    cmd --> bench["bench"]
    cmd --> test["test"]
    cmd --> build["build (+ --emit)"]
    cmd --> inspect["inspect"]
    cmd --> other["device · snap · diff · clean · config · update · init<br/>(pure CPU / IO — stay in-process)"]
    bench --> pr["ProjectRunner<br/>spawn __tile_runner"]
    test --> pr
    build --> pr
    inspect --> pr
```

`bench` / `test` / `build` / `inspect` route through **`ProjectRunner`**, which
spawns the runner subprocess. `device` (GPU query), `snap` (save baseline),
`diff` (compare baselines), `clean` (remove build artifacts), `config` (show
resolved config), `update` (self-update), and `init` (scaffold a new project)
are pure CPU / IO and run directly.

> **`emit` is a `build` flag, not a command.** `tile build --emit
> msl|metallib|swift|ir|all --out <dir>` writes per-kernel `.metal`, the
> compiled `kernels.metallib`, the `MetalTileKernels.swift` bindings, and/or the
> `manifest.json` IR descriptor. `build` / `inspect` always emit **MSL** (the
> `--backend` flag below applies only to `bench` / `test`).

## Subprocess execution

The `tile` CLI does **no GPU work itself**. `ProjectRunner` spawns
`__tile_runner` — a binary linked against the project's `metaltile-std`, so the
`#[kernel]`/`#[bench]`/`#[test_kernel]` `inventory` is populated inside the child
— and streams its stdout, parsing each line as a `ProtocolMessage`.

```mermaid
flowchart LR
    cli["tile CLI<br/>(thin protocol parser)"] -- "spawn + args" --> proc["__tile_runner<br/>RunnerHarness + GpuRunner + inventory"]
    proc -- "ProtocolMessage JSON lines (stdout)" --> cli
```

| Piece | Where | Purpose |
|---|---|---|
| `ProtocolMessage` (+ `runner_version`) | `metaltile-core::protocol` | Versioned JSON-line wire format (CLI ↔ runner). |
| `RunnerArgs` | `metaltile::runner::args` | Subprocess CLI arg parsing (`from_env_args`), incl. `--backend`. |
| `RunnerHarness` | `metaltile::runner::harness` | Orchestrates bench / test / build / inspect, emitting protocol messages. |
| `runner::backend` | `metaltile::runner::backend` | Routes `--backend cuda\|hip\|vulkan` through the matching feature-gated device. |
| `ProjectRunner` | `metaltile-cli::project_runner` | Spawns `__tile_runner`, streams + parses its stdout. |

## Multi-backend codegen

`backend.rs` defines `Target { Metal, Cuda, Hip, Spirv }` and the
`CodegenBackend` trait; `MslGenerator` is the `Metal` impl, with
`cuda::CudaGenerator` / `hip` / `spirv` alongside. A `TargetProfile` carries the
per-backend knobs the IR→source lowering needs — SIMD/warp **lane width** (32 on
Metal simdgroup / CUDA warp / RDNA wave32, 64 on CDNA wave64, variable on Vulkan
subgroup) and the **`MmaStrategy`** (Metal `simdgroup_matrix` 8×8, CUDA/CDNA/RDNA
tensor-core paths, Vulkan `VK_KHR_cooperative_matrix`).

- **Codegen** (emit) supports all four targets; `build`/`inspect` emit MSL.
- **Execution** is Metal by default. `tile bench|test --backend cuda|hip|vulkan`
  routes through the feature-gated `CudaDevice` / `HipDevice` / `VulkanDevice`
  in `metaltile-runtime`. These are validated by GPU-vs-GPU reference corpora
  (`tests/{cuda,hip}_kernel_corpus.rs`, `tests/vulkan_sdpa_multi.rs`) — the same
  `#[test_kernel]` inventory run on a non-Metal device and diffed against Metal.

See the backend specs ([`CUDA`](CUDA_BACKEND_SPEC.md) /
[`AMD`](AMD_BACKEND_SPEC.md) / [`VULKAN`](VULKAN_BACKEND_SPEC.md) /
[`ANE`](ANE_BACKEND_SPEC.md)) for each target's design and hazards.

## Bench runner

```mermaid
flowchart TD
    RH["RunnerHarness (in __tile_runner)"] --> loop["for each #[bench] × dtype (sequential)"]
    loop --> emit["backend emit + compile (PSO/module cache)"]
    emit --> timed["⏱ GpuRunner.bench(warmup, iters)<br/>→ BenchStats (min/mean µs)"]
    timed --> dm["metrics from stats — to_gflops · estimate_profile ·<br/>classify_bottleneck · device_specs::lookup (CPU-only, AFTER timing)"]
    dm --> res["ProfileInfo { GFLOP/s · %-peak · bottleneck }"]
    res --> proto["ProtocolMessage → CLI SuitePrinter<br/>(GB/s · GFLOP/s · %-peak · bottleneck)"]
```

The bench run loop is **sequential** — GPU dispatch + timing is serialized on
the device, so running benches concurrently would corrupt timings. (The CPU-only
work that *can* parallelize — `tile build` MSL emit, `tile test` oracles — uses
`rayon`; the bench run does not.)

**Timing isolation.** `GpuRunner.bench(…)` is the *only* timed region; it
returns the finalized `BenchStats`. Metric derivation — GFLOP/s
(`gpu::to_gflops`), the roofline / bottleneck verdict (`profile::estimate_profile`
+ `profile::classify_bottleneck`), and %-of-peak (`device_specs::lookup`) — runs
strictly afterward and *consumes* those stats; it never dispatches to the GPU. So
metric computation **cannot skew** the measured kernel performance.

## Test runner

`tile test` iterates the `#[test_kernel]` registry and dispatches each setup,
comparing GPU output against the test's CPU oracle within its tolerance. The CPU
oracle pass is `rayon`-parallel (order-preserving via `collect`); GPU dispatch of
the survivors is sequential. Under `--backend`, the same inventory runs on the
selected device and is compared against the Metal reference.

## Kernel profiling

The roofline / occupancy metrics shown under `tile bench -v` / `-vv`:

```mermaid
flowchart LR
    stats["BenchStats<br/>(measured µs)"] --> g["gpu::to_gflops"]
    ir["kernel IR"] --> prof["profile::estimate_profile<br/>+ classify_bottleneck<br/>(occupancy · registers · bottleneck)"]
    specs["device_specs::lookup<br/>(peak BW + FP32/FP16 TFLOPS)"] --> PI
    g --> PI
    prof --> PI
    PI["protocol::ProfileInfo<br/>latency µs · GFLOP/s · %-peak BW/compute · arith intensity · bottleneck"]
```

- `profile::estimate_profile` runs the pass pipeline + register/occupancy
  analysis on the IR (pure CPU).
- `device_specs::lookup` returns per-chip peak ceilings (bandwidth + FP32
  TFLOPS, with FP16 = 2× FP32 on the SIMD pipe, and the M5 Neural-Accelerator
  FP16 ceiling where applicable). Unknown devices return `None` → roofline
  columns blank, never an error.
- `protocol::ProfileInfo` is the serializable wire form (GFLOP/s, %-peak BW /
  compute, arithmetic intensity, bottleneck) the runner streams back to the CLI
  for rendering.

See [`BENCH_METRICS_SPEC.md`](BENCH_METRICS_SPEC.md) for the metric formulas and
device-spec sourcing.
