# MetalTile Toolchain Design

**Status:** Draft — refactor/bench-logic-3

---

## Problem with the current design

The old system compiled all bench logic — buffer allocation strategies, reference kernel names, dispatch shapes, correctness tolerances — directly into the `tile` CLI binary via `inventory::submit!`. This created three problems:

1. **Every kernel change required reinstalling the CLI.** The bench registration lived in `metaltile-std`, which `metaltile-cli` linked. `cargo install` was not optional.

2. **All policy was centralised.** `ClassKind`, `BenchDispatch`, `ShapeSpec`, and `run_spec` lived in toolchain crates. Kernel authors could not control how their kernel was benched — they filled in fields of a schema someone else defined.

3. **The CLI was a monolith.** Bench execution (GPU buffer allocation, timing loops, correctness checks, MLX comparison) was all in-process. Testing a new bench shape meant modifying `run_spec.rs`.

---

## Design Goals

| Goal | Description |
|---|---|
| **No reinstall** | `tile bench` / `tile test` run the user's project as a subprocess. Changing a kernel or its bench setup only requires recompiling the project, not the CLI. |
| **Kernel-local policy** | Every decision about how a kernel is benched or tested (buffer sizes, dtypes, tolerance, reference kernel) is authored next to the kernel, in the user's crate. |
| **Minimal toolchain surface** | The toolchain provides traits and a protocol. It does not define dispatch classes, buffer init strategies, or anything domain-specific. |
| **Foundry UX** | `cd my-kernels && tile bench` — the project directory is the unit of operation, like Cargo itself. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User project  (e.g. metaltile-std, or any external     │
│  crate with #[kernel] functions)                        │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │  kernel.rs   │  │  bench.rs    │  │  test.rs      │ │
│  │  #[kernel]   │  │  impl Bench  │  │  impl Test    │ │
│  │  fn mt_exp   │  │  for ExpBench│  │  for ExpTest  │ │
│  └──────────────┘  └──────────────┘  └───────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  src/bin/tile-runner.rs                          │  │
│  │  Thin binary: parses tile protocol commands,     │  │
│  │  iterates registered Bench/Test impls, streams   │  │
│  │  JSON results to stdout.                         │  │
│  └──────────────────────────────────────────────────┘  │
└──────────┬──────────────────────────────────────────────┘
           │ cargo run --bin tile-runner -- bench --filter exp
           │ (JSON lines on stdout)
           ▼
┌─────────────────────────────┐
│  tile CLI  (metaltile-cli)  │
│                             │
│  Detects tile.toml,         │
│  spawns subprocess,         │
│  streams + renders output.  │
│                             │
│  No GPU code, no kernel     │
│  knowledge, no bench logic. │
└─────────────────────────────┘
```

The CLI is a **rendering and orchestration** layer only. It knows nothing about kernel shapes, buffer allocation, or Metal.

---

## Project manifest — `tile.toml`

Every project that uses `tile` has a `tile.toml` at the workspace root:

```toml
[project]
name = "metaltile-std"

[runner]
# The binary in this workspace that implements the tile runner protocol.
bin = "tile-runner"
# Optional: extra cargo args forwarded when spawning the runner.
cargo_args = ["--release"]

[bench]
warmup_iters = 5
bench_iters  = 20

[test]
# Tolerance applied globally unless overridden per-kernel.
default_tol = 1e-4
```

---

## The `#[kernel]` macro

`#[kernel]` does exactly one thing: **convert the DSL function body into MetalTile IR** and register a `KernelEntry` in the inventory so `tile build` / `tile inspect` can find it.

```rust
#[kernel]
pub fn mt_exp<T>(input: Tensor<T>, out: Tensor<T>) {
    let idx = program_id::<0>();
    store(out[idx], exp(load(input[idx])));
}
```

That is all. No bench args, no dispatch class, no tolerance.

The macro generates:
- `mod mt_exp { pub fn kernel_ir_for(dt: DType) -> Kernel { … } }`
- A `KernelEntry` submitted to `metaltile_core::inventory` for `tile build`/`inspect`

---

## Bench registration — `KernelBench` trait

Bench logic lives in the user's crate, next to the kernel, as an implementation of the `KernelBench` trait provided by `metaltile`:

```rust
// metaltile-core (or metaltile facade)

pub trait KernelBench: Send + Sync {
    /// Short dotted identifier: "op/subop" or just "kernel_name".
    fn name(&self) -> &str;

    /// Dtypes to exercise. Kernel author decides — no hardcoded list.
    fn dtypes(&self) -> &[DType];

    /// Build an input/output buffer set for one (dtype, iteration).
    /// Returns a `BenchSetup` that the runner dispatches directly.
    fn setup(&self, dt: DType) -> BenchSetup;

    /// Optional: name of the Metal reference kernel to compare against.
    /// The reference is compiled from a `.metal` file the author provides.
    fn metal_reference(&self) -> Option<MetalRef> { None }

    /// Total bytes moved in one dispatch (used for GB/s reporting).
    /// Called after `setup` so the impl can consult the allocated sizes.
    fn bytes_moved(&self, setup: &BenchSetup) -> u64;
}
```

The runner iterates `inventory::iter::<KernelBenchEntry>`, calls `setup`, dispatches, times, and streams results.

### `BenchSetup`

Everything the runner needs to dispatch one kernel invocation:

```rust
pub struct BenchSetup {
    /// The compiled IR for this kernel at this dtype.
    pub kernel: Kernel,
    /// Named GPU buffers (inputs + outputs), in parameter order.
    pub buffers: Vec<BenchBuffer>,
    /// Constexpr values keyed by name.
    pub constexprs: Vec<(String, ConstValue)>,
    /// Grid dimensions [x, y, z].
    pub grid: [u32; 3],
    /// Threads-per-threadgroup [x, y, z].
    pub tpg: [u32; 3],
}

pub struct BenchBuffer {
    pub name: String,
    pub data: Vec<u8>,        // host-side init; runner uploads to GPU
    pub dtype: DType,
    pub is_output: bool,
}
```

No `ClassKind`. No `ShapeSpec`. No hardcoded rows/columns. The author fills in exactly what the kernel needs.

### Registration

```rust
// In the user crate, near the kernel:

struct ExpBench;

impl KernelBench for ExpBench {
    fn name(&self) -> &str { "unary/exp" }
    fn dtypes(&self) -> &[DType] { &[DType::F32, DType::F16, DType::BF16] }

    fn setup(&self, dt: DType) -> BenchSetup {
        const N: usize = 64 * 1024 * 1024;
        BenchSetup {
            kernel: mt_exp::kernel_ir_for(dt),
            buffers: vec![
                BenchBuffer::random("input",  N, dt),
                BenchBuffer::zeros("out",     N, dt).output(),
            ],
            constexprs: vec![("n".into(), ConstValue::U32(N as u32))],
            grid: [N.div_ceil(256) as u32, 1, 1],
            tpg:  [256, 1, 1],
        }
    }

    fn bytes_moved(&self, s: &BenchSetup) -> u64 {
        2 * s.buffers[0].data.len() as u64  // read + write
    }
}

// One macro call registers the impl:
register_bench!(ExpBench);
```

---

## Test registration — `KernelTest` trait

Same pattern, for CPU-oracle correctness checks:

```rust
pub trait KernelTest: Send + Sync {
    fn name(&self) -> &str;
    fn dtypes(&self) -> &[DType];

    /// Build inputs + expected outputs for one dtype.
    fn setup(&self, dt: DType) -> TestSetup;

    /// Element-wise tolerance (absolute).
    fn tolerance(&self, dt: DType) -> f64 { 1e-4 }
}

pub struct TestSetup {
    pub kernel: Kernel,
    pub inputs: Vec<TestBuffer>,
    /// Expected output values (computed by the CPU oracle).
    pub expected: Vec<TestBuffer>,
    pub constexprs: Vec<(String, ConstValue)>,
    pub grid: [u32; 3],
    pub tpg: [u32; 3],
}
```

The runner dispatches the kernel, reads back outputs, and diffs against `expected` within tolerance.

---

## Metal reference comparison

When `metal_reference()` returns `Some(MetalRef { .. })`, the runner:

1. Compiles the reference `.metal` file via `xcrun metal`
2. Allocates the same buffers
3. Dispatches the reference kernel with the same inputs
4. Compares GB/s (MT vs ref) and correctness

```rust
pub struct MetalRef {
    /// Path to the `.metal` source file, relative to the project root.
    pub metal_file: &'static str,
    /// Kernel function name inside the metal file.
    pub function: &'static str,
    /// Constexprs to pass to the reference (may differ from MT spelling).
    pub constexprs: Vec<(String, ConstValue)>,
}
```

---

## Runner protocol (JSON Lines)

The `tile-runner` binary writes newline-delimited JSON to stdout. The CLI reads this stream and renders it. This is the only contract between them.

```jsonc
// Announce the run
{"type":"start","runner_version":"0.1","total_benches":42}

// Per-bench result
{
  "type": "bench",
  "name": "unary/exp",
  "dtype": "f16",
  "mt_gbps": 1234.5,
  "ref_gbps": 1189.2,    // null if no metal_reference
  "mt_pct": 103.8,       // null if no ref
  "correct": true,
  "min_us": 12.3,
  "mean_us": 12.8
}

// Per-test result
{"type":"test","name":"unary/exp","dtype":"f16","passed":true,"max_err":3.2e-5}

// Non-fatal error
{"type":"error","name":"unary/exp","dtype":"f16","message":"buffer size mismatch"}

// Final summary
{"type":"done","bench_passed":41,"bench_failed":1,"test_passed":30,"test_failed":0}
```

The protocol is versioned. The CLI negotiates with the runner via the `runner_version` field and gracefully degrades for older runners.

---

## The `tile-runner` binary

A runner binary is a thin adapter the user's project provides:

```rust
// src/bin/tile-runner.rs  (user-authored, boilerplate)

fn main() {
    metaltile::runner::run(metaltile::runner::Args::from_env());
}
```

`metaltile::runner::run` iterates the `inventory`, handles `--filter`, `--bench`, `--test` sub-commands, and streams JSON. The user writes zero protocol code — they only implement `KernelBench` / `KernelTest`.

---

## CLI commands

### `tile bench`

```
tile bench [-f <filter>] [-v] [-o results.json]
```

1. Find `tile.toml` walking up from CWD.
2. Spawn `cargo run --bin <runner.bin> [runner.cargo_args] -- bench [--filter …]`.
3. Stream JSON lines → render live table.
4. Optionally write `results.json`.

### `tile test`

```
tile test [-f <filter>] [-v]
```

Same as bench but invokes `-- test`.

### `tile build`

```
tile build [-f <filter>] [--dtypes f32,f16,bf16] [--emit msl,metallib] [-o <dir>]
```

Invokes the runner with `-- build`. The runner iterates `KernelEntry` inventory, generates MSL via `metaltile-codegen`, optionally compiles a metallib, and streams artifacts over the protocol.

### `tile inspect`

```
tile inspect [<kernel>] [--ir] [--pass <name>] [--dtype f32]
```

Invokes `-- inspect`. Same kernel discovery path.

---

## What the toolchain owns vs the kernel author

| Concern | Toolchain (`metaltile`) | Kernel author |
|---|---|---|
| DSL → IR compilation | ✅ `#[kernel]` macro | |
| MSL codegen | ✅ `metaltile-codegen` | |
| GPU dispatch & timing | ✅ `runner::run` | |
| JSON protocol | ✅ defined in `metaltile` | |
| Buffer allocation | | ✅ `BenchSetup::buffers` |
| Dtypes to run | | ✅ `KernelBench::dtypes` |
| Dispatch shape (grid/tpg) | | ✅ `BenchSetup::grid/tpg` |
| Reference kernel | | ✅ `KernelBench::metal_reference` |
| Tolerance | | ✅ `KernelTest::tolerance` |
| CPU oracle | | ✅ `TestSetup::expected` |
| Bench iterations | ✅ `tile.toml [bench]` | override per-bench if needed |

---

## File layout in a kernel project

```
my-kernels/
├── tile.toml
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── bin/
    │   └── tile-runner.rs     # 2-line boilerplate
    └── ops/
        ├── unary.rs           # #[kernel] fn mt_exp ...
        ├── unary_bench.rs     # impl KernelBench for ExpBench
        └── unary_test.rs      # impl KernelTest for ExpTest
```

Bench and test code can live in the same file as the kernel, or in neighbouring files — the author decides.

---

## Implementation sequence

1. **`metaltile-core`**: add `KernelBench`, `KernelTest`, `BenchSetup`, `TestSetup`, `BenchBuffer`, `TestBuffer`, `MetalRef`, `ConstValue` types and `KernelBenchEntry` / `KernelTestEntry` inventory wrappers.
2. **`metaltile`**: re-export the new traits; add `register_bench!` / `register_test!` macros.
3. **`metaltile`**: implement `runner::run` — the protocol loop.
4. **`metaltile-cli`**: implement `tile bench`, `tile test` as subprocess launchers.
5. **`metaltile-std`**: add `src/bin/tile-runner.rs`; port existing bench specs to `impl KernelBench`.
6. **`tile.toml`**: add to `metaltile-std` root.
