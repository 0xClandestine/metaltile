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
| **Idiomatic Rust** | All toolchain and kernel-author code follows Rust best practices: builder pattern, opaque types, trait-based polymorphism, `Result`-propagating errors. |

---

## Coding Standards

These conventions apply to both the toolchain crates and kernel-author code.

### Encapsulation — no public fields on configuration types

Configuration structs (`BenchSetup`, `BenchBuffer`, `TestSetup`) have **private fields**. Construction goes through named constructors and builder methods only. This keeps the API stable across breaking internal changes.

```rust
// ✅ correct
BenchSetup::new(kernel)
    .buffer(BenchBuffer::random("input", N, dt))
    .grid_1d(N, 256)

// ❌ avoid — struct literal breaks on any field addition
BenchSetup { kernel, buffers: vec![…], grid: [N/256, 1, 1], tpg: [256,1,1] }
```

### Builder pattern for multi-field configuration

Every type that requires more than two fields to construct exposes a builder. Builders own the object and take `self` (not `&mut self`) so chains compose naturally and the intermediate object is never observable in a partial state.

```rust
impl BenchSetup {
    pub fn new(kernel: Kernel) -> Self { … }        // named constructor
    pub fn buffer(self, b: BenchBuffer) -> Self { … } // consuming chain
    pub fn grid_1d(self, n: usize, tpg: u32) -> Self { … }
}
```

### Traits for polymorphism, not inheritance

Shared behaviour is expressed as traits (`KernelBench`, `KernelTest`), not through struct hierarchies or `dyn Any` downcasts. The `#[bench]` / `#[test_kernel]` macros generate trait impls — the runner only knows the trait. All three macros (`#[kernel]`, `#[bench]`, `#[test_kernel]`) live in the same source file as the kernel they describe.

### Newtype wrappers for domain clarity

Prefer newtypes over raw primitives when the type carries domain meaning:

```rust
pub struct Gbps(pub f64);   // not f64
pub struct Microseconds(pub f64);
```

This prevents accidentally swapping throughput and latency values at call sites.

### Error handling — `Result`, never `panic` in library code

Toolchain library code (`metaltile`, `metaltile-core`) returns `Result<_, E>` and propagates errors with `?`. `unwrap()` and `expect()` are reserved for cases that are genuinely unreachable, with a comment explaining why. Proc-macro code returns `syn::Error` / `compile_error!` on bad input rather than panicking.

### `From` / `Into` for conversions

Implement `From<T> for U` rather than `.to_foo()` conversion methods wherever the target type is clearly the canonical form. This lets callers use `.into()` and enables `?`-based error coercions.

### Visibility — minimal surface

Use `pub(crate)` for cross-module implementation details. Only the stable author-facing API should be `pub`. Internal helpers are `fn` (private by default).

### Documentation — every public item has a doc comment

All `pub` items carry a `///` doc comment. Module files open with a `//!` crate/module-level doc. Doc comments follow this structure:

```rust
//! Module-level summary — what this module is for and what it contains.

/// One-line summary of what this does.
///
/// Longer explanation if the behaviour isn't obvious from the name. Describe
/// the contract, not the implementation. Note any preconditions, panics, or
/// edge cases the caller must know about.
///
/// # Examples
///
/// ```rust
/// let setup = BenchSetup::new(kernel)
///     .buffer(BenchBuffer::random("input", N, dt))
///     .grid_1d(N, 256);
/// ```
pub fn something() { … }
```

Rules:
- **One-line summary** on the first `///` line. No trailing period. Start with a verb: *Build*, *Return*, *Register*, *Dispatch*.
- **Blank `///` line** between the summary and any further paragraphs.
- **`# Examples`** on any public constructor or non-trivial method. The example must compile (`cargo test --doc`).
- **`# Panics`** section if the function can panic under reachable conditions.
- **`# Errors`** section if the function returns `Result`.
- **No implementation details** — describe what, not how.
- Private helpers use `//` line comments only when the logic isn't self-evident. Do not add `///` to private items.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User project  (e.g. metaltile-std, or any external     │
│  crate with #[kernel] functions)                        │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  unary.rs                                         │  │
│  │  #[kernel]      fn mt_exp<T>(…) { … }             │  │
│  │  #[bench(…)]    fn exp_bench(dt) -> BenchSetup    │  │
│  │  #[test_kernel] fn exp_test(dt)  -> TestSetup     │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  [auto-generated by tile — never authored]       │  │
│  │  Thin harness: parses tile protocol commands,    │  │
│  │  iterates registered Bench/Test impls, streams   │  │
│  │  JSON results to stdout.                         │  │
│  └──────────────────────────────────────────────────┘  │
└──────────┬──────────────────────────────────────────────┘
           │ cargo run --bin __tile_runner -- bench --filter exp
           │ (JSON lines on stdout; harness generated in $CARGO_TARGET_DIR)
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
# Optional: extra cargo args forwarded when spawning the auto-generated runner.
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

## Bench registration — `#[bench]`

Bench logic lives in the user's crate, next to the kernel, as an ordinary function annotated with `#[bench]`. No struct, no `impl`, no registration call — the macro generates all of that.

```rust
#[bench(name = "unary/exp", dtypes = [f32, f16, bf16])]
fn exp_bench(dt: DType) -> BenchSetup {
    const N: usize = 64 << 20;
    BenchSetup::new(mt_exp::kernel_ir_for(dt))
        .buffer(BenchBuffer::random("input", N, dt))
        .buffer(BenchBuffer::zeros("out",    N, dt).output())
        .constexpr("n", N as u32)
        .grid_1d(N, 256)
}
```

`#[bench]` expands to an anonymous struct that implements `KernelBench` and submits itself to the inventory. Authors never see or write the trait.

`bytes_moved` defaults to the sum of all buffer sizes. Override with the `bytes` key when the kernel's bandwidth figure differs (e.g. read-only inputs counted once):

```rust
#[bench(name = "unary/exp", dtypes = [f32, f16, bf16], bytes = |s| 2 * s.buffer_bytes("input"))]
fn exp_bench(dt: DType) -> BenchSetup { … }
```

### `BenchSetup` and `BenchBuffer`

Both types are **opaque** — fields are private, construction goes through named constructors only. This is intentional: call-site code is insulated from internal layout changes.

```rust
// BenchSetup — consuming builder
pub struct BenchSetup { /* private */ }

impl BenchSetup {
    pub fn new(kernel: Kernel) -> Self;
    pub fn buffer(self, b: BenchBuffer) -> Self;
    pub fn constexpr(self, name: &str, v: impl Into<ConstValue>) -> Self;
    pub fn grid_1d(self, n: usize, tpg: u32) -> Self; // [ceil(n/tpg),1,1] / [tpg,1,1]
    pub fn grid_2d(self, x: u32, y: u32, tpg: [u32; 2]) -> Self;
    pub fn grid_3d(self, x: u32, y: u32, z: u32, tpg: [u32; 3]) -> Self;
    pub fn buffer_bytes(&self, name: &str) -> u64;
}

// BenchBuffer — named constructors, no public fields
pub struct BenchBuffer { /* private */ }

impl BenchBuffer {
    pub fn random(name: &str, len: usize, dt: DType) -> Self;
    pub fn zeros(name: &str, len: usize, dt: DType) -> Self;
    pub fn from_vec(name: &str, data: Vec<u8>, dt: DType) -> Self;
    pub fn output(self) -> Self; // marks buffer as an output slot
}
```

No `ClassKind`. No `ShapeSpec`. No hardcoded rows/columns. The author fills in exactly what the kernel needs.

### Escape hatch — `KernelBench` trait

For kernels that need dynamic dispatch, shared setup logic across many dtypes, or other complexity the macro form can't express, implement the trait directly:

```rust
pub trait KernelBench: Send + Sync {
    fn name(&self) -> &str;
    fn dtypes(&self) -> &[DType];
    fn setup(&self, dt: DType) -> BenchSetup;
    fn metal_reference(&self) -> Option<MetalRef> { None }
    fn bytes_moved(&self, setup: &BenchSetup) -> u64;
}

// register_bench!(MyComplexBench) submits the impl to the inventory.
register_bench!(MyComplexBench);
```

The `#[bench]` macro is syntactic sugar over this trait — the runner only knows the trait.

---

## Test registration — `#[test_kernel]`

Same pattern as `#[bench]`, for CPU-oracle correctness checks:

```rust
#[test_kernel(name = "unary/exp", dtypes = [f32, f16, bf16])]
fn exp_test(dt: DType) -> TestSetup {
    const N: usize = 1024;
    // CPU oracle: generate input, compute expected output in f32, let the
    // runner handle dtype casting and element-wise comparison.
    let input = TestBuffer::random("input", N, dt);
    let expected = input.map_f32(f32::exp).rename("out");
    TestSetup::new(mt_exp::kernel_ir_for(dt))
        .input(input)
        .expected(expected)
        .grid_1d(N, 256)
}
```

`tolerance` defaults to `1e-4`. Override per-function:

```rust
#[test_kernel(name = "unary/exp", dtypes = [f32, f16, bf16], tol = 1e-5)]
fn exp_test(dt: DType) -> TestSetup { … }
```

The runner dispatches the kernel, reads back outputs, and diffs against `expected` within tolerance.

### Escape hatch — `KernelTest` trait

```rust
pub trait KernelTest: Send + Sync {
    fn name(&self) -> &str;
    fn dtypes(&self) -> &[DType];
    fn setup(&self, dt: DType) -> TestSetup;
    fn tolerance(&self, dt: DType) -> f64 { 1e-4 }
}
```

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

Pass one via the `ref` key in `#[bench]`:

```rust
#[bench(
    name   = "unary/exp",
    dtypes = [f32, f16, bf16],
    ref    = MetalRef { metal_file: "metal/exp.metal", function: "mt_exp_ref", constexprs: vec![] },
)]
fn exp_bench(dt: DType) -> BenchSetup { … }
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

## How the CLI discovers your kernels

Kernel authors write zero runner code. The subprocess wiring is entirely owned by the toolchain.

When `tile bench` is invoked, it:

1. Finds `tile.toml` walking up from CWD.
2. Generates a harness entry-point on the fly (in `$CARGO_TARGET_DIR/tile/`) — exactly like how `cargo test` generates a test harness without you writing a `fn main`.
3. Compiles it with `cargo build --bin __tile_runner` (the generated bin is invisible to the author).
4. Spawns the compiled binary and streams JSON.

The harness source is a single generated file:

```rust
// auto-generated by tile — do not edit, do not check in
fn main() {
    metaltile::runner::run(metaltile::runner::Args::from_env());
}
```

`metaltile::runner::run` iterates the `inventory`, handles `--filter`, `--bench`, `--test` sub-commands, and streams JSON. **Authors never see, write, or think about this file.**

---

## CLI commands

### `tile bench`

```
tile bench [-f <filter>] [-v] [-o results.json]
```

1. Find `tile.toml` walking up from CWD.
2. Generate runner harness source into `$CARGO_TARGET_DIR/tile/__runner.rs` if absent or stale.
3. Spawn `cargo run --bin __tile_runner [runner.cargo_args] -- bench [--filter …]`.
4. Stream JSON lines → render live table.
5. Optionally write `results.json`.

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
| Runner harness / subprocess wiring | ✅ auto-generated by `tile` | |
| Bench iterations | ✅ `tile.toml [bench]` | override per-bench if needed |

---

## File layout in a kernel project

```
my-kernels/
├── tile.toml
├── Cargo.toml
└── src/
    ├── lib.rs
    └── ops/
        └── unary.rs           # #[kernel], #[bench], #[test_kernel] all in one file
```

The kernel, its bench setup, and its correctness test live in the same file. There is no reason to split them — they share the same constants, the same buffer layout, and the same understanding of what the kernel does. Keeping them together makes that knowledge visible in one place.

No runner binary. No `src/bin/`. No protocol code. The harness is generated by `tile` at build time and lives entirely in `$CARGO_TARGET_DIR`.

---

## Implementation sequence

1. **`metaltile-core`**: add `KernelBench`, `KernelTest`, `BenchSetup`, `TestSetup`, `BenchBuffer`, `TestBuffer`, `MetalRef`, `ConstValue` types and `KernelBenchEntry` / `KernelTestEntry` inventory wrappers.
2. **`metaltile`**: re-export the new traits; add `register_bench!` / `register_test!` macros.
3. **`metaltile`**: implement `runner::run` — the protocol loop.
4. **`metaltile-cli`**: implement `tile bench`, `tile test` — harness generation + subprocess launch + JSON rendering.
5. **`metaltile-std`**: port existing bench specs to `impl KernelBench`; add `tile.toml` at the workspace root.

No step requires kernel authors to create a runner binary. Step 4 owns that entirely.
