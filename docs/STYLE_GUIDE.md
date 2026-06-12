# MetalTile Kernel Style Guide

The authority on **how to write one kernel** in `metaltile-std` — file shape,
naming, the `#[kernel(variants(...))]` axis, shared primitives, the CPU oracle,
and the bench. Follow it to produce a kernel that is consistent with the rest of
the library: correct, testable, benchmarkable, and maintainable. This is the
target style the whole library is migrating toward.

> **Companion:** [`KERNEL_CONSOLIDATION_PLAN.md`](specs/KERNEL_CONSOLIDATION_PLAN.md)
> owns *where files go* (the `kernels/<family>/` layout) and the family-by-family
> migration roadmap. This guide owns *how a kernel is written*. Where they
> overlap (naming, the per-file shape), this guide is authoritative.

---

## Table of Contents

1. [Where does my kernel live?](#1-where-does-my-kernel-live)
2. [File skeleton](#2-file-skeleton)
3. [The kernel function](#3-the-kernel-function)
4. [Dispatch modes and grid geometry](#4-dispatch-modes-and-grid-geometry)
5. [Compile-time variants](#5-compile-time-variants)
6. [Shared primitives (cross-kernel calling)](#6-shared-primitives-cross-kernel-calling)
7. [Writing the CPU oracle and tests](#7-writing-the-cpu-oracle-and-tests)
8. [Writing the bench](#8-writing-the-bench)
9. [Registering the kernel](#9-registering-the-kernel)
10. [DSL reference and known limitations](#10-dsl-reference-and-known-limitations)
11. [Worked example: elementwise scale](#11-worked-example-elementwise-scale)

---

## 1. Where does my kernel live?

Kernels are grouped by **operation family** under `kernels/`, not by whether an
upstream metal reference exists (that's a property of one optional bench, not of
the kernel):

```
crates/metaltile-std/src/kernels/
  ops/   core/gather/scatter/reduce/elementwise primitives
  gemm/ sdpa/ moe/ norm/ rope/ conv/ ssm/ quant/ audio/ vision/ sampling/ kv_cache/
  primitives.rs   cross-family decode/reduce ops inlined at codegen
```

Put the kernel in the folder for its **operation** (`rope_yarn` → `kernels/rope/`,
`gemv` → `kernels/gemm/`). The quantized form of an op is *not* a separate
family — it folds into the op's file as a format axis (see §5). See
[`KERNEL_CONSOLIDATION_PLAN.md`](specs/KERNEL_CONSOLIDATION_PLAN.md) for the full
family list and the migration state (the crate is mid-migration from the legacy
`mlx/` + `ffai/` split).

---

## 2. File skeleton

Every kernel file has the same four-section shape, in this order:

```rust
//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! One-line description — what this kernel computes and which upstream it ports.
//!
//! Longer description of the algorithm, layout, and any non-obvious choices.
//!
//! ## Layout            ← required for kernels with multiple buffers
//!
//! - `input  [rows, n]`   T   — description
//! - `output [rows, n]`   T
//!
//! ## DISPATCH INVARIANTS   ← required for Reduction / threadgroup kernels
//!
//! - **TPG = N / 4.** Each thread owns exactly 4 elements.
//! - **TPG must be a multiple of 32.**
//! - **Grid: 1 threadgroup per row.**

use metaltile::kernel;

// ── 1. Kernel function(s) ────────────────────────────────────────────────────

#[kernel]
pub fn mt_my_kernel<T>(...) { ... }

// ── 2. Correctness tests ─────────────────────────────────────────────────────

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};
    use super::mt_my_kernel;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(...) -> TestSetup { ... }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_mt_my_kernel(dt: DType) -> TestSetup { setup(..., dt) }
}

// ── 3. Benchmarks ────────────────────────────────────────────────────────────

pub mod kernel_benches {
    use metaltile::{bench, test::*};
    use super::mt_my_kernel;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mt_my_kernel(dt: DType) -> BenchSetup { ... }
}
```

### Module-level doc comment

The `//!` block at the top is the public documentation for the kernel. It must include:

- **What** the kernel computes (one line).
- **Why** key algorithmic choices were made (e.g. f32 accumulation for bf16 inputs).
- **Layout** — the shape and dtype of every buffer, if non-trivial.
- **DISPATCH INVARIANTS** — for Reduction and threadgroup kernels, the exact TPG / grid constraints, stated as hard requirements. Violations in Reduction kernels silently miscompute (the GPU won't error); this section is the contract the caller must uphold.

---

## 3. The kernel function

### Naming

| Pattern | Used for |
|---|---|
| `mt_<op>` | The operation (`mt_softmax`, `mt_copy`, `mt_rope_llama`) |
| `mt_<op>_<variant>` | A named variant (`mt_rms_norm_small`) |
| `mt_<family>_<variant>` | Variant families from `variants(...)` (`mt_hadamard_n64`, `mt_int4_conv2d`) |

**Name the operation, not the model or the source.** No model names
(`kokoro` → `adain1d`/`lstm`), no `mlx`/`ffai` coupling in the name. `ffai_<op>`
is a legacy prefix being phased toward `mt_<op>`; don't introduce new ones.

### Generic dtype parameter

Every kernel that operates on floating-point data is generic over `T`:

```rust
#[kernel]
pub fn mt_my_kernel<T>(inp: Tensor<T>, out: Tensor<T>, ...) { ... }
```

The convention for multi-type kernels (e.g. float input, u32 index output) is a single `T` for the primary floating-point type; integer buffers are typed concretely:

```rust
pub fn mt_dequant<T>(packed: Tensor<u32>, scales: Tensor<T>, out: Tensor<T>, ...)
```

### Output buffer detection

A `Tensor` parameter is treated as an output if:
- It is declared `mut`: `mut out: Tensor<T>` — **preferred**, explicit.
- Its name matches the legacy heuristic set (`out`, `output`, `result`, `y`).

Prefer `mut` for new kernels; it is unambiguous and survives refactors.

### `#[constexpr]` parameters

Scalar values baked in at dispatch time (shapes, bit-widths, thresholds):

```rust
pub fn mt_my_kernel<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n: u32,
    #[constexpr] eps: f32,
)
```

`#[constexpr]` parameters are declared in `kernel.constexprs` and passed as Metal `constant T&` buffers. They are available as plain variable names inside the kernel body (`n`, `eps`).

Use `u32` for counts and indices; `f32` for floating-point thresholds; `bool` (rendered as `u32`) for flags.

### Accumulation precision

Always accumulate in `f32`, regardless of the input dtype:

```rust
// Good
let v = load(inp[idx]).cast::<f32>();
let acc = acc + v * v;
store(out[idx], acc.cast::<T>());

// Bad — bf16 accumulation loses precision across a long reduction
let v = load(inp[idx]);
let acc = acc + v * v;
```

Load inputs with `.cast::<f32>()` at the load site; cast outputs back to `T` at the store site.

---

## 4. Dispatch modes and grid geometry

The kernel's dispatch mode determines how `program_id` maps to the grid.

| Mode | `KernelMode` | Grid | Use for |
|---|---|---|---|
| Elementwise | `Grid3D` (default) | `[ceil(N/TPG), 1, 1]` × `[TPG, 1, 1]` | One thread per output element |
| Reduction | `Reduction` | `[rows, 1, 1]` × `[TPG, 1, 1]` | One threadgroup per row; threads reduce |
| Tiled 2D | `Grid3D` | `[N/BN, M/BM, 1]` × `[TPG, 1, 1]` | Matmul tile geometry |

**Elementwise kernels** use `program_id::<0>()` (or `program_id(0)`) as the flat element index:

```rust
let idx = program_id::<0>();
store(out[idx], load(a[idx]) + load(b[idx]));
```

**Reduction kernels** use `tgid_x` / `tgid_y` for the row, and `tid` for the lane within the threadgroup:

```rust
let row = tgid_x;
let lane = tid;
```

**Grid3D kernels** use `program_id::<0>()` through `program_id::<2>()` for each axis, or the `tgid_x` / `tgid_y` / `tgid_z` aliases.

### Documenting the grid

Every kernel doc comment that uses a non-trivial grid shape must state:

```
/// Grid: Reduction, `[rows, 1, 1]` × `[N, 1, 1]` (one thread per element).
```

Tests and benches must wire the same geometry. Mismatched geometry is the most common source of silent GPU miscomputes.

---

## 5. Compile-time variants

Use `#[kernel(variants(...))]` when the same algorithm is needed at multiple values of a compile-time integer (bit-widths, transform sizes, tile shapes). This eliminates `macro_rules!` dispatch and lets the compiler constant-fold the variant-specific values.

```rust
/// Walsh–Hadamard transform — produces `mt_hadamard_n64`, `_n128`, … `_n1024`.
///
/// Produces kernels: `mt_hadamard_n64`, `mt_hadamard_n128`, `mt_hadamard_n256`,
/// `mt_hadamard_n512`, `mt_hadamard_n1024`.
#[kernel(variants(N = [64, 128, 256, 512, 1024], LOG_N = [6, 7, 8, 9, 10], suffix = "n{N}"))]
pub fn mt_hadamard<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] scale: f32) {
    threadgroup_alloc("buf", N, "f32");  // N is a literal at compile time
    for s in range(0u32, LOG_N, 1u32) { // LOG_N is a literal at compile time
        ...
    }
}
```

Rules:
- The `suffix` template uses `{PARAM}` interpolation: `"int{BITS}"` → `"int4"`.
- Multiple correlated params (like `N` and `LOG_N`) are listed as parallel arrays of the same length.
- Generated module names: `{fn_name}_{suffix}` — e.g. `mt_hadamard_n64`.
- **Use ALL_CAPS, multi-character names** for variant params (`BITS`, `LOG_N`, not `b`, `n`) to avoid substring collisions in ident-embedding.

### Variant tests and benches

`#[test_kernel]` and `#[bench]` accept the same `variants(...)` syntax. Use ident-embedding (`intBITS` → `int4`) to reference the generated module:

```rust
// Collapses 5 test functions into 1
#[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 1e-1],
              variants(BITS = [2, 4, 8], suffix = "int{BITS}"))]
fn test_dequant_gather(dt: DType) -> TestSetup {
    setup(dequant_gather_intBITS::kernel_ir_for(dt), BITS, dt)
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ident-embedding: BITS → 2/4/8
}

// Collapses 5 bench functions into 1; name is also a template
#[bench(name = "ffai/dequant_gather/int{BITS}", dtypes = [f32, f16, bf16],
        variants(BITS = [2, 4, 8], suffix = "int{BITS}"))]
fn bench_dequant_gather(dt: DType) -> BenchSetup {
    gb(dequant_gather_intBITS::kernel_ir_for(dt), BITS, 4096, 64, dt)
}
```

If different variants need different test parameters (e.g. different `hidden` sizes for pow-2 vs odd bit-widths), use two separate `#[test_kernel]` functions with different variant lists rather than trying to unify them.

---

## 6. Shared primitives (cross-kernel calling)

The single biggest LOC-reduction tool. A sub-expression that recurs across many kernels — a weight decode (E2M1 nibble, E8M0 scale, a straddle-aware N-bit unpack), a simdgroup reduction — is factored into its own `#[kernel]` and **called** from every kernel that needs it. `KernelInlinePass` inlines the call at codegen, so there is zero runtime overhead: it is exactly as fast as pasting the body inline, but written once.

```rust
// kernels/primitives.rs  (or quant/codec for the decode family)
#[kernel]
pub fn mt_decode_e2m1(nib: u32) -> f32 { /* 4-bit E2M1 codebook */ }

#[kernel]
pub fn mt_decode_e8m0(byte: u32) -> f32 { exp2(byte.cast::<f32>() - 127.0f32) }
```

```rust
// In a quantized matmul/conv body — call instead of re-deriving:
let nib   = (load(weight[w_pack + col / 8u32]) >> ((col % 8u32) * 4u32)) & 0xFu32;
let scale = mt_decode_e8m0(load(scales[w_blk + col / block_size]).cast::<u32>());
acc = acc + pix * (mt_decode_e2m1(nib) * scale);
```

Extract a primitive once it appears in **two or more** kernels. Keep primitives small and decode/reduce-focused. The host-side CPU oracle should call the *same* math (e.g. `quant::codec`) so the kernel and oracle can't drift apart.

---

## 7. Writing the CPU oracle and tests

### The oracle pattern

The CPU oracle is a plain Rust function that reimplements the kernel's math in `f32`. It must:

1. **Round all inputs through the target dtype** (`pack_f32` → `unpack_f32`) before computing, so the oracle sees the same precision losses as the GPU.
2. **Match the kernel's accumulation order** where precision matters. For reductions, left-fold in `f32`.
3. Be **algorithm-independent** — don't mirror the kernel's implementation; express the mathematical definition directly. This catches bugs in both.

```rust
// Good oracle: mathematical definition, independent of the kernel's butterfly order
for i in 0..n {
    let acc: f32 = (0..n)
        .map(|j| {
            let sign = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            sign * xd[r * n + j]
        })
        .sum();
    expected[r * n + i] = acc * scale;
}
```

### TestSetup builder

```rust
fn setup(n: usize, dt: DType) -> TestSetup {
    // 1. Build inputs in f32
    let x: Vec<f32> = (0..n).map(|i| ...).collect();

    // 2. Round through dtype — oracle sees what GPU loads
    let xd = unpack_f32(&pack_f32(&x, dt), dt);

    // 3. Compute expected in f32
    let expected: Vec<f32> = xd.iter().map(|&v| ...).collect();

    // 4. Assemble TestSetup
    TestSetup::new(mt_my_kernel::kernel_ir_for(dt))
        .mode(KernelMode::Reduction)          // omit for elementwise (default)
        .input(TestBuffer::from_vec("inp", pack_f32(&x, dt), dt))
        .input(TestBuffer::zeros("out", n, dt))
        .constexpr("n", n as u32)
        .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
        .grid_3d(rows as u32, 1, 1, [tpg as u32, 1, 1])  // matches kernel dispatch
}
```

Key builder methods:

| Method | Notes |
|---|---|
| `.mode(KernelMode::Reduction)` | Required for Reduction kernels; omit for elementwise |
| `.input(TestBuffer::from_vec(...))` | Inputs in declaration order |
| `.input(TestBuffer::zeros("out", n, dt))` | Output buffers initialised to zero |
| `.constexpr("name", value)` | One call per `#[constexpr]` param |
| `.expect(TestBuffer::from_vec(...))` | Only the output buffers need expected values |
| `.grid_1d(n, tpg)` | Elementwise shorthand: `ceil(n/tpg)` threadgroups |
| `.grid_3d(gx, gy, gz, [tpg, 1, 1])` | Explicit 3D grid |

### `#[test_kernel]` annotation

```rust
#[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
fn test_mt_my_kernel(dt: DType) -> TestSetup { setup(1024, dt) }
```

- `dtypes` — always test all three float dtypes unless the kernel only supports a subset.
- `tol` — per-dtype absolute tolerance. A reasonable starting point:
  - `f32`: `1e-4` (rounding + online-algorithm drift)
  - `f16`: `1e-2` (shorter mantissa; wide if the op has long reductions)
  - `bf16`: `5e-2` (only 7 mantissa bits)
- Test name = function name (the inventory key is `test_mt_my_kernel_f32`, etc.).

### What to test

Write at least two tests per kernel:

1. **Standard shape** — representative production size, exercises the main code path.
2. **Edge-case shape** — exercises a specific branch (e.g. large-magnitude inputs to pin the overflow guard, a shape that hits the spill path in a bit-stream read, single-simdgroup dispatch).

Name edge-case tests descriptively: `test_mt_softmax_large_values`, not `test_mt_softmax_2`.

---

## 8. Writing the bench

### BenchSetup builder

```rust
#[bench(dtypes = [f32, f16, bf16])]
fn bench_mt_my_kernel(dt: DType) -> BenchSetup {
    let n = 64 * 1024 * 1024usize;
    BenchSetup::new(mt_my_kernel::kernel_ir_for(dt))
        .mode(KernelMode::Reduction)              // omit for elementwise
        .buffer(BenchBuffer::random("inp", n, dt))
        .buffer(BenchBuffer::zeros("out", n, dt).output())  // .output() marks readback
        .constexpr("n", n as u32)
        .grid_3d(rows as u32, 1, 1, [tpg as u32, 1, 1])
        .bytes_moved((2 * n * dt.size_bytes()) as u64)  // reads + writes
}
```

### Bench name = function name

`#[bench]` has **no `name` key** — the registered bench name is always the
function name (for a `variants(...)` bench, the suffix-renamed name). So make the
function name descriptive: `bench_mt_softmax`, `bench_mxfp4_conv2d`. Do not encode
a `mlx/`/`ffai/` path in the name; the family is the folder, not the bench string.

### `bytes_moved`

Report the **memory bandwidth** that the kernel's working set consumes. For elementwise ops: `(reads + writes) * n * sizeof(T)`. For reductions: include both the input stream and the (typically smaller) output. For matmuls: include packed weights, scales/biases, input, and output.

Do **not** count the small `#[constexpr]` constant buffers.

### Adding a metal reference (optional)

A bench may carry an optional **metal reference** — a hand-written `.metal`
comparator for side-by-side throughput + correctness. It is not tied to `mlx/`:
any kernel can have one, and most have none. Add it with `.with_reference(...)`:

```rust
.with_reference(
    RefKernel::new(
        format!("looped_softmax_{tn}"),   // reference kernel function name
        include_str!(concat!(env!("OUT_DIR"), "/metal/softmax.metal")),
    )
    .buffer(BenchBuffer::zeros("inp", n, dt))   // reference buffer order may differ
    .buffer(BenchBuffer::zeros("out", n, dt).output())
    .buffer(BenchBuffer::from_vec("axis_size", (n as u32).to_le_bytes().to_vec(), DType::U32))
    .grid(Grid::new_3d(rows as u32, 1, 1, [1024, 1, 1]))
    .tol(dtype_tol(dt).max(1e-4)),
)
```

Buffers **shared by name** (e.g. `"inp"`) are filled with the same random data as the kernel — the runner overwrites the placeholder. Buffers unique to the reference (like `"axis_size"`) are provided as separate `from_vec` entries.

---

## 9. Registering the kernel

After creating the file, declare it in its family `mod.rs`:

```rust
// crates/metaltile-std/src/kernels/<family>/mod.rs
pub mod my_kernel;
```

The `#[kernel]` / `#[bench]` / `#[test_kernel]` proc-macros automatically submit the kernel and its tests/benches to the global inventory via `inventory::submit!`. No manual registration is needed beyond the `mod` declaration.

---

## 10. DSL reference and known limitations

### Built-in identifiers

| Identifier | Type | Description |
|---|---|---|
| `tid` | `u32` | Thread index within the threadgroup (`thread_position_in_threadgroup.x`) |
| `tgid_x` / `tgid_y` / `tgid_z` | `u32` | Threadgroup index per axis |
| `lsize` | `u32` | Threadgroup size (threads per group) |
| `program_id::<N>()` | `u32` | Global thread index on axis N (elementwise) |
| `program_id(N)` | `u32` | Same, non-const-generic form |

### DSL intrinsics (selection)

```rust
// Arithmetic
let y = sqrt(x);  exp(x);  log(x);  exp2(x);  log2(x);
let y = sin(x);   cos(x);  atan2(y, x);
let y = abs(x);   recip(x);  pow(a, b);

// Reductions (simdgroup-wide)
let m = simd_sum(v);   simd_max(v);   simd_min(v);

// Reductions (threadgroup-wide, Reduction mode only)
let m = reduce_max(local_max);   reduce_sum(local_sum);

// Control
let v = select(cond, a, b);   // branchless ternary
let v = max(a, b);   min(a, b);

// Threadgroup memory
threadgroup_alloc("name", SIZE);             // allocate SIZE f32 slots
threadgroup_alloc("name", SIZE, "u32");      // allocate SIZE u32 slots
threadgroup_store("name", idx, val);
let v = threadgroup_load("name", idx);
threadgroup_barrier();                        // synchronise all threads

// Atomic (on threadgroup memory)
atomic_or_tg("name", idx, mask);

// Casts
let f = val.cast::<f32>();
let t = acc.cast::<T>();
```

### Loops and branches

```rust
// Counted loop — use range()
for i in range(start, end, step) { ... }

// Conditional
if cond { ... }
if cond { ... } else { ... }
```

### Hard limitations

These patterns are **silently dropped** by the body parser — the kernel compiles but produces wrong output:

- **`while` loops** — rewrite as a `for` with `range`, or use explicit `if`-blocks.
- **`return` statements** — use `if/else` branching instead.
- **`macro_rules!` invocations inside a `#[kernel]` body** — inline all code; the proc-macro runs before declarative macros expand.

Additional constraints:

- **`threadgroup_alloc` is hoisted to function scope** — allocate names once, unconditionally. Never allocate the same name in two branches; the later allocation is a no-op and the first is used for both.
- **No stack-allocated arrays** — use threadgroup memory for per-thread scratch that needs to survive a barrier.
- **No `let mut` captured across loop iterations** — declare `let mut acc = 0.0f32` before the loop, update inside.

---

## 11. Worked example: elementwise scale

A complete, minimal kernel that multiplies every element of a tensor by a scalar constant. This is the "hello world" for the DSL.

```rust
//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Elementwise scale — multiplies every input element by a scalar `alpha`.

use metaltile::kernel;

/// Multiply each element of `inp` by `alpha`, writing to `out`.
///
/// Grid: Elementwise, `[ceil(n/256), 1, 1]` × `[256, 1, 1]`.
#[kernel]
pub fn mt_scale<T>(inp: Tensor<T>, mut out: Tensor<T>, #[constexpr] alpha: f32) {
    let idx = program_id::<0>();
    store(out[idx], (load(inp[idx]).cast::<f32>() * alpha).cast::<T>());
}

/// Correctness: GPU output must match `alpha * x` within dtype precision.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_scale;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, alpha: f32, dt: DType) -> TestSetup {
        let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
        let xd = unpack_f32(&pack_f32(&x, dt), dt);
        let expected: Vec<f32> = xd.iter().map(|&v| v * alpha).collect();
        TestSetup::new(mt_scale::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("inp", pack_f32(&x, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("alpha", alpha)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-3, 1e-3])]
    fn test_mt_scale(dt: DType) -> TestSetup { setup(1024, 2.5, dt) }

    // alpha = 0 collapses every output to zero — pins the zero-multiply path.
    #[test_kernel(dtypes = [f32], tol = [1e-6])]
    fn test_mt_scale_zero_alpha(dt: DType) -> TestSetup { setup(256, 0.0, dt) }
}

/// Benchmark: 64M-element scale, reads + writes one stream each.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_scale;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_mt_scale(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_scale::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("inp", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("alpha", 2.5f32)
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }
}
```

After creating the file, add one line to its family `mod.rs` (e.g.
`src/kernels/ops/mod.rs`):

```rust
pub mod scale;
```

Run `cargo test -p metaltile-std` to confirm the tests register and pass.
