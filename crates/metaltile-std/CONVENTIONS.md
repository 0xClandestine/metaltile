# metaltile-std kernel file conventions

Every file under `src/mlx/` and `src/ffai/` follows the same three-section
layout.  Keep the sections in order and separated by a blank line.

---

## 1 — Kernel definitions (file body)

Kernel functions live at the top of the file, outside any module.

```rust
//! Copyright 2026 …
//! SPDX-License-Identifier: Apache-2.0
//! One-line description of what this file contains.

use metaltile::kernel;

#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}
```

Rules:
- One `use metaltile::kernel;` at the top — the proc-macro prelude brings
  `Tensor`, `program_id`, `store`, `load`, and all builtins into scope.
- Kernels are generic over `T` unless a concrete dtype is required.
- No bench or test imports here; keep the top-level clean.

---

## 2 — `pub mod kernel_tests`

Correctness tests live in a single sub-module.  The module is `pub` so the
test runner discovers it via `metaltile-std`'s public surface.

```rust
pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile::core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;          // brings mt_exp, mt_log, … into scope

    // ── helpers ──────────────────────────────────────────────────────────

    fn pack_f32(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32  => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16  => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    // ── tests ─────────────────────────────────────────────────────────────

    /// `name` maps to the baseline JSON's `"op/subop"` key.
    /// Keep N small (≤ 8 192) — tests run on every CI push.
    #[test_kernel(name = "mlx/unary/exp", dtypes = [f32, f16, bf16], tol = 1e-3)]
    fn test_exp(dt: DType) -> TestSetup {
        let n = 512usize;
        let inp: Vec<f32> = (0..n).map(|i| (i % 32) as f32 * 0.1 - 1.6).collect();
        let exp_out: Vec<f32> = inp.iter().map(|&v| v.exp()).collect();
        TestSetup::new(mt_exp::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("a",   pack_f32(&inp,     dt), dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp_out, dt), dt))
            .grid_1d(n, 256)
    }
}
```

Rules:
- `#![allow(unused, dead_code, clippy::too_many_arguments)]` at the top of
  the module — auto-generated impls trigger these lints.
- `use super::*` pulls in all kernels defined above.
- `dtypes = [f32]` / `[f16, bf16]` / `[f32, f16, bf16]` — list only the
  dtypes actually tested; the macro expands one entry per dtype.
- `tol` is the max element-wise absolute error.  Typical values:
  - f32 exact copies: `1e-6`
  - f32 transcendentals: `1e-4`
  - f16: `1e-3`
  - bf16: `1e-2`
- Keep `N ≤ 8_192` in tests.  The runtime pools `MTLBuffer` allocations
  across dispatches so there are no per-test alloc/free cycles, but very
  large pool buckets still stress the unified DRAM bandwidth shared between
  the GPU and the display compositor on Apple Silicon.  Small N → small
  pool buckets → no display pressure.
- Use deterministic, bounded inputs (not `TestBuffer::random`) so failures
  reproduce reliably.
- Use `compare_against(ref_setup)` when comparing a new kernel against an
  existing one rather than computing CPU expected values.

---

## 3 — `pub mod kernel_benches`

Performance benchmarks live in a second sub-module.

```rust
pub mod kernel_benches {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::bench;
    use metaltile::core::{DType, bench::{BenchBuffer, BenchSetup}};

    use super::*;

    /// Bench sizes should match MLX for a fair side-by-side GB/s comparison.
    /// `name` maps to `"op/subop"` in the baseline JSON (baselines/*.json).
    #[bench(name = "unary/exp", dtypes = [f32, f16, bf16])]
    fn bench_exp(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;   // 64 M elements — matches MLX default
        BenchSetup::new(mt_exp::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a",   n, dt))
            .buffer(BenchBuffer::zeros("out",  n, dt).output())
            .grid_1d(n, 256)
    }
}
```

Rules:
- `#![allow(unused, dead_code, clippy::too_many_arguments)]` at the top.
- Bench setup is written **inline** — do not call `crate::mlx::benches::*`
  helpers.  The `benches/` sub-module is being removed; write the
  `BenchSetup` construction directly here.
- `name` must match the `"op/subop"` field in `baselines/apple-*.json` so
  the CLI can display the MLX reference GB/s alongside the MetalTile result.
- Use `BenchBuffer::random` for inputs, `BenchBuffer::zeros(...).output()`
  for write-back buffers.
- Mark outputs with `.output()` so the runner reads them back and the
  runtime-computed bandwidth is accurate.
- Use `grid_1d(n, tpg)` for elementwise, `grid_3d(rows, 1, 1, [tpg,1,1])`
  for row-parallel reductions, matching the dispatch used by the kernel.

---

## Complete annotated example

`src/mlx/copy.rs` is the canonical reference implementation.  It has all
three sections with working tests and a bench that matches the MLX baseline.

---

## `name` → baseline mapping

| Baseline `op`     | Expected `name` prefix in `#[bench]` |
|-------------------|--------------------------------------|
| `unary`           | `unary/<subop>` e.g. `unary/exp`     |
| `binary`          | `binary/<subop>`                     |
| `all_reduce`      | `all_reduce/<subop>`                 |
| `row_reduce`      | `row_reduce/<subop>`                 |
| `rms_norm`        | `rms_norm/<subop>`                   |
| `layer_norm`      | `layer_norm/layer_norm`              |
| `softmax`         | `softmax/softmax`                    |
| `gemv`            | `gemv/gemv`                          |
| `quantized`       | `quantized/<subop>`                  |
| `steel_gemm_fused`| `steel_gemm_fused/<tile_config>`     |
| `sdpa`            | `sdpa/<subop>`                       |
| `copy`            | `copy/copy`                          |
| `scan`            | `scan/scan`                          |
| `sort`            | `sort/sort`                          |

---

## What to avoid

- **Do not** add `#[bench]` or `#[test_kernel]` outside these two modules.
- **Do not** use `macro_rules!` inside a `#[kernel]` body — declarative
  macros are silently dropped by the proc-macro.
- **Do not** use `while` or `return` inside `#[kernel]` — silently dropped.
- **Do not** reference `crate::mlx::benches` — that module is being removed.
- **Do not** allocate buffers larger than ~32 M elements in `kernel_tests` —
  this causes GPU memory pressure and display compositor flicker on Apple
  Silicon while tests run.
