# Testing

How kernels and codegen are verified, what runs where, how to write a test, and — importantly — the **gaps** in the test infrastructure that let bugs through silently.

## The test layers

Correctness is checked at four layers, each catching what the layer above cannot:

| Layer | Catches | Where it lives | Runs in CI? |
|---|---|---|---|
| **DSL / codegen unit tests** | Pass correctness, body-parser arms, IR variants, emit paths; `trybuild` compile-fail fixtures | `crates/metaltile-codegen`, `metaltile-core`, `metaltile-macros` | ✅ |
| **MSL snapshots** (`insta`) | Codegen output drift — a reviewable text diff in the PR | `crates/metaltile-codegen/tests/msl_snapshots.rs` | ✅ |
| **GPU correctness** | Numeric disagreement vs a naive CPU oracle, on a real Metal device | `crates/metaltile-std/tests/<kernel>_gpu_correctness.rs` | ✅ (macOS runner) |
| **MLX side-by-side** (bench) | Throughput + numeric parity vs the upstream MLX kernel | `tile bench` | local-only (needs an MLX checkout) |

No single layer is sufficient. The unit tests never touch a GPU; snapshots pin *whatever* the codegen emits (including wrong output); `xcrun metal` only checks syntax. **GPU correctness tests are the floor** — see the gaps section below.

## Running tests — what runs where

```bash
make test        # whole workspace: codegen, runtime, GPU correctness (GPU on a Mac)
make clippy      # lint, -D warnings
make fmt-check   # formatting
make typos       # spell-check
make coverage    # HTML coverage report (needs cargo-llvm-cov)
make bench       # MLX side-by-side benchmark suite (macOS + Metal only)
```

Per-kernel, via `cargo` directly (these are the documented exceptions to "always use `make`"):

```bash
# One kernel's GPU correctness test:
cargo test -p metaltile-std --test <kernel>_gpu_correctness

# One kernel's perf bench (the #[ignore]'d companion test):
cargo test --release -p metaltile-std --test <kernel>_gpu_correctness -- --ignored --nocapture
```

### CI vs local

| Job | Workflow | What it runs |
|---|---|---|
| `typos` / `clippy` / tests | `.github/workflows/check.yml` | spell-check, lint `-D warnings`, `cargo test --workspace` (Ubuntu — no GPU) |
| build / test / bench | `.github/workflows/tile.yml` | `tile build`, `tile test` (GPU correctness vs CPU oracle), `tile bench` — on a **macOS GPU runner** |
| coverage | `.github/workflows/coverage.yml` | `cargo llvm-cov --workspace --codecov` on macOS, uploads to Codecov; runs on pushes touching `crates/`, `Cargo.*`, `rust-toolchain.toml`, `.github/configs/codecov.yml` |
| PR title | `.github/workflows/pr.yml` | validates the conventional-commit format |
| labels | `.github/workflows/auto-label.yml` | release-notes labels from the PR-title prefix |

- The DSL / codegen / GPU-correctness layers all run in CI — including on a macOS runner with a real GPU.
- **`tile bench` benches the metaltile kernels by default**; the MLX side-by-side A/B is opt-in via `tile bench --mlx` (it needs an MLX checkout, so the default CI bench runs metaltile-only).

### macOS runner environment

The macOS CI jobs (`tile.yml` build/test/bench, plus `coverage.yml` and
`release.yml`) run on the **`macos-26`** GitHub-hosted runner. To see exactly
what is installed (Xcode versions, SDKs, CLI tools), consult the image manifest:
[runner-images → macos-26-arm64 readme](https://github.com/actions/runner-images/blob/main/images/macos/macos-26-arm64-Readme.md).

> **NAX / MPP kernels need the macOS 26.5+ Metal toolchain.** The
> cooperative-tensor kernels (`mpp::tensor_ops::matmul2d`) compile via the Metal
> toolchain that ships with the **OS**, and the dynamic-extent `cooperative_tensor`
> they emit (inside Apple's MPP library) is rejected by the macOS-26.4 compiler
> at pipeline-build time ("unsupported deferred-static-alloca-size"); macOS 26.5
> compiles it. This is the *runtime* `newLibraryWithSource` compiler, which is the
> OS's — **selecting a newer Xcode (`DEVELOPER_DIR`) does not change it** (that
> only swaps the offline `metal` compiler). The `macos-26` runner image is
> currently macOS 26.4, so `tile test` **skips** any cooperative-tensor kernel
> whose pipeline won't build (`Kernel::requires_cooperative_tensors()` →
> `[SKIP]`), reporting them as skipped rather than failed. They still compile +
> get correctness-tested wherever the toolchain supports them (a 26.5+ box, and
> CI automatically once the image's OS reaches 26.5) — the skip self-re-enables.
> Numeric mismatches still fail; only build failures skip.

## Writing tests

### Every non-trivial kernel ships a GPU correctness test — same commit

The test runs the kernel on a real Metal device and compares against a naive CPU reference computed in `f32`. Shared helpers (`ramp`, dtype pack/unpack, `max_abs_diff`, `naive_*`) live in `crates/metaltile-std/tests/common/mod.rs`.

```rust
#![cfg(target_os = "macos")]
mod common;
use common::{ramp, pack_bytes, unpack_bytes, max_abs_diff};
use metaltile_runtime::Context;

#[test]
fn my_kernel_matches_naive_cpu_reference_f32() {
    // 1. Build small synthetic inputs (ramp / deterministic pattern).
    // 2. Compute a naive CPU reference in f32.
    // 3. Pack to bytes, populate the buffer map, dispatch via
    //    Context::dispatch_with_grid(&kernel, &buffers, &constexprs,
    //                                 grid_xyz, threadgroup_xyz).
    // 4. Unpack the output buffer; assert max_abs_diff < 1e-4.
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn my_kernel_perf_bench_f32() {
    // 20 warmup + 100 measure iterations; report median GPU µs + GB/s.
}
```

The naive CPU reference **is the contract**. If kernel and reference disagree, decide which is wrong before merging — don't loosen the tolerance to pass.

### New: declarative `#[test_kernel]` / `#[bench]` (additive, opt-in)

Alongside the hand-written `tests/*_gpu_correctness.rs` files, a kernel can now declare its correctness test and benchmark **next to the kernel** with the `#[test_kernel]` / `#[bench]` attributes. This is being introduced additively — the legacy `#[kernel(bench(...))]` registration and the `tests/*_gpu_correctness.rs` files keep working unchanged, and during migration a kernel can carry both so old and new are A/B-compared on the same IR. `crates/metaltile-std/src/mlx/arange.rs` is the first kernel ported; use it as the template.

```rust
use metaltile::kernel;

#[kernel]
pub fn mt_arange<T>(out: Tensor<T>, start: Tensor<T>, step: Tensor<T>, #[constexpr] n: u32) { /* … */ }

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};
    use super::mt_arange;
    use crate::utils::{pack_f32, scalar_bytes};

    fn setup(start: f32, step: f32, n: usize, dt: DType) -> TestSetup {
        let expected: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect(); // CPU oracle in f32
        TestSetup::new(mt_arange::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("out", vec![0u8; n * dt.size_bytes()], dt))
            .input(TestBuffer::from_vec("start", scalar_bytes(start, dt), dt))
            .input(TestBuffer::from_vec("step", scalar_bytes(step, dt), dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_arange_ascending(dt: DType) -> TestSetup { setup(0.0, 0.5, 64, dt) }

    // Per-dtype tolerances (order matches `dtypes`): f32, f16, bf16.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-2, 1e-1])]
    fn test_mt_arange_fractional_step(dt: DType) -> TestSetup { setup(0.0, 0.1, 64, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};
    use super::mt_arange;
    use crate::utils::scalar_bytes;

    #[bench(name = "mlx/arange", dtypes = [f32, f16, bf16])]
    fn bench_arange(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_arange::kernel_ir_for(dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .buffer(BenchBuffer::from_vec("start", scalar_bytes(0.0, dt), dt))
            .buffer(BenchBuffer::from_vec("step", scalar_bytes(1.0, dt), dt))
            .constexpr("n", n as u32)
            .grid_1d(n, 256)
            .bytes_moved((n * dt.size_bytes()) as u64)
    }
}
```

Notes:
- Buffers bind **by name** (matching the kernel parameter names); ordering of `.buffer()`/`.input()` calls doesn't matter. `#[constexpr]` scalars are passed as little-endian uniform buffers, same as the hand-written tests.
- The CPU oracle is the same contract as above — compute expected in `f32`, let the runner pack to the dtype and diff within tolerance. `tol` accepts a scalar, a per-dtype array, or a `{ f32: …, f16: … }` table.
- Run them with `tile test [-f <filter>]` and `tile bench [-f <filter>]`; the new benches render in the same table as legacy rows. The `tests/kernel_tests_harness.rs` cargo bridge runs every `#[test_kernel]` under `cargo test` so the new path is part of the commit gate without `tile test`.
- This in-process runner is deliberately simple (it re-dispatches per iteration rather than reusing the legacy `GpuRunner`'s resident-buffer + DVFS-pinning path), so new-syntax bench GB/s currently reads lower than the legacy rows — fidelity is a follow-up, correctness is not affected.

### MSL snapshots for new emit paths

A new DSL primitive, fusion pattern, or dtype path also lands an `insta` fixture in `crates/metaltile-codegen/tests/msl_snapshots.rs` — a hand-built kernel run through `MslGenerator`, with the full MSL pinned via `assert_snapshot!`. Any future codegen change then surfaces as a reviewable text diff. Refresh intentional changes with `cargo insta review` (interactive) or `cargo insta test --accept`.

Fixtures exist to **exercise distinct emit paths**, not to be exhaustive — add one when a new path lands that the existing snapshots don't cover.

## Coverage

`make coverage` (or `./.github/scripts/coverage.sh`) produces an HTML report at `target/llvm-cov/html/index.html`; `./.github/scripts/coverage.sh summary` prints the per-file table CI emits. Per-crate floors live in `.github/configs/codecov.yml`:

| Crate | Floor |
|---|---|
| `metaltile-macros` | 92% |
| `metaltile-codegen` / `metaltile-core` | 90% |
| `metaltile-runtime` | 85% |
| `metaltile-cli` | 80% |
| `metaltile-std` | line-coverage exempt — gated by bench-correctness instead |
| `metaltile` (facade) | excluded |

`metaltile-std`'s `ffai/` and `mlx/` kernel-body files are excluded from the line-coverage denominator: the `#[kernel]` proc-macro consumes the body at compile time, the Rust body never executes, so line coverage on them is structurally meaningless. **Their correctness is gated by GPU correctness tests and bench equivalence instead — not by line coverage.**

## ⚠️ Gaps in the test infrastructure

These are the holes a bug can slip through. Know them; close them when you can.

### ⚠️ A wrong kernel can pass every check except a GPU correctness test

A kernel that emits an **empty body** — from an inner `macro_rules!` or from a codegen pass dropping a loop body (see [Developing → kernel-authoring hazards](developing.md#kernel-authoring-hazards)) — produces all-zeros output. That output:

- **passes `xcrun metal`** — an empty body is valid MSL;
- **passes `tile build --emit` smoke** — same reason;
- **passes MSL-snapshot drift checks** — the snapshot just pins the wrong-but-stable empty body;
- **passes a loose integration test** if its tolerance absorbs the noise.

It fails **only** when actual GPU output is compared to an expected value. That is the GPU correctness test, and nothing else. This is exactly how a family of quantized-gather kernels shipped silently broken until a correctness test was added. **Do not rely on the smoke build or snapshots to catch a broken kernel.**

### ⚠️ Not every kernel has a GPU correctness test yet

Coverage of `crates/metaltile-std/tests/` is incomplete — some kernels have a bench row but no correctness test, and some have neither. A kernel with no correctness test has *no automated proof it computes the right answer*. When you touch such a kernel, add the test; when you add a kernel, add it in the same commit.

### ⚠️ Perf numbers can be harness artifacts

A bench number is only meaningful if the harness measures the kernel and not its own overhead. A latency that doesn't scale with input size is the tell. See [Developing → kernel-authoring hazards](developing.md#kernel-authoring-hazards) ("too flat to be physical") for the resident-buffer and GPU-clock-warmup fixes.
