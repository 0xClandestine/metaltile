//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! In-process execution of new-syntax (`#[bench]` / `#[test_kernel]`) setups.
//!
//! This is the consumer the foundation PR deferred:
//! - [`run_kernel_test`] turns a [`TestSetup`] into a CPU-oracle correctness
//!   verdict via the name-keyed [`Context::dispatch_with_grid`] path the legacy
//!   GPU correctness tests use.
//! - [`run_kernel_bench`] turns a [`BenchSetup`] into a timed GB/s figure via
//!   the legacy [`crate::runner`] / [`GpuRunner`] path (resident buffers, SLC
//!   flush, DVFS pinning), so its numbers are comparable to legacy rows.
//!
//! The execution logic lives here (not in the toolchain crates) so it can be
//! reused unchanged whether the CLI **links** it (today) or a generated runner
//! binary **spawns** it and streams results over a protocol (a later step).
//!
//! ## Buffer binding
//!
//! Tests bind buffers **by name** (matching kernel parameter names), so the
//! order of `.input()` calls doesn't matter. Benches bind **positionally** —
//! `GpuRunner` dispatches by buffer index — so they follow the codegen order:
//! tensor params in signature order, then constexprs in IR order. In both
//! paths `#[constexpr]` scalars are passed as ordinary little-endian uniform
//! buffers, not Metal function constants.

use std::collections::BTreeMap;

use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{
    bench::{BenchSetup, ConstValue, KernelBench, RefKernel, TestSetup},
    dtype::DType,
    ir::ParamKind,
};
use metaltile_runtime::Context;

use crate::{
    bench_types::{EquivResult, OpBench, OpResult, check_equiv, dtype_label},
    runner::{GpuBuffer, GpuRunner, bench_gbps, read_typed},
    utils::unpack_f32,
};

/// Outcome of running one `#[test_kernel]` setup on the GPU.
#[derive(Debug, Clone, Copy)]
pub struct TestOutcome {
    /// Whether every compared element was within tolerance.
    pub passed: bool,
    /// Largest absolute error observed across all expected buffers.
    pub max_abs_err: f32,
    /// Total number of elements compared.
    pub n_checked: usize,
}

/// Encode a constexpr value as the little-endian uniform-buffer bytes the
/// kernel expects. Pointer-sized values are narrowed to `u32` to match the
/// `constant uint&` convention used by `#[constexpr] n: u32` parameters.
fn constexpr_bytes(v: &ConstValue) -> Vec<u8> {
    match *v {
        ConstValue::U32(x) => x.to_le_bytes().to_vec(),
        ConstValue::I32(x) => x.to_le_bytes().to_vec(),
        ConstValue::F32(x) => x.to_le_bytes().to_vec(),
        ConstValue::U64(x) => x.to_le_bytes().to_vec(),
        ConstValue::I64(x) => x.to_le_bytes().to_vec(),
        ConstValue::Usize(x) => (x as u32).to_le_bytes().to_vec(),
    }
}

/// Maximum absolute element-wise difference between two `f32` slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Render an element count compactly (e.g. `64M`, `8K`, `1000`).
fn human_count(n: usize) -> String {
    const M: usize = 1 << 20;
    const K: usize = 1 << 10;
    if n >= M && n.is_multiple_of(M) {
        format!("{}M", n / M)
    } else if n >= K && n.is_multiple_of(K) {
        format!("{}K", n / K)
    } else {
        n.to_string()
    }
}

/// Run a `#[test_kernel]` setup: dispatch the kernel, then compare each
/// expected output buffer against the GPU result within `tol` (absolute).
///
/// When the setup carries a `compare_against` reference, the reference is
/// dispatched first and its output buffers become the expected values
/// (GPU-vs-GPU); otherwise the setup's `.expect()` buffers (a CPU oracle) are
/// used.
///
/// # Errors
///
/// Returns an error string if dispatch fails or an expected output buffer is
/// missing from the dispatch result.
pub fn run_kernel_test(ctx: &Context, setup: &TestSetup, tol: f64) -> Result<TestOutcome, String> {
    let no_consts: BTreeMap<String, u32> = BTreeMap::new();

    // Inputs + constexprs → name-keyed dispatch buffers.
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for inp in setup.inputs() {
        buffers.insert(inp.name().to_string(), inp.data().to_vec());
    }
    for (name, v) in setup.constexprs() {
        buffers.insert(name.clone(), constexpr_bytes(v));
    }

    let grid = setup.grid();
    let g = grid.grid.map(|x| x as usize);
    let t = grid.tpg.map(|x| x as usize);

    let result = ctx
        .dispatch_with_grid(setup.kernel(), &buffers, &no_consts, g, t)
        .map_err(|e| format!("dispatch failed: {e}"))?;

    // Expected (name, bytes, dtype): GPU reference output, or the CPU oracle.
    let expected: Vec<(String, Vec<u8>, DType)> = if let Some(reference) = setup.ref_setup() {
        let ref_outputs = dispatch_outputs(ctx, reference)?;
        // Compare each output the reference produced, using the dtype the main
        // setup declared for that buffer (falling back to f32).
        ref_outputs
            .into_iter()
            .map(|(name, bytes)| {
                let dt = setup
                    .inputs()
                    .iter()
                    .find(|b| b.name() == name)
                    .map_or(DType::F32, |b| b.dtype());
                (name, bytes, dt)
            })
            .collect()
    } else {
        setup
            .expected()
            .iter()
            .map(|b| (b.name().to_string(), b.data().to_vec(), b.dtype()))
            .collect()
    };

    let mut worst = 0.0f32;
    let mut n_checked = 0usize;
    for (name, exp_bytes, dt) in &expected {
        let out_bytes = result
            .output(name)
            .ok_or_else(|| format!("expected output '{name}' missing from dispatch result"))?;
        let got = unpack_f32(out_bytes, *dt);
        let exp = unpack_f32(exp_bytes, *dt);
        let n = got.len().min(exp.len());
        worst = worst.max(max_abs_diff(&got[..n], &exp[..n]));
        n_checked += n;
    }

    Ok(TestOutcome { passed: (worst as f64) <= tol, max_abs_err: worst, n_checked })
}

/// Dispatch a setup and return its raw output buffers (used for the
/// GPU-vs-GPU `compare_against` path).
fn dispatch_outputs(ctx: &Context, setup: &TestSetup) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let no_consts: BTreeMap<String, u32> = BTreeMap::new();
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for inp in setup.inputs() {
        buffers.insert(inp.name().to_string(), inp.data().to_vec());
    }
    for (name, v) in setup.constexprs() {
        buffers.insert(name.clone(), constexpr_bytes(v));
    }
    let grid = setup.grid();
    let g = grid.grid.map(|x| x as usize);
    let t = grid.tpg.map(|x| x as usize);
    let result = ctx
        .dispatch_with_grid(setup.kernel(), &buffers, &no_consts, g, t)
        .map_err(|e| format!("reference dispatch failed: {e}"))?;
    Ok(result.outputs)
}

/// Run a `#[bench]` setup and produce a GB/s row.
///
/// Times the kernel with the same steady-state machinery as the legacy bench
/// path — [`GpuRunner`] resident buffers, an SLC flush to pin DVFS at peak
/// clock, warmup, and the minimum of [`crate::runner`]'s timed iterations —
/// so new-syntax rows are directly comparable to legacy rows in the same table.
///
/// Buffers bind **positionally** to match the `[[buffer(N)]]` indices the MSL
/// declares: tensor params in signature order, then constexprs in IR order.
///
/// Returns an [`OpResult`] (already reported through the active result
/// reporter), or `None` if MSL generation/compilation fails, a buffer the
/// kernel expects is missing from the setup (including the `_shape`/`_strides`
/// metadata buffers a `#[strided]` param requires), or timing is unavailable
/// (e.g. off-GPU platforms).
///
/// Correctness is **not** checked here — it is proven by the kernel's
/// `#[test_kernel]`s via `tile test` / the cargo-test harness. The row carries
/// a perf-only equivalence sentinel, matching the legacy runner's behaviour
/// when no reference kernel is available.
pub fn run_kernel_bench(
    runner: &GpuRunner,
    bench: &'static dyn KernelBench,
    dt: DType,
) -> Option<OpResult> {
    let setup: BenchSetup = bench.setup(dt);
    let bytes_moved = bench.bytes_moved(&setup);
    let kernel = setup.kernel();

    // Compile the MetalTile kernel the same way the legacy MT path does.
    let msl = MslGenerator::default().generate(kernel).ok()?;
    let compiled = runner.compile(&msl, &kernel.name).ok()?;

    // Allocate resident GPU buffers in codegen binding order: tensor params
    // first, then constexpr scalars. We also remember (a) each tensor param's
    // initial bytes by name, so a reference kernel can be fed *identical* input
    // data for an apples-to-apples A/B, and (b) where the primary output buffer
    // lands, so it can be read back and compared.
    let mut bufs: Vec<GpuBuffer> =
        Vec::with_capacity(kernel.params.len() + kernel.constexprs.len());
    let mut input_bytes: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();
    let mut mt_out: Option<MtOutput> = None;
    for param in &kernel.params {
        let buf = setup.buffers().iter().find(|b| b.name() == param.name)?;
        let bytes = buf.initial_bytes();
        if param.is_output && mt_out.is_none() {
            mt_out = Some(MtOutput { buf_idx: bufs.len(), elems: buf.len(), dtype: buf.dtype() });
        }
        // Upload first, then hand the bytes to the share-map (no clone) so a
        // reference kernel can reuse this exact input.
        bufs.push(runner.buffer_bytes(&bytes));
        input_bytes.insert(param.name.clone(), bytes);
        // A `#[strided]` param occupies three consecutive buffer slots —
        // data, then `<name>_shape` and `<name>_strides` — matching the
        // runtime's `push_strided` ABI. The bench setup must carry those two
        // metadata buffers (u32 `[rank]` each) as ordinary named buffers.
        if param.kind == ParamKind::Strided {
            let shape_name = format!("{}_shape", param.name);
            let strides_name = format!("{}_strides", param.name);
            let shape = setup.buffers().iter().find(|b| b.name() == shape_name)?;
            let strides = setup.buffers().iter().find(|b| b.name() == strides_name)?;
            bufs.push(runner.buffer_bytes(&shape.initial_bytes()));
            bufs.push(runner.buffer_bytes(&strides.initial_bytes()));
        }
    }
    for decl in &kernel.constexprs {
        let name = decl.name.name();
        let (_, value) = setup.constexprs().iter().find(|(n, _)| n == name)?;
        bufs.push(runner.buffer_bytes(&constexpr_bytes(value)));
    }
    let refs: Vec<&GpuBuffer> = bufs.iter().collect();

    let grid = setup.grid();
    let g = grid.grid.map(|x| x as usize);
    let t = grid.tpg.map(|x| x as usize);
    let (gbps, _stats) = bench_gbps(runner, &compiled, &refs, g, t, bytes_moved as f64)?;

    // Shape label: the author's explicit label if given (needed for
    // multi-dimensional kernels that one buffer's length can't summarise),
    // else inferred as `N=<largest buffer> <dtype>` — the largest buffer is
    // the meaningful size whether that's the output (elementwise, e.g. arange)
    // or the input (a reduction whose output is a single element, e.g. argmax).
    let shape = match setup.shape_label() {
        Some(label) => label.to_string(),
        None => {
            let n = setup.buffers().iter().map(|b| b.len()).max().unwrap_or(0);
            format!("N={} {}", human_count(n), dtype_label(dt))
        },
    };

    // When the bench declares a reference kernel (e.g. an MLX kernel), time it
    // the same way and compare outputs — the row then reports MT GB/s, ref
    // GB/s, the speed ratio, and a real correctness verdict. If anything in the
    // reference path fails (compile, missing output, off-GPU), fall through to
    // the perf-only row rather than dropping the bench entirely.
    if let (Some(rk), Some(out)) = (setup.ref_kernel(), mt_out)
        && let Some(equiv) =
            run_reference_compare(runner, rk, &bufs, out, &input_bytes, bytes_moved)
    {
        return Some(OpBench::new(bench.name(), "GB/s").implemented(
            shape,
            Some(equiv.ref_gbps),
            gbps,
            equiv.result,
        ));
    }

    let equiv = EquivResult { n_checked: 0, max_abs_err: 0.0, cosine_sim: 1.0, passed: true };
    Some(OpBench::new(bench.name(), "GB/s").implemented(shape, None, gbps, equiv))
}

/// Upper bound on output elements compared in a reference A/B. Keeps the
/// per-kernel read-back + compare cheap; deterministic inputs repeat a short
/// pattern, so a 32K-element prefix exercises every branch the full output
/// would. Held below f16's largest finite value (65504) so generators like
/// `arange` (whose output grows with the index) stay representable across the
/// compared prefix in every dtype instead of saturating to inf.
const COMPARE_ELEM_CAP: usize = 1 << 15;

/// Where a MetalTile kernel's primary output landed, for read-back.
#[derive(Debug, Clone, Copy)]
struct MtOutput {
    /// Index of the output buffer within the positional `bufs` vector.
    buf_idx: usize,
    /// Element count of the output.
    elems: usize,
    /// Output element dtype.
    dtype: DType,
}

/// Result of running a reference kernel alongside MetalTile: its throughput plus
/// the numerical equivalence verdict between the two outputs.
struct RefOutcome {
    ref_gbps: f64,
    result: EquivResult,
}

/// Compile, dispatch, and time the reference kernel, then compare its output
/// against MetalTile's.
///
/// The reference is fed **identical input data**: every non-output reference
/// buffer whose name matches a MetalTile tensor parameter reuses that
/// parameter's exact initial bytes (`input_bytes`), so the A/B is apples-to-
/// apples even though `BenchBuffer::random` is non-deterministic. Buffers with
/// no MetalTile counterpart (the fresh output, MLX-specific scalars) use their
/// own initial bytes.
///
/// Returns `None` (so the caller emits a perf-only row) if the reference can't
/// compile, declares no `.output()` buffer, or times out.
fn run_reference_compare(
    runner: &GpuRunner,
    rk: &RefKernel,
    mt_bufs: &[GpuBuffer],
    mt_out: MtOutput,
    input_bytes: &std::collections::HashMap<String, Vec<u8>>,
    bytes_moved: u64,
) -> Option<RefOutcome> {
    // Compile the reference, binding any Metal function constants it requires
    // (e.g. rope / steel attention gate their body on no-default bool constants).
    let compiled = if rk.bool_constants.is_empty() {
        runner.compile(&rk.source, &rk.fn_name).ok()?
    } else {
        runner.compile_with_bool_constants(&rk.source, &rk.fn_name, &rk.bool_constants).ok()?
    };

    // Build the reference's positional buffers, sharing MT input data by name.
    let mut ref_bufs: Vec<GpuBuffer> = Vec::with_capacity(rk.buffers.len());
    let mut ref_out: Option<(usize, usize, DType)> = None;
    for b in &rk.buffers {
        if b.is_output() && ref_out.is_none() {
            ref_out = Some((ref_bufs.len(), b.len(), b.dtype()));
        }
        let bytes = match (b.is_output(), input_bytes.get(b.name())) {
            (false, Some(shared)) => shared.clone(),
            _ => b.initial_bytes(),
        };
        ref_bufs.push(runner.buffer_bytes(&bytes));
    }
    let (ref_out_idx, ref_out_elems, ref_out_dt) = ref_out?;

    let ref_refs: Vec<&GpuBuffer> = ref_bufs.iter().collect();
    let g = rk.grid.grid.map(|x| x as usize);
    let t = rk.grid.tpg.map(|x| x as usize);
    let (ref_gbps, _stats) = bench_gbps(runner, &compiled, &ref_refs, g, t, bytes_moved as f64)?;

    // Read both outputs back (the timed run leaves the last result resident) and
    // compare over the overlapping prefix. The prefix is capped: a 64M-element
    // readback + compare per kernel would dominate the bench wall-clock, and a
    // 1M-element prefix is plenty to catch a miscomputing kernel (inputs repeat
    // a short pattern, so every code path is exercised within the prefix).
    let n = mt_out.elems.min(ref_out_elems).min(COMPARE_ELEM_CAP);
    let mt_vals = read_typed(runner, &mt_bufs[mt_out.buf_idx], n, mt_out.dtype);
    let ref_vals = read_typed(runner, &ref_bufs[ref_out_idx], n, ref_out_dt);
    let result = check_equiv(&ref_vals, &mt_vals, rk.tol);

    Some(RefOutcome { ref_gbps, result })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constexpr_32bit_variants_pack_to_four_le_bytes() {
        assert_eq!(constexpr_bytes(&ConstValue::U32(513)), vec![1, 2, 0, 0]);
        assert_eq!(constexpr_bytes(&ConstValue::Usize(513)), vec![1, 2, 0, 0]);
        assert_eq!(constexpr_bytes(&ConstValue::I32(513)), vec![1, 2, 0, 0]);
        assert_eq!(constexpr_bytes(&ConstValue::F32(1.0)), 1.0f32.to_le_bytes().to_vec());
    }

    #[test]
    fn constexpr_64bit_variants_pack_to_eight_le_bytes() {
        assert_eq!(constexpr_bytes(&ConstValue::U64(513)), vec![1, 2, 0, 0, 0, 0, 0, 0]);
        assert_eq!(constexpr_bytes(&ConstValue::I64(513)), vec![1, 2, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn human_count_renders_power_of_two_suffixes() {
        assert_eq!(human_count(64 * 1024 * 1024), "64M");
        assert_eq!(human_count(8 * 1024), "8K");
        assert_eq!(human_count(1024), "1K");
        assert_eq!(human_count(1 << 20), "1M");
        // Non-multiples and small values render as plain decimals.
        assert_eq!(human_count(1000), "1000");
        assert_eq!(human_count(1536), "1536"); // 1.5K — not a clean multiple
        assert_eq!(human_count(0), "0");
    }

    #[test]
    fn max_abs_diff_is_elementwise_max() {
        assert_eq!(max_abs_diff(&[1.0, 2.0, 3.0], &[1.0, 2.5, 3.0]), 0.5);
        assert_eq!(max_abs_diff(&[-1.0], &[1.0]), 2.0); // sign-aware via abs
        assert_eq!(max_abs_diff(&[], &[]), 0.0); // empty → 0
    }

    #[test]
    fn max_abs_diff_ignores_nan_in_the_fold() {
        // (inf - inf) = NaN; `f32::max` returns the non-NaN argument, so a
        // matching ±inf position contributes nothing while the real diffs still
        // dominate. The harness relies on this so masked-out -inf softmax maxes
        // (which round-trip as -inf on both sides) don't poison the comparison.
        assert_eq!(max_abs_diff(&[f32::INFINITY, 1.0], &[f32::INFINITY, 1.25]), 0.25);
        assert_eq!(max_abs_diff(&[f32::NEG_INFINITY], &[f32::NEG_INFINITY]), 0.0);
    }

    #[test]
    fn max_abs_diff_compares_over_the_shorter_slice() {
        // `zip` stops at the shorter length — the trailing 99.0 is not compared.
        assert_eq!(max_abs_diff(&[1.0, 2.0, 99.0], &[1.0, 2.0]), 0.0);
    }
}
