//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! RMS normalization benchmark — #[kernel] DSL vs MLX metal/rms_norm.metal
//!
//! The kernel is generic over `N = tpg * 4` — each thread owns 4
//! consecutive elements, the partial sum-of-squares reduces across
//! the threadgroup. The bench wires `n=4096, tpg=1024` for the
//! hidden-axis case. For per-head normalisation (Qwen3-style q_norm
//! / k_norm pre-RoPE), the same kernel is dispatched as one
//! threadgroup per `(batch*token*n_heads)` row at `tpg = head_dim/4`
//! with the per-head_dim weight broadcast across all rows. The
//! per-head contract is pinned by
//! `tests/rms_norm_per_head_gpu.rs`.
//!
//! Models with head_dim < 128 (older 7B-class, head_dim=64) dispatch
//! [`mt_rms_norm_small`] instead, which uses a 2-elements-per-thread
//! layout so head_dim=64 still hits the tpg=32 minimum.
//!
//! ## DISPATCH INVARIANTS
//!
//! This kernel is reduction-mode and has STRICT threadgroup-geometry
//! requirements. Violating any of these silently miscomputes the
//! output (best case) or pins the GPU in an infinite loop (worst
//! case — see FFAI post-mortem 2026-05-19). Consumers MUST encode
//! these as preconditions in their wrappers.
//!
//! - **`N = TPG * 4`.** Each thread owns exactly 4 consecutive
//!   elements of the row, loaded unconditionally at offsets
//!   `tid*4 + {0..3}`. The wrapper computes `TPG = n / 4`.
//! - **`TPG` must be a multiple of 32** (one full Apple simdgroup).
//!   The cross-simdgroup combine reads `n_simd = TPG / 32` slots
//!   from threadgroup memory; with `TPG < 32` the combine reads
//!   zero everywhere and `tg_ssq` silently collapses to 0.
//! - **`TPG ≤ 1024`** (Apple's max-threads-per-threadgroup cap on
//!   M-series). Combined with `N = TPG*4`, this means `N ≤ 4096`;
//!   larger rows need the multi-row dispatch variant + chunking.
//! - **Combined**: `n` must be a multiple of 128 and `n ≤ 4096`.
//! - **Grid: 1 threadgroup per row.** Multi-row dispatch uses
//!   `grid = (nRows * TPG, 1, 1)`, `tg = (TPG, 1, 1)`; Metal slices
//!   that into `nRows` threadgroups of `TPG` threads each.

use metaltile::kernel;

/// Cross-kernel callee: threadgroup-wide RMS inverse.
///
/// Given each thread's pre-computed `partial_ssq` (sum of squares for its
/// slice of the row), reduces across the threadgroup and returns:
///
/// ```text
///   rsqrt(reduce_sum(partial_ssq) / n + eps)
/// ```
///
/// This kernel exists **only** as a cross-kernel callee. Kernels that fuse
/// RMSNorm with a second operation (residual add, RoPE, quantized GEMV) call
/// it via the DSL cross-kernel syntax so that the reduction + rsqrt body is
/// expressed once and inlined by `KernelInlinePass` rather than copy-pasted.
///
/// ## Calling convention
///
/// ```rust
/// // In the caller kernel body (after computing per-thread partial_ssq):
/// let inv_rms = mt_rms_inv_scalar(partial_ssq, eps_buf, n);
/// ```
///
/// - `partial_ssq` → `KernelCallArg::Value`: the callee's param-load is
///   replaced by the caller's pre-computed scalar. No memory round-trip.
/// - `eps_buf`, `n` → `KernelCallArg::Tensor`: the callee's loads are kept
///   but renamed to the caller's buffer/constexpr names, so the inlined code
///   reads the correct per-kernel eps and row length.
/// - The output param `out` receives no arg; its store is skipped and the
///   stored `inv_rms` value is returned as the call result.
///
/// ## Standalone vs inlined semantics
///
/// `mt_rms_inv_scalar` is a **valid standalone kernel**: `partial_ssq` is a
/// real 1-element `Tensor<f32>` and `load(partial_ssq[0u32])` is a legal
/// memory access. It can be dispatched directly (e.g. in tests) by passing a
/// 1-element buffer containing the pre-summed partial sum.
///
/// When called via the cross-kernel DSL (`let inv = mt_rms_inv_scalar(g, ...)`)
/// the caller passes `g` as a `KernelCallArg::Value` — a pre-computed scalar
/// already in registers. `KernelInlinePass` detects the `Value` arg, skips the
/// load, and substitutes `g` directly, eliminating the memory round-trip.
/// This is load-forwarding: the callee is correct both ways.
#[kernel]
pub fn mt_rms_inv_scalar(
    partial_ssq: Tensor<f32>,
    eps_buf: Tensor<f32>,
    mut out: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let v = load(partial_ssq[0u32]); // replaced by Value arg at inline time
    let tg_ssq = reduce_sum(v);
    let eps = load(eps_buf[0u32]);
    store(out[0u32], rsqrt(tg_ssq / n + eps));
}

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
    // Each thread owns exactly 4 consecutive elements (N = TPG * 4).
    // The wrapper enforces this — but as belt-and-braces (the original
    // 2026-05-19 freeze came from a wrong-TPG dispatch in a sibling
    // kernel), clamp the load base for OOB threads and mask their SSQ
    // contribution + skip their stores. Threads with `col >= n` re-read
    // row[0..3] (benign, since `partial_ssq` for them is forced to 0),
    // participate in `reduce_sum` (required — Apple simdgroup
    // primitives need all lanes active), and skip their stores so
    // they don't trample a neighbouring row.
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col; // only used inside the in_bounds-guarded store block.
    // Read x once, cache in registers, reuse for both ssq and output — 3 reads total.
    let x0 = load(x[safe_base]).cast::<f32>();
    let x1 = load(x[safe_base + 1u32]).cast::<f32>();
    let x2 = load(x[safe_base + 2u32]).cast::<f32>();
    let x3 = load(x[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    // Mask OOB lanes to 0 contribution so `mean(x²) = tg_ssq / n` stays
    // correct: in-bounds lanes contribute their real x² values, the
    // sum/n divisor is unchanged. Only valid when the wrapper has
    // ensured the in-bounds lanes cover the full row exactly once;
    // duplicate / missing coverage is a wrapper bug we can't repair here.
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
        store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
        store(out[base + 2u32], (x2 * rms * load(w[col + 2u32]).cast::<f32>()).cast::<T>());
        store(out[base + 3u32], (x3 * rms * load(w[col + 3u32]).cast::<f32>()).cast::<T>());
    }
}

/// Small-head RMSNorm — 2 consecutive elements per thread, so
/// `N = tpg * 2`. Covers per-head dispatch at head_dim ∈ {64, 128,
/// 192, 256} (head_dim=64 → tpg=32 hits the single-simdgroup
/// minimum that the 4-element variant misses). At head_dim ≥ 128
/// the 4-element [`mt_rms_norm`] has better ILP per lane and is
/// preferred; this variant exists to cover the small-head_dim
/// regime (older 7B-class architectures) without a dispatch-time
/// fallback.
///
/// Algorithm-identical to `mt_rms_norm`: f32 accumulator for the
/// sum-of-squares, threadgroup-wide `reduce_sum`, `rsqrt(ssq/n + eps)`
/// scaling, per-element output store rounded through `T`.
#[kernel]
pub fn mt_rms_norm_small<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // 2 elements per thread → tpg = n / 2. The minimum supported is
    // tpg = 32 (one full simdgroup) → n ≥ 64.
    let base = rs + tid * 2u32;
    let col = tid * 2u32;
    let x0 = load(x[base]).cast::<f32>();
    let x1 = load(x[base + 1u32]).cast::<f32>();
    let partial_ssq = x0 * x0 + x1 * x1;
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
    store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
}

/// Wide-row RMSNorm — handles rows wider than the 4096-element cap of
/// [`mt_rms_norm`]. Where `mt_rms_norm` fixes `N = TPG * 4` (so a
/// 1024-thread group tops out at 4096), this kernel has each thread
/// *stride* over the row in steps of one full threadgroup, so any `n`
/// is covered regardless of the threadgroup size. Needed for
/// large-hidden models (e.g. Gemma 4 31B, hidden 5376).
///
/// Two passes over device memory: pass 1 accumulates the strided
/// sum-of-squares and reduces it threadgroup-wide; pass 2 re-reads `x`
/// and writes the scaled output. The per-thread element count is
/// `ceil(n / TPG)` and varies with `n`, so the `x` values cannot be
/// held in registers across the reduction the way `mt_rms_norm` does
/// — hence the re-read. RMSNorm is memory-bound; the extra `x` read is
/// the price of unbounded `n`.
///
/// ## DISPATCH INVARIANTS
///
/// - **TPG a multiple of 32** (one full Apple simdgroup) so the
///   `reduce_sum` cross-simdgroup combine is well-defined. The wrapper
///   uses TPG = 1024. The stride is derived as `n_simd * 32`, so the
///   kernel is correct for any such TPG.
/// - **Grid: 1 threadgroup per row.** Multi-row dispatch uses
///   `grid = (nRows * TPG, 1, 1)`, `tg = (TPG, 1, 1)`.
/// - **`n` may be any positive value.** The strided loops bound on
///   `n`, so no `N = TPG * k` relationship is required; threads whose
///   stride walks past `n` simply stop. Unlike `mt_rms_norm` there is
///   no 128-alignment or `n ≤ 4096` requirement.
#[kernel]
pub fn mt_rms_norm_wide<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    // One full threadgroup of threads; every thread strides by this.
    let tpg = n_simd * 32u32;
    // Pass 1: strided sum-of-squares. A thread with `tid >= n` runs
    // zero iterations and contributes 0 — still required to reach
    // `reduce_sum` (Apple simdgroup reductions need all lanes active).
    let mut acc = 0.0f32;
    for i in range(tid, n, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        acc = acc + xi * xi;
    }
    let tg_ssq = reduce_sum(acc);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    // Pass 2: strided scaled store. `x` is re-read from device memory
    // (see the doc note above).
    for i in range(tid, n, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        let wi = load(w[i]).cast::<f32>();
        store(out[rs + i], (xi * rms * wi).cast::<T>());
    }
}

/// Fused gated-mixer-norm: `out = rms_norm(y, w) · silu(z)`. Per-row
/// across `[Hv, Dv]` — one row per threadgroup. Used by the FFAI
/// Qwen3.5 / Qwen3.6 GDN mixer's phase-2 step (`y` is the recurrence
/// output in fp32; `z` is the gate from `in_proj_z` in the model
/// dtype; `w` is `mixer.norm.weight`). Folding RMSNorm + weight +
/// `silu(z)` into one dispatch kills the host round-trip the legacy
/// path needed to compute this on the CPU between phases — 30 host
/// commit+waits per Qwen3.6-A3B decode token recovered.
///
/// Math (one row):
///   rms = rsqrt(mean(y²) + eps)
///   y_normed[i] = y[i] * rms * w[i]
///   silu(z)[i]  = z[i] / (1 + exp(-z[i]))
///   out[i] = y_normed[i] * silu(z)[i]
///
/// Same `N = TPG * 4` invariant as `mt_rms_norm` — Dv is multiple of
/// 4 on every shipped Qwen3 hybrid (128 / 256 / 512). One thread owns
/// 4 consecutive `Dv`-axis elements; the OOB clamp + mask copies the
/// `mt_rms_norm` template so a wrong-TPG dispatch fails loudly rather
/// than silently miscomputing.
#[kernel]
pub fn mt_gated_mixer_norm<T>(
    y: Tensor<f32>,
    z: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col;
    // y is already fp32, but mirror the mt_rms_norm load pattern
    // (`.cast::<f32>()` after each load) — the vectorize pass on this
    // codegen reads the cast as the consumer hook for the float4
    // load+extract emit. Removing the cast leaves the vectorize pass
    // half-finished (load merges into a float4, scalar y_n references
    // never get rewritten into VectorExtract — see emit + bug-report
    // in metaltile codegen `vectorize.rs`).
    let y0 = load(y[safe_base]).cast::<f32>();
    let y1 = load(y[safe_base + 1u32]).cast::<f32>();
    let y2 = load(y[safe_base + 2u32]).cast::<f32>();
    let y3 = load(y[safe_base + 3u32]).cast::<f32>();
    let raw_ssq = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        let w0 = load(w[col]).cast::<f32>();
        let w1 = load(w[col + 1u32]).cast::<f32>();
        let w2 = load(w[col + 2u32]).cast::<f32>();
        let w3 = load(w[col + 3u32]).cast::<f32>();
        let z0 = load(z[base]).cast::<f32>();
        let z1 = load(z[base + 1u32]).cast::<f32>();
        let z2 = load(z[base + 2u32]).cast::<f32>();
        let z3 = load(z[base + 3u32]).cast::<f32>();
        // silu(z) = z / (1 + exp(-z)). Inlined per the `mt_sigmoid`
        // precedent — Activation::Sigmoid folds into FusedElementwise
        // and the per-kernel feature analyzer would miss it, so the
        // emitted MSL stays self-contained without an `mt_sigmoid`
        // helper. Same as `mt_gated_delta_prep_step`'s `beta` path.
        let silu0 = z0 / (1.0f32 + exp(0.0f32 - z0));
        let silu1 = z1 / (1.0f32 + exp(0.0f32 - z1));
        let silu2 = z2 / (1.0f32 + exp(0.0f32 - z2));
        let silu3 = z3 / (1.0f32 + exp(0.0f32 - z3));
        store(out[base], ((y0 * rms * w0) * silu0).cast::<T>());
        store(out[base + 1u32], ((y1 * rms * w1) * silu1).cast::<T>());
        store(out[base + 2u32], ((y2 * rms * w2) * silu2).cast::<T>());
        store(out[base + 3u32], ((y3 * rms * w3) * silu3).cast::<T>());
    }
}

#[cfg(test)]
mod wide_tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::{DType, ir::KernelMode};

    use super::mt_rms_norm_wide;

    fn msl_for(dt: DType) -> String {
        let mut k = mt_rms_norm_wide::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("mt_rms_norm_wide codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void mt_rms_norm_wide"),
                "MSL for {dt:?} should declare mt_rms_norm_wide:\n{src}",
            );
        }
    }
}

mod tests_support {
    #![allow(unused, dead_code)]
    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
        ir::KernelMode,
    };

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn dt_round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
    }

    fn naive_rms_norm(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
        assert_eq!(x.len() % n, 0);
        assert_eq!(w.len(), n);
        let rows = x.len() / n;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let base = r * n;
            let ssq: f32 = x[base..base + n].iter().map(|v| v * v).sum();
            let rms = (ssq / n as f32 + eps).sqrt().recip();
            for d in 0..n {
                out[base + d] = x[base + d] * rms * w[d];
            }
        }
        out
    }

    fn make_rms_norm_setup(n: usize, rows: usize, eps: f32, dt: DType) -> TestSetup {
        let tpg = n / 4;
        let x: Vec<f32> =
            (0..rows * n).map(|i| dt_round(((i % 23) as f32 - 11.0) * 0.3, dt)).collect();
        let w: Vec<f32> = (0..n).map(|i| dt_round(1.0 + (i % 7) as f32 * 0.05, dt)).collect();
        let expected = naive_rms_norm(&x, &w, n, eps);
        let mut kernel = mt_rms_norm::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [tpg as u32, 1, 1])
    }

    fn make_rms_norm_small_setup(n: usize, rows: usize, eps: f32, dt: DType) -> TestSetup {
        let tpg = n / 2;
        let x: Vec<f32> = (0..rows * n)
            .map(|i| dt_round(0.5 + ((i % 17) as f32) * 0.03 - ((i % 11) as f32) * 0.02, dt))
            .collect();
        let w: Vec<f32> = (0..n).map(|i| dt_round(1.0 + (i % 13) as f32 * 0.01, dt)).collect();
        let expected = naive_rms_norm(&x, &w, n, eps);
        let mut kernel = mt_rms_norm_small::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [tpg as u32, 1, 1])
    }

    fn make_rms_norm_wide_setup(n: usize, rows: usize, eps: f32, dt: DType) -> TestSetup {
        const TPG: usize = 1024;
        let x: Vec<f32> =
            (0..rows * n).map(|i| dt_round(((i % 37) as f32 - 18.0) * 0.05, dt)).collect();
        let w: Vec<f32> = (0..n).map(|i| dt_round(1.0 + (i % 23) as f32 * 0.02, dt)).collect();
        let expected = naive_rms_norm(&x, &w, n, eps);
        let mut kernel = mt_rms_norm_wide::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("x", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [TPG as u32, 1, 1])
    }

    fn oracle_gated_mixer_norm(
        y: &[f32],
        z: &[f32],
        w: &[f32],
        hv: usize,
        dv: usize,
        eps: f32,
        dt: DType,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; hv * dv];
        for h in 0..hv {
            let base = h * dv;
            let mut ssq = 0.0f32;
            for i in 0..dv {
                let v = y[base + i];
                ssq += v * v;
            }
            let inv = 1.0 / (ssq / dv as f32 + eps).sqrt();
            for i in 0..dv {
                let normed = y[base + i] * inv * dt_round(w[i], dt);
                let zq = dt_round(z[base + i], dt);
                let silu = zq / (1.0 + (-zq).exp());
                out[base + i] = dt_round(normed * silu, dt);
            }
        }
        out
    }

    fn make_gated_mixer_norm_setup(hv: usize, dv: usize, eps: f32, dt: DType) -> TestSetup {
        let tpg = dv / 4;
        let y: Vec<f32> = ramp(hv * dv, 17, 8.0).iter().map(|v| 0.1 * v).collect();
        let z: Vec<f32> = ramp(hv * dv, 11, 5.0).iter().map(|v| 0.2 * v - 1.0).collect();
        let w: Vec<f32> = ramp(dv, 7, 3.0).iter().map(|v| 1.0 + 0.05 * v).collect();
        let expected = oracle_gated_mixer_norm(&y, &z, &w, hv, dv, eps, dt);
        let mut kernel = mt_gated_mixer_norm::kernel_ir_for(dt);
        kernel.mode = KernelMode::Reduction;
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("y", pack(&y, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("z", pack(&z, dt), dt))
            .input(TestBuffer::from_vec("w", pack(&w, dt), dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .constexpr("n", dv as u32)
            .grid_3d(hv as u32, 1, 1, [tpg as u32, 1, 1])
    }

    // ── rms_norm: n=128 (minimum), single row, f32 ────────────────────
    #[test_kernel(name = "mlx/rms_norm/n128_f32", dtypes = [f32], tol = 1e-4)]
    fn test_rms_norm_n128_f32(dt: DType) -> TestSetup { make_rms_norm_setup(128, 1, 1e-5, dt) }

    // ── rms_norm: n=512, 4 rows, f32 ─────────────────────────────────
    #[test_kernel(name = "mlx/rms_norm/n512_rows4_f32", dtypes = [f32], tol = 1e-4)]
    fn test_rms_norm_n512_rows4_f32(dt: DType) -> TestSetup {
        make_rms_norm_setup(512, 4, 1e-5, dt)
    }

    // ── rms_norm: n=4096 (max), single row, f32 ──────────────────────
    #[test_kernel(name = "mlx/rms_norm/n4096_f32", dtypes = [f32], tol = 5e-4)]
    fn test_rms_norm_n4096_f32(dt: DType) -> TestSetup { make_rms_norm_setup(4096, 1, 1e-5, dt) }

    // ── rms_norm_small: head_dim=64 (older 7B architectures), f32/f16/bf16 ─
    #[test_kernel(name = "mlx/rms_norm_small/hd64_f32", dtypes = [f32], tol = 1e-4)]
    fn test_rms_norm_small_hd64_f32(dt: DType) -> TestSetup {
        make_rms_norm_small_setup(64, 512, 1e-6, dt)
    }

    #[test_kernel(name = "mlx/rms_norm_small/hd64_f16", dtypes = [f16], tol = 5e-3)]
    fn test_rms_norm_small_hd64_f16(dt: DType) -> TestSetup {
        make_rms_norm_small_setup(64, 512, 1e-6, dt)
    }

    #[test_kernel(name = "mlx/rms_norm_small/hd64_bf16", dtypes = [bf16], tol = 5e-2)]
    fn test_rms_norm_small_hd64_bf16(dt: DType) -> TestSetup {
        make_rms_norm_small_setup(64, 512, 1e-6, dt)
    }

    // ── rms_norm: head_dim=128 (Qwen3-class), f32/f16/bf16 ───────────
    #[test_kernel(name = "mlx/rms_norm/hd128_f32", dtypes = [f32], tol = 1e-4)]
    fn test_rms_norm_hd128_f32(dt: DType) -> TestSetup { make_rms_norm_setup(128, 1024, 1e-6, dt) }

    #[test_kernel(name = "mlx/rms_norm/hd128_f16", dtypes = [f16], tol = 5e-3)]
    fn test_rms_norm_hd128_f16(dt: DType) -> TestSetup { make_rms_norm_setup(128, 1024, 1e-6, dt) }

    #[test_kernel(name = "mlx/rms_norm/hd128_bf16", dtypes = [bf16], tol = 5e-2)]
    fn test_rms_norm_hd128_bf16(dt: DType) -> TestSetup { make_rms_norm_setup(128, 1024, 1e-6, dt) }

    // ── rms_norm: head_dim=256 (Gemma-2/3, Phi-3-medium), f32/f16/bf16
    #[test_kernel(name = "mlx/rms_norm/hd256_f32", dtypes = [f32], tol = 1e-4)]
    fn test_rms_norm_hd256_f32(dt: DType) -> TestSetup { make_rms_norm_setup(256, 256, 1e-6, dt) }

    #[test_kernel(name = "mlx/rms_norm/hd256_f16", dtypes = [f16], tol = 5e-3)]
    fn test_rms_norm_hd256_f16(dt: DType) -> TestSetup { make_rms_norm_setup(256, 256, 1e-6, dt) }

    #[test_kernel(name = "mlx/rms_norm/hd256_bf16", dtypes = [bf16], tol = 5e-2)]
    fn test_rms_norm_hd256_bf16(dt: DType) -> TestSetup { make_rms_norm_setup(256, 256, 1e-6, dt) }

    // ── rms_norm_wide: Gemma 4 31B hidden (n=5376), single row ────────
    #[test_kernel(name = "mlx/rms_norm_wide/n5376_f32", dtypes = [f32], tol = 5e-4)]
    fn test_rms_norm_wide_n5376_f32(dt: DType) -> TestSetup {
        make_rms_norm_wide_setup(5376, 1, 1e-6, dt)
    }

    // ── rms_norm_wide: n=5376, 3 rows ─────────────────────────────────
    #[test_kernel(name = "mlx/rms_norm_wide/n5376_rows3_f32", dtypes = [f32], tol = 5e-4)]
    fn test_rms_norm_wide_n5376_rows3_f32(dt: DType) -> TestSetup {
        make_rms_norm_wide_setup(5376, 3, 1e-6, dt)
    }

    // ── rms_norm_wide: n=8192 (exact TPG multiple) ────────────────────
    #[test_kernel(name = "mlx/rms_norm_wide/n8192_f32", dtypes = [f32], tol = 5e-4)]
    fn test_rms_norm_wide_n8192_f32(dt: DType) -> TestSetup {
        make_rms_norm_wide_setup(8192, 1, 1e-6, dt)
    }

    // ── gated_mixer_norm: Qwen3.6-A3B (Hv=32, Dv=128), f32/f16/bf16 ─
    #[test_kernel(name = "mlx/gated_mixer_norm/qwen36_f32", dtypes = [f32], tol = 1e-5)]
    fn test_gated_mixer_norm_qwen36_f32(dt: DType) -> TestSetup {
        make_gated_mixer_norm_setup(32, 128, 1e-5, dt)
    }

    #[test_kernel(name = "mlx/gated_mixer_norm/qwen36_f16", dtypes = [f16], tol = 5e-4)]
    fn test_gated_mixer_norm_qwen36_f16(dt: DType) -> TestSetup {
        make_gated_mixer_norm_setup(32, 128, 1e-5, dt)
    }

    #[test_kernel(name = "mlx/gated_mixer_norm/qwen36_bf16", dtypes = [bf16], tol = 5e-3)]
    fn test_gated_mixer_norm_qwen36_bf16(dt: DType) -> TestSetup {
        make_gated_mixer_norm_setup(32, 128, 1e-5, dt)
    }

    // ── gated_mixer_norm: wider Dv=256 ────────────────────────────────
    #[test_kernel(name = "mlx/gated_mixer_norm/dv256_bf16", dtypes = [bf16], tol = 5e-3)]
    fn test_gated_mixer_norm_dv256_bf16(dt: DType) -> TestSetup {
        make_gated_mixer_norm_setup(8, 256, 1e-5, dt)
    }
}
