//! 4-row-batched dense GEMV for bf16/f16/f32 decode (BM=4).
//!
//! Processes 4 output rows per threadgroup, sharing one load of the input
//! vector across all four dot products. For single-token decode the input
//! activation (`vec`) is small — `hidden_dim × dtype_size` bytes (e.g. 4KB
//! for Llama-3.2-1B bf16). The single-row `mt_gemv` loads that vector once
//! per row, so for N output rows the total vec bandwidth is N × 4KB. BM4
//! loads it once per tile-of-4, giving a **4× reduction in activation
//! bandwidth** — the dominant bottleneck for bandwidth-limited decode.
//!
//! ## Algorithm
//!
//! Each threadgroup covers one tile of 4 consecutive output rows
//! `[r0, r0+1, r0+2, r0+3]`. Each of the `TPG` threads iterates over the
//! `k`-element dot product in steps of `lsize` (= TPG), loading
//! `vec[col]` once and multiplying it against the four corresponding weight
//! elements before advancing to the next column batch:
//!
//! ```text
//! for col in (tid, tid+TPG, tid+2*TPG, ...) < k:
//!   v    = vec[col]
//!   acc0 += mat[r0 * k + col] * v
//!   acc1 += mat[r1 * k + col] * v
//!   acc2 += mat[r2 * k + col] * v
//!   acc3 += mat[r3 * k + col] * v
//! ```
//!
//! After the loop, four `reduce_sum` calls (each triggering a threadgroup
//! barrier) reduce each accumulator to a scalar.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel.
//!
//! - **Grid: `[ceil(rows / 4), 1, 1]` threadgroups.**
//!   `program_id::<0>()` = tile index; tile × 4 = r0.
//! - **`TPG` a multiple of 32 and ≤ 1024.**  The `reduce_sum` threadgroup
//!   reduction allocates `ceil(TPG/32)` simdgroup slots; `TPG > 1024`
//!   overflows the 32-slot buffer.
//! - **`k` a multiple of `TPG`.**  If not, the `if col < k` guard handles
//!   the tail safely, but ILP is slightly reduced in the last iteration.
//! - **`rows` accurate.**  The constexpr is used to bounds-check the last
//!   tile; if `rows` is wrong the last partial tile stores to wrong
//!   addresses.  For Llama-family models all dims are multiples of 4, so
//!   the last tile is never partial.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Dense GEMV with 4 output rows per threadgroup.
///
/// `out[r] = mat[r, :] · vec[:]` for `r` in `[tile*4, tile*4+4)`.
///
/// Rows beyond `rows` in the last tile are skipped (no OOB write).
#[kernel]
pub fn ffai_gemv_bm4<T>(
    mat: Tensor<T>,
    vec: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] rows: u32,
) {
    let tile = program_id::<0>();
    let r0 = tile * 4u32;
    let r1 = r0 + 1u32;
    let r2 = r0 + 2u32;
    let r3 = r0 + 3u32;

    // Clamp OOB row indices for the last partial tile so mat loads stay
    // within the allocated buffer.  Accumulated values for clamped rows
    // are never stored (the if-r<rows guards below skip them).
    let r1_safe = select(r1 < rows, r1, r0);
    let r2_safe = select(r2 < rows, r2, r0);
    let r3_safe = select(r3 < rows, r3, r0);

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let n_iters = (k + lsize - 1u32) / lsize;
    for _iter in range(0u32, n_iters, 1u32) {
        let col = _iter * lsize + tid;
        if col < k {
            // Load the input vector element once; share across all 4 rows.
            let v = load(vec[col]).cast::<f32>();
            acc0 = acc0 + load(mat[r0 * k + col]).cast::<f32>() * v;
            acc1 = acc1 + load(mat[r1_safe * k + col]).cast::<f32>() * v;
            acc2 = acc2 + load(mat[r2_safe * k + col]).cast::<f32>() * v;
            acc3 = acc3 + load(mat[r3_safe * k + col]).cast::<f32>() * v;
        }
    }

    // Cross-threadgroup reduction — one barrier per row (4 total).
    let res0 = reduce_sum(acc0);
    let res1 = reduce_sum(acc1);
    let res2 = reduce_sum(acc2);
    let res3 = reduce_sum(acc3);

    if tid == 0u32 {
        store(out[r0], res0.cast::<T>());
        if r1 < rows {
            store(out[r1], res1.cast::<T>());
        }
        if r2 < rows {
            store(out[r2], res2.cast::<T>());
        }
        if r3 < rows {
            store(out[r3], res3.cast::<T>());
        }
    }
}

inventory::submit! {
    BenchSpec {
        op: "gemv",
        subop: "bm4",
        kernel_name: "ffai_gemv_bm4",
        kernel_ir: ffai_gemv_bm4::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-2,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::ffai_gemv_bm4;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_gemv_bm4::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_gemv_bm4 codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_gemv_bm4"),
                "MSL for {dt:?} should declare ffai_gemv_bm4:\n{src}",
            );
        }
    }

    #[test]
    fn msl_contains_four_accumulators() {
        let src = msl_for(DType::F16);
        assert!(src.contains("acc0"), "should have acc0");
        assert!(src.contains("acc1"), "should have acc1");
        assert!(src.contains("acc2"), "should have acc2");
        assert!(src.contains("acc3"), "should have acc3");
    }
}
