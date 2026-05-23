//! Head-dimension-batched dense GEMV for bf16/f16/f32 decode (BM = head_dim).
//!
//! Processes one head per threadgroup. Each TG produces exactly `bm`
//! output elements (one head's worth), iterating over the `bm` output rows
//! sequentially. Unlike `gemv/bm4` which hardcodes 4 accumulators, this
//! kernel uses a single accumulator per row, which is fine because `bm`
//! (head_dim) is typically 64–128 and the inner K dot product dominates.
//!
//! The BM = head_dim layout is required for q_chain fusion with RoPE
//! (Pattern 6): RoPE reads two elements `half_dim` apart from the same
//! head, which with BM = head_dim are produced by the same TG.
//!
//! ## Dispatch invariants
//!
//! Reduction mode. Grid: `[rows, 1, 1]` threadgroups where `rows = n_heads`
//! (one TG per head). TPG a multiple of 32. `k` a multiple of TPG preferred.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Dense GEMV with BM = head_dim (one head per threadgroup).
///
/// `out[base + row] = mat[base + row, :] · vec[:]` for `row` in `0..bm`,
/// where `base = program_id::<0>() * bm`.
#[kernel]
pub fn ffai_gemv_bm_hd<T>(
    mat: Tensor<T>,
    vec: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
    #[constexpr] rows: u32,
    #[constexpr] bm: u32,
) {
    let tile = program_id::<0>();
    let base = tile * bm;

    let n_iters = (k + lsize - 1u32) / lsize;

    for _row in range(0u32, bm, 1u32) {
        let r = base + _row;

        let mut acc = 0.0f32;
        for _c in range(0u32, n_iters, 1u32) {
            let col = _c * lsize + tid;
            if col < k {
                let v = load(vec[col]).cast::<f32>();
                acc = acc + load(mat[r * k + col]).cast::<f32>() * v;
            }
        }

        let res = reduce_sum(acc);

        if tid == 0u32 {
            store(out[r], res.cast::<T>());
        }
    }
}

inventory::submit! {
    BenchSpec {
        op: "gemv",
        subop: "bm_hd",
        kernel_name: "ffai_gemv_bm_hd",
        kernel_ir: ffai_gemv_bm_hd::kernel_ir_for,
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

    use super::ffai_gemv_bm_hd;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_gemv_bm_hd::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_gemv_bm_hd codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_gemv_bm_hd"),
                "MSL for {dt:?} should declare ffai_gemv_bm_hd:\n{src}",
            );
        }
    }

    #[test]
    fn msl_contains_outer_row_loop() {
        let src = msl_for(DType::F16);
        let loop_count = src.matches("for (uint ").count();
        assert!(loop_count >= 2, "should have outer row loop and inner k loop: {src}");
    }

    #[test]
    fn msl_contains_reduce_sum() {
        let src = msl_for(DType::F32);
        assert!(src.contains("simd_sum"), "should use simd_sum for reduction");
    }
}
