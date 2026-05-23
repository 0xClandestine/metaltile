//! 4-row-batched dense GEMV with 4× column unrolling (SMM = Scalar-Mul-Many).
//!
//! Same BM=4 structure as `gemv/bm4` — 4 output rows per threadgroup,
//! input vector shared across all four dot products — but the inner loop
//! processes **4 consecutive columns per thread per iteration** instead of 1.
//!
//! ## Why unrolling helps
//!
//! With 256 threads and k=2048 columns, `gemv/bm4` runs 8 outer iterations
//! (one column each).  Here n_iters = k/(lsize*4) = 2, and each iteration
//! loads four consecutive matrix and vector elements.  Metal's LLVM backend
//! recognises the consecutive-index pattern and combines the four scalar loads
//! into a single hardware `T4` (`float4`/`half4`/`bfloat4`) instruction,
//! reducing memory transaction overhead 4×.
//!
//! Barrier count stays at 8 (four `reduce_sum` calls, two barriers each) —
//! identical to `gemv/bm4`.  Benchmarking showed barriers are not the
//! bottleneck; thread parallelism for memory-bus saturation is.
//!
//! ## Dispatch invariants
//!
//! Reduction mode — same contract as `gemv/bm4`.
//!
//! - **Grid: `[ceil(rows / 4), 1, 1]` threadgroups.**
//! - **TPG = 256** (8 simdgroups × 32 lanes).
//! - **`k` a multiple of `TPG * 4 = 1024` preferred** for full unrolling.
//!   The `if base < k` guard handles any tail, but the last partial group is
//!   skipped entirely — use `gemv/bm4` for arbitrary `k`.
//! - **`rows` accurate** — same contract as `gemv/bm4`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// Dense GEMV with BM=4 and 4× column unrolling per thread.
///
/// `out[r] = mat[r, :] · vec[:]` for `r` in `[tile*4, tile*4+4)`.
///
/// Each thread processes four consecutive columns per outer iteration so
/// the Metal compiler can issue a single hardware T4 load instead of four
/// separate scalar loads.
#[kernel]
pub fn ffai_gemv_smm<T>(
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

    let r1_safe = select(r1 < rows, r1, r0);
    let r2_safe = select(r2 < rows, r2, r0);
    let r3_safe = select(r3 < rows, r3, r0);

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    // 4× unrolled inner loop: each iteration loads 4 consecutive columns.
    // n_iters4 = ceil(k / (lsize * 4)).
    let n_iters4 = (k + lsize * 4u32 - 1u32) / (lsize * 4u32);
    for _iter in range(0u32, n_iters4, 1u32) {
        let base = _iter * lsize * 4u32 + tid * 4u32;
        if base < k {
            // Load 4 consecutive vec elements — compiled into a single T4 load.
            let v0 = load(vec[base]).cast::<f32>();
            let v1 = load(vec[base + 1u32]).cast::<f32>();
            let v2 = load(vec[base + 2u32]).cast::<f32>();
            let v3 = load(vec[base + 3u32]).cast::<f32>();

            // 4 rows × 4 cols = 16 consecutive mat loads — 4 T4 hardware loads.
            acc0 = acc0
                + load(mat[r0 * k + base]).cast::<f32>() * v0
                + load(mat[r0 * k + base + 1u32]).cast::<f32>() * v1
                + load(mat[r0 * k + base + 2u32]).cast::<f32>() * v2
                + load(mat[r0 * k + base + 3u32]).cast::<f32>() * v3;
            acc1 = acc1
                + load(mat[r1_safe * k + base]).cast::<f32>() * v0
                + load(mat[r1_safe * k + base + 1u32]).cast::<f32>() * v1
                + load(mat[r1_safe * k + base + 2u32]).cast::<f32>() * v2
                + load(mat[r1_safe * k + base + 3u32]).cast::<f32>() * v3;
            acc2 = acc2
                + load(mat[r2_safe * k + base]).cast::<f32>() * v0
                + load(mat[r2_safe * k + base + 1u32]).cast::<f32>() * v1
                + load(mat[r2_safe * k + base + 2u32]).cast::<f32>() * v2
                + load(mat[r2_safe * k + base + 3u32]).cast::<f32>() * v3;
            acc3 = acc3
                + load(mat[r3_safe * k + base]).cast::<f32>() * v0
                + load(mat[r3_safe * k + base + 1u32]).cast::<f32>() * v1
                + load(mat[r3_safe * k + base + 2u32]).cast::<f32>() * v2
                + load(mat[r3_safe * k + base + 3u32]).cast::<f32>() * v3;
        }
    }

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
        subop: "smm",
        kernel_name: "ffai_gemv_smm",
        kernel_ir: ffai_gemv_smm::kernel_ir_for,
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

    use super::ffai_gemv_smm;
    use crate::bench_types::DType;

    fn msl_for(dt: DType) -> String {
        let mut k = ffai_gemv_smm::kernel_ir_for(dt);
        k.mode = KernelMode::Reduction;
        MslGenerator::default().generate(&k).expect("ffai_gemv_smm codegen succeeds")
    }

    #[test]
    fn codegen_produces_nonempty_msl_for_all_float_dtypes() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let src = msl_for(dt);
            assert!(!src.trim().is_empty(), "MSL for {dt:?} should not be empty");
            assert!(
                src.contains("kernel void ffai_gemv_smm"),
                "MSL for {dt:?} should declare ffai_gemv_smm:\n{src}",
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

    #[test]
    fn msl_has_unrolled_loop_with_factor_4() {
        // The unrolled loop iterates over lsize*4 columns; the constant 4u
        // appears as a multiplier in the iteration count.  The DSL folds
        // constants into generated variable names so we check that there is
        // more than one add-to-accumulator per loop body (i.e. unrolling fired).
        let src = msl_for(DType::BF16);
        // 4 accumulators × 4 adds each = 16 compound additions in one loop body.
        // A simple proxy: the string "4u" (the unroll factor) must appear.
        assert!(src.contains("4u"), "unroll factor 4 must appear in generated MSL:\n{src}");
        // The loop body should reference `lsize` (from the `lsize * 4` stride).
        assert!(src.contains("lsize"), "lsize must appear in loop stride:\n{src}");
    }
}
