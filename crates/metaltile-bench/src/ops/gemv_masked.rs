//! Masked GEMV benchmark — #[kernel] DSL (no MLX reference)
//!
//! MLX reference: unavailable — MLX gemv_masked has no nomask/nomask variant;
//!   all variants require explicit out_mask/op_mask buffers with stride arrays.
//!   Algorithm: GEMV with element-wise mask applied: out[row] = Σ mat[row*K+i]*vec[i]*mask[i]
//!
//! MetalTile: mt_gemv_masked — per-row reduction with mask multiply via #[kernel] DSL.
//!   KernelMode::Reduction

use metaltile::kernel;

use crate::{
    ops::{
        DType,
        DtypeCtx,
        OpBench,
        OpResult,
        bench_all_dtypes,
        bench_gbps,
        buffer_typed,
        check_equiv,
        generate_reduction_msl,
        quantize_roundtrip,
        run_typed_once,
        zeros_typed,
    },
    runner::GpuRunner,
};

const BENCH: OpBench = OpBench::new("gemv_masked", "GB/s");
const SHAPES: &[(usize, usize)] = &[(4096, 4096)];
const TPG: usize = 256;
const N_CHECK_M: usize = 64;
const N_CHECK_K: usize = 256;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Masked GEMV: accumulate mat[row*k + i] * vec[i] * mask[i] for i in 0..k.
/// One threadgroup per output row; threads cooperatively stride over K.
#[kernel]
pub fn mt_gemv_masked<T>(
    mat: Tensor<T>,
    vec: Tensor<T>,
    mask: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] k: u32,
) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let mut acc = 0.0f32;
    for _i in range(rs + tid, re, lsize) {
        let col = _i - rs;
        let m_val = load(mask[col]).cast::<f32>();
        acc = acc + load(mat[_i]).cast::<f32>() * load(vec[col]).cast::<f32>() * m_val;
    }
    let result = reduce_sum(acc);
    store(out[row], result.cast::<T>());
}

// ── Bench ─────────────────────────────────────────────────────────────────────

fn gemv_masked_msl_for(dt: DType) -> String {
    generate_reduction_msl(|| mt_gemv_masked::kernel_ir_for(dt), "gemv_masked")
}

fn cpu_gemv_masked(mat: &[f32], vec: &[f32], mask: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    for row in 0..m {
        let base = row * k;
        out[row] =
            (0..k).filter(|&col| mask[col] != 0.0).map(|col| mat[base + col] * vec[col]).sum();
    }
    out
}

pub fn bench_gemv_masked(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_gemv_masked_for)
}

fn bench_gemv_masked_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let (dlabel, eb) = (ctx.label, ctx.eb);
    let tol = ctx.tol.max(1e-2);

    let msl = gemv_masked_msl_for(dt);
    let mk = runner.compile(&msl, "mt_gemv_masked").ok();

    let mut results = Vec::new();
    for &(m, k) in SHAPES {
        // ── Correctness ──────────────────────────────────────────────────────
        let equiv = mk.as_ref().map(|mk| {
            let cm = N_CHECK_M;
            let ck = N_CHECK_K;
            let sm: Vec<f32> = (0..cm * ck).map(|i| (i % 13) as f32 * 0.01).collect();
            let sv: Vec<f32> = (0..ck).map(|i| (i % 7) as f32 * 0.01).collect();
            let mask_vals: Vec<f32> = (0..ck).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();

            let sm_q = quantize_roundtrip(&sm, dt);
            let sv_q = quantize_roundtrip(&sv, dt);
            let ref_out = cpu_gemv_masked(&sm_q, &sv_q, &mask_vals, cm, ck);

            let mat_b = buffer_typed(runner, &sm, dt);
            let vec_b = buffer_typed(runner, &sv, dt);
            let mask_b = buffer_typed(runner, &mask_vals, dt);
            let out_b = zeros_typed(runner, cm, dt);
            let k_b = runner.buffer_u32(ck as u32);

            let mt_vals = run_typed_once(
                runner,
                mk,
                &[&mat_b, &vec_b, &mask_b, &out_b, &k_b],
                &out_b,
                cm,
                [cm, 1, 1],
                [TPG, 1, 1],
                dt,
            );
            check_equiv(&ref_out, &mt_vals, tol)
        });

        // ── Perf ─────────────────────────────────────────────────────────────
        let mat_vals: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let vec_vals: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.01).collect();
        let mask_perf: Vec<f32> = (0..k).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();

        let mat_buf = buffer_typed(runner, &mat_vals, dt);
        let vec_buf = buffer_typed(runner, &vec_vals, dt);
        let mask_buf = buffer_typed(runner, &mask_perf, dt);
        let k_buf = runner.buffer_u32(k as u32);

        // MLX gemv_masked has no nomask/nomask variant; all variants require explicit mask
        // buffers with complex strides — no direct comparable reference available.
        let bytes = (m * k * eb + k * eb * 2 + m * eb) as f64; // mat + vec + mask + out
        let ref_perf: Option<f64> = None;

        let mt_out = zeros_typed(runner, m, dt);
        let mt_perf = mk.as_ref().and_then(|mk| {
            bench_gbps(
                runner,
                mk,
                &[&mat_buf, &vec_buf, &mask_buf, &mt_out, &k_buf],
                [m, 1, 1],
                [TPG, 1, 1],
                bytes,
            )
        });

        let shape = format!("M={m} K={k} {dlabel}");
        results.push(BENCH.result(shape, ref_perf, mt_perf, equiv));
    }
    results
}

crate::bench_tests!(msl_fn: gemv_masked_msl_for, kernel_name: "mt_gemv_masked");

use crate::ops::{FLOAT_DTYPE_STRS, KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "gemv_masked",
        mt_kernel: "mt_gemv_masked".into(),
        metal_file: "gemv_masked.metal",
        ref_spec: RefSpec::None(
            "no nomask/nomask variant in instantiate_gemv_base;              all MLX variants require explicit mask buffers",
        ),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
