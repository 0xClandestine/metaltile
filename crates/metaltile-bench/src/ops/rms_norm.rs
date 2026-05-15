//! RMS normalization benchmark — #[kernel] DSL vs MLX metal/rms_norm.metal
//!
//! MLX kernel: rmsfloat32 / rmsfloat16 / rmsbfloat16 (rms_norm.metal)
//!   Params: (x: device T*, w: device T*, out: device T*, axis_size: constant uint&,
//!            w_stride: constant uint&) — slots [0, 1, 2, 3, 4]
//!   Function constant: slot 20 = true  (needed to compile the kernel variant)
//!   Grid: [B, 1, 1] × [1024, 1, 1]  (one threadgroup per row)
//!   Algorithm: per-row: sum-of-squares → reduce → rsqrt(mean_sq + eps) → scale by weight.
//!
//! MetalTile: mt_rms_norm — stride-reduce sum-of-squares + N_READS=4 write-back.
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
        run_typed_once,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/rms_norm.metal");
const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
const BENCH: OpBench = OpBench::new("rms_norm", "GB/s");
const CHECK_B: usize = 4;
const CHECK_N: usize = 4_096;
const REF_TPG: usize = 1_024;
const MT_TPG: usize = 1024;

/// RMS norm with N_READS=4 write-back:
///   sum-of-squares via stride-reduce (N_READS=4) → reduce → rsqrt(mean_sq + eps)
///   write-back: N_READS=4 loop reads x + w, writes out.
///
/// Dispatch: [B, 1, 1] x [1024, 1, 1]  — matches MLX REF_TPG for max parallelism
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
    let re = rs + n;
    let ssq = strided_reduce_dot(x, x, rs, 0, re);
    let tg_ssq = reduce_sum(ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    let n_full = n / (lsize * 4u32);
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = load(x[base]).cast::<f32>() * rms * load(w[col]).cast::<f32>();
        let n1 = load(x[base + 1u32]).cast::<f32>() * rms * load(w[col + 1u32]).cast::<f32>();
        let n2 = load(x[base + 2u32]).cast::<f32>() * rms * load(w[col + 2u32]).cast::<f32>();
        let n3 = load(x[base + 3u32]).cast::<f32>() * rms * load(w[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let ni = load(x[_i]).cast::<f32>() * rms * load(w[_i - rs]).cast::<f32>();
        store(out[_i], ni.cast::<T>());
    }
}

fn rms_norm_msl_for(dt: DType) -> String {
    generate_reduction_msl(|| mt_rms_norm::kernel_ir_for(dt), "rms_norm")
}

pub fn bench_rms_norm(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_rms_norm_for)
}

fn bench_rms_norm_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::reduce(dt);
    let (tn, dlabel, eb, tol) = (ctx.tn, ctx.label, ctx.eb, ctx.tol);

    let mt_msl = rms_norm_msl_for(dt);
    let mk = runner.compile(&mt_msl, "mt_rms_norm").ok();
    // has_w [[function_constant(20)]] = true; lsize = N/N_READS = 4096/4 = 1024
    let rk = runner.compile_with_bool_constants(SRC, &format!("rms{tn}"), &[(20, true)]).ok();

    // Correctness check
    let x_vals: Vec<f32> = (0..CHECK_B * CHECK_N)
        .map(|i| {
            let row = i / CHECK_N;
            let col = i % CHECK_N;
            ((col % 29) as f32 - 14.0) * 0.03125 + row as f32 * 0.125
        })
        .collect();
    let w_vals: Vec<f32> = (0..CHECK_B * CHECK_N)
        .map(|i| {
            let row = i / CHECK_N;
            let col = i % CHECK_N;
            0.5 + row as f32 * 0.0625 + (col % 17) as f32 * 0.03125
        })
        .collect();
    let x = buffer_typed(runner, &x_vals, dt);
    let w = buffer_typed(runner, &w_vals, dt);
    let eps = runner.buffer_f32_scalar(1e-6_f32);
    let ns = runner.buffer_u32(CHECK_N as u32);
    let w_stride = runner.buffer_u32(1u32);

    let ref_check = rk.as_ref().map(|rk| {
        let ref_out = zeros_typed(runner, CHECK_B * CHECK_N, dt);
        run_typed_once(
            runner,
            rk,
            &[&x, &w, &ref_out, &eps, &ns, &w_stride],
            &ref_out,
            CHECK_B * CHECK_N,
            [CHECK_B, 1, 1],
            [REF_TPG, 1, 1],
            dt,
        )
    });
    let mt_check = mk.as_ref().map(|mk| {
        let mt_out = zeros_typed(runner, CHECK_B * CHECK_N, dt);
        run_typed_once(
            runner,
            mk,
            &[&x, &w, &mt_out, &eps, &ns],
            &mt_out,
            CHECK_B * CHECK_N,
            [CHECK_B, 1, 1],
            [MT_TPG, 1, 1],
            dt,
        )
    });
    let equiv = match (ref_check, mt_check) {
        (Some(r), Some(m)) => check_equiv(&r, &m, tol),
        (None, Some(_)) | (_, None) => return vec![],
    };

    let mut results = Vec::new();
    for &(b, n) in SHAPES {
        let xp = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
        let wp = buffer_typed(runner, &vec![1.0f32; n], dt);
        let eps_p = runner.buffer_f32_scalar(1e-6_f32);
        let ns_p = runner.buffer_u32(n as u32);
        let bytes = (b * n * eb * 2) as f64; // read x + write out (w is small)
        let ref_w_stride = runner.buffer_u32(1u32);

        let ref_perf = rk.as_ref().and_then(|r| {
            let out = zeros_typed(runner, b * n, dt);
            bench_gbps(
                runner,
                r,
                &[&xp, &wp, &out, &eps_p, &ns_p, &ref_w_stride],
                [b, 1, 1],
                [1024, 1, 1],
                bytes,
            )
        });
        let mt_perf = mk.as_ref().and_then(|m| {
            let out = zeros_typed(runner, b * n, dt);
            bench_gbps(runner, m, &[&xp, &wp, &out, &eps_p, &ns_p], [b, 1, 1], [256, 1, 1], bytes)
        });
        let shape = format!("B={b} N={n} {dlabel}");
        results.push(BENCH.result(shape, ref_perf, mt_perf, Some(equiv)));
    }
    results
}

crate::bench_tests!(msl_fn: rms_norm_msl_for, kernel_name: "mt_rms_norm");

use crate::ops::{FLOAT_DTYPE_STRS, KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "rms_norm",
        mt_kernel: "mt_rms_norm".into(),
        metal_file: "rms_norm.metal",
        ref_spec: RefSpec::Format("rms{tn}"),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
