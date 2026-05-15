//! Layer normalization benchmark — #[kernel] DSL vs MLX metal/layer_norm.metal
//!
//! MLX kernel: layer_norm_loopedfloat32 / ...float16 / ...bfloat16 (layer_norm.metal)
//!   Params: (x: device T*, w: device T*, b: device T*, out: device T*,
//!            axis_size: constant uint&) — slots [0, 1, 2, 3, 4]
//!   Grid: [B, 1, 1] × [256, 1, 1]  (one threadgroup per row)
//!   Algorithm: 2-pass per-row normalization: (1) mean + variance via strided reduce,
//!              (2) write-back: (x - mean) / sqrt(var + eps) * weight + bias.
//!
//! MetalTile: mt_layer_norm — same 2-pass algorithm via #[kernel] DSL.
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

static SRC: &str = include_str!("../metal/layer_norm.metal");
const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
const BENCH: OpBench = OpBench::new("layer_norm", "GB/s");
const CHECK_B: usize = 2;
const CHECK_N: usize = 512;
const REF_TPG: usize = 256;
const MT_TPG: usize = 1024; // more parallelism; threadgroup buf supports 32 simdgroups

/// Layer norm: single-pass stats (reads x once), N_READS=4 write-back.
///   mean = sum(x) / n,  variance = E[x²] − E[x]²
///   out[i] = (x[i] - mean) * rsqrt(variance + eps) * w[i] + b[i]
///
/// Stats: one combined loop accumulates sum and sum_sq simultaneously (N_READS=4
/// for full 256*4 chunks, N_READS=1 for remainder).  Two separate reduce_sum calls
/// then compute s_total and sq_total.
/// Write-back: N_READS=4 loop reads x, w, b and writes out.
/// Dispatch: [B, 1, 1] × [256, 1, 1]
#[kernel]
pub fn mt_layer_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    // Single-pass stats: accumulate sum and sum_sq in one loop (reads x once).
    // N_READS=4 for full lsize*4 chunks + N_READS=1 remainder for correctness.
    let n_full = n / (lsize * 4u32);
    let mut s_acc = 0.0f32;
    let mut sq_acc = 0.0f32;
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(x[base]).cast::<f32>();
        let v1 = load(x[base + 1u32]).cast::<f32>();
        let v2 = load(x[base + 2u32]).cast::<f32>();
        let v3 = load(x[base + 3u32]).cast::<f32>();
        s_acc = s_acc + v0 + v1 + v2 + v3;
        sq_acc = sq_acc + v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        s_acc = s_acc + xi;
        sq_acc = sq_acc + xi * xi;
    }
    let s_total = reduce_sum(s_acc);
    let sq_total = reduce_sum(sq_acc);
    let mean = s_total / n;
    let variance = sq_total / n - mean * mean;
    let eps = load(eps_buf[0]);
    let inv_std = rsqrt(variance + eps);
    // Write-back: N_READS=4 for full chunks + N_READS=1 remainder.
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = (load(x[base]).cast::<f32>() - mean) * inv_std * load(w[col]).cast::<f32>()
            + load(b[col]).cast::<f32>();
        let n1 = (load(x[base + 1u32]).cast::<f32>() - mean)
            * inv_std
            * load(w[col + 1u32]).cast::<f32>()
            + load(b[col + 1u32]).cast::<f32>();
        let n2 = (load(x[base + 2u32]).cast::<f32>() - mean)
            * inv_std
            * load(w[col + 2u32]).cast::<f32>()
            + load(b[col + 2u32]).cast::<f32>();
        let n3 = (load(x[base + 3u32]).cast::<f32>() - mean)
            * inv_std
            * load(w[col + 3u32]).cast::<f32>()
            + load(b[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let xi = load(x[_i]).cast::<f32>();
        let ci = _i - rs;
        let norm = (xi - mean) * inv_std * load(w[ci]).cast::<f32>() + load(b[ci]).cast::<f32>();
        store(out[_i], norm.cast::<T>());
    }
}

fn layer_norm_msl_for(dt: DType) -> String {
    generate_reduction_msl(|| mt_layer_norm::kernel_ir_for(dt), "layer_norm")
}

pub fn bench_layer_norm(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_layer_norm_for)
}

fn bench_layer_norm_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::reduce(dt);
    let (tn, dlabel, eb, tol) = (ctx.tn, ctx.label, ctx.eb, ctx.tol);

    let mt_msl = layer_norm_msl_for(dt);
    let mk = runner.compile(&mt_msl, "mt_layer_norm").ok();
    // layer_norm_looped{tn}: (x, w, b, out, eps:f32, axis_size:u32, w_stride:u32, b_stride:u32)
    let rk = runner.compile(SRC, &format!("layer_norm_looped{tn}")).ok();

    // Correctness check
    let x_vals: Vec<f32> = (0..CHECK_B * CHECK_N)
        .map(|i| {
            let row = i / CHECK_N;
            let col = i % CHECK_N;
            row as f32 * 0.125 + ((col % 29) as f32 - 14.0) * 0.25
        })
        .collect();
    let w_vals: Vec<f32> = (0..CHECK_B * CHECK_N)
        .map(|i| {
            let row = i / CHECK_N;
            let col = i % CHECK_N;
            0.75 + row as f32 * 0.0625 + (col % 17) as f32 * 0.03125
        })
        .collect();
    let b_vals: Vec<f32> = (0..CHECK_B * CHECK_N)
        .map(|i| {
            let row = i / CHECK_N;
            let col = i % CHECK_N;
            row as f32 * 0.03125 + ((col % 11) as f32 - 5.0) * 0.03125
        })
        .collect();

    let xc = buffer_typed(runner, &x_vals, dt);
    let wc = buffer_typed(runner, &w_vals, dt);
    let bc = buffer_typed(runner, &b_vals, dt);
    let eps = runner.buffer_f32_scalar(1e-6_f32);
    let ns = runner.buffer_u32(CHECK_N as u32);
    let stride = runner.buffer_u32(1u32);

    let ref_check = rk.as_ref().map(|rk| {
        let out = zeros_typed(runner, CHECK_B * CHECK_N, dt);
        run_typed_once(
            runner,
            rk,
            &[&xc, &wc, &bc, &out, &eps, &ns, &stride, &stride],
            &out,
            CHECK_B * CHECK_N,
            [CHECK_B, 1, 1],
            [REF_TPG, 1, 1],
            dt,
        )
    });
    let mt_check = mk.as_ref().map(|mk| {
        let out = zeros_typed(runner, CHECK_B * CHECK_N, dt);
        run_typed_once(
            runner,
            mk,
            &[&xc, &wc, &bc, &out, &eps, &ns],
            &out,
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
        let x = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
        let w = buffer_typed(runner, &vec![1.0f32; n], dt);
        let bi = buffer_typed(runner, &vec![0.0f32; n], dt);
        let eps_p = runner.buffer_f32_scalar(1e-6_f32);
        let ns_p = runner.buffer_u32(n as u32);
        let ref_stride = runner.buffer_u32(1u32);
        let bytes = (b * n * eb * 2) as f64;

        let ref_perf = rk.as_ref().and_then(|r| {
            let out = zeros_typed(runner, b * n, dt);
            bench_gbps(
                runner,
                r,
                &[&x, &w, &bi, &out, &eps_p, &ns_p, &ref_stride, &ref_stride],
                [b, 1, 1],
                [REF_TPG, 1, 1],
                bytes,
            )
        });
        let mt_perf = mk.as_ref().and_then(|m| {
            let out = zeros_typed(runner, b * n, dt);
            bench_gbps(
                runner,
                m,
                &[&x, &w, &bi, &out, &eps_p, &ns_p],
                [b, 1, 1],
                [MT_TPG, 1, 1],
                bytes,
            )
        });
        let shape = format!("B={b} N={n} {dlabel}");
        results.push(BENCH.result(shape, ref_perf, mt_perf, Some(equiv)));
    }
    results
}

crate::bench_tests!(msl_fn: layer_norm_msl_for, kernel_name: "mt_layer_norm");

use crate::ops::{FLOAT_DTYPE_STRS, KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "layer_norm",
        mt_kernel: "mt_layer_norm".into(),
        metal_file: "layer_norm.metal",
        ref_spec: RefSpec::Format("layer_norm_looped{tn}"),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
