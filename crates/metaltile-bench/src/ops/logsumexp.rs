//! LogSumExp benchmark — #[kernel] DSL vs MLX metal/logsumexp.metal
//!
//! MLX kernel: looped_logsumexp_float32 (logsumexp.metal, line ~150)
//!   Params: (inp: device T*, out: device T*, n: constant int&) — slots [0, 1, 2]
//!   Grid: [B, 1, 1] × [256, 1, 1]  (one threadgroup per row)
//!   Algorithm: 2-pass online log-sum-exp. Each thread strides over its row
//!              accumulating (max, sum) with the numerically-stable Welford merge.
//!              SIMD-group tree reduction (simd_sum / simd_shuffle_down), then
//!              threadgroup merge across SIMD groups. Thread 0 writes
//!              log(sum(exp(row))) = row_max + log(row_sum).
//!
//! MetalTile: mt_logsumexp — same algorithm via #[kernel] DSL.
//!   KernelMode::Reduction

use metaltile::kernel;

use crate::{
    ops::{
        DType,
        DtypeCtx,
        OpBench,
        OpResult,
        bench_all_dtypes,
        buffer_typed,
        check_equiv,
        generate_reduction_msl,
        bench_gbps,
        run_typed_once,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/logsumexp.metal");
const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
const BENCH: OpBench = OpBench::new("logsumexp", "GB/s");

const CHECK_B: usize = 8;
const CHECK_N: usize = 512;
const TPG: usize = 256;

/// log(sum(exp(x[i]))) computed as log_max + log(sum(exp(x[i] - max)))
/// Dispatch: [B, 1, 1] x [256, 1, 1]
#[kernel]
pub fn mt_logsumexp<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let acc_max = strided_reduce(inp, rs, re, max);
    let row_max = reduce_max(acc_max);
    let acc_sum = strided_reduce_exp_sub(inp, rs, re, row_max);
    let row_sum = reduce_sum(acc_sum);
    let lse = row_max + log(row_sum);
    store(out[row], lse);
}

fn logsumexp_msl_for(dt: DType) -> String {
    generate_reduction_msl(|| mt_logsumexp::kernel_ir_for(dt), "logsumexp")
}

pub fn bench_logsumexp(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_logsumexp_for)
}

fn bench_logsumexp_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::reduce(dt);
    let (tn, dlabel, eb, tol) = (ctx.tn, ctx.label, ctx.eb, ctx.tol);

    let msl = logsumexp_msl_for(dt);
    let mk = runner.compile(&msl, "mt_logsumexp").ok();
    let rk = runner.compile(SRC, &format!("looped_logsumexp_{tn}")).ok();

    // Correctness check
    let inp_vals: Vec<f32> = (0..CHECK_B * CHECK_N)
        .map(|i| {
            let row = i / CHECK_N;
            let col = i % CHECK_N;
            row as f32 * 0.0625 + ((col % 41) as f32 - 20.0) * 0.125
        })
        .collect();
    let inp_check = buffer_typed(runner, &inp_vals, dt);
    let ref_ns = runner.buffer_i32(CHECK_N as i32);
    let mt_ns = runner.buffer_u32(CHECK_N as u32);

    // logsumexp output is always f32 (scalar reduction result)
    let ref_check = rk.as_ref().map(|rk| {
        let out = zeros_typed(runner, CHECK_B, dt);
        run_typed_once(
            runner,
            rk,
            &[&inp_check, &out, &ref_ns],
            &out,
            CHECK_B,
            [CHECK_B, 1, 1],
            [TPG, 1, 1],
            dt,
        )
    });
    let mt_check = mk.as_ref().map(|mk| {
        let out = zeros_typed(runner, CHECK_B, dt);
        run_typed_once(
            runner,
            mk,
            &[&inp_check, &out, &mt_ns],
            &out,
            CHECK_B,
            [CHECK_B, 1, 1],
            [TPG, 1, 1],
            dt,
        )
    });
    let equiv = match (ref_check, mt_check) {
        (Some(r), Some(m)) => check_equiv(&r, &m, tol),
        (None, Some(_)) | (_, None) => return vec![],
    };

    let mut results = Vec::new();
    for &(b, n) in SHAPES {
        let inp = buffer_typed(runner, &vec![1.0f32 / n as f32; b * n], dt);
        let bytes = (b * n * eb) as f64;
        let ref_n = runner.buffer_i32(n as i32);
        let mt_n = runner.buffer_u32(n as u32);

        let ref_perf = rk.as_ref().and_then(|r| {
            let out = zeros_typed(runner, b, dt);
            bench_gbps(runner, r, &[&inp, &out, &ref_n], [b, 1, 1], [256, 1, 1], bytes)
        });
        let mt_perf = mk.as_ref().and_then(|m| {
            let out = zeros_typed(runner, b, dt);
            bench_gbps(runner, m, &[&inp, &out, &mt_n], [b, 1, 1], [256, 1, 1], bytes)
        });
        let shape = format!("B={b} N={n} {dlabel}");
        results.push(BENCH.result(shape, ref_perf, mt_perf, Some(equiv)));
    }
    results
}

crate::bench_tests!(msl_fn: logsumexp_msl_for, kernel_name: "mt_logsumexp");

use crate::ops::{KernelSpec, RefSpec, FLOAT_DTYPE_STRS};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "logsumexp",
        mt_kernel: "mt_logsumexp".into(),
        metal_file: "logsumexp.metal",
        ref_spec: RefSpec::Format("looped_logsumexp_{tn}"),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
