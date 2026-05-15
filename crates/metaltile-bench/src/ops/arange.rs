//! Arange benchmark — #[kernel] DSL vs MLX metal/arange.metal
//!
//! MLX kernel: arangefloat32 / arangefloat16 / arangebfloat16 (arange.metal)
//!   Params: (start: constant T&, step: constant T&, out: device T*) — slots [0, 1, 2]
//!   Grid: [ceil(N/1024), 1, 1] × [1024, 1, 1]  (TPG=1024)
//!   Algorithm: out[index] = start + index * step  (one thread per element)
//!
//! MetalTile: mt_arange — same one-thread-per-element algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

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
        generate_elementwise_msl,
        quantize_roundtrip,
        bench_gbps,
        run_typed_once,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/arange.metal");

const BENCH: OpBench = OpBench::new("arange", "GB/s");
const N_ELEM: usize = 64 * 1024 * 1024;
const N_CHECK: usize = 4_096;
const TPG: usize = 1_024;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Arange: out[idx] = start + idx * step
///
/// `start` and `step` are passed as single-element typed buffers.
/// Dispatch: [ceil(N/TPG), 1, 1] x [TPG, 1, 1]
#[kernel]
pub fn mt_arange<T>(out: Tensor<T>, start: Tensor<T>, step: Tensor<T>, #[constexpr] n: u32) {
    let idx = program_id(0);
    let s = load(start[0]);
    let st = load(step[0]);
    store(out[idx], s + idx.cast::<T>() * st);
}

fn arange_msl_for(dt: DType) -> String {
    generate_elementwise_msl(|| mt_arange::kernel_ir_for(dt), "arange")
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_arange(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_arange_for)
}

fn bench_arange_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let (tn, dlabel, eb, tol) = (ctx.tn, ctx.label, ctx.eb, ctx.tol);

    let msl = arange_msl_for(dt);
    let mk = runner.compile(&msl, "mt_arange").ok();

    // MLX ref (may not exist for all dtypes — silently skip)
    let ref_name = format!("arange{tn}");
    let rk = runner.compile(SRC, &ref_name).ok();

    let start = 0.0f32;
    let step = 1.0f32;

    // Correctness: compare MT against CPU reference using quantize_roundtrip
    let equiv = mk.as_ref().map(|mk| {
        let inp_f32: Vec<f32> = (0..N_CHECK).map(|i| i as f32).collect();
        let cpu_ref: Vec<f32> =
            inp_f32.iter().map(|&i| quantize_roundtrip(&[start + step * i], dt)[0]).collect();
        let s_buf = buffer_typed(runner, &[start], dt);
        let st_buf = buffer_typed(runner, &[step], dt);
        let out_buf = zeros_typed(runner, N_CHECK, dt);
        let n_buf = runner.buffer_u32(N_CHECK as u32);
        let mt_vals = run_typed_once(
            runner,
            mk,
            &[&out_buf, &s_buf, &st_buf, &n_buf],
            &out_buf,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            dt,
        );
        check_equiv(&cpu_ref, &mt_vals, tol)
    });

    let bytes = (N_ELEM * eb) as f64; // write-only

    // Reference perf (f32-only)
    let ref_perf = rk.as_ref().and_then(|rk| {
        let ref_start = runner.buffer_f32_scalar(start);
        let ref_step = runner.buffer_f32_scalar(step);
        let ref_out = runner.buffer_zeros(N_ELEM * 4);
        bench_gbps(runner, rk, &[&ref_start, &ref_step, &ref_out], [N_ELEM.div_ceil(TPG), 1, 1], [TPG, 1, 1], bytes)
    });

    // MT perf
    let mt_start = buffer_typed(runner, &[start], dt);
    let mt_step = buffer_typed(runner, &[step], dt);
    let mt_out = zeros_typed(runner, N_ELEM, dt);
    let mt_n = runner.buffer_u32(N_ELEM as u32);
    let mt_perf = mk.as_ref().and_then(|mk| {
        bench_gbps(runner, mk, &[&mt_out, &mt_start, &mt_step, &mt_n], [N_ELEM.div_ceil(TPG), 1, 1], [TPG, 1, 1], bytes)
    });

    let shape = format!("N={N_ELEM} {dlabel}");
    vec![BENCH.result(shape, ref_perf, mt_perf, equiv)]
}

crate::bench_tests!(msl_fn: arange_msl_for, kernel_name: "mt_arange");

use crate::ops::{KernelSpec, RefSpec, FLOAT_DTYPE_STRS};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "arange",
        mt_kernel: "mt_arange".into(),
        metal_file: "arange.metal",
        ref_spec: RefSpec::Format("arange{tn}"),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
