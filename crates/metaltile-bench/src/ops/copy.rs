//! Copy benchmark — #[kernel] DSL vs MLX metal/copy.metal
//!
//! MLX kernel: v_copyfloat32float32 / v_copyfloat16float16 / v_copybfloat16bfloat16
//!   (copy.metal, copy_v with N=1, same-type variant)
//!   Params: (src: device T*, dst: device T*, size: constant uint&) — slots [0, 1, 2]
//!   Grid: [ceil(N/TPG), 1, 1] × [TPG, 1, 1]
//!   Algorithm: dst[i] = src[i]  (one thread per element)
//!
//! MetalTile: mt_copy — same algorithm via #[kernel] DSL.
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

static SRC: &str = include_str!("../metal/copy.metal");
const BENCH: OpBench = OpBench::new("copy", "GB/s");
pub const N_ELEM: usize = 64 * 1024 * 1024;
const N_CHECK: usize = 4_096;
const TPG: usize = 256;

/// Copy: b[i] = a[i]
///
/// Matches MLX `v_copy{tn}{tn}` — dispatch one thread per element.
#[kernel]
pub fn mt_copy<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]));
}

fn copy_msl_for(dt: DType) -> String {
    generate_elementwise_msl(|| mt_copy::kernel_ir_for(dt), "copy")
}

pub fn bench_copy(runner: &GpuRunner) -> Vec<OpResult> { bench_all_dtypes(runner, bench_copy_for) }

fn bench_copy_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let (tn, dlabel, eb, tol) = (ctx.tn, ctx.label, ctx.eb, ctx.tol);

    let msl = copy_msl_for(dt);
    let mk = runner.compile(&msl, "mt_copy").ok();
    let rk = runner.compile(SRC, &format!("v_copy{tn}{tn}")).ok();

    let vals: Vec<f32> = (0..N_CHECK).map(|i| i as f32 * 0.25 - 17.0).collect();
    let check_in = buffer_typed(runner, &vals, dt);
    let check_out = zeros_typed(runner, N_CHECK, dt);
    let ref_size = runner.buffer_u32(N_CHECK as u32);

    let ref_check: Vec<f32> = rk
        .as_ref()
        .map(|rk| {
            run_typed_once(
                runner,
                rk,
                &[&check_in, &check_out, &ref_size],
                &check_out,
                N_CHECK,
                [N_CHECK.div_ceil(TPG), 1, 1],
                [TPG, 1, 1],
                dt,
            )
        })
        .unwrap_or_default();
    let mt_check_out = zeros_typed(runner, N_CHECK, dt);
    let mt_check = mk
        .as_ref()
        .map(|mk| {
            run_typed_once(
                runner,
                mk,
                &[&check_in, &mt_check_out],
                &mt_check_out,
                N_CHECK,
                [N_CHECK.div_ceil(TPG), 1, 1],
                [TPG, 1, 1],
                dt,
            )
        })
        .unwrap_or_default();

    let equiv = if !ref_check.is_empty() && !mt_check.is_empty() {
        check_equiv(&ref_check, &mt_check, tol)
    } else if !mt_check.is_empty() {
        // No MLX ref — MT output should equal quantize_roundtrip of input
        let expected = quantize_roundtrip(&vals, dt);
        check_equiv(&expected, &mt_check, tol)
    } else {
        return vec![];
    };

    let src = buffer_typed(runner, &vec![1.0f32; N_ELEM], dt);
    let bytes = (N_ELEM * eb * 2) as f64;
    let ref_size_perf = runner.buffer_u32(N_ELEM as u32);

    let ref_perf = rk.as_ref().and_then(|rk| {
        let ref_out = zeros_typed(runner, N_ELEM, dt);
        bench_gbps(runner, rk, &[&src, &ref_out, &ref_size_perf], [N_ELEM.div_ceil(TPG), 1, 1], [TPG, 1, 1], bytes)
    });
    let mt_perf = mk.as_ref().and_then(|mk| {
        let mt_out = zeros_typed(runner, N_ELEM, dt);
        bench_gbps(runner, mk, &[&src, &mt_out], [N_ELEM.div_ceil(TPG), 1, 1], [TPG, 1, 1], bytes)
    });

    let shape = format!("N={N_ELEM} {dlabel}");
    vec![BENCH.result(shape, ref_perf, mt_perf, Some(equiv))]
}

crate::bench_tests!(msl_fn: copy_msl_for, kernel_name: "mt_copy");

use crate::ops::{KernelSpec, RefSpec, FLOAT_DTYPE_STRS};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "copy",
        mt_kernel: "mt_copy".into(),
        metal_file: "copy.metal",
        ref_spec: RefSpec::Format("v_copy{tn}{tn}"),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
