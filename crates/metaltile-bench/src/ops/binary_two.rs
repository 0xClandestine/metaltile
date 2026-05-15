//! binary_two benchmark — #[kernel] DSL vs MLX metal/binary_two.metal
//!
//! MLX kernel: no single reference kernel (no MLX equiv for fused two-output);
//!   SRC included for completeness but ref_perf is always None.
//!   Grid would be: [ceil(N/TPG), 1, 1] × [TPG, 1, 1]
//!   Algorithm: compute two binary ops (add and mul) over the same inputs in a
//!              single kernel pass — avoids two separate memory round-trips.
//!
//! MetalTile: mt_binary_two — fused elementwise (add + mul) via #[kernel] DSL.
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

#[allow(dead_code)]
static SRC: &str = include_str!("../metal/binary_two.metal");

const BENCH: OpBench = OpBench::new("binary_two", "GB/s");
const N_ELEM: usize = 64 * 1024 * 1024;
const N_CHECK: usize = 2_048;
const TPG: usize = 1_024;

#[kernel]
pub fn mt_binary_two<T>(a: Tensor<T>, b: Tensor<T>, mut c: Tensor<T>, mut d: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    let y = load(b[idx]);
    store(c[idx], x + y);
    store(d[idx], x * y);
}

fn binary_two_msl_for(dt: DType) -> String {
    generate_elementwise_msl(|| mt_binary_two::kernel_ir_for(dt), "binary_two")
}

pub fn bench_binary_two(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_binary_two_for)
}

fn bench_binary_two_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let (dlabel, eb, tol) = (ctx.label, ctx.eb, ctx.tol);

    let msl = binary_two_msl_for(dt);
    let mk = runner.compile(&msl, "mt_binary_two").ok();

    // Keep inputs in [-1, 0.5] so products stay < 1: avoids f16/bf16 ULP issues at
    // large magnitudes (e.g. a=18, b=-39 → product≈-700, ULP≈0.5 in f16).
    let a_f32: Vec<f32> = (0..N_CHECK).map(|i| (i as f32 * 2.0 / N_CHECK as f32) - 1.0).collect();
    let b_f32: Vec<f32> = (0..N_CHECK).map(|i| 0.5 - (i as f32 / N_CHECK as f32)).collect();
    let a_q = quantize_roundtrip(&a_f32, dt);
    let b_q = quantize_roundtrip(&b_f32, dt);
    let ref_add: Vec<f32> = a_q.iter().zip(&b_q).map(|(&a, &b)| a + b).collect();
    let ref_mul: Vec<f32> = a_q.iter().zip(&b_q).map(|(&a, &b)| a * b).collect();

    // Correctness: run each output separately via run_typed_once
    let equiv = mk.as_ref().map(|mk| {
        let a_check = buffer_typed(runner, &a_f32, dt);
        let b_check = buffer_typed(runner, &b_f32, dt);

        let c_check = zeros_typed(runner, N_CHECK, dt);
        let d_check = zeros_typed(runner, N_CHECK, dt);
        // Run once to populate both outputs
        let mt_add = run_typed_once(
            runner,
            mk,
            &[&a_check, &b_check, &c_check, &d_check],
            &c_check,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            dt,
        );
        let add_ok = check_equiv(&ref_add, &mt_add, tol);

        let mt_mul = run_typed_once(
            runner,
            mk,
            &[&a_check, &b_check, &c_check, &d_check],
            &d_check,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            dt,
        );
        let mul_ok = check_equiv(&ref_mul, &mt_mul, tol);

        // Return the worst of the two
        if add_ok.max_abs_err > mul_ok.max_abs_err { add_ok } else { mul_ok }
    });

    let a = buffer_typed(runner, &vec![1.0f32; N_ELEM], dt);
    let b = buffer_typed(runner, &vec![2.0f32; N_ELEM], dt);
    let c = zeros_typed(runner, N_ELEM, dt);
    let d = zeros_typed(runner, N_ELEM, dt);
    let bytes = (N_ELEM * eb * 4) as f64; // 2 reads + 2 writes
    let mt_perf = mk.as_ref().and_then(|mk| {
        bench_gbps(runner, mk, &[&a, &b, &c, &d], [N_ELEM.div_ceil(TPG), 1, 1], [TPG, 1, 1], bytes)
    });

    let shape = format!("N={N_ELEM} {dlabel}");
    vec![BENCH.result(shape, None, mt_perf, equiv)]
}

crate::bench_tests!(msl_fn: binary_two_msl_for, kernel_name: "mt_binary_two");

use crate::ops::{KernelSpec, RefSpec, FLOAT_DTYPE_STRS};

pub fn kernel_specs() -> Vec<KernelSpec> {
    vec![KernelSpec {
        op: "binary_two",
        mt_kernel: "mt_binary_two".into(),
        metal_file: "binary_two.metal",
        ref_spec: RefSpec::None(
            "no MLX equivalent — MT benchmarks 2-output fused pass that MLX doesn't expose",
        ),
        dtypes: FLOAT_DTYPE_STRS,
    }]
}
