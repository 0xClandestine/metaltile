//! elementwise binary ops — #[kernel] DSL vs MLX.
//!
//! Reference: metal/binary.metal  (MLX, Apache-2.0)
//! Kernel pattern: vvn_{Op}float32  (binary_vv, N=2, f32×f32→f32)
//!
//! Algorithm: out[i] = a[i] op b[i]
//! Dispatch (ref): [N/(N_PER_THREAD*TPG), 1, 1] x [TPG, 1, 1]
//! Dispatch (MT):  [ceil(N/TPG), 1, 1] x [TPG, 1, 1]

use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{
        DType,
        DtypeCtx,
        OpBench,
        OpResult,
        bench_all_dtypes,
        buffer_typed,
        check_equiv,
        quantize_roundtrip,
        bench_gbps,
        run_typed_once,
        zeros_typed,
    },
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/binary.metal");

pub const N_ELEM: usize = 64 * 1024 * 1024;
const N_PER_THREAD: usize = 2;
const TPG: usize = 1_024;
const N_CHECK: usize = 2_048;

// ── Kernels ──────────────────────────────────────────────────────────────────

#[kernel]
pub fn vector_add<T>(a: Tensor<T>, b: Tensor<T>, c: Tensor<T>) {
    let idx = program_id(0);
    store(c[idx], load(a[idx]) + load(b[idx]));
}

#[kernel]
pub fn mt_mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) * load(b[idx]));
}

#[kernel]
pub fn mt_sub<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) - load(b[idx]));
}

#[kernel]
pub fn mt_div<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) / load(b[idx]));
}

#[kernel]
pub fn mt_max_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], max(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_min_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], min(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_pow<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], pow(load(a[idx]), load(b[idx])));
}

#[kernel]
pub fn mt_logaddexp<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(exp(load(a[idx])) + exp(load(b[idx]))));
}

// ── Entry table ──────────────────────────────────────────────────────────────

static BENCH_ADD: OpBench = OpBench::new("binary_add", "GB/s");
static BENCH_MUL: OpBench = OpBench::new("binary_mul", "GB/s");
static BENCH_SUB: OpBench = OpBench::new("binary_sub", "GB/s");
static BENCH_DIV: OpBench = OpBench::new("binary_div", "GB/s");
static BENCH_MAX: OpBench = OpBench::new("binary_maximum", "GB/s");
static BENCH_MIN: OpBench = OpBench::new("binary_minimum", "GB/s");
static BENCH_POW: OpBench = OpBench::new("binary_pow", "GB/s");
static BENCH_LOGADDEXP: OpBench = OpBench::new("binary_logaddexp", "GB/s");

struct BinaryEntry {
    bench: &'static OpBench,
    ref_fn: String,
    mt_name: &'static str,
    msl: String,
    cpu: fn(f32, f32) -> f32,
}

fn make_entries(dt: DType) -> Vec<BinaryEntry> {
    let tn = DtypeCtx::elementwise(dt).tn;
    vec![
        BinaryEntry {
            bench: &BENCH_ADD,
            ref_fn: format!("vvn_Add{tn}"),
            mt_name: "vector_add",
            msl: MslGenerator::default().generate(&vector_add::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| a + b,
        },
        BinaryEntry {
            bench: &BENCH_MUL,
            ref_fn: format!("vvn_Multiply{tn}"),
            mt_name: "mt_mul",
            msl: MslGenerator::default().generate(&mt_mul::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| a * b,
        },
        BinaryEntry {
            bench: &BENCH_SUB,
            ref_fn: format!("vvn_Subtract{tn}"),
            mt_name: "mt_sub",
            msl: MslGenerator::default().generate(&mt_sub::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| a - b,
        },
        BinaryEntry {
            bench: &BENCH_DIV,
            ref_fn: format!("vvn_Divide{tn}"),
            mt_name: "mt_div",
            msl: MslGenerator::default().generate(&mt_div::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| a / b,
        },
        BinaryEntry {
            bench: &BENCH_MAX,
            ref_fn: format!("vvn_Maximum{tn}"),
            mt_name: "mt_max_elem",
            msl: MslGenerator::default().generate(&mt_max_elem::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| a.max(b),
        },
        BinaryEntry {
            bench: &BENCH_MIN,
            ref_fn: format!("vvn_Minimum{tn}"),
            mt_name: "mt_min_elem",
            msl: MslGenerator::default().generate(&mt_min_elem::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| a.min(b),
        },
        BinaryEntry {
            bench: &BENCH_POW,
            ref_fn: format!("vvn_Power{tn}"),
            mt_name: "mt_pow",
            msl: MslGenerator::default().generate(&mt_pow::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| a.powf(b),
        },
        BinaryEntry {
            bench: &BENCH_LOGADDEXP,
            ref_fn: format!("vvn_LogAddExp{tn}"),
            mt_name: "mt_logaddexp",
            msl: MslGenerator::default().generate(&mt_logaddexp::kernel_ir_for(dt)).unwrap(),
            cpu: |a, b| (a.exp() + b.exp()).ln(),
        },
    ]
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_elementwise(runner: &GpuRunner) -> Vec<OpResult> { bench_binary_ops(runner) }

pub fn bench_binary_ops(runner: &GpuRunner) -> Vec<OpResult> {
    bench_all_dtypes(runner, bench_binary_ops_for)
}

fn bench_binary_ops_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let ctx = DtypeCtx::elementwise(dt);
    let (dlabel, eb, tol) = (ctx.label, ctx.eb, ctx.tol);
    let entries = make_entries(dt);
    let tpg = [TPG, 1, 1];
    let bytes = (N_ELEM * eb * 3) as f64; // 2 reads + 1 write

    // Shared typed input buffers for perf
    let a_vals: Vec<f32> = (0..N_ELEM).map(|i| 1.0 + i as f32 * 1e-6).collect();
    let b_vals: Vec<f32> = (0..N_ELEM).map(|i| 0.5 + i as f32 * 1e-6).collect();
    let a_buf = buffer_typed(runner, &a_vals, dt);
    let b_buf = buffer_typed(runner, &b_vals, dt);

    let mut results = Vec::new();

    for entry in &entries {
        let Some(mk) = runner.compile(&entry.msl, entry.mt_name).ok() else {
            continue;
        };

        // Correctness: use quantize-roundtrip inputs so tolerance is tight
        let a_check_f32: Vec<f32> = (0..N_CHECK).map(|i| 1.0 + i as f32 * 0.001).collect();
        let b_check_f32: Vec<f32> = (0..N_CHECK).map(|i| 0.5 + i as f32 * 0.001).collect();
        let a_q = quantize_roundtrip(&a_check_f32, dt);
        let b_q = quantize_roundtrip(&b_check_f32, dt);
        let cpu_ref: Vec<f32> = a_q.iter().zip(&b_q).map(|(&a, &b)| (entry.cpu)(a, b)).collect();
        let a_s = buffer_typed(runner, &a_check_f32, dt);
        let b_s = buffer_typed(runner, &b_check_f32, dt);
        let mt_out = zeros_typed(runner, N_CHECK, dt);
        let n_buf = runner.buffer_u32(N_CHECK as u32);
        let mt_vals = run_typed_once(
            runner,
            &mk,
            &[&a_s, &b_s, &mt_out, &n_buf],
            &mt_out,
            N_CHECK,
            [N_CHECK.div_ceil(TPG), 1, 1],
            tpg,
            dt,
        );
        let equiv = check_equiv(&cpu_ref, &mt_vals, tol);

        // Ref perf
        let ref_perf = runner.compile(SRC, &entry.ref_fn).ok().and_then(|rk| {
            let out_ref = zeros_typed(runner, N_ELEM, dt);
            let size = runner.buffer_u32(N_ELEM as u32);
            bench_gbps(runner, &rk, &[&a_buf, &b_buf, &out_ref, &size], [N_ELEM / (N_PER_THREAD * TPG), 1, 1], tpg, bytes)
        });

        // MT perf
        let out_mt = zeros_typed(runner, N_ELEM, dt);
        let n_perf = runner.buffer_u32(N_ELEM as u32);
        let mt_perf = bench_gbps(runner, &mk, &[&a_buf, &b_buf, &out_mt, &n_perf], [N_ELEM.div_ceil(TPG), 1, 1], tpg, bytes);

        let shape = format!("N={N_ELEM} {dlabel}");
        results.push(entry.bench.result(shape, ref_perf, mt_perf, Some(equiv)));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::FLOAT_DTYPES;

    #[test]
    fn msl_generates_for_all_dtypes() {
        for &dt in FLOAT_DTYPES {
            let entries = make_entries(dt);
            for entry in &entries {
                assert!(
                    !entry.msl.trim().is_empty(),
                    "MSL empty for op {} dtype {dt:?}",
                    entry.mt_name
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn kernels_compile() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        for &dt in FLOAT_DTYPES {
            let entries = make_entries(dt);
            for entry in &entries {
                runner.compile(&entry.msl, entry.mt_name).unwrap();
            }
        }
    }
}

use crate::ops::{KernelSpec, RefSpec, FLOAT_DTYPE_STRS};

/// Reference kernel name templates — mirrors the patterns in `make_entries()`.
static BINARY_REF_PATTERNS: &[(&str, &str)] = &[
    ("vector_add",   "vvn_Add{tn}"),
    ("mt_mul",       "vvn_Multiply{tn}"),
    ("mt_sub",       "vvn_Subtract{tn}"),
    ("mt_div",       "vvn_Divide{tn}"),
    ("mt_max_elem",  "vvn_Maximum{tn}"),
    ("mt_min_elem",  "vvn_Minimum{tn}"),
    ("mt_pow",       "vvn_Power{tn}"),
    ("mt_logaddexp", "vvn_LogAddExp{tn}"),
];

pub fn kernel_specs() -> Vec<KernelSpec> {
    BINARY_REF_PATTERNS
        .iter()
        .map(|&(mt_kernel, pat)| KernelSpec {
            op: "binary",
            mt_kernel: mt_kernel.into(),
            metal_file: "binary.metal",
            ref_spec: RefSpec::Format(pat),
            dtypes: FLOAT_DTYPE_STRS,
        })
        .collect()
}
