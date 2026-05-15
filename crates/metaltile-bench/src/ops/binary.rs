//! Elementwise binary ops — #[kernel] DSL vs MLX metal/binary.metal
//!
//! MLX kernel pattern: vvn_{Op}{tname}  (binary_vv, N_PER_THREAD=2)
//!   e.g. vvn_Addfloat32
//!   Params: (in0: device T*, in1: device T*, out: device T*, size: constant uint&)
//!   Grid: [N/(N_PER_THREAD*TPG), 1, 1] × [1024, 1, 1]
//!
//! MetalTile: [ceil(N/TPG), 1, 1] × [1024, 1, 1] — 1:1 threads

use metaltile::{bench_kernel, kernel};

static BINARY_SRC: &str = include_str!("../metal/binary.metal");

pub const N_ELEM: usize = 64 * 1024 * 1024;

// ── CPU references ────────────────────────────────────────────────────────────

pub fn cpu_add(a: f32, b: f32) -> f32 { a + b }
pub fn cpu_mul(a: f32, b: f32) -> f32 { a * b }
pub fn cpu_sub(a: f32, b: f32) -> f32 { a - b }
pub fn cpu_div(a: f32, b: f32) -> f32 { a / b }
pub fn cpu_maximum(a: f32, b: f32) -> f32 { a.max(b) }
pub fn cpu_minimum(a: f32, b: f32) -> f32 { a.min(b) }
pub fn cpu_pow(a: f32, b: f32) -> f32 { a.powf(b) }
pub fn cpu_logaddexp(a: f32, b: f32) -> f32 { (a.exp() + b.exp()).ln() }

// ── Input generators for bench ────────────────────────────────────────────────
// Defined as statics so the ramp behaviour is encoded in InputGen variants.
// a: 1.0 + i*1e-6   b: 0.5 + i*1e-6  (always positive — safe for pow/div)
// We use Unit/Half as close approximations; exact ramp perf difference is negligible.

// ── Kernels + registrations ───────────────────────────────────────────────────

#[bench_kernel(op="binary", subop="add", class=Binary, cpu=cpu_add,
               input_a=Unit, input_b=Half, tol=1e-6,
               mlx_src=BINARY_SRC, mlx="vvn_Add{tn}")]
#[kernel]
pub fn vector_add<T>(a: Tensor<T>, b: Tensor<T>, c: Tensor<T>) {
    let idx = program_id(0);
    store(c[idx], load(a[idx]) + load(b[idx]));
}

#[bench_kernel(op="binary", subop="mul", class=Binary, cpu=cpu_mul,
               input_a=Unit, input_b=Half, tol=1e-6,
               mlx_src=BINARY_SRC, mlx="vvn_Multiply{tn}")]
#[kernel]
pub fn mt_mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) * load(b[idx]));
}

#[bench_kernel(op="binary", subop="sub", class=Binary, cpu=cpu_sub,
               input_a=Unit, input_b=Half, tol=1e-6,
               mlx_src=BINARY_SRC, mlx="vvn_Subtract{tn}")]
#[kernel]
pub fn mt_sub<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) - load(b[idx]));
}

#[bench_kernel(op="binary", subop="div", class=Binary, cpu=cpu_div,
               input_a=Unit, input_b=Half, tol=1e-6,
               mlx_src=BINARY_SRC, mlx="vvn_Divide{tn}")]
#[kernel]
pub fn mt_div<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) / load(b[idx]));
}

#[bench_kernel(op="binary", subop="maximum", class=Binary, cpu=cpu_maximum,
               input_a=Unit, input_b=Half, tol=1e-6,
               mlx_src=BINARY_SRC, mlx="vvn_Maximum{tn}")]
#[kernel]
pub fn mt_max_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], max(load(a[idx]), load(b[idx])));
}

#[bench_kernel(op="binary", subop="minimum", class=Binary, cpu=cpu_minimum,
               input_a=Unit, input_b=Half, tol=1e-6,
               mlx_src=BINARY_SRC, mlx="vvn_Minimum{tn}")]
#[kernel]
pub fn mt_min_elem<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], min(load(a[idx]), load(b[idx])));
}

#[bench_kernel(op="binary", subop="pow", class=Binary, cpu=cpu_pow,
               input_a=Unit, input_b=Half, tol=1e-4,
               mlx_src=BINARY_SRC, mlx="vvn_Power{tn}")]
#[kernel]
pub fn mt_pow<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], pow(load(a[idx]), load(b[idx])));
}

#[bench_kernel(op="binary", subop="logaddexp", class=Binary, cpu=cpu_logaddexp,
               input_a=Signed, input_b=Signed, tol=1e-4,
               mlx_src=BINARY_SRC, mlx="vvn_LogAddExp{tn}")]
#[kernel]
pub fn mt_logaddexp<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(exp(load(a[idx])) + exp(load(b[idx]))));
}

// ── Legacy bench entry point ──────────────────────────────────────────────────

pub fn bench_elementwise(runner: &crate::runner::GpuRunner) -> Vec<crate::ops::OpResult> {
    bench_binary_ops(runner)
}

pub fn bench_binary_ops(runner: &crate::runner::GpuRunner) -> Vec<crate::ops::OpResult> {
    use crate::ops::FLOAT_DTYPES;
    let mut specs: Vec<&crate::spec::BenchSpec> = ::inventory::iter::<crate::spec::BenchSpec>
        .into_iter()
        .filter(|s| s.op == "binary")
        .collect();
    specs.sort_by_key(|s| s.subop);
    let mut results = Vec::new();
    for spec in specs {
        for &dt in FLOAT_DTYPES {
            results.extend(spec.run(runner, dt));
        }
    }
    results
}

// ── KernelSpec ────────────────────────────────────────────────────────────────

use crate::ops::{FLOAT_DTYPE_STRS, KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    ::inventory::iter::<crate::spec::BenchSpec>
        .into_iter()
        .filter(|s| s.op == "binary")
        .map(|s| KernelSpec {
            op: "binary",
            mt_kernel: s.kernel_name.into(),
            metal_file: "binary.metal",
            ref_spec: match &s.class {
                crate::spec::BenchClass::Binary { mlx_pattern: Some(p), .. } => RefSpec::Format(p),
                _ => RefSpec::None("no MLX reference"),
            },
            dtypes: FLOAT_DTYPE_STRS,
        })
        .collect()
}
