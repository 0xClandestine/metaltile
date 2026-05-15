//! Unary elementwise benchmark — #[kernel] DSL vs MLX metal/unary.metal
//!
//! MLX kernel: v_{Op}{tname}{tname} (unary.metal)
//!   e.g. v_Expfloat32float32, v_Logfloat16float16, v_Sqrtbfloat16bfloat16
//!   Params: (in: device T*, out: device T*, size: constant uint&) — slots [0, 1, 2]
//!   Grid: [ceil(N/TPG), 1, 1] × [256, 1, 1]  (one thread per element)
//!   Algorithm: out[i] = op(in[i])  (elementwise unary)
//!
//! MetalTile: mt_{op} — same elementwise algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::{bench_kernel, kernel};

static UNARY_SRC: &str = include_str!("../metal/unary.metal");

// ── CPU references ────────────────────────────────────────────────────────────

pub fn cpu_exp(x: f32) -> f32 { x.exp() }
pub fn cpu_log(x: f32) -> f32 { x.ln() }
pub fn cpu_sqrt(x: f32) -> f32 { x.sqrt() }
pub fn cpu_rsqrt(x: f32) -> f32 { x.sqrt().recip() }
pub fn cpu_abs(x: f32) -> f32 { x.abs() }
pub fn cpu_silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
pub fn cpu_gelu(x: f32) -> f32 {
    let k = 0.797_884_6_f32;
    0.5 * x * (1.0 + (k * (x + 0.044_715 * x * x * x)).tanh())
}
pub fn cpu_relu(x: f32) -> f32 { x.max(0.0) }
pub fn cpu_cos(x: f32) -> f32 { x.cos() }
pub fn cpu_sin(x: f32) -> f32 { x.sin() }
pub fn cpu_ceil(x: f32) -> f32 { x.ceil() }
pub fn cpu_floor(x: f32) -> f32 { x.floor() }
pub fn cpu_exp2(x: f32) -> f32 { x.exp2() }
pub fn cpu_log2(x: f32) -> f32 { x.log2() }
pub fn cpu_neg(x: f32) -> f32 { -x }
pub fn cpu_recip(x: f32) -> f32 { x.recip() }
pub fn cpu_erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let p = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let s = if x < 0.0 { -1.0f32 } else { 1.0 };
    s * (1.0 - p * (-x * x).exp())
}
pub fn cpu_sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
pub fn cpu_round(x: f32) -> f32 {
    // Match Metal rint() semantics: round-half-to-even (IEEE 754 default).
    let fl = x.floor();
    let diff = x - fl;
    if diff < 0.5 {
        fl
    } else if diff > 0.5 {
        fl + 1.0
    } else if fl % 2.0 == 0.0 {
        fl
    } else {
        fl + 1.0
    }
}
pub fn cpu_square(x: f32) -> f32 { x * x }
pub fn cpu_sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
pub fn cpu_log1p(x: f32) -> f32 { x.ln_1p() }

// ── Kernels + bench registrations ─────────────────────────────────────────────

#[bench_kernel(op="unary", subop="exp", class=Unary, cpu=cpu_exp,
               input=Signed, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Exp{tn}{tn}")]
#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[bench_kernel(op="unary", subop="log", class=Unary, cpu=cpu_log,
               input=Positive, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Log{tn}{tn}")]
#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}

#[bench_kernel(op="unary", subop="sqrt", class=Unary, cpu=cpu_sqrt,
               input=Positive, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Sqrt{tn}{tn}")]
#[kernel]
pub fn mt_sqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sqrt(load(a[idx])));
}

#[bench_kernel(op="unary", subop="rsqrt", class=Unary, cpu=cpu_rsqrt,
               input=Positive, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Rsqrt{tn}{tn}")]
#[kernel]
pub fn mt_rsqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], rsqrt(load(a[idx])));
}

#[bench_kernel(op="unary", subop="abs", class=Unary, cpu=cpu_abs,
               input=Signed, tol=1e-6, mlx_src=UNARY_SRC, mlx="v_Abs{tn}{tn}")]
#[kernel]
pub fn mt_abs<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], abs(load(a[idx])));
}

#[bench_kernel(op="unary", subop="silu", class=Unary, cpu=cpu_silu,
               input=Signed, tol=1e-4)]
#[kernel]
pub fn mt_silu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], silu(load(a[idx])));
}

#[bench_kernel(op="unary", subop="gelu", class=Unary, cpu=cpu_gelu,
               input=Signed, tol=1e-4)]
#[kernel]
pub fn mt_gelu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], gelu(load(a[idx])));
}

#[bench_kernel(op="unary", subop="relu", class=Unary, cpu=cpu_relu,
               input=Signed, tol=1e-6)]
#[kernel]
pub fn mt_relu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], relu(load(a[idx])));
}

#[bench_kernel(op="unary", subop="cos", class=Unary, cpu=cpu_cos,
               input=Signed, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Cos{tn}{tn}")]
#[kernel]
pub fn mt_cos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cos(load(a[idx])));
}

#[bench_kernel(op="unary", subop="sin", class=Unary, cpu=cpu_sin,
               input=Signed, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Sin{tn}{tn}")]
#[kernel]
pub fn mt_sin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sin(load(a[idx])));
}

#[bench_kernel(op="unary", subop="ceil", class=Unary, cpu=cpu_ceil,
               input=Signed, tol=1e-6, mlx_src=UNARY_SRC, mlx="v_Ceil{tn}{tn}")]
#[kernel]
pub fn mt_ceil<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], ceil(load(a[idx])));
}

#[bench_kernel(op="unary", subop="floor", class=Unary, cpu=cpu_floor,
               input=Signed, tol=1e-6, mlx_src=UNARY_SRC, mlx="v_Floor{tn}{tn}")]
#[kernel]
pub fn mt_floor<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], floor(load(a[idx])));
}

#[bench_kernel(op="unary", subop="erf", class=Unary, cpu=cpu_erf,
               input=Signed, tol=1e-3, mlx_src=UNARY_SRC, mlx="v_Erf{tn}{tn}")]
#[kernel]
pub fn mt_erf<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erf(load(a[idx])));
}

#[bench_kernel(op="unary", subop="exp2", class=Unary, cpu=cpu_exp2,
               input=Signed, tol=1e-4)]
#[kernel]
pub fn mt_exp2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp2(load(a[idx])));
}

#[bench_kernel(op="unary", subop="log2", class=Unary, cpu=cpu_log2,
               input=Positive, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Log2{tn}{tn}")]
#[kernel]
pub fn mt_log2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log2(load(a[idx])));
}

#[bench_kernel(op="unary", subop="sign", class=Unary, cpu=cpu_sign,
               input=Signed, tol=0.0, mlx_src=UNARY_SRC, mlx="v_Sign{tn}{tn}")]
#[kernel]
pub fn mt_sign<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sign(load(a[idx])));
}

#[bench_kernel(op="unary", subop="round", class=Unary, cpu=cpu_round,
               input=Signed, tol=0.0, mlx_src=UNARY_SRC, mlx="v_Round{tn}{tn}")]
#[kernel]
pub fn mt_round<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], round(load(a[idx])));
}

#[bench_kernel(op="unary", subop="neg", class=Unary, cpu=cpu_neg,
               input=Signed, tol=1e-6, mlx_src=UNARY_SRC, mlx="v_Negative{tn}{tn}")]
#[kernel]
pub fn mt_neg<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], -load(a[idx]));
}

#[bench_kernel(op="unary", subop="recip", class=Unary, cpu=cpu_recip,
               input=Positive, tol=1e-4)]
#[kernel]
pub fn mt_recip<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], 1.0f32.cast::<T>() / load(a[idx]));
}

#[bench_kernel(op="unary", subop="square", class=Unary, cpu=cpu_square,
               input=Signed, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Square{tn}{tn}")]
#[kernel]
pub fn mt_square<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], x * x);
}

#[bench_kernel(op="unary", subop="sigmoid", class=Unary, cpu=cpu_sigmoid,
               input=Signed, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Sigmoid{tn}{tn}")]
#[kernel]
pub fn mt_sigmoid<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], 1.0f32.cast::<T>() / (1.0f32.cast::<T>() + exp(-x)));
}

#[bench_kernel(op="unary", subop="log1p", class=Unary, cpu=cpu_log1p,
               input=Positive, tol=1e-4, mlx_src=UNARY_SRC, mlx="v_Log1p{tn}{tn}")]
#[kernel]
pub fn mt_log1p<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], log(1.0f32.cast::<T>() + x));
}

// ── Legacy bench entry point (forwards to inventory-based runner) ─────────────

pub fn bench_all_unary(runner: &crate::runner::GpuRunner) -> Vec<crate::ops::OpResult> {
    use crate::ops::FLOAT_DTYPES;
    // Collect specs for this op group, sorted subop-primary
    let mut specs: Vec<&crate::spec::BenchSpec> = ::inventory::iter::<crate::spec::BenchSpec>
        .into_iter()
        .filter(|s| s.op == "unary")
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

// ── KernelSpec (for kernel_table) ─────────────────────────────────────────────

use crate::ops::{FLOAT_DTYPE_STRS, KernelSpec, RefSpec};

pub fn kernel_specs() -> Vec<KernelSpec> {
    // Derive from inventory — same source of truth as the bench registrations
    ::inventory::iter::<crate::spec::BenchSpec>
        .into_iter()
        .filter(|s| s.op == "unary")
        .map(|s| {
            let ref_spec = match &s.class {
                crate::spec::BenchClass::Unary { mlx_pattern: Some(pat), .. } => {
                    // Pattern like "v_Exp{tn}{tn}" → RefSpec::UnaryV("Exp")
                    // Extract the op name between "v_" and "{tn}"
                    let inner = pat.trim_start_matches("v_");
                    let op = inner.split('{').next().unwrap_or(inner);
                    RefSpec::UnaryV(op)
                },
                _ => RefSpec::None("no standalone MLX kernel"),
            };
            KernelSpec {
                op: "unary",
                mt_kernel: s.kernel_name.into(),
                metal_file: "unary.metal",
                ref_spec,
                dtypes: FLOAT_DTYPE_STRS,
            }
        })
        .collect()
}
