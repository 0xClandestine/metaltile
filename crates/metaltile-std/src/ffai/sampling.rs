//! GPU sampling kernels — softmax + categorical inverse-CDF walk used
//! by FFAI's `gpu-categorical` decode path (T > 0, no filters). The
//! greedy fast path uses `argmax` instead.
//!
//! Codegen-only. End-to-end sampling correctness lives in FFAI's
//! harness.
//!
//! ## Known DSL limitations (body parser)
//! - `while` loops are silently dropped → use explicit if-blocks for reductions
//! - `return` is silently dropped → use `if/else` style branching
//! - Rust declarative macros inside #[kernel] are dropped → inline all code
//! - threadgroup_alloc declarations are hoisted to function scope → names must
//!   be unique across all branches (greedy uses tg_gmax/tg_gidx, temperature
//!   uses tg_max/tg_sum)

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

// Softmax + categorical sample over a 1D logits tensor. Cooperative
// reduction (256 threads) for max + sum-exp; single-thread inverse
// CDF walk for the categorical pick.
//
// Inputs:
//   inp            — logits [n]
//   out            — token id [1] (u32)
//   temperature_in — temperature [1] (f32, ≥ 0; 0 → greedy argmax)
//   uniform_in     — uniform draw in [0, 1) [1] (f32)
//
// When temperature < 1e-7 the kernel runs argmax instead to avoid
// 1/0 = Inf propagating through the softmax.
//
// Both paths use a 256-thread cooperative tree reduction.
// Greedy uses tg_gmax + tg_gidx; temperature uses tg_max + tg_sum.
// All four are allocated at the top level (hoisted by the DSL) so the
// names must not collide.
#[kernel]
pub fn softmax_categorical_sample<T>(
    inp: Tensor<T>,
    out: Tensor<u32>,
    temperature_in: Tensor<f32>,
    uniform_in: Tensor<f32>,
    #[constexpr] n: u32,
) {
    // Hoist all four threadgroup arrays (DSL lifts threadgroup_alloc to
    // function scope regardless of which branch uses them).
    threadgroup_alloc("tg_gmax", 256); // greedy: per-thread local max
    threadgroup_alloc("tg_gidx", 256); // greedy: index of per-thread max
    threadgroup_alloc("tg_max",  256); // temperature: per-thread local max (scaled)
    threadgroup_alloc("tg_sum",  256); // temperature: per-thread local sum-exp

    let lid  = tid;
    let temp = load(temperature_in[0]);
    let n_iters = (n + lsize - 1u32) / lsize;

    // ─── Greedy (argmax) path — T ≈ 0 ───────────────────────────────
    // All 256 threads read the same temp → uniform branch, so barriers
    // inside this block are safe.
    if temp < 1e-7f32 {
        let mut glocal_max = neg_infinity();
        let mut glocal_idx = 0u32;
        for _r in range(0u32, n_iters, 1u32) {
            let pos = _r * lsize + lid;
            if pos < n {
                let v = load(inp[pos]).cast::<f32>();
                let better = v > glocal_max;
                glocal_max = select(better, v,   glocal_max);
                glocal_idx = select(better, pos, glocal_idx);
            }
        }
        threadgroup_store("tg_gmax", lid, glocal_max);
        threadgroup_store("tg_gidx", lid, glocal_idx);
        threadgroup_barrier();

        // 8-stage binary tree: reduce max, carrying the winning index.
        // tg_gidx stores float(u32) — exact for indices < 2^24.
        if lid < 128u32 {
            let og = threadgroup_load("tg_gmax", lid + 128u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 128u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();
        if lid < 64u32 {
            let og = threadgroup_load("tg_gmax", lid + 64u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 64u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();
        if lid < 32u32 {
            let og = threadgroup_load("tg_gmax", lid + 32u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 32u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();
        if lid < 16u32 {
            let og = threadgroup_load("tg_gmax", lid + 16u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 16u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();
        if lid < 8u32 {
            let og = threadgroup_load("tg_gmax", lid + 8u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 8u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();
        if lid < 4u32 {
            let og = threadgroup_load("tg_gmax", lid + 4u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 4u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();
        if lid < 2u32 {
            let og = threadgroup_load("tg_gmax", lid + 2u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 2u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();
        if lid < 1u32 {
            let og = threadgroup_load("tg_gmax", lid + 1u32);
            let tg = threadgroup_load("tg_gmax", lid);
            let bt = og > tg;
            threadgroup_store("tg_gmax", lid, select(bt, og, tg));
            threadgroup_store("tg_gidx", lid, select(bt,
                threadgroup_load("tg_gidx", lid + 1u32),
                threadgroup_load("tg_gidx", lid)));
        }
        threadgroup_barrier();

        if lid == 0u32 {
            // tg_gidx stores float(u32) — cast back to u32 for output.
            store(out[0], threadgroup_load("tg_gidx", 0u32).cast::<u32>());
        }
    }

    // ─── Temperature sampling path — T > 0 ──────────────────────────
    // `return` is dropped by the body parser, so guard with if temp >= 1e-7.
    if temp >= 1e-7f32 {
        let inv_t = 1.0f32 / temp;

        // Pass 1: cooperative max reduce
        let mut local_max = neg_infinity();
        for _r in range(0u32, n_iters, 1u32) {
            let pos = _r * lsize + lid;
            if pos < n {
                let v = load(inp[pos]).cast::<f32>() * inv_t;
                local_max = select(v > local_max, v, local_max);
            }
        }
        threadgroup_store("tg_max", lid, local_max);
        threadgroup_barrier();

        // 8-stage binary tree reduction for global max
        if lid < 128u32 {
            let ov = threadgroup_load("tg_max", lid + 128u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
        if lid < 64u32 {
            let ov = threadgroup_load("tg_max", lid + 64u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
        if lid < 32u32 {
            let ov = threadgroup_load("tg_max", lid + 32u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
        if lid < 16u32 {
            let ov = threadgroup_load("tg_max", lid + 16u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
        if lid < 8u32 {
            let ov = threadgroup_load("tg_max", lid + 8u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
        if lid < 4u32 {
            let ov = threadgroup_load("tg_max", lid + 4u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
        if lid < 2u32 {
            let ov = threadgroup_load("tg_max", lid + 2u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();
        if lid < 1u32 {
            let ov = threadgroup_load("tg_max", lid + 1u32);
            let tv = threadgroup_load("tg_max", lid);
            threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
        }
        threadgroup_barrier();

        let max_val = threadgroup_load("tg_max", 0u32);

        // Pass 2: cooperative sum-exp reduce
        let mut local_sum = 0.0f32;
        for _r in range(0u32, n_iters, 1u32) {
            let pos = _r * lsize + lid;
            if pos < n {
                let v = load(inp[pos]).cast::<f32>() * inv_t;
                local_sum = local_sum + exp(v - max_val);
            }
        }
        threadgroup_store("tg_sum", lid, local_sum);
        threadgroup_barrier();

        // 8-stage binary tree reduction for global sum
        if lid < 128u32 {
            let ov = threadgroup_load("tg_sum", lid + 128u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();
        if lid < 64u32 {
            let ov = threadgroup_load("tg_sum", lid + 64u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();
        if lid < 32u32 {
            let ov = threadgroup_load("tg_sum", lid + 32u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();
        if lid < 16u32 {
            let ov = threadgroup_load("tg_sum", lid + 16u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();
        if lid < 8u32 {
            let ov = threadgroup_load("tg_sum", lid + 8u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();
        if lid < 4u32 {
            let ov = threadgroup_load("tg_sum", lid + 4u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();
        if lid < 2u32 {
            let ov = threadgroup_load("tg_sum", lid + 2u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();
        if lid < 1u32 {
            let ov = threadgroup_load("tg_sum", lid + 1u32);
            let tv = threadgroup_load("tg_sum", lid);
            threadgroup_store("tg_sum", lid, ov + tv);
        }
        threadgroup_barrier();

        let total = threadgroup_load("tg_sum", 0u32);

        // Pass 3: single-thread inverse CDF walk
        if lid == 0u32 {
            let target = load(uniform_in[0]) * total;
            let mut cum = 0.0f32;
            let mut found_idx = n - 1u32; // fallback to last index
            let mut done = 0u32;
            for i in range(0u32, n, 1u32) {
                let v = load(inp[i]).cast::<f32>() * inv_t;
                cum = cum + exp(v - max_val);
                let hit = (cum >= target) & (done == 0u32);
                found_idx = select(hit, i, found_idx);
                done = select(hit, 1u32, done);
            }
            store(out[0], found_idx);
        }
    }
}

inventory::submit! {
    BenchSpec {
        op: "sampling",
        subop: "softmax_categorical_sample",
        kernel_name: "softmax_categorical_sample",
        kernel_ir: softmax_categorical_sample::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Reduction),
    }
}
