//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Softmax benchmark — #[kernel] DSL vs MLX metal/softmax.metal

use metaltile::kernel;

#[kernel]
pub fn mt_softmax<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let nf = n / (lsize * 4u32);
    let mut lm = neg_infinity();
    let mut ls = 0.0f32;
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(inp[base]).cast::<f32>();
        let v1 = load(inp[base + 1u32]).cast::<f32>();
        let v2 = load(inp[base + 2u32]).cast::<f32>();
        let v3 = load(inp[base + 3u32]).cast::<f32>();
        let cm = max(max(v0, v1), max(v2, v3));
        let nm = max(lm, cm);
        let sc = exp(lm - nm);
        let e0 = exp(v0 - nm);
        let e1 = exp(v1 - nm);
        let e2 = exp(v2 - nm);
        let e3 = exp(v3 - nm);
        ls = ls * sc + e0 + e1 + e2 + e3;
        lm = nm;
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(inp[_i]).cast::<f32>();
        let nm = max(lm, xi);
        ls = ls * exp(lm - nm) + exp(xi - nm);
        lm = nm;
    }
    let rm = reduce_max(lm);
    let rsl = ls * exp(lm - rm);
    let rs_sum = reduce_sum(rsl);
    let is = recip(rs_sum);
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let f0 = exp(load(inp[base]).cast::<f32>() - rm) * is;
        let f1 = exp(load(inp[base + 1u32]).cast::<f32>() - rm) * is;
        let f2 = exp(load(inp[base + 2u32]).cast::<f32>() - rm) * is;
        let f3 = exp(load(inp[base + 3u32]).cast::<f32>() - rm) * is;
        store(out[base], f0.cast::<T>());
        store(out[base + 1u32], f1.cast::<T>());
        store(out[base + 2u32], f2.cast::<T>());
        store(out[base + 3u32], f3.cast::<T>());
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let fi = exp(load(inp[_i]).cast::<f32>() - rm) * is;
        store(out[_i], fi.cast::<T>());
    }
}

/// New-syntax correctness for `mt_softmax` (Reduction mode, one threadgroup per
/// row, tpg=256 — `n` must be a multiple of 1024 for the 4-elems/thread loop).
/// Per-row oracle: `exp(x - max) / sum(exp(x - max))` over dtype-rounded inputs.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_softmax;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        setup_with(rows, n, dt, |r, i| ((i % 17) as f32 - 8.0) * 0.1 + r as f32 * 0.05)
    }

    // Shared builder so the standard and large-value tests reuse one oracle.
    fn setup_with(
        rows: usize,
        n: usize,
        dt: DType,
        genf: impl Fn(usize, usize) -> f32,
    ) -> TestSetup {
        let mut inp = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows * n);
        for r in 0..rows {
            let row: Vec<f32> = (0..n).map(|i| genf(r, i)).collect();
            let rd = unpack_f32(&pack_f32(&row, dt), dt);
            let m = rd.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = rd.iter().map(|&x| (x - m).exp()).collect();
            let s: f32 = exps.iter().sum();
            expected.extend(exps.iter().map(|&e| e / s));
            inp.extend_from_slice(&row);
        }
        TestSetup::new(mt_softmax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", rows * n, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-2, 5e-2])]
    fn test_mt_softmax(dt: DType) -> TestSetup { setup(4, 1024, dt) }

    // Large-magnitude logits (50..120): pins the max-subtraction overflow
    // guard. Without it `exp(120)` is +inf; the online-softmax running-max
    // path keeps every term finite and the distribution summing to 1. The
    // CPU oracle subtracts the same max, so GPU and oracle agree exactly
    // (a non-finite GPU result would blow past tol). f32-only — at large
    // magnitudes f16/bf16 input rounding dominates and isn't the contract
    // under test. (Ported from the legacy softmax_large_values_no_overflow.)
    #[test_kernel(dtypes = [f32], tol = [1e-4])]
    fn test_mt_softmax_large_values(dt: DType) -> TestSetup {
        setup_with(2, 1024, dt, |_r, i| 50.0 + (i % 7) as f32 * 10.0)
    }
}

/// New-syntax benchmark for `mt_softmax` (vs MLX `metal/softmax.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_softmax;
    use crate::bench_types::{InputDomain, dtype_tol, input_buffer, mlx_tname};

    // MLX `looped_softmax_*` buffer order: `in`[[buffer(0)]], `out`[[buffer(1)]],
    // `axis_size`(int)[[buffer(2)]]. `inp` is shared by name with the MT input.
    //
    // MLX dispatch geometry: one threadgroup per row (grid=[rows,1,1]) at
    // tpg=1024 — NOT the MT tpg=256. `softmax_looped` uses
    // `threadgroup AccT local_max[32]` / `local_normalizer[32]` without
    // zero-initialising slots past `simd_group_id`; the later
    // `simd_max(local_max[simd_lane_id])` reads all 32. MLX always dispatches
    // these at 1024 threads (n_simd==32, every slot live). At MT's tpg=256
    // (n_simd==8) the 24 stale slots produce NaN. Pin tpg=1024 (mirrors the
    // legacy RowNorm `mlx_tpg: 1024`).
    #[bench(name = "mlx/softmax", dtypes = [f32, f16, bf16])]
    fn bench_softmax(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 1024usize);
        let tn = mlx_tname(dt);
        BenchSetup::new(mt_softmax::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(input_buffer("inp", rows * n, dt, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
            .bytes_moved((2 * rows * n * dt.size_bytes()) as u64)
            .with_reference(
                RefKernel::new(
                    format!("looped_softmax_{tn}"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/softmax.metal")),
                )
                // `inp` shared by name with the MT input above (placeholder).
                .buffer(BenchBuffer::zeros("inp", rows * n, dt))
                .buffer(BenchBuffer::zeros("out", rows * n, dt).output())
                .buffer(BenchBuffer::from_vec(
                    "axis_size",
                    (n as u32).to_le_bytes().to_vec(),
                    DType::U32,
                ))
                .grid(Grid::new_3d(rows as u32, 1, 1, [1024, 1, 1]))
                .tol(dtype_tol(dt).max(1e-4)),
            )
    }
}
