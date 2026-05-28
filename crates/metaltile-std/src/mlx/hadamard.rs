//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Walsh–Hadamard transform along the last axis (size N = 2^k) —
//! port of MLX's `hadamard_n`.
//!
//! Computes `y = H_N · x` where `H_N` is the order-N Hadamard matrix,
//! then scales by `scale`. Used by the Walsh–Hadamard quantization /
//! rotation path (relevant to AURA's rotation matrix).
//!
//! Expressed as the fast Walsh–Hadamard transform: `log2(N)` in-place
//! butterfly passes over a threadgroup buffer. The MLX kernel uses a
//! radix-decomposed multi-step form for register efficiency; this port
//! keeps the plain butterfly — the codegen handles the rest, and one
//! threadgroup per row covers any `N ≤ 1024`. The non-power-of-2
//! `hadamard_m` factor (M ∈ {12,20,28}) is a follow-up.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Reduction mode**, `grid = [rows, 1, 1]`, `tg = [N, 1, 1]`.
//! - `N` a power of two, `32 ≤ N ≤ 1024`; one thread per element.
//!
//! Codegen-only; correctness pinned by
//! `tests/hadamard_gpu_correctness.rs`.

use metaltile::kernel;

#[rustfmt::skip]
macro_rules! hadamard_kernel {
    ($name:ident, $n:literal, $log_n:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] scale: f32) {
            let row = program_id::<0>();
            let base = row * $n;
            threadgroup_alloc("buf", $n, "f32");
            threadgroup_store("buf", tid, load(inp[base + tid]).cast::<f32>());
            threadgroup_barrier();

            // log2(N) butterfly passes; stride h doubles each pass.
            for s in range(0u32, $log_n, 1u32) {
                let h = 1u32 << s;
                if (tid & h) == 0u32 {
                    let a = threadgroup_load("buf", tid);
                    let b = threadgroup_load("buf", tid + h);
                    threadgroup_store("buf", tid, a + b);
                    threadgroup_store("buf", tid + h, a - b);
                }
                threadgroup_barrier();
            }

            store(out[base + tid], (threadgroup_load("buf", tid) * scale).cast::<T>());
        }
    };
}

hadamard_kernel!(mt_hadamard_n64, 64u32, 6u32, "n64");
hadamard_kernel!(mt_hadamard_n128, 128u32, 7u32, "n128");
hadamard_kernel!(mt_hadamard_n256, 256u32, 8u32, "n256");
hadamard_kernel!(mt_hadamard_n512, 512u32, 9u32, "n512");
hadamard_kernel!(mt_hadamard_n1024, 1024u32, 10u32, "n1024");

// ── bottom of source file ─────────────────────────────────────────────────

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    /// `y[i] = scale · Σ_j (-1)^popcount(i&j) · x[j]` per row.
    fn naive_hadamard(x: &[f32], rows: usize, n: usize, scale: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; rows * n];
        for r in 0..rows {
            for i in 0..n {
                let mut acc = 0.0_f32;
                for (j, &xj) in x[r * n..(r + 1) * n].iter().enumerate() {
                    acc += if (i & j).count_ones() % 2 == 0 { xj } else { -xj };
                }
                out[r * n + i] = acc * scale;
            }
        }
        out
    }

    fn ramp(rows: usize, n: usize) -> Vec<f32> {
        (0..rows * n).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect()
    }

    #[test_kernel(name = "mlx/hadamard_n64_f32", dtypes = [f32], tol = 1e-4)]
    fn test_hadamard_n64_f32(dt: DType) -> TestSetup {
        let (rows, n) = (3usize, 64usize);
        let scale = 0.125f32;
        let x = ramp(rows, n);
        let expected = naive_hadamard(&x, rows, n, scale);
        let mut k = mt_hadamard_n64::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; rows * n], dt), dt))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [n as u32, 1, 1])
    }

    #[test_kernel(name = "mlx/hadamard_n128_f32", dtypes = [f32], tol = 1e-4)]
    fn test_hadamard_n128_f32(dt: DType) -> TestSetup {
        let (rows, n) = (4usize, 128usize);
        let scale = 1.0f32;
        let x = ramp(rows, n);
        let expected = naive_hadamard(&x, rows, n, scale);
        let mut k = mt_hadamard_n128::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; rows * n], dt), dt))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [n as u32, 1, 1])
    }

    #[test_kernel(name = "mlx/hadamard_n256_f32", dtypes = [f32], tol = 2e-4)]
    fn test_hadamard_n256_f32(dt: DType) -> TestSetup {
        let (rows, n) = (2usize, 256usize);
        let scale = 0.0625f32;
        let x = ramp(rows, n);
        let expected = naive_hadamard(&x, rows, n, scale);
        let mut k = mt_hadamard_n256::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; rows * n], dt), dt))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [n as u32, 1, 1])
    }

    #[test_kernel(name = "mlx/hadamard_n64_f16", dtypes = [f16], tol = 1e-2)]
    fn test_hadamard_n64_f16(dt: DType) -> TestSetup {
        let (rows, n) = (2usize, 64usize);
        let scale = 0.125f32;
        let x: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
        let expected = naive_hadamard(&x, rows, n, scale);
        let mut k = mt_hadamard_n64::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; rows * n], dt), dt))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [n as u32, 1, 1])
    }

    #[test_kernel(name = "mlx/hadamard_n128_bf16", dtypes = [bf16], tol = 5e-2)]
    fn test_hadamard_n128_bf16(dt: DType) -> TestSetup {
        let (rows, n) = (2usize, 128usize);
        let scale = 1.0f32 / 128.0f32.sqrt();
        let x: Vec<f32> = (0..rows * n).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
        let expected = naive_hadamard(&x, rows, n, scale);
        let mut k = mt_hadamard_n128::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Reduction;
        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&x, dt), dt))
            .input(TestBuffer::from_vec("out", pack(&vec![0.0f32; rows * n], dt), dt))
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [n as u32, 1, 1])
    }
}
