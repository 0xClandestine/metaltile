//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Dynamic-M qmm path — host-side driver for batched-T quantized matmul.
//!
//! Closes the bandwidth-bound prefill gap in FFAI. The existing
//! `mt_qmm_mma`, `mt_qmm_mma_m16`, `mt_qmm_bm4`, `mt_qmm_bm2`, `mt_qmm`
//! kernels each handle a *fixed* M class — `mt_qmm_mma` requires `M % 32 == 0`,
//! `mt_qmm_mma_m16` is hard-wired to `M = 16`, etc. The model-level
//! prefill entry point (`Qwen35Model.forwardMany`) has a logical
//! token count `T` that is arbitrary (T=1 decode, T=37 ragged
//! chunk, T=4096 production prefill cell). Without a path that
//! accepts any `T` per dispatch, the model is forced into a
//! per-token loop and reads the full int4 weight tile once per
//! token — bandwidth-bound, 70× slower than MLX at T=32K.
//!
//! This module provides the host-side dispatcher that pads `T` to
//! the next multiple of 32 and routes to `mt_qmm_mma` for the full
//! BM=BN=BK=32 simdgroup-matrix tile. The kernel itself is unchanged
//! (one dispatch reads each int4 weight tile once and produces
//! `m_padded × N` outputs). The caller discards the trailing
//! `m_padded - T` rows of the output. Padding the X buffer with
//! zeros makes the masked rows valid (zero contribution to the
//! valid outputs) and the trailing rows numerically defined.
//!
//! Routing: `mt_qmm_mma` over `mt_qmm_mma_mpp`. The MPP variant only
//! ships fp32 / fp16 (see `quantized_mpp.rs` — the InlineMSL is
//! per-dtype templated and asserts `F32 | F16`). Production prefill
//! for Qwen3.6-A3B runs bf16, so we use the DSL-generic `mt_qmm_mma`
//! that supports `F32 | F16 | BF16` via `#[kernel]` generics. The
//! `dispatch_padded_grid` helper is dtype-agnostic.
//!
//! ## Composition with FFAI's `Ops.dequantGemm`
//!
//! The Swift-side `Ops.dequantGemm(x, w, scales, biases, ...)` calls
//! `mt_qmm_for(dtype, m)` today — which only handles fixed-class M.
//! After this lands, `Ops.dequantGemm` can call into the dynamic-M
//! path by:
//!   1. Padding `x` to `m_padded` rows (`(T + 31) / 32 * 32`).
//!   2. Calling `dispatch_padded_grid` with the padded shape.
//!   3. Slicing the first `T` rows of the output.
//!
//! No changes to the kernel binaries or the per-dtype emit are
//! needed — `mt_qmm_mma` is already in the kernel pack at every
//! shipped dtype.

use metaltile_core::{dtype::DType, ir::Kernel};

use crate::mlx::quantized::{mt_qmm_mma, patch_qmm_mma_dtype_aware_skew};

/// Tile geometry mirrors `mt_qmm_mma`. Exposed for callers sizing
/// the dispatch grid + the M-padding step.
pub const BM_TILE: u32 = 32;
pub const BN_TILE: u32 = 32;
pub const BK_TILE: u32 = 32;
/// Threads per group — 4 SG × 32 lanes — matches `mt_qmm_mma`.
pub const TPG: u32 = 128;

/// Round `t` up to the next multiple of [`BM_TILE`] (32). The
/// padded value is the `m` we hand to the kernel; the caller
/// discards the trailing `m_padded - t` output rows.
///
/// ```ignore
/// assert_eq!(pad_t_to_bm(1), 32);
/// assert_eq!(pad_t_to_bm(32), 32);
/// assert_eq!(pad_t_to_bm(33), 64);
/// assert_eq!(pad_t_to_bm(4096), 4096);
/// ```
pub const fn pad_t_to_bm(t: usize) -> usize {
    let bm = BM_TILE as usize;
    t.div_ceil(bm) * bm
}

/// Pad an X buffer `[t, k]` to `[m_padded, k]` by appending zero
/// rows. `x` is a row-major fp byte stream (`f32 = 4B`, `f16 = 2B`,
/// `bf16 = 2B`). The trailing rows are zero-filled so their
/// contribution to any output column is exactly zero — the kernel's
/// `Σ q · x_row + bias · Σ x_row` term collapses to `bias · 0 + 0`
/// on padded rows. (We discard those output rows anyway, but zero
/// padding is the defensible value.)
pub fn pad_x_rows_bytes(x_bytes: &[u8], t: usize, k: usize, bytes_per_elem: usize) -> Vec<u8> {
    let m_padded = pad_t_to_bm(t);
    let row_bytes = k * bytes_per_elem;
    assert_eq!(x_bytes.len(), t * row_bytes, "x_bytes must be t * k * bytes_per_elem");
    let mut out = Vec::with_capacity(m_padded * row_bytes);
    out.extend_from_slice(x_bytes);
    out.resize(m_padded * row_bytes, 0);
    out
}

/// Build the kernel IR for the dynamic-M path. Returns
/// `mt_qmm_mma::kernel_ir_for(dtype)` with the dtype-aware TG skew
/// patch applied (matches the path in `mt_qmm_for` for `M % 32 == 0`).
/// The caller dispatches with grid `[N / 32, m_padded / 32, 1]`
/// and `tpg = 128`.
pub fn kernel_ir_for(dtype: DType) -> Kernel {
    let mut k = mt_qmm_mma::kernel_ir_for(dtype);
    patch_qmm_mma_dtype_aware_skew(&mut k, dtype);
    k.mode = metaltile_core::ir::KernelMode::Reduction;
    k
}

/// Dispatch grid for the dynamic-M path given a *logical* token
/// count `t` and a row width `n`. Returns the threadgroup grid
/// `[N / 32, m_padded / 32, 1]`. Caller still owns `tpg = [128, 1, 1]`.
pub fn dispatch_grid(t: usize, n: usize) -> [usize; 3] {
    assert!(n.is_multiple_of(BN_TILE as usize), "n must be multiple of {} (BN tile)", BN_TILE);
    let m_padded = pad_t_to_bm(t);
    [n / BN_TILE as usize, m_padded / BM_TILE as usize, 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_t_to_bm_rounds_up_to_multiple_of_32() {
        assert_eq!(pad_t_to_bm(0), 0);
        assert_eq!(pad_t_to_bm(1), 32);
        assert_eq!(pad_t_to_bm(31), 32);
        assert_eq!(pad_t_to_bm(32), 32);
        assert_eq!(pad_t_to_bm(33), 64);
        assert_eq!(pad_t_to_bm(37), 64);
        assert_eq!(pad_t_to_bm(64), 64);
        assert_eq!(pad_t_to_bm(4096), 4096);
        assert_eq!(pad_t_to_bm(4097), 4128);
    }

    #[test]
    fn dispatch_grid_pads_m_axis() {
        // T=1 decode → 1 TG in M, N/32 TGs in N.
        assert_eq!(dispatch_grid(1, 128), [4, 1, 1]);
        // T=37 ragged → ceil(37/32) = 2 TGs in M.
        assert_eq!(dispatch_grid(37, 128), [4, 2, 1]);
        // T=4096 production → 128 TGs in M.
        assert_eq!(dispatch_grid(4096, 2048), [64, 128, 1]);
    }

    #[test]
    fn pad_x_rows_zero_fills_trailing() {
        // T=2, K=4, 2 bytes/elem (f16/bf16) → 16 bytes input.
        let x = vec![0x01u8; 16];
        let padded = pad_x_rows_bytes(&x, 2, 4, 2);
        // m_padded = 32, k=4, 2B → 256 bytes total.
        assert_eq!(padded.len(), 32 * 4 * 2);
        // First 16 bytes preserved.
        assert!(padded[..16].iter().all(|&b| b == 0x01));
        // Rest zero.
        assert!(padded[16..].iter().all(|&b| b == 0));
    }

    #[test]
    fn kernel_ir_for_returns_mt_qmm_mma_per_dtype() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = kernel_ir_for(dt);
            assert_eq!(k.name, "mt_qmm_mma", "dynamic-M routes to mt_qmm_mma for dtype {:?}", dt);
            assert_eq!(k.mode, metaltile_core::ir::KernelMode::Reduction);
        }
    }
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    use super::*;

    fn pack_f32(vals: &[f32]) -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() }
    fn pack_f16(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect()
    }
    fn pack_bf16(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect()
    }
    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => pack_f32(vals),
            DType::F16 => pack_f16(vals),
            DType::BF16 => pack_bf16(vals),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round_dt(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn bpe(dt: DType) -> usize {
        match dt {
            DType::F32 => 4,
            _ => 2,
        }
    }

    /// Triple-loop CPU oracle — same algorithm as the kernel.
    #[allow(clippy::too_many_arguments)]
    fn cpu_qmm_reference(
        w: &[u32],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for m_row in 0..m {
            for n_col in 0..n {
                let mut acc = 0.0f32;
                for g in 0..gs_per_row {
                    let s = scales[n_col * gs_per_row + g];
                    let bias = biases[n_col * gs_per_row + g];
                    let mut q_dot = 0.0f32;
                    let mut x_sum = 0.0f32;
                    for p in 0..8usize {
                        let packed = w[n_col * k / 8 + g * 8 + p];
                        for bit in 0..8u32 {
                            let q = ((packed >> (bit * 4)) & 0xF) as f32;
                            let xv = x[m_row * k + g * group_size + p * 8 + bit as usize];
                            q_dot += q * xv;
                            x_sum += xv;
                        }
                    }
                    acc += s * q_dot + bias * x_sum;
                }
                out[m_row * n + n_col] = acc;
            }
        }
        out
    }

    fn build_quant_inputs(
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let w: Vec<u32> = (0..n * k / 8)
            .map(|i| {
                let mut v = 0u32;
                for bit in 0..8u32 {
                    v |= ((i as u32 + bit) & 0xF) << (bit * 4);
                }
                v
            })
            .collect();
        let scales: Vec<f32> = (0..n * gs_per_row).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let biases: Vec<f32> = (0..n * gs_per_row).map(|i| (i as f32) * 0.0001).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 1.0 + (i as f32) * 0.001).collect();
        (w, scales, biases, x)
    }

    fn build_quant_inputs_small_mag(
        m: usize,
        n: usize,
        k: usize,
        gs_per_row: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let w: Vec<u32> =
            (0..n * k / 8).map(|i| ((i as u32) % 17).wrapping_mul(0x12345678u32)).collect();
        let scales: Vec<f32> =
            (0..n * gs_per_row).map(|i| 0.005 + ((i % 7) as f32) * 0.0007).collect();
        let biases: Vec<f32> = (0..n * gs_per_row).map(|i| ((i % 5) as f32) * 0.00005).collect();
        let x: Vec<f32> = (0..m * k).map(|i| 0.05 + ((i % 23) as f32) * 0.003).collect();
        (w, scales, biases, x)
    }

    fn make_setup(dt: DType, t: usize, n: usize, k: usize, small_mag: bool) -> TestSetup {
        let group_size = 64usize;
        let gs_per_row = k / group_size;
        let (w, scales_f32, biases_f32, x_f32) = if small_mag {
            build_quant_inputs_small_mag(t, n, k, gs_per_row)
        } else {
            build_quant_inputs(t, n, k, gs_per_row)
        };
        let scales: Vec<f32> = scales_f32.iter().map(|&v| round_dt(v, dt)).collect();
        let biases: Vec<f32> = biases_f32.iter().map(|&v| round_dt(v, dt)).collect();
        let x: Vec<f32> = x_f32.iter().map(|&v| round_dt(v, dt)).collect();
        let expected = cpu_qmm_reference(&w, &scales, &biases, &x, t, n, k, gs_per_row, group_size);

        let m_padded = pad_t_to_bm(t);
        let be = bpe(dt);
        let x_padded = pad_x_rows_bytes(&pack(x_f32.as_slice(), dt), t, k, be);
        let out_zeros = vec![0u8; m_padded * n * be];

        let grid = dispatch_grid(t, n);
        let (gx, gy, gz) = (grid[0] as u32, grid[1] as u32, grid[2] as u32);

        TestSetup::new(kernel_ir_for(dt))
            .input(TestBuffer::from_vec("w", pack_u32(&w), DType::U32))
            .input(TestBuffer::from_vec("scales", pack(&scales, dt), dt))
            .input(TestBuffer::from_vec("biases", pack(&biases, dt), dt))
            .input(TestBuffer::from_vec("x", x_padded, dt))
            .input(TestBuffer::from_vec("out", out_zeros, dt))
            .expect(TestBuffer::from_vec(
                "out",
                {
                    // Pad oracle with zeros for trailing rows, then take first T*N
                    let mut out_padded = pack(&vec![0.0f32; m_padded * n], dt);
                    let expected_bytes = pack(&expected, dt);
                    out_padded[..expected_bytes.len()].copy_from_slice(&expected_bytes);
                    out_padded
                },
                dt,
            ))
            .constexpr("k", k as u32)
            .constexpr("n", n as u32)
            .constexpr("gs_per_row", gs_per_row as u32)
            .grid_3d(gx, gy, gz, [128, 1, 1])
    }

    #[test_kernel(name = "mlx/qmm_mma_dynamic_m_f16_t1", dtypes = [f16], tol = 5e-1)]
    fn test_dynamic_m_f16_t1(dt: DType) -> TestSetup { make_setup(dt, 1, 128, 128, false) }

    #[test_kernel(name = "mlx/qmm_mma_dynamic_m_f16_t8", dtypes = [f16], tol = 5e-1)]
    fn test_dynamic_m_f16_t8(dt: DType) -> TestSetup { make_setup(dt, 8, 128, 128, false) }

    #[test_kernel(name = "mlx/qmm_mma_dynamic_m_f16_t64", dtypes = [f16], tol = 5e-1)]
    fn test_dynamic_m_f16_t64(dt: DType) -> TestSetup { make_setup(dt, 64, 512, 2048, true) }

    #[test_kernel(name = "mlx/qmm_mma_dynamic_m_f32_t32", dtypes = [f32], tol = 1e-2)]
    fn test_dynamic_m_f32_t32(dt: DType) -> TestSetup { make_setup(dt, 32, 64, 128, false) }

    #[test_kernel(name = "mlx/qmm_mma_dynamic_m_f16_t37_ragged", dtypes = [f16], tol = 5e-1)]
    fn test_dynamic_m_f16_t37(dt: DType) -> TestSetup { make_setup(dt, 37, 128, 128, false) }
}
