//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! SwiGLU activation — `silu(gate) * up`.
//!
//! Fused element-wise activation used in every modern transformer MLP
//! (Llama 4, Qwen3 dense + MoE, Gemma, Mistral families): given two
//! equally-sized inputs `gate` and `up` (the two halves of an MLP's
//! `w_gate · x` and `w_up · x` outputs), produce
//!
//! ```text
//!   out[i] = silu(gate[i]) * up[i]
//!         = (gate[i] * sigmoid(gate[i])) * up[i]
//! ```
//!
//! Existing baseline: two separate kernel launches — one applies
//! `silu(gate)` elementwise (`mt_silu` in `unary.rs`), the second
//! multiplies by `up` (`mt_binary` mul). Each load+store cycles the
//! intermediate `silu(gate)` value through device memory.
//!
//! Fusion saves one full-tensor RMW: the intermediate value stays in
//! registers, halving global memory traffic on the activation path.
//! At Qwen3-MoE expert intermediate=768 × prefill 512 tokens =
//! ~400KB per layer per expert; across 48 layers × 8 active experts
//! the saved bandwidth adds up.
//!
//! MLX reference: `mx.fast.swiglu` lives in
//! `mlx/mlx/backend/metal/kernels/fast.metal` as a single launch with
//! `silu(g) * u` in the body. We mirror that pattern.
//!
//! ## Cross-kernel calling
//!
//! `mt_swiglu` calls `mt_silu` via the DSL cross-kernel call syntax
//! (just the kernel name). `KernelInlinePass` splices the silu body
//! inline before MSL emission — no extra memory round-trip, same code
//! quality as a manual inline, with a clear compositional structure
//! that future fusion passes can reason about.
//!
//! Type-efficiency: `g` and `u` are loaded and cast to f32 before the
//! call. `KernelInlinePass` replaces `mt_silu`'s input-param load with
//! the actual f32 arg, so all arithmetic stays in f32 regardless of T.
//! No T→f32→T precision loss in the silu path.

use metaltile::kernel;

#[kernel]
pub fn mt_swiglu<T>(gate: Tensor<T>, up: Tensor<T>, out: Tensor<T>) {
    let idx = tid;
    let g = load(gate[idx]).cast::<f32>();
    let u = load(up[idx]).cast::<f32>();
    // Cross-kernel call: KernelInlinePass splices mt_silu's scalar body
    // here. mt_silu's input-param load is replaced by g (already f32),
    // so silu runs in f32. Future fusion passes can identify the
    // (silu, mul) → swiglu composition pattern from this call site.
    let s = mt_silu(g);
    store(out[idx], (s * u).cast::<T>());
}

// ── bottom of source file ────────────────────────────────────────────────

mod tests_support {
    #![allow(unused, dead_code)]
    use super::*;
    use metaltile::test_kernel;
    use metaltile_core::{DType, bench::{TestSetup, TestBuffer}};

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        use DType::*;
        match dt {
            F32  => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            F16  => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            BF16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _    => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn round(v: f32, dt: DType) -> f32 {
        match dt {
            DType::F32 => v,
            DType::F16 => half::f16::from_f32(v).to_f32(),
            DType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        }
    }

    fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect()
    }

    #[test_kernel(name = "mlx/swiglu", dtypes = [f32], tol = 1e-5)]
    fn test_swiglu_f32(dt: DType) -> TestSetup {
        let n = 1024usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017) % 6.0 - 3.0).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.029) % 4.0 - 2.0).collect();
        let expected = cpu_swiglu(&gate, &up);
        TestSetup::new(mt_swiglu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack(&up, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/swiglu_f16", dtypes = [f16], tol = 5e-3)]
    fn test_swiglu_f16(dt: DType) -> TestSetup {
        let n = 2048usize;
        let gate: Vec<f32> =
            (0..n).map(|i| round((i as f32 * 0.013) % 8.0 - 4.0, dt)).collect();
        let up: Vec<f32> =
            (0..n).map(|i| round((i as f32 * 0.021) % 3.0 - 1.5, dt)).collect();
        let expected = cpu_swiglu(&gate, &up);
        TestSetup::new(mt_swiglu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack(&up, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "mlx/swiglu_bf16", dtypes = [bf16], tol = 2e-2)]
    fn test_swiglu_bf16(dt: DType) -> TestSetup {
        let n = 1024usize;
        let gate: Vec<f32> =
            (0..n).map(|i| round((i as f32 * 0.019) % 6.0 - 3.0, dt)).collect();
        let up: Vec<f32> =
            (0..n).map(|i| round((i as f32 * 0.023) % 4.0 - 2.0, dt)).collect();
        let expected = cpu_swiglu(&gate, &up);
        TestSetup::new(mt_swiglu::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("gate", pack(&gate, dt), dt))
            .input(TestBuffer::from_vec("up", pack(&up, dt), dt))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }
}
