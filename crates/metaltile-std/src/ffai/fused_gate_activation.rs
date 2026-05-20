//! Fused dense gate-activation — `out[r, i] = act(gateUp[r, i]) * up`,
//! where `up = gateUp[r, hidden + i]`.
//!
//! Replaces the unfused Split + activation + Multiply (≥ 2 dispatches)
//! with a single dispatch. Hot path in every FFN: the gate and up
//! projections are produced interleaved as one `[rows, 2*hidden]`
//! tensor, and this kernel consumes both halves in place.
//!
//! Layout:
//!   gate_up: `[rows, 2*hidden]` — gate half then up half per row.
//!   out:     `[rows, hidden]`.
//!
//! Three activation variants are separate kernels (no runtime branch):
//!   - `ffai_fused_gate_silu`   — `silu(g) * u`.
//!   - `ffai_fused_gate_gelu`   — `gelu_tanh(g) * u` (tanh approximation).
//!   - `ffai_fused_gate_swiglu` — clipped SwiGLU (GPT-OSS): both halves
//!     clamped to ±7, gate side `g·sigmoid(1.702·g)`, up side `+1` bias.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D, one thread per output element.** `program_id::<0>()` is
//!   the global element index over `rows * hidden`; the consumer picks
//!   `grid = [ceil(rows*hidden / TPG), 1, 1]`, `tg = [TPG, 1, 1]`.
//! - The MLX reference splits into `single_row` / `looped` kernels for
//!   manual `N_READS=4` vectorization. metaltile leaves that to the
//!   codegen vectorize pass, so one Grid3D kernel covers any `hidden`.
//!
//! Codegen-only (the `#[kernel]` body is consumed at compile time);
//! correctness is pinned by `tests/fused_gate_activation_gpu_correctness.rs`.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

/// SiLU gate: `out = silu(gate) * up`.
#[kernel]
pub fn ffai_fused_gate_silu<T>(gate_up: Tensor<T>, out: Tensor<T>, #[constexpr] hidden: u32) {
    let idx = program_id::<0>();
    let row = idx / hidden;
    let col = idx - row * hidden;
    let base = row * 2u32 * hidden;
    let g = load(gate_up[base + col]).cast::<f32>();
    let u = load(gate_up[base + hidden + col]).cast::<f32>();
    store(out[idx], (silu(g) * u).cast::<T>());
}

/// GELU gate (tanh approximation): `out = gelu(gate) * up`.
#[kernel]
pub fn ffai_fused_gate_gelu<T>(gate_up: Tensor<T>, out: Tensor<T>, #[constexpr] hidden: u32) {
    let idx = program_id::<0>();
    let row = idx / hidden;
    let col = idx - row * hidden;
    let base = row * 2u32 * hidden;
    let g = load(gate_up[base + col]).cast::<f32>();
    let u = load(gate_up[base + hidden + col]).cast::<f32>();
    store(out[idx], (gelu(g) * u).cast::<T>());
}

/// Clipped SwiGLU (GPT-OSS): both halves clamped to ±7, gate side uses
/// `sigmoid(1.702·g)`, up side carries a `+1` bias before the multiply.
#[kernel]
pub fn ffai_fused_gate_swiglu<T>(gate_up: Tensor<T>, out: Tensor<T>, #[constexpr] hidden: u32) {
    let idx = program_id::<0>();
    let row = idx / hidden;
    let col = idx - row * hidden;
    let base = row * 2u32 * hidden;
    let g = load(gate_up[base + col]).cast::<f32>();
    let u = load(gate_up[base + hidden + col]).cast::<f32>();
    let gc = min(max(g, -7.0f32), 7.0f32);
    let uc = min(max(u, -7.0f32), 7.0f32);
    let s = sigmoid(1.702f32 * gc);
    store(out[idx], (gc * s * (uc + 1.0f32)).cast::<T>());
}

macro_rules! fga_spec {
    ($name:ident, $subop:literal) => {
        inventory::submit! {
            BenchSpec {
                op: "fused_gate_activation",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: &[DType::F32, DType::F16, DType::BF16],
                tol: 1e-4,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: BenchDispatch::Generic,
                kernel_mode: Some(KernelMode::Grid3D),
            }
        }
    };
}

fga_spec!(ffai_fused_gate_silu, "silu");
fga_spec!(ffai_fused_gate_gelu, "gelu");
fga_spec!(ffai_fused_gate_swiglu, "swiglu");
