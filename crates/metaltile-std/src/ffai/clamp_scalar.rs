//! Element-wise scalar clamp: `out[i] = min(max(x[i], lo), hi)`.
//!
//! The clamp bound for Gemma 4's `Gemma4ClippableLinear` (vision tower +
//! audio encoder, `use_clipped_linears=True`) is a single scalar per
//! linear — each projection clamps its input to `[input_min, input_max]`
//! and its output to `[output_min, output_max]`, with bounds baked into
//! the checkpoint. This kernel applies one such clamp on the GPU so the
//! vision/audio attention pipeline stays GPU-resident (a CPU clamp would
//! force a commit + readback per projection).
//!
//! `lo` / `hi` are 1-element `f32` buffers (runtime scalars, like the
//! RMSNorm `eps`), so a single compiled kernel serves every distinct bound
//! pair instead of specialising per value.
//!
//! Grid3D — one thread per element, no cross-thread cooperation, so there
//! is no reduction TPG to get wrong (no machine-freeze hazard).
//!
//! Codegen-only. Correctness validated by the inline `kernel_tests` oracle.
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D: grid = `[n, 1, 1]` threadgroups, tpg = `[1, 1, 1]` (one
//!     thread per element). NEVER a reduction TPG.
//!   - `input` and `out` element counts both == `n` (the grid width).
//!   - `lo` / `hi` are 1-element `f32` buffers; `lo <= hi` (caller-enforced
//!     — a degenerate `lo > hi` collapses every element to `hi`).

use metaltile::kernel;

#[kernel]
pub fn ffai_clamp_scalar<T>(input: Tensor<T>, out: Tensor<T>, lo: Tensor<f32>, hi: Tensor<f32>) {
    let i = program_id::<0>();
    let lo_v = load(lo[0]);
    let hi_v = load(hi[0]);
    let x = load(input[i]).cast::<f32>();
    // clamp = min(max(x, lo), hi), via `select` (the DSL lowers method-style
    // `.min()`/`.max()` on f32 inconsistently; `select` is reliable).
    let lo_clamped = select(x < lo_v, lo_v, x);
    let clamped = select(lo_clamped > hi_v, hi_v, lo_clamped);
    store(out[i], clamped.cast::<T>());
}

/// New-syntax correctness for `ffai_clamp_scalar`. Grid3D, grid `[n,1,1]`,
/// tpg `[1,1,1]`. Oracle clamps each element to `[lo, hi]`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_clamp_scalar;
    use crate::utils::{pack_f32, unpack_f32};

    fn f32_bytes(v: f32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-2, 5e-2])]
    fn test_clamp_scalar(dt: DType) -> TestSetup {
        let n = 256usize;
        let lo = -6.375f32;
        let hi = 6.312f32;
        // A ramp that straddles both bounds.
        let input_f: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let exp: Vec<f32> = input.iter().map(|&x| x.max(lo).min(hi)).collect();
        TestSetup::new(ffai_clamp_scalar::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("lo", f32_bytes(lo), DType::F32))
            .input(TestBuffer::from_vec("hi", f32_bytes(hi), DType::F32))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n as u32, 1, 1, [1, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_clamp_scalar`.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_clamp_scalar;

    #[bench(name = "ffai/elementwise/clamp_scalar", dtypes = [f32, f16, bf16])]
    fn bench_clamp_scalar(dt: DType) -> BenchSetup {
        let n = 576 * 768usize; // SigLIP patch-grid activation size
        BenchSetup::new(ffai_clamp_scalar::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::random("lo", 1, DType::F32))
            .buffer(BenchBuffer::random("hi", 1, DType::F32))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .with_shape_label(format!("n{n} {}", crate::bench_types::dtype_label(dt)))
            .grid_3d(n as u32, 1, 1, [1, 1, 1])
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }
}
