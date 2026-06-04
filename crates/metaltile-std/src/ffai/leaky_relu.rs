//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Element-wise Leaky ReLU: `out[i] = x[i] > 0 ? x[i] : slope·x[i]`.
//!
//! The activation between every conv in the StyleTTS2 / Kokoro decoder and
//! iSTFTNet generator (`nn.LeakyReLU(0.2)` in the AdaIN res-blocks, `0.1`
//! inside the generator up-sample loop, `0.01` at the generator tail). The
//! slope varies per call site, so it is a runtime 1-element `f32` buffer
//! (like the RMSNorm `eps`) — one compiled kernel serves every slope rather
//! than specialising per value. Keeps the vocoder pipeline GPU-resident (a
//! CPU activation would force a commit + readback per conv over thousands of
//! frames).
//!
//! Grid3D — one thread per element, no cross-thread cooperation, so there is
//! no reduction TPG to get wrong (no machine-freeze hazard).
//!
//! ## DISPATCH INVARIANTS
//!   - Grid3D: grid = `[n, 1, 1]` threadgroups, tpg = `[1, 1, 1]` (one
//!     thread per element). NEVER a reduction TPG.
//!   - `input` and `out` element counts both == `n` (the grid width).
//!   - `slope` is a 1-element `f32` buffer.

use metaltile::kernel;

#[kernel]
pub fn ffai_leaky_relu<T>(input: Tensor<T>, out: Tensor<T>, slope: Tensor<f32>) {
    let i = program_id::<0>();
    let s = load(slope[0]);
    let x = load(input[i]).cast::<f32>();
    // out = x > 0 ? x : slope·x
    let y = select(x > 0.0f32, x, x * s);
    store(out[i], y.cast::<T>());
}

/// New-syntax correctness for `ffai_leaky_relu`. Grid3D, grid `[n,1,1]`, tpg
/// `[1,1,1]`. Oracle applies the leaky rectifier per element.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_leaky_relu;
    use crate::utils::{pack_f32, unpack_f32};

    fn f32_bytes(v: f32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 1e-2, 5e-2])]
    fn test_leaky_relu(dt: DType) -> TestSetup {
        let n = 257usize;
        let slope = 0.2f32;
        // A ramp straddling zero so both branches are exercised.
        let input_f: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let input = unpack_f32(&pack_f32(&input_f, dt), dt);
        let exp: Vec<f32> = input.iter().map(|&x| if x > 0.0 { x } else { slope * x }).collect();
        TestSetup::new(ffai_leaky_relu::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("input", pack_f32(&input_f, dt), dt))
            .input(TestBuffer::from_vec("slope", f32_bytes(slope), DType::F32))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&exp, dt), dt))
            .grid_3d(n as u32, 1, 1, [1, 1, 1])
    }
}

/// New-syntax bench: a generator-tail activation (128 ch × 7801 frames).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_leaky_relu;

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_leaky_relu(dt: DType) -> BenchSetup {
        let n = 128usize * 7801usize;
        BenchSetup::new(ffai_leaky_relu::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::from_vec("slope", 0.1f32.to_le_bytes().to_vec(), DType::F32))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
            // 1 compare + 1 mul per element.
            .flops((n as u64) * 2)
    }
}
