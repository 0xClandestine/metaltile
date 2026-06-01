//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Ternary select benchmark — #[kernel] DSL vs MLX metal/ternary.metal
//!
//! MLX kernel: v_Selectfloat32 / v_Selectfloat16 / v_Selectbfloat16 (ternary.metal)
//!   Params: (cond: device T*, a: device T*, b: device T*, dst: device T*,
//!            size: constant uint&) — slots [0, 1, 2, 3, 4]
//!   Grid: [ceil(N/TPG), 1, 1] × [TPG, 1, 1]
//!   Algorithm: dst[i] = cond[i] != 0 ? a[i] : b[i]  (one thread per element)
//!
//! MetalTile: mt_select — same algorithm via #[kernel] DSL.
//!   KernelMode::Elementwise

use metaltile::kernel;

#[kernel]
pub fn mt_select<T>(cond: Tensor<u8>, on_true: Tensor<T>, on_false: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let c = load(cond[idx]);
    let t = load(on_true[idx]);
    let f = load(on_false[idx]);
    store(out[idx], select(c, t, f));
}

/// New-syntax correctness for `mt_select` (elementwise, exact — picks an input
/// verbatim, so the result is bit-exact in every dtype).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_select;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(n: usize, dt: DType) -> TestSetup {
        let cond: Vec<u8> = (0..n).map(|i| (i % 3 != 0) as u8).collect();
        let t: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.1 - 0.5).collect();
        let f: Vec<f32> = (0..n).map(|i| -((i % 13) as f32) * 0.1 + 0.3).collect();
        let t_dt = unpack_f32(&pack_f32(&t, dt), dt);
        let f_dt = unpack_f32(&pack_f32(&f, dt), dt);
        let expected: Vec<f32> =
            (0..n).map(|i| if cond[i] != 0 { t_dt[i] } else { f_dt[i] }).collect();
        TestSetup::new(mt_select::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("cond", cond, DType::U8))
            .input(TestBuffer::from_vec("on_true", pack_f32(&t, dt), dt))
            .input(TestBuffer::from_vec("on_false", pack_f32(&f, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
    fn test_mt_select(dt: DType) -> TestSetup { setup(512, dt) }
}

/// New-syntax benchmark for `mt_select` (vs MLX `metal/ternary.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_select;
    use crate::bench_types::{InputDomain, dtype_tol, input_buffer, mlx_tname};

    // MLX `metal/ternary.metal` `v_Select<tn>` (`ternary_v`, 1 element/thread):
    //   a=cond (device const bool*) [[buffer(0)]], b=on_true [[buffer(1)]],
    //   c=on_false [[buffer(2)]], d=out [[buffer(3)]], size (uint) [[buffer(4)]].
    // The reference binds cond/on_true/on_false by name (shared with the MT
    // inputs, identical data), its fresh `out`, then the U32 element count.
    //
    // Select picks an input verbatim, so it's bit-exact in every dtype; legacy
    // spec used tol=1e-4 (covered by the per-dtype `dtype_tol` floor). Inputs are
    // seeded deterministically: `cond`'s `Positive` 0.25.. ramp value-casts to a
    // repeating mix of 0 and nonzero u8 bytes (so both select branches are
    // exercised), and on_true/on_false use the `Signed` pattern. Both kernels see
    // identical cond/on_true/on_false bytes, so the A/B is exact.
    #[bench(name = "mlx/select", dtypes = [f32, f16, bf16])]
    fn bench_select(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        let tn = mlx_tname(dt);
        BenchSetup::new(mt_select::kernel_ir_for(dt))
            .buffer(input_buffer("cond", n, DType::U8, InputDomain::Positive))
            .buffer(input_buffer("on_true", n, dt, InputDomain::Signed))
            .buffer(input_buffer("on_false", n, dt, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            // cond (1 byte) + two reads + one write of the dtype.
            .bytes_moved((n + 3 * n * dt.size_bytes()) as u64)
            .with_reference(
                RefKernel::new(
                    format!("v_Select{tn}"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/ternary.metal")),
                )
                // cond/on_true/on_false shared by name with the MT inputs above
                // (placeholders; the runner overrides them with the MT bytes).
                .buffer(BenchBuffer::zeros("cond", n, DType::U8))
                .buffer(BenchBuffer::zeros("on_true", n, dt))
                .buffer(BenchBuffer::zeros("on_false", n, dt))
                .buffer(BenchBuffer::zeros("out", n, dt).output())
                .buffer(BenchBuffer::from_vec("n", (n as u32).to_le_bytes().to_vec(), DType::U32))
                .grid_1d(n, 256)
                .tol(dtype_tol(dt)),
            )
    }
}
