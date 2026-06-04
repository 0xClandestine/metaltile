//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! RoPE benchmark — #[kernel] DSL vs MLX metal/rope.metal

use metaltile::kernel;

#[kernel]
pub fn mt_rope<T>(
    inp: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] h_stride: u32,
    #[constexpr] seq_stride: u32,
    #[constexpr] grid_x: u32,
    #[constexpr] base: f32,
) {
    let px = program_id::<0>();
    let py = program_id::<1>();
    let pz = program_id::<2>();
    let px_f = px.cast::<f32>();
    let gx_f = grid_x.cast::<f32>();
    let d_norm = px_f / gx_f;
    let inv_freq = exp2(-(d_norm * base));
    let theta = py.cast::<f32>() * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    let head_base = pz * 4;
    for i in range(0, 4, 1) {
        let head = head_base + i;
        let idx1 = py * seq_stride + head * h_stride + px;
        let idx2 = idx1 + grid_x;
        let x1 = load(inp[idx1]).cast::<f32>();
        let x2 = load(inp[idx2]).cast::<f32>();
        let rx1 = x1 * cos_t - x2 * sin_t;
        let rx2 = x1 * sin_t + x2 * cos_t;
        store(out[idx1], rx1.cast::<T>());
        store(out[idx2], rx2.cast::<T>());
    }
}

/// New-syntax correctness for `mt_rope` (Grid3D, single threadgroup with
/// `tpg = [grid_x, seq_len, n_heads/4]`). Oracle = rotate-half RoPE on
/// dtype-rounded input; constexprs derived as in the legacy test.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_rope;
    use crate::utils::{pack_f32, unpack_f32};

    fn naive_rope(
        inp: &[f32],
        n_heads: u32,
        seq_len: u32,
        head_dim: u32,
        theta_base: f32,
    ) -> Vec<f32> {
        let grid_x = head_dim / 2;
        let h_stride = seq_len * head_dim;
        let seq_stride = head_dim;
        let base = theta_base.log2();
        let mut out = inp.to_vec();
        for pz in 0..n_heads / 4 {
            for py in 0..seq_len {
                for px in 0..grid_x {
                    let inv_freq = (-(px as f32 / grid_x as f32 * base)).exp2();
                    let theta = py as f32 * inv_freq;
                    let (c, s) = (theta.cos(), theta.sin());
                    for i in 0..4 {
                        let head = pz * 4 + i;
                        let idx1 = (py * seq_stride + head * h_stride + px) as usize;
                        let idx2 = idx1 + grid_x as usize;
                        let (x1, x2) = (inp[idx1], inp[idx2]);
                        out[idx1] = x1 * c - x2 * s;
                        out[idx2] = x1 * s + x2 * c;
                    }
                }
            }
        }
        out
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_mt_rope(dt: DType) -> TestSetup {
        let (n_heads, seq_len, head_dim, theta_base) = (4u32, 8u32, 16u32, 10000.0f32);
        let n = (n_heads * seq_len * head_dim) as usize;
        let inp: Vec<f32> = (0..n).map(|i| ((i % 23) as f32 - 11.0) * 0.1).collect();
        let inp_dt = unpack_f32(&pack_f32(&inp, dt), dt);
        let expected = naive_rope(&inp_dt, n_heads, seq_len, head_dim, theta_base);
        let grid_x = head_dim / 2;
        TestSetup::new(mt_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("h_stride", seq_len * head_dim)
            .constexpr("seq_stride", head_dim)
            .constexpr("grid_x", grid_x)
            .constexpr("base", theta_base.log2())
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(1, 1, 1, [grid_x, seq_len, n_heads / 4])
    }
}

/// New-syntax benchmark for `mt_rope` (vs MLX `metal/rope.metal`). Multi-TG:
/// `tpg = [grid_x, 8, 1]`, grid splits seq_len and the head-quads.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_rope;
    use crate::utils::{InputDomain, dtype_tol, input_buffer, mlx_tname};

    // Attaches the MLX `metal/rope.metal` `rope_<tn>` reference (the non-freqs,
    // 32-bit-index `rope<T, int32_t>` instantiation, `instantiate_rope_g`).
    //
    // That kernel gates its body on three Metal *function constants* with no
    // defaults (verified against the metal source, lines 7-9):
    //   `forward      [[function_constant(1)]]` → true  (forward rotation)
    //   `traditional  [[function_constant(2)]]` → false (rotate-half, like MT)
    //   `hs_transpose [[function_constant(3)]]` → false (contiguous head/seq)
    // bound via `.bool_constant(idx, val)`, which the runner threads through
    // `compile_with_bool_constants` (mirrors MLX's `MTLFCList` in rope.cpp and
    // the legacy `compile_with_bool_constants(src, "rope_<tn>", &[(1,true),
    // (2,false),(3,false)])`).
    //
    // Buffer ABI (derived from MLX `rope.cpp::eval_gpu` `set_*`/`set_bytes`
    // calls and the `[[kernel]] void rope(...)` signature in rope.metal):
    //   in[[0]]            = `inp`  (shared by name with the MT input)
    //   out[[1]]           = `out`  (.output(), read back + compared)
    //   offset[[2]]        = `device int*`, value 0  → L = scale·pos.y      (4B)
    //   scale[[3]]         = float 1.0                                       (4B)
    //   strides[[4]]       = int64[3] = [T·D, D, 1] (head, seq, elem)       (24B)
    //   out_strides[[5]]   = int64[3] = [T·D, D, 1]                         (24B)
    //   offset_stride[[6]] = int64 1                                        (8B)
    //   n_head[[7]]        = int  = n_heads                                 (4B)
    //   <gap [[8]],[[9]]>  = 1-elem placeholders; the rope kernel declares no
    //                        buffer(8)/(9) (those indices are freqs-only), but
    //                        `base` is explicitly `[[buffer(10)]]`, so the
    //                        positional binder needs fillers to reach slot 10.
    //   base[[10]]         = float = log2(theta_base)                       (4B)
    //
    // Strides are MLX's real `eval_gpu` values `[mat_size=T·D, D, 1]` (NOT the
    // legacy `run_rope`'s hand-rolled `[D, H·D, 1]`, which misrepresented the
    // row-contiguous `[H, T, D]` layout). With these, MLX's index
    // `pos.y·strides[1] + (pz·4+i)·strides[0] + pos.x` == MT's
    // `py·seq_stride + head·h_stride + px` exactly.
    //
    // Grid: MLX `dispatch_threads(grid_dims=(D/2, T, B·ceil(H/4)), group_dims)`,
    // so `threads_per_grid = (grid_x, seq_len, n_heads/4)`. The MT runner
    // dispatches `threadgroups × tpg`, so tgs=[1, seq_len/8, n_heads/4] with
    // tpg=[grid_x, 8, 1] yields the same total `(grid_x, seq_len, n_heads/4)`.
    //
    // `inp` is seeded `Signed` (period-8 `[-3..3]`, nan-free) and shared by name
    // so MT and the reference rotate identical data; tol floor 0.01 is the
    // legacy rope tolerance (MT folds in f32, MLX uses `fast::cos/sin`).
    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_rope(dt: DType) -> BenchSetup {
        let (n_heads, seq_len, head_dim, theta_base) = (32u32, 512u32, 128u32, 10000.0f32);
        let grid_x = head_dim / 2; // 64
        let n = (n_heads * seq_len * head_dim) as usize;
        let tn = mlx_tname(dt);
        let base = theta_base.log2();
        // MLX real strides: row-contiguous [n_heads, seq_len, head_dim] →
        // [head, seq, elem] = [seq_len·head_dim, head_dim, 1].
        let strides: Vec<u8> = [(seq_len * head_dim) as i64, head_dim as i64, 1i64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        BenchSetup::new(mt_rope::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(input_buffer("inp", n, dt, InputDomain::Signed))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .constexpr("h_stride", seq_len * head_dim)
            .constexpr("seq_stride", head_dim)
            .constexpr("grid_x", grid_x)
            .constexpr("base", base)
            // tpg [64, 8, 1] (=512 lanes); grid covers [grid_x, seq_len, n_heads/4].
            .grid_3d(1, seq_len / 8, n_heads / 4, [grid_x, 8, 1])
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
            .with_reference(
                RefKernel::new(
                    format!("rope_{tn}"),
                    include_str!(concat!(env!("OUT_DIR"), "/metal/rope.metal")),
                )
                // in[[0]] shared by name with the MT `inp` (placeholder bytes
                // overwritten by the runner with MT's exact input).
                .buffer(BenchBuffer::zeros("inp", n, dt))
                .buffer(BenchBuffer::zeros("out", n, dt).output())
                // offset[[2]] = int* (value 0 → no positional offset).
                .buffer(BenchBuffer::from_vec("offset", 0i32.to_le_bytes().to_vec(), DType::I32))
                // scale[[3]] = float 1.0.
                .buffer(BenchBuffer::from_vec("scale", 1.0f32.to_le_bytes().to_vec(), DType::F32))
                // strides[[4]] / out_strides[[5]] = int64[3] = [T·D, D, 1].
                .buffer(BenchBuffer::from_vec("strides", strides.clone(), DType::U64))
                .buffer(BenchBuffer::from_vec("out_strides", strides, DType::U64))
                // offset_stride[[6]] = int64 1; n_head[[7]] = int = n_heads.
                .buffer(BenchBuffer::from_vec("offset_stride", 1i64.to_le_bytes().to_vec(), DType::U64))
                .buffer(BenchBuffer::from_vec("n_head", (n_heads as i32).to_le_bytes().to_vec(), DType::I32))
                // [[8]]/[[9]] gap fillers (freqs-only buffers, unbound by `rope`).
                .buffer(BenchBuffer::zeros("_gap8", 1, DType::I32))
                .buffer(BenchBuffer::zeros("_gap9", 1, DType::I32))
                // base[[10]] = float = log2(theta_base).
                .buffer(BenchBuffer::from_vec("base", base.to_le_bytes().to_vec(), DType::F32))
                // threads_per_grid = tgs × tpg = (grid_x, seq_len, n_heads/4),
                // matching MLX's `dispatch_threads((D/2, T, ceil(H/4)), …)`.
                .grid(Grid::new_3d(1, seq_len / 8, n_heads / 4, [grid_x, 8, 1]))
                // forward / traditional / hs_transpose function constants.
                .bool_constant(1, true)
                .bool_constant(2, false)
                .bool_constant(3, false)
                .tol(dtype_tol(dt).max(0.01)),
            )
    }
}
