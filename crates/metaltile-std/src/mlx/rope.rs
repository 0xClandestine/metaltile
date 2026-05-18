//! RoPE benchmark — #[kernel] DSL vs MLX metal/rope.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="rope",
    subop="rope",
    class=Rope,
    b=1,
    h=32,
    l=512,
    d=128,
    n_per_group=4,
    tol=0.01,
    metal_file="rope.metal",
    dtypes=crate::spec::F16_ONLY,
)]
#[kernel]
pub fn mt_rope_f16(
    inp: Tensor<f16>,
    out: Tensor<f16>,
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
    let head_base = pz * 4u32;
    let row_base = py * seq_stride + px;

    // Hand-unrolled across `n_per_thread = 4` heads. The previous
    // `for i in range(0, 4, 1)` body was getting mangled by the unroll
    // pass: the IR remap dropped `seq_stride`/`h_stride`/`i` references
    // and the emitted MSL ended up with `bad sub-op ref` placeholders
    // (cf. `crates/metaltile-codegen/src/passes/unroll.rs` — same
    // BlockId-snapshot bug that bit `mt_affine_quantize_int8`).
    let h0 = head_base * h_stride;
    let h1 = (head_base + 1u32) * h_stride;
    let h2 = (head_base + 2u32) * h_stride;
    let h3 = (head_base + 3u32) * h_stride;

    let i1_0 = row_base + h0;
    let i2_0 = i1_0 + grid_x;
    let i1_1 = row_base + h1;
    let i2_1 = i1_1 + grid_x;
    let i1_2 = row_base + h2;
    let i2_2 = i1_2 + grid_x;
    let i1_3 = row_base + h3;
    let i2_3 = i1_3 + grid_x;

    let x1_0 = load(inp[i1_0]).cast::<f32>();
    let x2_0 = load(inp[i2_0]).cast::<f32>();
    store(out[i1_0], (x1_0 * cos_t - x2_0 * sin_t).cast::<f16>());
    store(out[i2_0], (x1_0 * sin_t + x2_0 * cos_t).cast::<f16>());

    let x1_1 = load(inp[i1_1]).cast::<f32>();
    let x2_1 = load(inp[i2_1]).cast::<f32>();
    store(out[i1_1], (x1_1 * cos_t - x2_1 * sin_t).cast::<f16>());
    store(out[i2_1], (x1_1 * sin_t + x2_1 * cos_t).cast::<f16>());

    let x1_2 = load(inp[i1_2]).cast::<f32>();
    let x2_2 = load(inp[i2_2]).cast::<f32>();
    store(out[i1_2], (x1_2 * cos_t - x2_2 * sin_t).cast::<f16>());
    store(out[i2_2], (x1_2 * sin_t + x2_2 * cos_t).cast::<f16>());

    let x1_3 = load(inp[i1_3]).cast::<f32>();
    let x2_3 = load(inp[i2_3]).cast::<f32>();
    store(out[i1_3], (x1_3 * cos_t - x2_3 * sin_t).cast::<f16>());
    store(out[i2_3], (x1_3 * sin_t + x2_3 * cos_t).cast::<f16>());
}
