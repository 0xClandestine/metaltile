//! Softmax benchmark — #[kernel] DSL vs MLX metal/softmax.metal

use metaltile::{bench_kernel, kernel};

use crate::spec::ExtraInput;

static SRC: &str = include_str!("../metal/softmax.metal");
static SOFTMAX_SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
static NO_EXTRAS: &[ExtraInput] = &[];

#[bench_kernel(op="softmax", subop="softmax", class=RowNorm,
               shapes=&SOFTMAX_SHAPES, tpg=256, reads=2, out_elements=4096, extra=&NO_EXTRAS,
               tol=1e-4,
               mlx_src=SRC, mlx="looped_softmax_{tn}", mlx_extra_slots=0,
               metal_file="softmax.metal")]
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
    let rs = reduce_sum(rsl);
    let is = recip(rs);
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
