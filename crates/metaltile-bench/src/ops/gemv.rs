//! GEMV benchmark — #[kernel] DSL vs MLX metal/gemv.metal

use metaltile::{bench_kernel, kernel};

static SRC: &str = include_str!("../metal/gemv.metal");
static GEMV_SHAPES: &[(usize, usize)] = &[(4096, 4096)];

#[bench_kernel(op="gemv", subop="gemv", class=MatVec,
               shapes=&GEMV_SHAPES, tpg=256, tol=1e-2,
               mlx_src=SRC, mlx="gemv_{tn}_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
               metal_file="gemv.metal")]
#[kernel]
pub fn mt_gemv<T>(mat: Tensor<T>, vec: Tensor<T>, out: Tensor<T>, #[constexpr] k: u32) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let acc = strided_reduce_dot(mat, vec, rs, rs, re);
    let result = reduce_sum(acc);
    store(out[row], result);
}
