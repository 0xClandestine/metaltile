//! GEMV benchmark — #[kernel] DSL vs MLX metal/gemv.metal
//!
//! Tuned for K=4096 via tpg sweep (64,128,256,512,1024): tpg=512 gives the best f16
//! throughput (+1.8% vs tpg=256) by giving each thread 2 iterations
//! of the 4-wide unroll (8 elements/thread), enough ILP to hide
//! load latency. tpg=1024 regresses −20% on f16 (only 1 iteration,
//! zero latency hiding). f32/bf16 are flat across tpgs.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="gemv",
    subop="gemv",
    class=MatVec,
    b=4096,
    n=4096,
    tpg=512,
    tol=1e-2,
    mlx="gemv_{tn}_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
    metal_file="gemv.metal",
)]
#[kernel]
pub fn mt_gemv<T>(mat: Tensor<T>, vec: Tensor<T>, out: Tensor<T>, #[constexpr] k: u32) {
    let row = program_id::<0>();
    let rs = row * k;
    let re = rs + k;
    let acc = strided_reduce_dot(mat, vec, rs, rs, re);
    let result = reduce_sum(acc);
    store(out[row], result);
}
