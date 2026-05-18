//! Steel tiled GEMM — #[kernel] DSL vs MLX steel/gemm/kernels/steel_gemm_fused.metal
//!
//! Production tiled matrix multiply via simdgroup matrix ops matching
//! the MLX `steel_gemm_fused` algorithm: threadgroup-cooperative load of
//! A and B tiles into shared memory, SIMD-group matrix multiply-accumulate
//! in an 8×8 fragment, K-dimension tiling loop.
//!
//! Block shape: (BM×BN×BK, WM×WN) = 64×64×16 / 2×2
//! Threadgroup: 256 threads (2×2×32)
//! Dtype: f16 input → f32 accumulator → f16 output

use metaltile::kernel;

#[kernel]
pub fn mt_steel_gemm_64x64x16_2x2<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>, #[constexpr] m: u32, #[constexpr] n: u32, #[constexpr] k: u32) {
    let tg_col = program_id::<0>();
    let tg_row = program_id::<1>();
    let sg_id = simd_group_id();
    let sg_m = sg_id / 2u32;
    let sg_n = sg_id % 2u32;
    let lane = simd_lane_id();

    threadgroup_alloc("As", 1088u32);
    threadgroup_alloc("Bs", 1088u32);
    threadgroup_barrier();

    let acc = simdgroup_alloc::<f32, 8, 8>();

    // Lane → 8×8 element mapping (Metal simdgroup_matrix convention)
    let qid = lane / 4u32;
    let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
    let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
    let fn1 = fn0 + 1u32;

    // Sub-tile origin for this SIMD group
    let sub_m0 = sg_m * 32u32;
    let sub_n0 = sg_n * 32u32;

    // Cooperative load helpers: reuse threadgroup memory as flat buffer
    let n_total = 256u32;
    let flat_tid = sg_id * 32u32 + lane;

    let n_steps = k / 16u32;
    for _kk in range(0u32, n_steps, 1u32) {
        let k_off = _kk * 16u32;

        // Base positions recomputed inside loop to avoid dead-store elimination
        let _row0 = tg_row * 64u32;
        let _col0 = tg_col * 64u32;

        let a_elems = 4u32;
        for ei in range(0u32, a_elems, 1u32) {
            let f_idx = flat_tid + ei * n_total;
            let _ar = f_idx / 16u32;
            let _ac = f_idx % 16u32;
            let a_src = (_row0 + _ar) * k + (k_off + _ac);
            threadgroup_store("As", f_idx, load(a[a_src]));
        }

        let b_elems = 4u32;
        for ei in range(0u32, b_elems, 1u32) {
            let f_idx = flat_tid + ei * n_total;
            let _br = f_idx / 64u32;
            let _bc = f_idx % 64u32;
            let b_src = (k_off + _br) * n + (_col0 + _bc);
            threadgroup_store("Bs", f_idx, load(b[b_src]));
        }

        threadgroup_barrier();

        let sub_a = simdgroup_alloc::<f16, 8, 8>();
        let sub_b = simdgroup_alloc::<f16, 8, 8>();

        let a_idx0 = (sub_m0 + fm) * 16u32 + fn0;
        let a_idx1 = a_idx0 + 1u32;
        let b_idx0 = fn0 * 64u32 + (sub_n0 + fm);
        let b_idx1 = b_idx0 + 64u32;

        simdgroup_elem_store(sub_a, 0u32, threadgroup_load("As", a_idx0));
        simdgroup_elem_store(sub_a, 1u32, threadgroup_load("As", a_idx1));
        simdgroup_elem_store(sub_b, 0u32, threadgroup_load("Bs", b_idx0));
        simdgroup_elem_store(sub_b, 1u32, threadgroup_load("Bs", b_idx1));

        simdgroup_matmul(sub_a, sub_b, acc);
    }

    // Store output (row0/col0 recomputed here since they're loop-local above)
    let out_row0 = tg_row * 64u32;
    let out_col0 = tg_col * 64u32;

    let r0 = simdgroup_elem_load(acc, 0u32);
    let r1 = simdgroup_elem_load(acc, 1u32);

    let out_r = out_row0 + sub_m0 + fm;
    let out_c0 = out_col0 + sub_n0 + fn0;
    let out_c1 = out_c0 + 1u32;

    store(out[out_r * n + out_c0], r0.cast::<T>());
    store(out[out_r * n + out_c1], r1.cast::<T>());
}

inventory::submit! {
    crate::spec::BenchSpec {
        op: "steel_gemm_fused",
        subop: "bm64_bn64_bk16_wm2_wn2",
        kernel_name: "mt_steel_gemm_64x64x16_2x2",
        kernel_ir: mt_steel_gemm_64x64x16_2x2::kernel_ir_for,
        dtypes: crate::bench_types::FLOAT_DTYPES,
        tol: 1e-2f32,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: crate::spec::BenchDispatch::Generic,
        kernel_mode: Some(metaltile_core::ir::KernelMode::SimdGroup2D),
    }
}
