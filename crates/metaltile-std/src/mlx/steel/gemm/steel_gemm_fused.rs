//! Steel tiled GEMM — #[kernel] DSL vs MLX steel/gemm/kernels/steel_gemm_fused.metal
//!
//! Production tiled matrix multiply via simdgroup matrix ops matching
//! all 5 MLX `steel_gemm_fused` block shapes (NN transpose only):
//!   (BM×BN×BK, WM×WN): 64×64×16/2×2, 64×32×32/2×2, 32×64×16/1×2,
//!                        32×32×16/2×2, 64×32×8/4×1
//!
//! Each variant is generated from a shared body macro to avoid code
//! duplication.  Metal requires compile-time threadgroup memory sizes
//! so constexpr parameters can't substitute — separate PSOs per shape
//! are needed (same as MLX's C++ template instantiation).

use metaltile::kernel;

// ── Shared body (macro-expanded per variant) ────────────────────────────
macro_rules! steel_gemm_body {
    ($BM:literal, $BN:literal, $BK:literal, $WM:literal, $WN:literal,
     $tg_size:literal, $as_pad:literal, $bs_pad:literal,
     $a_elems:literal, $b_elems:literal,
     $sub_m_stride:literal, $sub_n_stride:literal) => {{
        let tg_col = program_id::<0>();
        let tg_row = program_id::<1>();
        let sg_id = simd_group_id();
        let sg_m = sg_id / $WM;
        let sg_n = sg_id % $WN;
        let lane = simd_lane_id();

        threadgroup_alloc("As", $as_pad);
        threadgroup_alloc("Bs", $bs_pad);
        threadgroup_barrier();

        let acc = simdgroup_alloc::<f32, 8, 8>();

        let qid = lane / 4u32;
        let fm = (qid & 4u32) + ((lane / 2u32) % 4u32);
        let fn0 = (qid & 2u32) * 2u32 + (lane % 2u32) * 2u32;
        let fn1 = fn0 + 1u32;

        let sub_m0 = sg_m * $sub_m_stride;
        let sub_n0 = sg_n * $sub_n_stride;

        let n_total = $tg_size;
        let flat_tid = sg_id * 32u32 + lane;

        let n_steps = k / $BK;
        for _kk in range(0u32, n_steps, 1u32) {
            let k_off = _kk * $BK;
            let _row0 = tg_row * $BM;
            let _col0 = tg_col * $BN;

            for ei in range(0u32, $a_elems, 1u32) {
                let f_idx = flat_tid + ei * n_total;
                let _ar = f_idx / $BK;
                let _ac = f_idx % $BK;
                threadgroup_store("As", f_idx, load(a[(_row0 + _ar) * k + (k_off + _ac)]));
            }

            for ei in range(0u32, $b_elems, 1u32) {
                let f_idx = flat_tid + ei * n_total;
                let _br = f_idx / $BN;
                let _bc = f_idx % $BN;
                threadgroup_store("Bs", f_idx, load(b[(k_off + _br) * n + (_col0 + _bc)]));
            }

            threadgroup_barrier();

            let sub_a = simdgroup_alloc::<f16, 8, 8>();
            let sub_b = simdgroup_alloc::<f16, 8, 8>();

            let a_idx0 = (sub_m0 + fm) * $BK + fn0;
            let a_idx1 = a_idx0 + 1u32;
            let b_idx0 = fn0 * $BN + (sub_n0 + fm);
            let b_idx1 = b_idx0 + $BN;

            simdgroup_elem_store(sub_a, 0u32, threadgroup_load("As", a_idx0));
            simdgroup_elem_store(sub_a, 1u32, threadgroup_load("As", a_idx1));
            simdgroup_elem_store(sub_b, 0u32, threadgroup_load("Bs", b_idx0));
            simdgroup_elem_store(sub_b, 1u32, threadgroup_load("Bs", b_idx1));

            simdgroup_matmul(sub_a, sub_b, acc);
        }

        let r0 = simdgroup_elem_load(acc, 0u32);
        let r1 = simdgroup_elem_load(acc, 1u32);
        let out_r = tg_row * $BM + sub_m0 + fm;
        let out_c0 = tg_col * $BN + sub_n0 + fn0;
        let out_c1 = out_c0 + 1u32;

        store(out[out_r * n + out_c0], r0.cast::<T>());
        store(out[out_r * n + out_c1], r1.cast::<T>());
    }};
}

// ── Variant generator macro (one #[kernel] fn + inventory::submit! each) ──
macro_rules! steel_gemm_variant {
    ($name:ident, $subop:literal,
     $BM:literal, $BN:literal, $BK:literal, $WM:literal, $WN:literal,
     $tg_size:literal, $as_pad:literal, $bs_pad:literal,
     $a_elems:literal, $b_elems:literal,
     $sub_m_stride:literal, $sub_n_stride:literal) => {
        #[kernel]
        pub fn $name<T>(
            a: Tensor<T>,
            b: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] m: u32,
            #[constexpr] n: u32,
            #[constexpr] k: u32,
        ) {
            steel_gemm_body!(
                $BM, $BN, $BK, $WM, $WN,
                $tg_size, $as_pad, $bs_pad,
                $a_elems, $b_elems,
                $sub_m_stride, $sub_n_stride
            )
        }

        inventory::submit! {
            crate::spec::BenchSpec {
                op: "steel_gemm_fused",
                subop: $subop,
                kernel_name: stringify!($name),
                kernel_ir: $name::kernel_ir_for,
                dtypes: crate::bench_types::FLOAT_DTYPES,
                tol: 1e-2f32,
                mlx_src: None,
                mlx_pattern: None,
                shapes: &[],
                dispatch: crate::spec::BenchDispatch::Generic,
                kernel_mode: Some(metaltile_core::ir::KernelMode::SimdGroup2D),
            }
        }
    };
}

// ── All 5 MLX-visible variants (NN transpose, matching instantiate_gemm) ──
// BM  BN  BK  WM WN  tg   as_pad bs_pad a_elems b_elems sub_m sub_n
steel_gemm_variant!(mt_steel_gemm_64x64x16_2x2, "bm64_bn64_bk16_wm2_wn2",
    64, 64, 16, 2, 2, 128, 1024, 1024, 8, 8, 32, 32);
steel_gemm_variant!(mt_steel_gemm_64x32x32_2x2, "bm64_bn32_bk32_wm2_wn2",
    64, 32, 32, 2, 2, 128, 2048, 1024, 16, 8, 32, 16);
steel_gemm_variant!(mt_steel_gemm_32x64x16_1x2, "bm32_bn64_bk16_wm1_wn2",
    32, 64, 16, 1, 2, 64,   512, 1024, 8, 16, 32, 32);
steel_gemm_variant!(mt_steel_gemm_32x32x16_2x2, "bm32_bn32_bk16_wm2_wn2",
    32, 32, 16, 2, 2, 128,  512,  512, 4, 4, 16, 16);
steel_gemm_variant!(mt_steel_gemm_64x32x8_4x1,  "bm64_bn32_bk8_wm4_wn1",
    64, 32,  8, 4, 1, 128,  512,  256, 4, 2, 16, 32);
