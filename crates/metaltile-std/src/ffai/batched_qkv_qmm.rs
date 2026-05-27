//! Batched Q/K/V 4-bit quantized QMM (M>1) — fuses the three independent
//! Q, K, V projection matmuls of a *prefill* step into one dispatch.
//!
//! This is the M>1 sibling of `ffai_batched_qkv_qgemv_fast`. At
//! `program_id::<1>() = m` we load row `m` of the batched input
//! `x: [M, in_dim]` and produce row `m` of THREE separate output tensors:
//!   q_buf: [M, out_q] T
//!   k_buf: [M, out_k] T
//!   v_buf: [M, out_v] T
//!
//! Three separate buffers keep each projection contiguous in memory so
//! the downstream attention pipeline reads `[M, dim]` per projection.
//! Callers can alias all three into one backing allocation if they want;
//! the kernel only sees three base pointers.
//!
//! Dispatch geometry mirrors `ffai_batched_qkv_qgemv_fast`:
//!   * `program_id::<2>()` selects matrix (0 = Q, 1 = K, 2 = V).
//!   * `program_id::<1>()` selects batched row m (0..M).
//!   * `tgid_x` selects an 8-row output tile. TPG = 64 = 2 SG × 32 lanes.
//!
//! Grid: `[ceil(max(out_q,out_k,out_v)/8), M, 3]`, TPG = `[64,1,1]`.
//!
//! The inner loop is the same `stack_alloc` + `range(0,4)` pattern as
//! `dequant_gemv_int4_fast` — DSL unrolls at codegen. The x-preload is
//! hoisted before the per-matrix dispatch and shared across all branches.
//!
//! Constraints (same as the fast GEMV variant):
//!   * `in_dim % 512 == 0`
//!   * `out_q`, `out_k`, `out_v` each a multiple of 8
//!   * `group_size == 64`
//!
//! Quantized-weight layout (N = `in_dim`, G = `group_size`, int4):
//!   x         [M, N]          T
//!   w_*       [out_*, N/8]    uint32
//!   scales_*  [out_*, N/G]    T
//!   biases_*  [out_*, N/G]    T
//!   *_buf     [M, out_*]      T
//!
//! Codegen-only; correctness pinned by
//! `tests/batched_qkv_qmm_gpu_correctness.rs`.

use metaltile::kernel;

/// Perf-tuned fused Q/K/V int4 QMM (M>1) — 8 output rows per TG.
///
/// Grid: `[ceil(max(out_q,out_k,out_v)/8), M, 3]`. See module docs for
/// the full geometry contract. TGs past a matrix's `out_*` rows no-op.
#[kernel]
pub fn ffai_batched_qkv_qmm_fast<T>(
    x: Tensor<T>,
    w_q: Tensor<u32>,
    scales_q: Tensor<T>,
    biases_q: Tensor<T>,
    w_k: Tensor<u32>,
    scales_k: Tensor<T>,
    biases_k: Tensor<T>,
    w_v: Tensor<u32>,
    scales_v: Tensor<T>,
    biases_v: Tensor<T>,
    mut q_buf: Tensor<T>,
    mut k_buf: Tensor<T>,
    mut v_buf: Tensor<T>,
    #[constexpr] out_q: u32,
    #[constexpr] out_k: u32,
    #[constexpr] out_v: u32,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let matrix = program_id::<2>();
    let m = program_id::<1>();
    let tg = tgid_x;
    let sg = simd_id;
    let lane = simd_lane;
    let base_row = tg * 8u32 + sg * 4u32;
    let gs_per_row = in_dim / group_size;
    let packs_per_row = in_dim / 8u32;
    let lane_x_off = lane * 16u32;
    let lane_pack_off = lane * 2u32;
    // Row-m offsets into x and per-projection output buffers.
    let x_row_off = m * in_dim;
    let q_row_off = m * out_q;
    let k_row_off = m * out_k;
    let v_row_off = m * out_v;
    // Mask-without-shift constants — eliminates 56 shifts per block.
    let s_16 = 0.0625f32;
    let s_256 = 0.00390625f32;
    let s_4096 = 0.000244140625f32;
    // Route the row guard to the output size for this matrix slice.
    let out_limit = select(matrix == 0u32, out_q, select(matrix == 1u32, out_k, out_v));
    // Per-row partial-sum accumulators. `stack_alloc` lowers to a
    // thread-private array; DSL unrolls range(0,4) loops at codegen.
    stack_alloc("accs", 4, "f32");
    for _r in range(0u32, 4u32, 1u32) {
        stack_store("accs", _r, 0.0f32);
    }
    if base_row < out_limit {
        for _b in range(0u32, in_dim, 512u32) {
            // 16 x loads per K-block, shared across all three matrix branches.
            // xb includes the batch-row offset; group index uses the
            // in-row column offset only (scales/biases are per weight row).
            let xb = x_row_off + _b + lane_x_off;
            let x0 = load(x[xb]).cast::<f32>();
            let x1r = load(x[xb + 1u32]).cast::<f32>();
            let x2r = load(x[xb + 2u32]).cast::<f32>();
            let x3r = load(x[xb + 3u32]).cast::<f32>();
            let x4 = load(x[xb + 4u32]).cast::<f32>();
            let x5r = load(x[xb + 5u32]).cast::<f32>();
            let x6r = load(x[xb + 6u32]).cast::<f32>();
            let x7r = load(x[xb + 7u32]).cast::<f32>();
            let x8 = load(x[xb + 8u32]).cast::<f32>();
            let x9r = load(x[xb + 9u32]).cast::<f32>();
            let x10r = load(x[xb + 10u32]).cast::<f32>();
            let x11r = load(x[xb + 11u32]).cast::<f32>();
            let x12 = load(x[xb + 12u32]).cast::<f32>();
            let x13r = load(x[xb + 13u32]).cast::<f32>();
            let x14r = load(x[xb + 14u32]).cast::<f32>();
            let x15r = load(x[xb + 15u32]).cast::<f32>();
            // xs = Σ x[i] over the 16-element block (bias term).
            let xs = x0
                + x1r
                + x2r
                + x3r
                + x4
                + x5r
                + x6r
                + x7r
                + x8
                + x9r
                + x10r
                + x11r
                + x12
                + x13r
                + x14r
                + x15r;
            // Pre-scale nibble positions 1/2/3 for mask-without-shift.
            let x1 = x1r * s_16;
            let x2 = x2r * s_256;
            let x3 = x3r * s_4096;
            let x5 = x5r * s_16;
            let x6 = x6r * s_256;
            let x7 = x7r * s_4096;
            let x9 = x9r * s_16;
            let x10 = x10r * s_256;
            let x11 = x11r * s_4096;
            let x13 = x13r * s_16;
            let x14 = x14r * s_256;
            let x15 = x15r * s_4096;
            // Group index uses the in-row column offset (not the batched
            // global offset) since scales/biases are per weight row × group.
            let g = (_b + lane_x_off) / group_size;
            let pack_off = _b / 8u32 + lane_pack_off;
            // Per-matrix dispatch. Only tensor names differ across branches.
            if matrix == 0u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_q[wb + pack_off]);
                    let p_hi = load(w_q[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_q[sb + g]).cast::<f32>();
                    let bi = load(biases_q[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
            if matrix == 1u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_k[wb + pack_off]);
                    let p_hi = load(w_k[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_k[sb + g]).cast::<f32>();
                    let bi = load(biases_k[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
            if matrix == 2u32 {
                for _r in range(0u32, 4u32, 1u32) {
                    let row = base_row + _r;
                    let wb = row * packs_per_row;
                    let sb = row * gs_per_row;
                    let p_lo = load(w_v[wb + pack_off]);
                    let p_hi = load(w_v[wb + pack_off + 1u32]);
                    let p_lo_hi = p_lo >> 16u32;
                    let p_hi_hi = p_hi >> 16u32;
                    let s = load(scales_v[sb + g]).cast::<f32>();
                    let bi = load(biases_v[sb + g]).cast::<f32>();
                    let qd = (p_lo & 15u32).cast::<f32>() * x0
                        + (p_lo & 240u32).cast::<f32>() * x1
                        + (p_lo & 3840u32).cast::<f32>() * x2
                        + (p_lo & 61440u32).cast::<f32>() * x3
                        + (p_lo_hi & 15u32).cast::<f32>() * x4
                        + (p_lo_hi & 240u32).cast::<f32>() * x5
                        + (p_lo_hi & 3840u32).cast::<f32>() * x6
                        + (p_lo_hi & 61440u32).cast::<f32>() * x7
                        + (p_hi & 15u32).cast::<f32>() * x8
                        + (p_hi & 240u32).cast::<f32>() * x9
                        + (p_hi & 3840u32).cast::<f32>() * x10
                        + (p_hi & 61440u32).cast::<f32>() * x11
                        + (p_hi_hi & 15u32).cast::<f32>() * x12
                        + (p_hi_hi & 240u32).cast::<f32>() * x13
                        + (p_hi_hi & 3840u32).cast::<f32>() * x14
                        + (p_hi_hi & 61440u32).cast::<f32>() * x15;
                    let prev = stack_load("accs", _r);
                    stack_store("accs", _r, prev + s * qd + bi * xs);
                }
            }
        }
        // Cross-lane reduce + store. out_* are multiples of 8 so all four
        // rows are valid whenever base_row < out_limit.
        //
        // The simd_sums are hoisted into per-row locals BEFORE the lane==0
        // gate so the per-matrix branches can reference them by name. The
        // DSL unroller mangles SSA bindings when a `simd_sum` result is
        // referenced inside a nested `if lane == 0 { if matrix == N { ... } }`
        // block emitted from a `range(0,4)` body (the cast op keeps a stale
        // pre-unroll SSA handle), so the 4 reductions are written by hand
        // here rather than looped. The accumulation loop above keeps the
        // stack_alloc + range dedup intact.
        let r0 = simd_sum(stack_load("accs", 0u32));
        let r1 = simd_sum(stack_load("accs", 1u32));
        let r2 = simd_sum(stack_load("accs", 2u32));
        let r3 = simd_sum(stack_load("accs", 3u32));
        if lane == 0u32 {
            if matrix == 0u32 {
                store(q_buf[q_row_off + base_row + 0u32], r0.cast::<T>());
                store(q_buf[q_row_off + base_row + 1u32], r1.cast::<T>());
                store(q_buf[q_row_off + base_row + 2u32], r2.cast::<T>());
                store(q_buf[q_row_off + base_row + 3u32], r3.cast::<T>());
            }
            if matrix == 1u32 {
                store(k_buf[k_row_off + base_row + 0u32], r0.cast::<T>());
                store(k_buf[k_row_off + base_row + 1u32], r1.cast::<T>());
                store(k_buf[k_row_off + base_row + 2u32], r2.cast::<T>());
                store(k_buf[k_row_off + base_row + 3u32], r3.cast::<T>());
            }
            if matrix == 2u32 {
                store(v_buf[v_row_off + base_row + 0u32], r0.cast::<T>());
                store(v_buf[v_row_off + base_row + 1u32], r1.cast::<T>());
                store(v_buf[v_row_off + base_row + 2u32], r2.cast::<T>());
                store(v_buf[v_row_off + base_row + 3u32], r3.cast::<T>());
            }
        }
    }
}
