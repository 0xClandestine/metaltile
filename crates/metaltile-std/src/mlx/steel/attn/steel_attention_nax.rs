//! `mt_sdpa_prefill_nax` — flash attention via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the steel flash-attention prefill
//! kernel. Gated behind the `nax` Cargo feature (Metal 4 / macOS 26+).
//!
//! Expressed in the `#[kernel]` DSL via the `coop_tile_*` intrinsics —
//! no `Op::InlineMsl`. The cooperative-tensor counterpart of
//! `mt_sdpa_prefill_mma`: the standard FlashAttention-2 online-softmax
//! loop, but the two matmuls — `S = Q·Kᵀ` and `O += P·V` — are each one
//! cooperative `matmul2d` instead of an 8×8 `simdgroup_matmul` ladder.
//!
//! ## Tile geometry
//!
//! - **BQ = 16** queries/TG, **BK = 16** keys/block, **BD = 32** head dim.
//! - **tpg = 32** (one simdgroup). The 16×16 S tile and 16×32 O tile are
//!   each one cooperative `matmul2d`.
//! - Grid: `[q_len/16, n_q_heads, batch]` — `tgid_x` Q-tile, `tgid_y`
//!   Q-head, `tgid_z` batch.
//!
//! `BD = 32` makes the QK descriptor's K-dim exactly 32 (Apple's
//! "at least one of M/N/K = 32" rule); larger head dims are a follow-up.
//!
//! ## Per K-block flash loop
//!
//! 1. Coop-load the 16×32 K and V tiles into TG memory.
//! 2. `S = Q·Kᵀ` — `matmul2d(16, 16, 32)`, `tb=true` (Kᵀ). Overwrite mode.
//! 3. Lane `r` owns S-row `r`: causal-mask, online-softmax max/sum
//!    rescale, write the exp-weights `P`.
//! 4. `O += P·V` — `matmul2d(16, 32, 16)`. The per-block max-rescale
//!    makes the accumulation explicit in `Os` scratch (the `matmul2d`
//!    runs overwrite into `Obk`, which is then added into `Os`).
//!
//! ## Dispatch invariants
//!
//! - TPG 32 (1 SG); grid `[q_len/16, n_q_heads, batch]`.
//! - `q_len % 16 == 0`, `k_len % 16 == 0`, `head_dim == 32`.
//! - `KernelMode::Reduction`.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 — see
//! `crates/metaltile-std/tests/steel_attention_nax_gpu_correctness.rs`.

use metaltile::kernel;

/// Tile geometry — keep in lock-step with the codegen-emitted MSL.
pub const BQ: u32 = 16;
pub const BK: u32 = 16;
pub const BD: u32 = 32;
/// Threads per group (1 SG × 32 lanes).
pub const TPG: u32 = 32;
/// Row skew past the inner extent — scatters 32-bank conflicts on the
/// column-strided frag loads inside `matmul2d`.
pub const TG_SKEW: u32 = 4;
/// Leading dim of the BQ/BK × BD tiles (BD + skew).
pub const TG_LD_D: u32 = BD + TG_SKEW; // 36
/// Leading dim of the BQ × BK S/P scratch (BK + skew).
pub const TG_LD_K: u32 = BK + TG_SKEW; // 20

/// Flash-attention prefill via cooperative `matmul2d`. Generic over
/// `T ∈ {f32, f16}`.
///
/// Params: `q`/`k`/`v`/`out` are `[batch, heads, len, head_dim]` slabs
/// (`q`/`out` use `n_q_heads`, `k`/`v` use `n_kv_heads`). Constexprs:
/// `q_len`, `k_len`, `gqa_factor`, `n_q_heads`, `n_kv_heads`, `scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_sdpa_prefill_nax<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] q_len: u32,
    #[constexpr] k_len: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] n_kv_heads: u32,
    #[constexpr] scale: f32,
) {
    let q_tile = tgid_x;
    let q_head = tgid_y;
    let batch = tgid_z;
    let kv_head = q_head / gqa_factor;
    let lane = simd_lane;

    let head_dim = 32u32;
    let q_len_off = k_len - q_len;

    // Slab offsets — q/out: [batch, n_q_heads, q_len, D]; k/v: [batch,
    // n_kv_heads, k_len, D].
    let kv_row_base = batch * n_kv_heads * k_len * head_dim + kv_head * k_len * head_dim;
    let q_head_row_off = batch * n_q_heads * q_len * head_dim + q_head * q_len * head_dim;
    let q_tile_first = q_tile * 16u32;

    // Threadgroup tiles (skewed). Qs/Ks/Vs/Ps are T; Ss/Os/Obk are fp32.
    threadgroup_alloc("Qs", 576, T); // 16 × 36
    threadgroup_alloc("Ks", 576, T);
    threadgroup_alloc("Vs", 576, T);
    threadgroup_alloc("Ps", 320, T); // 16 × 20
    threadgroup_alloc("Ss", 320, f32);
    threadgroup_alloc("Os", 576, f32);
    threadgroup_alloc("Obk", 576, f32);

    // Coop-load the 16×32 Q tile (lane fills column `lane`); zero Os.
    for _r in range(0u32, 16u32, 1u32) {
        let q_dev = q_head_row_off + (q_tile_first + _r) * head_dim + lane;
        let qv = load(q[q_dev]).cast::<f32>() * scale;
        threadgroup_store("Qs", _r * 36u32 + lane, qv.cast::<T>());
        threadgroup_store("Os", _r * 36u32 + lane, 0.0f32);
    }

    // Per-row online-softmax state — lane `r` (r < 16) owns row r.
    let mut row_m = -1.0e30f32;
    let mut row_s = 0.0f32;
    let owns_row = lane < 16u32;
    let my_row = lane;
    let q_abs = q_tile_first + my_row + q_len_off;

    // QK: matmul2d(16, 16, 32), ta=false tb=true, overwrite (S fresh).
    coop_tile_setup("qk", 16, 16, 32, T, "overwrite", "simdgroup", f32, false, true, false);
    // PV: matmul2d(16, 32, 16), ta=false tb=false, overwrite (per-block).
    coop_tile_setup("pv", 16, 32, 16, T, "overwrite", "simdgroup", f32, false, false, false);

    // Causal trim — last K-block touched by the tile's last query.
    let q_tile_last_abs = q_tile_first + 15u32 + q_len_off;
    let kb_lim = q_tile_last_abs / 16u32 + 1u32;

    for kb in range(0u32, kb_lim, 1u32) {
        let kb_off = kb * 16u32;

        // 1. Coop-load the 16×32 K and V tiles.
        for _r in range(0u32, 16u32, 1u32) {
            let kv_dev = kv_row_base + (kb_off + _r) * head_dim + lane;
            threadgroup_store("Ks", _r * 36u32 + lane, load(k[kv_dev]).cast::<T>());
            threadgroup_store("Vs", _r * 36u32 + lane, load(v[kv_dev]).cast::<T>());
        }
        threadgroup_barrier();

        // 2. S = Q·Kᵀ — extents inner-first: tQ/tK [TG_LD_D=36, 16],
        //    tS [TG_LD_K=20, 16].
        coop_tile_load_a("qk", "Qs", true, T, 36, 16);
        coop_tile_load_b("qk", "Ks", true, T, 36, 16);
        coop_tile_run("qk");
        coop_tile_store_c("qk", "Ss", true, f32, 20, 16);
        threadgroup_barrier();

        // 3. Online softmax — each owning lane processes its S row.
        if owns_row {
            let mut blk_m = -1.0e30f32;
            for _c in range(0u32, 16u32, 1u32) {
                let k_abs = kb_off + _c;
                let raw = threadgroup_load("Ss", my_row * 20u32 + _c);
                let sc = select(k_abs > q_abs, -1.0e30f32, raw);
                threadgroup_store("Ss", my_row * 20u32 + _c, sc);
                blk_m = select(sc > blk_m, sc, blk_m);
            }
            let new_m = select(blk_m > row_m, blk_m, row_m);
            let rescale = exp(row_m - new_m);
            let mut blk_s = 0.0f32;
            for _c in range(0u32, 16u32, 1u32) {
                let p = exp(threadgroup_load("Ss", my_row * 20u32 + _c) - new_m);
                threadgroup_store("Ss", my_row * 20u32 + _c, p);
                blk_s = blk_s + p;
            }
            row_s = row_s * rescale + blk_s;
            // Rescale the running O accumulator by exp(m_old - m_new).
            for _d in range(0u32, 32u32, 1u32) {
                let o = threadgroup_load("Os", my_row * 36u32 + _d);
                threadgroup_store("Os", my_row * 36u32 + _d, o * rescale);
            }
            row_m = new_m;
        }
        threadgroup_barrier();

        // 4. Stage the fp32 exp-weights P into the T-typed Ps tile.
        if owns_row {
            for _c in range(0u32, 16u32, 1u32) {
                let p = threadgroup_load("Ss", my_row * 20u32 + _c);
                threadgroup_store("Ps", my_row * 20u32 + _c, p.cast::<T>());
            }
        }
        threadgroup_barrier();

        // O_blk = P·V — tP [TG_LD_K=20, 16], tV [TG_LD_D=36, 16],
        // tObk [TG_LD_D=36, 16].
        coop_tile_load_a("pv", "Ps", true, T, 20, 16);
        coop_tile_load_b("pv", "Vs", true, T, 36, 16);
        coop_tile_run("pv");
        coop_tile_store_c("pv", "Obk", true, f32, 36, 16);
        threadgroup_barrier();

        // Add the per-block P·V product into the running Os accumulator.
        if owns_row {
            for _d in range(0u32, 32u32, 1u32) {
                let o = threadgroup_load("Os", my_row * 36u32 + _d);
                let ob = threadgroup_load("Obk", my_row * 36u32 + _d);
                threadgroup_store("Os", my_row * 36u32 + _d, o + ob);
            }
        }
        threadgroup_barrier();
    }

    // 5. Normalize by the softmax denominator and store O.
    if owns_row {
        let inv_s = select(row_s > 0.0f32, 1.0f32 / row_s, 0.0f32);
        for _d in range(0u32, 32u32, 1u32) {
            let o_dev = q_head_row_off + (q_tile_first + my_row) * head_dim + _d;
            let o = threadgroup_load("Os", my_row * 36u32 + _d);
            store(out[o_dev], (o * inv_s).cast::<T>());
        }
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::{dtype::DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16] {
            let k = mt_sdpa_prefill_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_sdpa_prefill_nax");
            assert_eq!(k.params.len(), 4);
            assert!(k.params[3].is_output);
            assert_eq!(k.constexprs.len(), 6);
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            // No raw inline MSL — both matmuls are CoopTile* ops.
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            // Two distinct cooperative-matmul setups (qk + pv).
            let n_setup = all_ops().filter(|op| matches!(op, Op::CoopTileSetup { .. })).count();
            assert_eq!(n_setup, 2, "expected qk + pv CoopTileSetup ops");
        }
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half")] {
            let mut k = mt_sdpa_prefill_nax::kernel_ir_for(dt);
            let suffix = if dt == DType::F32 { "f32" } else { "f16" };
            k.name = format!("mt_sdpa_prefill_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_sdpa_prefill_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name}")));
        }
    }
}
