//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Q8_0 inline-dequant GEMV — `out[r] = Σ_k dequant(W[r,k]) · x[k]`,
//! reading the Q8_0 weight straight from its split buffers so the dense
//! attention/shared-expert projections stay 1 byte/weight resident
//! instead of being pre-expanded to f16 (2 bytes). The DSv4 attention
//! block is bandwidth-bound on these projections (q_b / output_a /
//! output_b are Q8_0 on disk, ~100M weights/layer); halving their bytes
//! roughly halves the attn GPU time.
//!
//! ## Q8_0 block (32 values)
//!   d (f16 scale) + 32 int8 quants;  value[i] = d · q_i8[i]
//!
//! ## Split inputs (loader produces these once, resident)
//!   qs   [m_out * (k_in/32) * 8]  u32  — 32 int8/block packed as 8 LE u32
//!   d    [m_out * (k_in/32)]      f32  — per-block scale (fp16→f32)
//!   x    [k_in]                   T
//!   out  [m_out]                  T
//!
//! Dispatch (Reduction): grid (threadgroups) = [m_out, 1, 1], tg=[32,1,1].

use metaltile::kernel;

#[kernel]
pub fn ffai_gemv_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let x_base = b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                // sign-extend the byte to int8 range in the float domain
                // (avoids ambiguous integer `select` in MSL).
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                let val = d * qf;
                acc = acc + val * load(x[x_base + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Grouped Q8_0 gemv — `out[r] = Σ_k dequant(W[r,k]) · x[(r/rows_per_group)*k_in + k]`.
/// Each contiguous block of `rows_per_group` output rows reads its own
/// `k_in`-slice of `x`. Fuses the DSv4 grouped O-LoRA (8 groups × a
/// [1024,4096] Q8 slice, each on a different 4096-slice of the attention
/// output) into a SINGLE dispatch instead of 8.
#[kernel]
pub fn ffai_grouped_gemv_q8<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let x_blk = x_base + b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                acc = acc + d * qf * load(x[x_blk + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// COALESCED per-token grouped Q8 gemv — same math as `ffai_grouped_gemv_q8`
/// but the warp walks the row's `u32` words contiguously (lane j, j+32, …) so
/// consecutive lanes hit consecutive addresses. The original strided by 8 u32
/// per lane (each lane owned a whole 32-int8 block), which only reached ~45% of
/// DRAM bandwidth on GB10; this coalesced pattern is the decode-GEMV fast path.
#[kernel]
pub fn ffai_gemv_q8_coalesced<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let nwords = bpr * 8u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for j in range(lane, nwords, 32u32) {
        let block = j / 8u32;
        let w = j % 8u32;
        let packed = load(qs[qs_base + j]);
        let d = load(d_f32[d_base + block]).cast::<f32>();
        let x_blk = x_base + block * 32u32 + w * 4u32;
        for i in range(0u32, 4u32, 1u32) {
            let by = (packed >> (i * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            acc = acc + d * qf * load(x[x_blk + i]).cast::<f32>();
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[row], total.cast::<T>());
    }
}

/// Coalesced Q8 gemv with a fused ReLU² on the output: `out[r] = max(0, Wq·x)²`.
/// Fuses a MoE expert's `up` projection and its activation into one dispatch
/// (was gemv + a separate relu² kernel), keeping per-row occupancy.
#[kernel]
pub fn ffai_gemv_q8_coalesced_relu2<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let nwords = bpr * 8u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut dot = 0.0f32;
    for j in range(lane, nwords, 32u32) {
        let block = j / 8u32;
        let w = j % 8u32;
        let packed = load(qs[qs_base + j]);
        let d = load(d_f32[d_base + block]).cast::<f32>();
        let x_blk = x_base + block * 32u32 + w * 4u32;
        for i in range(0u32, 4u32, 1u32) {
            let by = (packed >> (i * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            dot = dot + d * qf * load(x[x_blk + i]).cast::<f32>();
        }
    }
    let total = simd_sum(dot);
    if lane == 0u32 {
        let r = select(total > 0.0f32, total, 0.0f32);
        store(out[row], (r * r).cast::<T>());
    }
}

/// Coalesced Q8 gemv that SCALES + ACCUMULATES in place: `acc[r] += scale[0] ·
/// Σ_k dequant(W[r,k])·x[k]`. Lets a MoE expert's `down` projection fold its
/// router weight and sum into the layer accumulator in ONE kernel — no separate
/// scalar-broadcast upload or `fma` dispatch per expert. `scale` is a 1-element
/// device buffer (the router weight); loaded once per output row.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_gemv_q8_coalesced_accum<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut acc: Tensor<T>,
    scale: Tensor<f32>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let nwords = bpr * 8u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut dot = 0.0f32;
    for j in range(lane, nwords, 32u32) {
        let block = j / 8u32;
        let w = j % 8u32;
        let packed = load(qs[qs_base + j]);
        let d = load(d_f32[d_base + block]).cast::<f32>();
        let x_blk = x_base + block * 32u32 + w * 4u32;
        for i in range(0u32, 4u32, 1u32) {
            let by = (packed >> (i * 8u32)) & 0xffu32;
            let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
            dot = dot + d * qf * load(x[x_blk + i]).cast::<f32>();
        }
    }
    let total = simd_sum(dot);
    if lane == 0u32 {
        let s = load(scale[0]);
        let prev = load(acc[row]).cast::<f32>();
        store(acc[row], (prev + s * total).cast::<T>());
    }
}

/// Copy a contiguous device slice `dst[i] = src[off + i]` — lets the Mamba
/// in_proj output be split (z / xBC / dt) ON-DEVICE instead of via a host
/// download, so the layer runs pure-async. `offbuf[0]` is the start offset.
#[kernel]
pub fn ffai_slice<T>(src: Tensor<T>, mut dst: Tensor<T>, #[constexpr] off: u32, #[constexpr] len: u32) {
    let i = program_id::<0>();
    if i < len {
        store(dst[i], load(src[off + i]));
    }
}

/// Device dt for Mamba2: `dt[i] = softplus(dt_raw[i] + dt_bias[i])` (stable form).
/// Keeps the Mamba dt computation ON-DEVICE (no host round-trip).
#[kernel]
pub fn ffai_softplus_add(a: Tensor<f32>, b: Tensor<f32>, mut out: Tensor<f32>, #[constexpr] n: u32) {
    let i = program_id::<0>();
    if i < n {
        let x = load(a[i]) + load(b[i]);
        let ax = select(x > 0.0f32, x, 0.0f32 - x);
        let pos = select(x > 0.0f32, x, 0.0f32);
        store(out[i], pos + log(1.0f32 + exp(0.0f32 - ax)));
    }
}

/// NemotronH/Zamba2 gated GROUPED RMSNorm (ON-DEVICE; removes the per-Mamba-layer
/// dl→host-norm→up sync). Gate-BEFORE-norm, per group of `gs`: g = y·silu(z);
/// out = g · rsqrt(mean_group(g²)+eps) · w. `y` fp32, z/w/out = T. One TG/group,
/// 4 elems/thread (block = gs/4), threadgroup reduce.
#[kernel]
pub fn ffai_gated_group_rmsnorm<T>(
    y: Tensor<f32>, z: Tensor<T>, w: Tensor<T>, mut out: Tensor<T>, eps_buf: Tensor<f32>,
    #[constexpr] gs: u32,
) {
    let grp = program_id::<0>();
    let rs = grp * gs;
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < gs;
    let safe_col = select(in_bounds, col, 0u32);
    let sb = rs + safe_col;
    let y0 = load(y[sb]).cast::<f32>();
    let y1 = load(y[sb + 1u32]).cast::<f32>();
    let y2 = load(y[sb + 2u32]).cast::<f32>();
    let y3 = load(y[sb + 3u32]).cast::<f32>();
    let z0 = load(z[sb]).cast::<f32>();
    let z1 = load(z[sb + 1u32]).cast::<f32>();
    let z2 = load(z[sb + 2u32]).cast::<f32>();
    let z3 = load(z[sb + 3u32]).cast::<f32>();
    let g0 = y0 * (z0 / (1.0f32 + exp(0.0f32 - z0)));
    let g1 = y1 * (z1 / (1.0f32 + exp(0.0f32 - z1)));
    let g2 = y2 * (z2 / (1.0f32 + exp(0.0f32 - z2)));
    let g3 = y3 * (z3 / (1.0f32 + exp(0.0f32 - z3)));
    let raw = g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3;
    let partial = select(in_bounds, raw, 0.0f32);
    let ssq = reduce_sum(partial);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(ssq / (gs.cast::<f32>()) + eps);
    if in_bounds {
        let base = rs + col;
        store(out[base], (g0 * rms * load(w[base]).cast::<f32>()).cast::<T>());
        store(out[base + 1u32], (g1 * rms * load(w[base + 1u32]).cast::<f32>()).cast::<T>());
        store(out[base + 2u32], (g2 * rms * load(w[base + 2u32]).cast::<f32>()).cast::<T>());
        store(out[base + 3u32], (g3 * rms * load(w[base + 3u32]).cast::<f32>()).cast::<T>());
    }
}

/// MoE router pre-scores (NemotronH / DeepSeek-V3 noaux, sigmoid variant):
/// `unbiased[i] = sigmoid(logit[i])`, `biased[i] = unbiased[i] + e_score_correction_bias[i]`.
/// Feeds `mt_dsv4_router_topk` (top-k by biased, weights from unbiased) so the whole
/// router stays ON-DEVICE — no per-MoE-layer dl(gate)+host-topk+up(idx) sync round-trip.
#[kernel]
pub fn ffai_moe_sigmoid_bias(logits: Tensor<f32>, bias: Tensor<f32>, mut unbiased: Tensor<f32>, mut biased: Tensor<f32>, #[constexpr] n: u32) {
    let i = program_id::<0>();
    if i < n {
        let s = 1.0f32 / (1.0f32 + exp(0.0f32 - load(logits[i])));
        store(unbiased[i], s);
        store(biased[i], s + load(bias[i]));
    }
}

/// Scale a vector in place by a scalar (router weights × routed_scaling_factor).
#[kernel]
pub fn ffai_vscale(mut buf: Tensor<f32>, #[constexpr] scale: f32, #[constexpr] n: u32) {
    let i = program_id::<0>();
    if i < n {
        store(buf[i], load(buf[i]) * scale);
    }
}

/// Elementwise dtype cast f32 → f16. Compacts the attention KV cache to half
/// precision: at 32K context the sdpa read is bandwidth-bound, so halving the
/// cache bytes roughly halves the per-layer attention cost. One thread / elem.
#[kernel]
pub fn ffai_cast_f32_f16(src: Tensor<f32>, mut dst: Tensor<f16>, #[constexpr] n: u32) {
    let i = program_id::<0>();
    if i < n {
        store(dst[i], load(src[i]).cast::<f16>());
    }
}

/// Elementwise dtype cast f16 → f32 (reverse): the sdpa f16 output is widened
/// back to f32 for the downstream o_proj Q4 GEMV, which consumes f32 activations.
#[kernel]
pub fn ffai_cast_f16_f32(src: Tensor<f16>, mut dst: Tensor<f32>, #[constexpr] n: u32) {
    let i = program_id::<0>();
    if i < n {
        store(dst[i], load(src[i]).cast::<f32>());
    }
}

/// Roll a causal-conv state ON-DEVICE: `new = [old[conv_dim..], xbc]` (drop the
/// oldest conv_dim, append the current input) — keeps the Mamba conv history on
/// the GPU. `keep = (kc-2)*conv_dim`; indices clamped so both select branches
/// are in-bounds.
#[kernel]
pub fn ffai_conv_roll<T>(old: Tensor<T>, xbc: Tensor<T>, mut newst: Tensor<T>,
    #[constexpr] conv_dim: u32, #[constexpr] keep: u32, #[constexpr] n: u32) {
    let i = program_id::<0>();
    if i < n {
        let oi = select(i < keep, i + conv_dim, 0u32);
        let xi = select(i < keep, 0u32, i - keep);
        let v = select(i < keep, load(old[oi]), load(xbc[xi]));
        store(newst[i], v);
    }
}

/// Batched MoE expert UP-projection + ReLU²: gathers the `top_k` selected
/// experts (indices in `idx`) from one contiguous `[n_exp*inter, hid]` Q4 weight
/// and computes all of them in ONE big GEMV — small per-expert matrices run at
/// ~52% DRAM bandwidth, but a [top_k*inter, hid] batch runs at ~90%. `out` is
/// `[top_k*inter]`. grid = top_k*inter threadgroups.
#[kernel]
pub fn ffai_moe_gather_q4_relu2<T>(
    qs: Tensor<u32>, d_f32: Tensor<f16>, x: Tensor<T>, idx: Tensor<u32>, mut out: Tensor<T>,
    #[constexpr] k_in: u32, #[constexpr] inter: u32, #[constexpr] rows_per_tg: u32,
) {
    // 2D grid [inter/rows_per_tg, top_k]: slot = tgid_y; `rows_per_tg` warps per
    // TG each own one inter-row (multi-warp hides global-load latency, same as
    // the dense gemv). rows_per_tg=1 is bit-identical (warp=0, lane=tid).
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let local = tgid_x * rows_per_tg + warp;
    let slot = tgid_y;
    if local < inter {
        let e = load(idx[slot]);
        let row = e * inter + local;
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let dd = load(d_f32[d_base + block]).cast::<f32>();
            let xb = block * 32u32 + sub * 8u32;
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                blk = blk + (nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32)) * load(x[xb + i]).cast::<f32>();
            }
            dot = dot + dd * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            let rr = select(total > 0.0f32, total, 0.0f32);
            store(out[slot * inter + local], (rr * rr).cast::<T>());
        }
    }
}

/// Batched MoE expert DOWN-projection + router-weighted accumulate: for each
/// output row `h`, sums the `top_k` experts' `down[e,h]·x_slot` weighted by
/// `wts[slot]`, into `acc[h]`. One dispatch for all experts. `x` is the
/// `[top_k*inter]` up-relu² output; `qs` is the contiguous `[n_exp*hid, inter]`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_moe_gather_q4_down_accum<T>(
    qs: Tensor<u32>, d_f32: Tensor<f32>, x: Tensor<T>, idx: Tensor<u32>, wts: Tensor<f32>, mut acc: Tensor<T>,
    #[constexpr] inter: u32, #[constexpr] hid: u32, #[constexpr] top_k: u32,
) {
    let h = tgid_x;
    let lane = tid;
    let bpr = inter / 32u32;
    let nwords = bpr * 4u32;
    let mut total = 0.0f32;
    for slot in range(0u32, top_k, 1u32) {
        let e = load(idx[slot]);
        let row = e * hid + h;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let xoff = slot * inter;
        let w = load(wts[slot]);
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let dd = load(d_f32[d_base + block]).cast::<f32>();
            let xb = xoff + block * 32u32 + sub * 8u32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                dot = dot + dd * (nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32)) * load(x[xb + i]).cast::<f32>();
            }
        }
        total = total + w * simd_sum(dot);
    }
    if lane == 0u32 {
        store(acc[h], (load(acc[h]).cast::<f32>() + total).cast::<T>());
    }
}

/// Batched MoE DOWN gather (no accumulate): `out[slot*hid + h] = down[e_slot, h]·
/// x_slot`, one big `[top_k*hid]` GEMV (grid top_k*hid ⇒ high occupancy, vs the
/// fused-accum variant's grid[hid] which serialized top_k experts at ~50% bw).
#[kernel]
pub fn ffai_moe_gather_q4_down<T>(
    qs: Tensor<u32>, d_f32: Tensor<f16>, x: Tensor<T>, idx: Tensor<u32>, mut out: Tensor<T>,
    #[constexpr] inter: u32, #[constexpr] hid: u32, #[constexpr] rows_per_tg: u32,
) {
    // 2D grid [hid/rows_per_tg, top_k]: rows_per_tg warps/TG, one hid-row each
    // (multi-warp latency hiding). rows_per_tg=1 is bit-identical.
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let local = tgid_x * rows_per_tg + warp;
    let slot = tgid_y;
    if local < hid {
        let e = load(idx[slot]);
        let row = e * hid + local;
        let bpr = inter / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let xoff = slot * inter;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let dd = load(d_f32[d_base + block]).cast::<f32>();
            let xb = xoff + block * 32u32 + sub * 8u32;
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                blk = blk + (nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32)) * load(x[xb + i]).cast::<f32>();
            }
            dot = dot + dd * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 { store(out[slot * hid + local], total.cast::<T>()); }
    }
}

/// Router-weighted sum of the per-expert down outputs into `acc`:
/// `acc[h] += Σ_slot wts[slot]·downs[slot*hid + h]`. Cheap (grid hid).
#[kernel]
pub fn ffai_moe_weighted_sum<T>(
    downs: Tensor<T>, wts: Tensor<f32>, mut acc: Tensor<T>,
    #[constexpr] hid: u32, #[constexpr] top_k: u32,
) {
    let h = program_id::<0>();
    if h < hid {
        let mut t = load(acc[h]).cast::<f32>();
        for s in range(0u32, top_k, 1u32) {
            t = t + load(wts[s]) * load(downs[s * hid + h]).cast::<f32>();
        }
        store(acc[h], t.cast::<T>());
    }
}

// ── Q4 (4-bit) coalesced gemv family — half the weight DRAM of Q8, the decode
// bandwidth lever (decode reads cold weights: 35GB resident ≫ L2). Block 32,
// symmetric int4 in [-7,7], one f32 scale/block. qs packs 8 nibbles per u32
// (4 u32/block). Same coalesced walk + warp reduce as the Q8 variants. ──

/// Plain Q4 coalesced matvec: `out[r] = Σ_k dequant4(W[r,k]) · x[...]`.
#[kernel]
pub fn ffai_gemv_q4_coalesced<T>(
    qs: Tensor<u32>, d_f32: Tensor<f16>, x: Tensor<T>, mut out: Tensor<T>,
    #[constexpr] k_in: u32, #[constexpr] m_out: u32, #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    // Multi-warp: `rows_per_tg` warps per threadgroup, each warp owns one row.
    // The single-warp gemv (rows_per_tg=1) is memory-LATENCY-bound (ncu: ~80%
    // scoreboard stalls, <50% occupancy) — packing several warps per TG keeps
    // more global loads in flight to hide that latency. rows_per_tg=1 is
    // bit-identical to the original (warp=0, lane=tid, row=tgid_x).
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row = tgid_x * rows_per_tg + warp;
    if row < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32; // 4 u32 per 32-value block
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let x_base = (row / rows_per_group) * k_in;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let d = load(d_f32[d_base + block]).cast::<f32>();
            let x_blk = x_base + block * 32u32 + sub * 8u32;
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                let q = nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32);
                blk = blk + q * load(x[x_blk + i]).cast::<f32>();
            }
            dot = dot + d * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 { store(out[row], total.cast::<T>()); }
    }
}

/// Q4 GEMV, VECTORIZED weight load: each lane owns whole Q4 blocks and reads the
/// block's 4 packed words as 4 CONSECUTIVE loads → the codegen Vectorize pass
/// collapses them into one 128-bit `VectorLoad` (vs the strided scalar-u32 load,
/// which never vectorizes). 4× fewer weight-load instructions → fewer scoreboard
/// stalls (ncu: the latency-bound GEMV's actual bottleneck). Coalesced: adjacent
/// lanes read adjacent 16-byte blocks.
#[kernel]
pub fn ffai_gemv_q4_vec<T>(
    qs: Tensor<u32>, d_f32: Tensor<f32>, x: Tensor<T>, mut out: Tensor<T>,
    #[constexpr] k_in: u32, #[constexpr] m_out: u32, #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 4u32;
    let d_base = row * bpr;
    let x_base = (row / rows_per_group) * k_in;
    let mut dot = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let wbase = qs_base + b * 4u32;
        // Precompute the 4 index VIDs BEFORE the loads (sdpa_decode trick) so the
        // Vectorize pass sees 4 consecutive bare Op::Load → collapses to one uint4.
        let i0 = wbase;
        let i1 = wbase + 1u32;
        let i2 = wbase + 2u32;
        let i3 = wbase + 3u32;
        let p0 = load(qs[i0]);
        let p1 = load(qs[i1]);
        let p2 = load(qs[i2]);
        let p3 = load(qs[i3]);
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let xb = x_base + b * 32u32;
        let mut acc = 0.0f32;
        for i in range(0u32, 8u32, 1u32) {
            let n0 = (p0 >> (i * 4u32)) & 0xfu32;
            acc = acc + (n0.cast::<f32>() - select(n0 > 7u32, 16.0f32, 0.0f32)) * load(x[xb + i]).cast::<f32>();
        }
        for i in range(0u32, 8u32, 1u32) {
            let n1 = (p1 >> (i * 4u32)) & 0xfu32;
            acc = acc + (n1.cast::<f32>() - select(n1 > 7u32, 16.0f32, 0.0f32)) * load(x[xb + 8u32 + i]).cast::<f32>();
        }
        for i in range(0u32, 8u32, 1u32) {
            let n2 = (p2 >> (i * 4u32)) & 0xfu32;
            acc = acc + (n2.cast::<f32>() - select(n2 > 7u32, 16.0f32, 0.0f32)) * load(x[xb + 16u32 + i]).cast::<f32>();
        }
        for i in range(0u32, 8u32, 1u32) {
            let n3 = (p3 >> (i * 4u32)) & 0xfu32;
            acc = acc + (n3.cast::<f32>() - select(n3 > 7u32, 16.0f32, 0.0f32)) * load(x[xb + 24u32 + i]).cast::<f32>();
        }
        dot = dot + d * acc;
    }
    let total = simd_sum(dot);
    if lane == 0u32 { store(out[row], total.cast::<T>()); }
}

/// Q4 GEMV, 2 output rows per warp: load the shared activation `x` ONCE and run
/// TWO independent weight streams (rows 2r, 2r+1) → 2× memory-level-parallelism on
/// the latency-bound Q4 weight read (ncu: scoreboard-stalled, <50% BW), plus the x
/// read is shared (halved). Stacks with multi-warp (`rows_per_tg` warps/TG).
#[kernel]
pub fn ffai_gemv_q4_coalesced_2row<T>(
    qs: Tensor<u32>, d_f32: Tensor<f32>, x: Tensor<T>, mut out: Tensor<T>,
    #[constexpr] k_in: u32, #[constexpr] m_out: u32, #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row_a = (tgid_x * rows_per_tg + warp) * 2u32;
    let row_b = row_a + 1u32;
    if row_a < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qa = row_a * bpr * 4u32;
        let qb = row_b * bpr * 4u32;
        let da = row_a * bpr;
        let db = row_b * bpr;
        let x_base = (row_a / rows_per_group) * k_in;
        let mut dot_a = 0.0f32;
        let mut dot_b = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let pa = load(qs[qa + j]);
            let pb = load(qs[qb + j]);
            let dda = load(d_f32[da + block]);
            let ddb = load(d_f32[db + block]);
            let xb = x_base + block * 32u32 + sub * 8u32;
            let mut ba = 0.0f32;
            let mut bb = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let xv = load(x[xb + i]).cast::<f32>();
                let na = (pa >> (i * 4u32)) & 0xfu32;
                let nb = (pb >> (i * 4u32)) & 0xfu32;
                ba = ba + (na.cast::<f32>() - select(na > 7u32, 16.0f32, 0.0f32)) * xv;
                bb = bb + (nb.cast::<f32>() - select(nb > 7u32, 16.0f32, 0.0f32)) * xv;
            }
            dot_a = dot_a + dda * ba;
            dot_b = dot_b + ddb * bb;
        }
        let ta = simd_sum(dot_a);
        let tb = simd_sum(dot_b);
        if lane == 0u32 {
            store(out[row_a], ta.cast::<T>());
            if row_b < m_out { store(out[row_b], tb.cast::<T>()); }
        }
    }
}

/// Q4 coalesced matvec with fused ReLU² (MoE expert up).
/// Multi-warp: `rows_per_tg` warps per threadgroup, each warp owns one output
/// row. The single-warp form (`rows_per_tg=1`) is memory-LATENCY-bound (small
/// shared-expert matrices: ~50% BW); packing several warps per TG keeps more
/// global Q4 loads in flight to hide that latency. `rows_per_tg=1` is
/// bit-identical to the original (warp=0, lane=tid, row=tgid_x).
#[kernel]
pub fn ffai_gemv_q4_coalesced_relu2<T>(
    qs: Tensor<u32>, d_f32: Tensor<f16>, x: Tensor<T>, mut out: Tensor<T>,
    #[constexpr] k_in: u32, #[constexpr] m_out: u32, #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row = tgid_x * rows_per_tg + warp;
    if row < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let x_base = (row / rows_per_group) * k_in;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let d = load(d_f32[d_base + block]).cast::<f32>();
            let x_blk = x_base + block * 32u32 + sub * 8u32;
            // Scale `d` is constant across the block's 8 nibbles — factor it OUT of
            // the inner loop (`d·Σ q·x` not `Σ d·q·x`): the dequant is ALU-bound
            // (~56 ALU ops/word vs 1 load), so dropping 7 mul/word is a real win.
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                let q = nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32);
                blk = blk + q * load(x[x_blk + i]).cast::<f32>();
            }
            dot = dot + d * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            let r = select(total > 0.0f32, total, 0.0f32);
            store(out[row], (r * r).cast::<T>());
        }
    }
}

/// Q4 coalesced matvec, scale + accumulate in place (MoE expert down).
/// Multi-warp (`rows_per_tg` warps/TG, one output row each) — same latency-
/// hiding rationale as the relu2 variant; `rows_per_tg=1` is bit-identical.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_gemv_q4_coalesced_accum<T>(
    qs: Tensor<u32>, d_f32: Tensor<f16>, x: Tensor<T>, mut acc: Tensor<T>, scale: Tensor<f32>,
    #[constexpr] k_in: u32, #[constexpr] m_out: u32, #[constexpr] rows_per_group: u32,
    #[constexpr] rows_per_tg: u32,
) {
    let warp = tid / 32u32;
    let lane = tid % 32u32;
    let row = tgid_x * rows_per_tg + warp;
    if row < m_out {
        let bpr = k_in / 32u32;
        let nwords = bpr * 4u32;
        let qs_base = row * bpr * 4u32;
        let d_base = row * bpr;
        let x_base = (row / rows_per_group) * k_in;
        let mut dot = 0.0f32;
        for j in range(lane, nwords, 32u32) {
            let block = j / 4u32;
            let sub = j % 4u32;
            let packed = load(qs[qs_base + j]);
            let d = load(d_f32[d_base + block]).cast::<f32>();
            let x_blk = x_base + block * 32u32 + sub * 8u32;
            // Scale `d` is constant across the block's 8 nibbles — factor it OUT of
            // the inner loop (`d·Σ q·x` not `Σ d·q·x`): the dequant is ALU-bound
            // (~56 ALU ops/word vs 1 load), so dropping 7 mul/word is a real win.
            let mut blk = 0.0f32;
            for i in range(0u32, 8u32, 1u32) {
                let nib = (packed >> (i * 4u32)) & 0xfu32;
                let q = nib.cast::<f32>() - select(nib > 7u32, 16.0f32, 0.0f32);
                blk = blk + q * load(x[x_blk + i]).cast::<f32>();
            }
            dot = dot + d * blk;
        }
        let total = simd_sum(dot);
        if lane == 0u32 {
            let s = load(scale[0]);
            let prev = load(acc[row]).cast::<f32>();
            store(acc[row], (prev + s * total).cast::<T>());
        }
    }
}

/// Append a decode step's K (or V) into an IN-PLACE device KV cache, so the
/// growing context never round-trips through the host. `src` is `[nkv*hd]` (the
/// new token's per-head vectors), `dst` is the cache `[nkv, cap, hd]`, `posbuf[0]`
/// is the current position. Writes `dst[h, pos, :] = src[h, :]`. Runtime `pos`
/// rides in a buffer (NOT constexpr) so the kernel is compiled once, not per step.
#[kernel]
pub fn ffai_kv_append<T>(
    src: Tensor<T>,
    mut dst: Tensor<T>,
    posbuf: Tensor<u32>,
    #[constexpr] hd: u32,
    #[constexpr] cap: u32,
) {
    let idx = program_id::<0>();
    let pos = load(posbuf[0]);
    let h = idx / hd;
    let dd = idx % hd;
    store(dst[h * cap * hd + pos * hd + dd], load(src[idx]));
}

/// BATCHED grouped Q8_0 gemv — ffai_grouped_gemv_q8 over `n_tokens` rows in
/// ONE dispatch (grid z/y = token). Prefill O-LoRA looped the per-token
/// grouped gemv N times; this folds it. x is [n_tokens, n_groups*k_in],
/// out is [n_tokens, m_out]; n_groups = m_out/rows_per_group.
/// Grid (Reduction): [m_out, n_tokens, 1], tg=[32,1,1].
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_grouped_gemv_q8_rows<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
) {
    let row = tgid_x;
    let token = tgid_y;
    let lane = tid;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let n_groups = m_out / rows_per_group;
    let x_base = token * n_groups * k_in + (row / rows_per_group) * k_in;
    let mut acc = 0.0f32;
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let x_blk = x_base + b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]);
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                acc = acc + d * qf * load(x[x_blk + w * 4u32 + i]).cast::<f32>();
            }
        }
    }
    let total = simd_sum(acc);
    if lane == 0u32 {
        store(out[token * m_out + row], total.cast::<T>());
    }
}

/// TOKEN-TILED grouped Q8 gemv — the amortized fix for the prefill O-LoRA-A
/// hotspot. `ffai_grouped_gemv_q8_rows` re-reads each weight row from DRAM
/// once PER TOKEN (no amortization); at N=512 that's the single biggest
/// op in the attention block (~47 ms/layer). Here each threadgroup owns one
/// output row and a TILE of `tokens_per_tile` tokens: the Q8 weight block
/// (d + 8 packed = 32 int8) is loaded ONCE and applied to all T tokens, so
/// the weight DRAM traffic drops T-fold. T accumulators in a register stack.
/// grid (threadgroups) = [m_out, ceil(n_tokens/T), 1], threadgroup [32,1,1].
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn ffai_grouped_gemv_q8_rows_tiled<T>(
    qs: Tensor<u32>,
    d_f32: Tensor<f32>,
    x: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] rows_per_group: u32,
    #[constexpr] n_tokens: u32,
) {
    // Tile size is a compile-time LITERAL (stack_alloc can't size on a
    // constexpr param — codegen emits the array decl before the constant
    // is bound). T=8 tokens/tile → 8-fold weight-DRAM amortization.
    let tokens_per_tile = 8u32;
    let row = tgid_x;
    let ttile = tgid_y;
    let lane = tid;
    let tok0 = ttile * tokens_per_tile;
    let bpr = k_in / 32u32;
    let qs_base = row * bpr * 8u32;
    let d_base = row * bpr;
    let n_groups = m_out / rows_per_group;
    let group_off = (row / rows_per_group) * k_in;

    stack_alloc("acc", 8, "f32");
    for t in range(0u32, tokens_per_tile, 1u32) {
        stack_store("acc", t, 0.0f32);
    }
    for b in range(lane, bpr, 32u32) {
        let d = load(d_f32[d_base + b]).cast::<f32>();
        let blk = b * 32u32;
        for w in range(0u32, 8u32, 1u32) {
            let packed = load(qs[qs_base + b * 8u32 + w]); // 4 int8 weights, read ONCE
            for i in range(0u32, 4u32, 1u32) {
                let by = (packed >> (i * 8u32)) & 0xffu32;
                let qf = by.cast::<f32>() - select(by > 127u32, 256.0f32, 0.0f32);
                let wv = d * qf; // weight value, reused across the T tokens
                let kk = blk + w * 4u32 + i;
                for t in range(0u32, tokens_per_tile, 1u32) {
                    let tok = tok0 + t;
                    if tok < n_tokens {
                        let xb = tok * n_groups * k_in + group_off + kk;
                        let prev = stack_load("acc", t);
                        stack_store("acc", t, prev + wv * load(x[xb]).cast::<f32>());
                    }
                }
            }
        }
    }
    for t in range(0u32, tokens_per_tile, 1u32) {
        let tok = tok0 + t;
        let total = simd_sum(stack_load("acc", t));
        if (lane == 0u32) & (tok < n_tokens) {
            store(out[tok * m_out + row], total.cast::<T>());
        }
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        ffai_gemv_q8,
        ffai_grouped_gemv_q8,
        ffai_grouped_gemv_q8_rows,
        ffai_grouped_gemv_q8_rows_tiled,
    };

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_gemv_q8(dt: DType) -> BenchSetup {
        let m_out = 4096usize;
        let k_in = 8192usize;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_gemv_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", k_in, dt))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + k_in * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_grouped_gemv_q8::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", 8 * k_in, dt))
            .buffer(BenchBuffer::zeros("out", m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .grid_3d(m_out as u32, 1, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + 8 * k_in * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8_rows(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let n_tokens = 256usize;
        let n_groups = m_out / 1024;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_grouped_gemv_q8_rows::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", n_tokens * n_groups * k_in, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .grid_3d(m_out as u32, n_tokens as u32, 1, [32, 1, 1])
            .bytes_moved((m_out * bpr * 36 + n_tokens * n_groups * k_in * dt.size_bytes()) as u64)
    }

    #[bench(dtypes = [f32, f16, bf16])]
    fn bench_grouped_gemv_q8_rows_tiled(dt: DType) -> BenchSetup {
        let m_out = 8192usize;
        let k_in = 4096usize;
        let n_tokens = 256usize;
        let tokens_per_tile = 8usize;
        let n_groups = m_out / 1024;
        let bpr = k_in / 32;
        BenchSetup::new(ffai_grouped_gemv_q8_rows_tiled::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("qs", m_out * bpr * 8, DType::U32))
            .buffer(BenchBuffer::random("d_f32", m_out * bpr, DType::F32))
            .buffer(BenchBuffer::random("x", n_tokens * n_groups * k_in, dt))
            .buffer(BenchBuffer::zeros("out", n_tokens * m_out, dt).output())
            .constexpr("k_in", k_in as u32)
            .constexpr("m_out", m_out as u32)
            .constexpr("rows_per_group", 1024u32)
            .constexpr("n_tokens", n_tokens as u32)
            .grid_3d(m_out as u32, (n_tokens as u32).div_ceil(tokens_per_tile as u32), 1, [
                32, 1, 1,
            ])
            .bytes_moved((m_out * bpr * 36 + n_tokens * n_groups * k_in * dt.size_bytes()) as u64)
    }
}
