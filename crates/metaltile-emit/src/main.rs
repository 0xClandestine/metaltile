//! metaltile-emit
//!
//! Build-time codegen tool. Walks a registry of `#[kernel]` definitions and
//! produces three artifacts under `<out>/`:
//!
//!   Resources/kernels/<name>.metal   — MSL source per kernel (debug aid)
//!   Resources/kernels.metallib       — compiled Metal library
//!   Resources/manifest.json          — per-kernel metadata
//!   Generated/MetalTileKernels.swift — typed Swift dispatch wrappers
//!
//! Phase 0 ships two kernels: `vector_add` (proof-of-life) and `rms_norm`
//! across f32/f16/bf16. Add more in `register_kernels()` as later phases land.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail};
use clap::Parser;
use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;
use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode, Param, ParamKind},
};
// Bring high-perf kernels from metaltile-std into the emit registry.
// All canonical sources — no inline duplicates here; metaltile-emit
// just registers, names, and bakes mode/dispatch metadata.
use metaltile_std::{
    ffai::{
        arg_reduce::ffai_argmax,
        gated_delta::{mt_gated_delta_chunk, mt_gated_delta_step},
        gated_delta_prep::mt_gated_delta_prep_step,
        gather::ffai_gather,
        moe::mt_moe_gather_qmm_mma_int4_bm16,
        moe_mpp,
        moe_mpp_bm8,
        moe_mpp_bm64,
        rope_llama::ffai_rope_llama,
        sdpa_decode::ffai_sdpa_decode,
    },
    mlx::{
        binary::{mt_mul, vector_add},
        gemv::mt_gemv,
        quantized::mt_qmm_mma,
        quantized_mpp,
        rms_norm::mt_rms_norm,
        steel::attn::steel_attention_mma::mt_sdpa_prefill_mma,
        unary::{mt_cast_to_f32, mt_gelu, mt_relu, mt_sigmoid, mt_silu, mt_softplus},
    },
    probe::mpp_matmul_smoke,
};
use serde::Serialize;

// ─── CLI ──────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "metaltile-emit", about = "Emit metallib + manifest + Swift wrappers")]
struct Cli {
    /// Output directory (typically `Sources/MetalTileSwift/` of a Swift package).
    #[arg(long)]
    out: PathBuf,

    /// SDK to use for `xcrun metal` invocation.
    #[arg(long, default_value = "macosx")]
    sdk: String,

    /// Skip the metallib compile step (still emit .metal + manifest + Swift).
    /// Useful when running on a host without the Metal toolchain.
    #[arg(long)]
    no_compile: bool,
}
//   temperature_in — temperature [1] (f32, must be > 0)
//   uniform_in     — uniform draw in [0, 1) [1] (f32)
//
// Output is the smallest index `i` such that the cumulative softmax
// (in fp32) up to and including `i` is ≥ `uniform_in * sum_exp`.
//
// The greedy fast path (T == 0) is the separate `argmax` kernel —
// this kernel is for the pure-temperature pipeline that bypasses the
// CPU logits readback. Top-K / top-P / min-P / rep-penalty still go
// through the CPU `Sampling.sample(...)` path until those kernels
// land separately.
//
// Cost: ~150µs at vocab=152K on M-class GPU (1% overhead per token
// at 60 tok/s decode). The cooperative max + sum-exp passes are
// fast; the single-thread CDF walk is the bottleneck, but still
// cheaper than the full vocab readback the CPU path requires.
#[kernel]
fn softmax_categorical_sample<T>(
    inp: Tensor<T>,
    out: Tensor<u32>,
    temperature_in: Tensor<f32>,
    uniform_in: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let lid = tid;
    let inv_t = 1.0f32 / load(temperature_in[0]);

    // ─── Pass 1: cooperative max reduce ─────────────────────────────
    let mut local_max = neg_infinity();
    threadgroup_alloc("tg_max", 256);
    let n_iters = (n + lsize - 1u32) / lsize;
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>() * inv_t;
            local_max = select(v > local_max, v, local_max);
        }
    }
    threadgroup_store("tg_max", lid, local_max);
    threadgroup_barrier();

    if lid < 128u32 {
        let ov = threadgroup_load("tg_max", lid + 128u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 64u32 {
        let ov = threadgroup_load("tg_max", lid + 64u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 32u32 {
        let ov = threadgroup_load("tg_max", lid + 32u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 16u32 {
        let ov = threadgroup_load("tg_max", lid + 16u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 8u32 {
        let ov = threadgroup_load("tg_max", lid + 8u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 4u32 {
        let ov = threadgroup_load("tg_max", lid + 4u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid < 2u32 {
        let ov = threadgroup_load("tg_max", lid + 2u32);
        let tv = threadgroup_load("tg_max", lid);
        threadgroup_store("tg_max", lid, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    if lid == 0u32 {
        let ov = threadgroup_load("tg_max", 1u32);
        let tv = threadgroup_load("tg_max", 0u32);
        threadgroup_store("tg_max", 0u32, select(ov > tv, ov, tv));
    }
    threadgroup_barrier();
    let max_val = threadgroup_load("tg_max", 0u32);

    // ─── Pass 2: cooperative sum-exp reduce ─────────────────────────
    let mut local_sum = 0.0f32;
    threadgroup_alloc("tg_sum", 256);
    for _r in range(0u32, n_iters, 1u32) {
        let pos = _r * lsize + lid;
        if pos < n {
            let v = load(inp[pos]).cast::<f32>() * inv_t;
            local_sum = local_sum + exp(v - max_val);
        }
    }
    threadgroup_store("tg_sum", lid, local_sum);
    threadgroup_barrier();

    if lid < 128u32 {
        let ov = threadgroup_load("tg_sum", lid + 128u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 64u32 {
        let ov = threadgroup_load("tg_sum", lid + 64u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 32u32 {
        let ov = threadgroup_load("tg_sum", lid + 32u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 16u32 {
        let ov = threadgroup_load("tg_sum", lid + 16u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 8u32 {
        let ov = threadgroup_load("tg_sum", lid + 8u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 4u32 {
        let ov = threadgroup_load("tg_sum", lid + 4u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid < 2u32 {
        let ov = threadgroup_load("tg_sum", lid + 2u32);
        let tv = threadgroup_load("tg_sum", lid);
        threadgroup_store("tg_sum", lid, ov + tv);
    }
    threadgroup_barrier();
    if lid == 0u32 {
        let ov = threadgroup_load("tg_sum", 1u32);
        let tv = threadgroup_load("tg_sum", 0u32);
        threadgroup_store("tg_sum", 0u32, ov + tv);
    }
    threadgroup_barrier();
    let total = threadgroup_load("tg_sum", 0u32);

    // ─── Pass 3: single-thread inverse CDF walk ─────────────────────
    if lid == 0u32 {
        let target = load(uniform_in[0]) * total;
        let mut cum = 0.0f32;
        let mut found_idx = n - 1u32; // fallback to last index
        let mut done = 0u32;
        for i in range(0u32, n, 1u32) {
            let v = load(inp[i]).cast::<f32>() * inv_t;
            cum = cum + exp(v - max_val);
            let hit = (cum >= target) & (done == 0u32);
            found_idx = select(hit, i, found_idx);
            done = select(hit, 1u32, done);
        }
        store(out[0], found_idx);
    }
}

// Affine quantize a one-token K (or V) row into an int8-packed KV
// cache slot at `position`. One thread per group.
//
// Source layout : [n_kv_heads, head_dim]                 (fp16 / bf16)
// Dest layouts  :
//   weights     : [n_kv_heads, max_seq, head_dim / 4]    (u32, 4 u8 per word)
//   scales      : [n_kv_heads, max_seq, head_dim / group_size]  (T)
//   biases      : [n_kv_heads, max_seq, head_dim / group_size]  (T)
//
// Per group of `group_size` values: find min/max, derive
// scale = (max - min) / 255, bias = min, quantize to u8, pack 4 per
// uint32. Defends against zero-range groups (all-zero K/V) by
// forcing scale = 1 — those reconstruct as bias regardless of q.
//
// Grid: nKVHeads * (head_dim / group_size) threads. Tiny dispatch
// (for Qwen3 1.7B: 8 * 128/64 = 16 threads per token).
#[kernel]
fn quantize_kv_int8<T>(
    src: Tensor<T>,
    mut out_w: Tensor<u32>,
    mut out_s: Tensor<T>,
    mut out_b: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] group_size: u32,
    #[constexpr] position: u32,
) {
    let g_global = program_id::<0>();
    let groups_per_head = head_dim / group_size;
    let h = g_global / groups_per_head;
    let g_in_h = g_global - h * groups_per_head;
    let d_start = g_in_h * group_size;
    let src_base = h * head_dim;

    // Pass 1: find min + max over the group.
    let mut mn = load(src[src_base + d_start]).cast::<f32>();
    let mut mx = mn;
    for i in range(1u32, group_size, 1u32) {
        let v = load(src[src_base + d_start + i]).cast::<f32>();
        mn = select(v < mn, v, mn);
        mx = select(v > mx, v, mx);
    }
    let range = mx - mn;
    let safe_scale = select(range == 0.0f32, 1.0f32, range / 255.0f32);
    let inv_scale = 1.0f32 / safe_scale;

    // Store scale + bias for the group.
    let dst_sb_idx = (h * max_seq + position) * groups_per_head + g_in_h;
    store(out_s[dst_sb_idx], safe_scale.cast::<T>());
    store(out_b[dst_sb_idx], mn.cast::<T>());

    // Pass 2: quantize + pack 4 u8 per u32.
    let dst_w_base = (h * max_seq + position) * (head_dim / 4u32) + d_start / 4u32;
    for p in range(0u32, group_size / 4u32, 1u32) {
        let mut packed = 0u32;
        for i in range(0u32, 4u32, 1u32) {
            let v = load(src[src_base + d_start + p * 4u32 + i]).cast::<f32>();
            // Round half-up via +0.5 + truncating cast.
            let q_f = (v - mn) * inv_scale + 0.5f32;
            let q_clamped_f = select(q_f > 255.0f32, 255.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_clamped_f.cast::<u32>();
            packed = packed | (q << (i * 8u32));
        }
        store(out_w[dst_w_base + p], packed);
    }
}

// Bulk dequant the live slice of an int8-quantized K (or V) cache
// into a fp16/bf16 working buffer that SDPA can read directly. One
// thread per output element.
//
// Output layout : [n_kv_heads, max_seq, head_dim] T  (same as raw KVCache)
// Only positions [0, n_positions) are written — SDPA's `n_kv` is
// the live length, `kv_stride = max_seq`.
//
// Grid: nKVHeads * n_positions * head_dim threads.
#[kernel]
fn bulk_dequant_kv_int8<T>(
    in_w: Tensor<u32>,
    in_s: Tensor<T>,
    in_b: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] group_size: u32,
    #[constexpr] n_positions: u32,
) {
    let idx = program_id::<0>();
    let total_per_head = n_positions * head_dim;
    let h = idx / total_per_head;
    let rest = idx - h * total_per_head;
    let pos = rest / head_dim;
    let d = rest - pos * head_dim;

    let groups_per_head = head_dim / group_size;
    let g = d / group_size;
    let scale = load(in_s[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();
    let bias = load(in_b[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();

    let pack_idx = (h * max_seq + pos) * (head_dim / 4u32) + d / 4u32;
    let lane = d & 3u32;
    let packed = load(in_w[pack_idx]);
    let q = (packed >> (lane * 8u32)) & 255u32;
    let w_real = q.cast::<f32>() * scale + bias;

    let dst_idx = h * max_seq * head_dim + pos * head_dim + d;
    store(out[dst_idx], w_real.cast::<T>());
}

// Mamba 2 / Mamba 1D depthwise causal-conv step — streaming-decode
// form. Each channel has its own kernel of size `kernel_size` (K,
// typically 4). The convention is causal: the output at time t
// depends only on inputs at t-K+1..t.
//
//   y[d] = bias[d]
//        + w[K-1][d] * x[d]
//        + Σ_{k=0..K-2} w[k][d] * state[k][d]
//
// `state` holds the K-1 most recent inputs (state[k][d] is in[t-K+1+k][d]).
// After computing y, the kernel shifts state in-place:
//
//   state[k][d] = state[k+1][d]   for k in [0, K-2)
//   state[K-2][d] = x[d]
//
// Each (channel d) is owned by exactly one thread, so the read-then-
// write shift is safe within the thread without barriers. Activation
// (Mamba 2 typically follows the conv with SiLU) is the caller's
// concern — kept separate for composability.
//
// Inputs:
//   x     [n_channels]           T  — current timestep input
//   w     [K, n_channels]        T  — per-channel kernel weights
//   b     [n_channels]           T  — per-channel bias
// In/out:
//   state [K-1, n_channels]      T  — rolling window of last K-1 inputs
// Output:
//   y     [n_channels]           T
//
// Grid: n_channels threads (one per channel). For Mamba 2 with
// conv_dim ~1500 channels and K=4, this is a tiny dispatch.
#[kernel]
fn conv1d_causal_step<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    b: Tensor<T>,
    mut state: Tensor<T>,
    mut y: Tensor<T>,
    #[constexpr] n_channels: u32,
    #[constexpr] kernel_size: u32,
) {
    let d = program_id::<0>();
    let x_d = load(x[d]).cast::<f32>();
    let b_d = load(b[d]).cast::<f32>();

    // Convolution: w[K-1] pairs with current input x[d]; w[0]..w[K-2]
    // pair with state[0]..state[K-2].
    let w_last = load(w[(kernel_size - 1u32) * n_channels + d]).cast::<f32>();
    let mut acc = b_d + w_last * x_d;
    for k in range(0u32, kernel_size - 1u32, 1u32) {
        let s_kd = load(state[k * n_channels + d]).cast::<f32>();
        let w_kd = load(w[k * n_channels + d]).cast::<f32>();
        acc = acc + w_kd * s_kd;
    }
    store(y[d], acc.cast::<T>());

    // Shift state up by one (drop state[0], append x[d] at the tail).
    // Sequential within the thread → safe even though state[k] is
    // read after being written in the prior iteration's write to
    // state[k-1] (we read state[k+1] each iteration, never state[k]).
    for k in range(0u32, kernel_size - 2u32, 1u32) {
        let next = load(state[(k + 1u32) * n_channels + d]);
        store(state[k * n_channels + d], next);
    }
    store(state[(kernel_size - 2u32) * n_channels + d], load(x[d]));
}

// Mamba 2 selective-scan single-token decode step. Updates the
// recurrent state `h` in-place and emits the output channel vector
// `y`. State `h` is stored in fp32 because it accumulates over many
// decode steps and bf16's 7-bit mantissa drifts fast.
//
// Per Mamba 2's SSD form, restricted to single-token decode:
//
//   h[head, n, d]_new = exp(A[head] * dt) * h[head, n, d]_old
//                       + dt * B[n] * x[head, d]
//   y[head, d]         = Σ_n  C[n] * h[head, n, d]_new
//
// Where:
//   x  [n_heads, head_dim]                T  — input channels
//   a  [n_heads]                          T  — per-head selective coeff (negative; controls decay rate)
//   b  [state_dim]                        T  — state-input projection (shared across heads)
//   c  [state_dim]                        T  — state-output projection (shared across heads)
//   dt [n_heads]                          T  — per-head time delta (Mamba 2 spec)
//   h  [n_heads, state_dim, head_dim]     f32 — recurrent state (read + written in place)
//   y  [n_heads, head_dim]                T  — output channels
//
// One thread per (head, d) — total n_heads * head_dim threads. Each
// thread walks the state_dim axis once: loads h[head, n, d], computes
// the updated value, writes it back, and accumulates C[n] * new_h
// into y[head, d]. No cross-thread sync needed because each (head, d)
// column of h is owned by exactly one thread.
//
// Note: this is the decode (single-token) form. Chunked prefill uses
// a parallel-scan variant that's a separate kernel — not needed for
// the inference path Phase 5e ships.
#[kernel]
fn ssm_step<T>(
    x: Tensor<T>,
    a: Tensor<T>,
    b: Tensor<T>,
    c: Tensor<T>,
    dt: Tensor<T>,
    mut h: Tensor<f32>,
    mut y: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] state_dim: u32,
) {
    let idx = program_id::<0>();
    let h_id = idx / head_dim;
    let d = idx - h_id * head_dim;

    let dt_val = load(dt[h_id]).cast::<f32>();
    let a_val = load(a[h_id]).cast::<f32>();
    let decay = exp(a_val * dt_val);
    let x_d = load(x[h_id * head_dim + d]).cast::<f32>();

    let mut y_d = 0.0f32;
    let h_base = h_id * state_dim * head_dim;
    for n in range(0u32, state_dim, 1u32) {
        let h_idx = h_base + n * head_dim + d;
        let h_old = load(h[h_idx]);
        let b_n = load(b[n]).cast::<f32>();
        let new_h = decay * h_old + dt_val * b_n * x_d;
        store(h[h_idx], new_h);
        let c_n = load(c[n]).cast::<f32>();
        y_d = y_d + c_n * new_h;
    }
    store(y[h_id * head_dim + d], y_d.cast::<T>());
}

// Same shape as `quantize_kv_int8` but at 4 bits per element —
// pack 8 nibbles per uint32 and use 0..15 quantization levels.
// Row of head_dim values → head_dim/8 uint32s of weights.
//
// Cache layouts:
//   weights [n_kv_heads, max_seq, head_dim / 8]            u32
//   scales  [n_kv_heads, max_seq, head_dim / group_size]   T
//   biases  [n_kv_heads, max_seq, head_dim / group_size]   T
#[kernel]
fn quantize_kv_int4<T>(
    src: Tensor<T>,
    mut out_w: Tensor<u32>,
    mut out_s: Tensor<T>,
    mut out_b: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] group_size: u32,
    #[constexpr] position: u32,
) {
    let g_global = program_id::<0>();
    let groups_per_head = head_dim / group_size;
    let h = g_global / groups_per_head;
    let g_in_h = g_global - h * groups_per_head;
    let d_start = g_in_h * group_size;
    let src_base = h * head_dim;

    let mut mn = load(src[src_base + d_start]).cast::<f32>();
    let mut mx = mn;
    for i in range(1u32, group_size, 1u32) {
        let v = load(src[src_base + d_start + i]).cast::<f32>();
        mn = select(v < mn, v, mn);
        mx = select(v > mx, v, mx);
    }
    let range = mx - mn;
    let safe_scale = select(range == 0.0f32, 1.0f32, range / 15.0f32);
    let inv_scale = 1.0f32 / safe_scale;

    let dst_sb_idx = (h * max_seq + position) * groups_per_head + g_in_h;
    store(out_s[dst_sb_idx], safe_scale.cast::<T>());
    store(out_b[dst_sb_idx], mn.cast::<T>());

    // Pack 8 nibbles per uint32.
    let dst_w_base = (h * max_seq + position) * (head_dim / 8u32) + d_start / 8u32;
    for p in range(0u32, group_size / 8u32, 1u32) {
        let mut packed = 0u32;
        for i in range(0u32, 8u32, 1u32) {
            let v = load(src[src_base + d_start + p * 8u32 + i]).cast::<f32>();
            let q_f = (v - mn) * inv_scale + 0.5f32;
            let q_clamped_f = select(q_f > 15.0f32, 15.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
            let q = q_clamped_f.cast::<u32>();
            packed = packed | (q << (i * 4u32));
        }
        store(out_w[dst_w_base + p], packed);
    }
}

// int4 bulk dequant. Output layout matches the raw cache:
// [n_kv_heads, max_seq, head_dim]. Only positions [0, n_positions)
// are written. One thread per output element.
#[kernel]
fn bulk_dequant_kv_int4<T>(
    in_w: Tensor<u32>,
    in_s: Tensor<T>,
    in_b: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] group_size: u32,
    #[constexpr] n_positions: u32,
) {
    let idx = program_id::<0>();
    let total_per_head = n_positions * head_dim;
    let h = idx / total_per_head;
    let rest = idx - h * total_per_head;
    let pos = rest / head_dim;
    let d = rest - pos * head_dim;

    let groups_per_head = head_dim / group_size;
    let g = d / group_size;
    let scale = load(in_s[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();
    let bias = load(in_b[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();

    let pack_idx = (h * max_seq + pos) * (head_dim / 8u32) + d / 8u32;
    let lane = d & 7u32;
    let packed = load(in_w[pack_idx]);
    let q = (packed >> (lane * 4u32)) & 15u32;
    let w_real = q.cast::<f32>() * scale + bias;

    let dst_idx = h * max_seq * head_dim + pos * head_dim + d;
    store(out[dst_idx], w_real.cast::<T>());
}

// KV cache update — write a one-token K (or V) slice into the
// per-head cache slot at `position`. Source layout: [n_kv_heads, head_dim].
// Dest layout: [n_kv_heads, max_seq, head_dim]. One thread per output
// element (n_kv_heads * head_dim total threads).
#[kernel]
fn kv_cache_update<T>(
    src: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] position: u32,
) {
    let idx = program_id::<0>();
    let h = idx / head_dim;
    let d = idx - h * head_dim;
    let dst_idx = h * max_seq * head_dim + position * head_dim + d;
    store(out[dst_idx], load(src[idx]));
}
// look up packed_w[token_id, d/8], extract the right nibble, then
// dequantize via w_real = q * scale + bias (scale/bias for the group d
// belongs to). One thread per output element.
#[kernel]
fn dequant_gather_int4<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let packs_per_row = hidden / 8u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let pack_idx = token_id * packs_per_row + d / 8u32;
    let nibble = d & 7u32;
    let packed = load(weight[pack_idx]);
    let q = (packed >> (nibble * 4u32)) & 15u32;
    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}

// MLX-format int4 dequantizing GEMV — sub-group cooperative version.
// Reduction-mode kernel; one threadgroup per output row. Threads
// stride across PACKS (not groups), giving in_dim/8-way parallelism
// per row instead of in_dim/group_size-way. For Qwen3 4B (in_dim=2560,
// group_size=64): 320 packs per row vs 40 groups — 8× more thread work.
//
// Layouts:
//   weight  [out_dim, in_dim / 8]            uint32
//   scales  [out_dim, in_dim / group_size]   T
//   biases  [out_dim, in_dim / group_size]   T
//   input   [in_dim]                         T
//   output  [out_dim]                        T
#[kernel]
fn dequant_gemv_int4<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 8u32;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / 8u32;
    let row_pack_off = row * n_packs_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    // Each thread handles one pack at a time, striding by lsize.
    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();

            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 8u32;

            let q0 = (packed >> 0u32) & 15u32;
            let q1 = (packed >> 4u32) & 15u32;
            let q2 = (packed >> 8u32) & 15u32;
            let q3 = (packed >> 12u32) & 15u32;
            let q4 = (packed >> 16u32) & 15u32;
            let q5 = (packed >> 20u32) & 15u32;
            let q6 = (packed >> 24u32) & 15u32;
            let q7 = (packed >> 28u32) & 15u32;

            acc = acc + (q0.cast::<f32>() * scale + bias) * load(input[p_off + 0u32]).cast::<f32>();
            acc = acc + (q1.cast::<f32>() * scale + bias) * load(input[p_off + 1u32]).cast::<f32>();
            acc = acc + (q2.cast::<f32>() * scale + bias) * load(input[p_off + 2u32]).cast::<f32>();
            acc = acc + (q3.cast::<f32>() * scale + bias) * load(input[p_off + 3u32]).cast::<f32>();
            acc = acc + (q4.cast::<f32>() * scale + bias) * load(input[p_off + 4u32]).cast::<f32>();
            acc = acc + (q5.cast::<f32>() * scale + bias) * load(input[p_off + 5u32]).cast::<f32>();
            acc = acc + (q6.cast::<f32>() * scale + bias) * load(input[p_off + 6u32]).cast::<f32>();
            acc = acc + (q7.cast::<f32>() * scale + bias) * load(input[p_off + 7u32]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// MLX-format int3 dequantizing GEMV. 3-bit values: 8 values in 3 bytes
// (24 bits). uint32 cycle: 4 chunks span 3 uint32 (4×3=12 bytes →
// 4×8=32 values per cycle). Same byte-stream layout as int6 but
// different intra-chunk value extraction.
#[kernel]
fn dequant_gemv_int3<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_groups = in_dim / group_size;
    let u32_per_row = in_dim * 3u32 / 32u32;
    let u32_per_group = group_size * 3u32 / 32u32;
    let row_u32_off = row * u32_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let g_iters = (n_groups + lsize - 1u32) / lsize;
    for g_iter in range(0u32, g_iters, 1u32) {
        let g = g_iter * lsize + tid;
        if g < n_groups {
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();
            let g_start = g * group_size;
            let g_u32_off = row_u32_off + g * u32_per_group;
            let cycles = group_size / 32u32;

            for c in range(0u32, cycles, 1u32) {
                let cy = g_u32_off + c * 3u32;
                let u0 = load(weight[cy]);
                let u1 = load(weight[cy + 1u32]);
                let u2 = load(weight[cy + 2u32]);
                let xo = g_start + c * 32u32;

                // Chunk 0 — bytes 0,1,2 of u0
                let v0 = u0 & 7u32;
                let v1 = (u0 >> 3u32) & 7u32;
                let v2 = ((u0 >> 6u32) & 3u32) | (((u0 >> 8u32) & 1u32) << 2u32);
                let v3 = (u0 >> 9u32) & 7u32;
                let v4 = (u0 >> 12u32) & 7u32;
                let v5 = ((u0 >> 15u32) & 1u32) | (((u0 >> 16u32) & 3u32) << 1u32);
                let v6 = (u0 >> 18u32) & 7u32;
                let v7 = (u0 >> 21u32) & 7u32;
                acc =
                    acc + (v0.cast::<f32>() * scale + bias) * load(input[xo + 0u32]).cast::<f32>();
                acc =
                    acc + (v1.cast::<f32>() * scale + bias) * load(input[xo + 1u32]).cast::<f32>();
                acc =
                    acc + (v2.cast::<f32>() * scale + bias) * load(input[xo + 2u32]).cast::<f32>();
                acc =
                    acc + (v3.cast::<f32>() * scale + bias) * load(input[xo + 3u32]).cast::<f32>();
                acc =
                    acc + (v4.cast::<f32>() * scale + bias) * load(input[xo + 4u32]).cast::<f32>();
                acc =
                    acc + (v5.cast::<f32>() * scale + bias) * load(input[xo + 5u32]).cast::<f32>();
                acc =
                    acc + (v6.cast::<f32>() * scale + bias) * load(input[xo + 6u32]).cast::<f32>();
                acc =
                    acc + (v7.cast::<f32>() * scale + bias) * load(input[xo + 7u32]).cast::<f32>();

                // Chunk 1 — byte 3 of u0, bytes 0,1 of u1
                let v8 = (u0 >> 24u32) & 7u32;
                let v9 = (u0 >> 27u32) & 7u32;
                let v10 = ((u0 >> 30u32) & 3u32) | ((u1 & 1u32) << 2u32);
                let v11 = (u1 >> 1u32) & 7u32;
                let v12 = (u1 >> 4u32) & 7u32;
                let v13 = ((u1 >> 7u32) & 1u32) | (((u1 >> 8u32) & 3u32) << 1u32);
                let v14 = (u1 >> 10u32) & 7u32;
                let v15 = (u1 >> 13u32) & 7u32;
                acc =
                    acc + (v8.cast::<f32>() * scale + bias) * load(input[xo + 8u32]).cast::<f32>();
                acc =
                    acc + (v9.cast::<f32>() * scale + bias) * load(input[xo + 9u32]).cast::<f32>();
                acc = acc
                    + (v10.cast::<f32>() * scale + bias) * load(input[xo + 10u32]).cast::<f32>();
                acc = acc
                    + (v11.cast::<f32>() * scale + bias) * load(input[xo + 11u32]).cast::<f32>();
                acc = acc
                    + (v12.cast::<f32>() * scale + bias) * load(input[xo + 12u32]).cast::<f32>();
                acc = acc
                    + (v13.cast::<f32>() * scale + bias) * load(input[xo + 13u32]).cast::<f32>();
                acc = acc
                    + (v14.cast::<f32>() * scale + bias) * load(input[xo + 14u32]).cast::<f32>();
                acc = acc
                    + (v15.cast::<f32>() * scale + bias) * load(input[xo + 15u32]).cast::<f32>();

                // Chunk 2 — bytes 2,3 of u1, byte 0 of u2
                let v16 = (u1 >> 16u32) & 7u32;
                let v17 = (u1 >> 19u32) & 7u32;
                let v18 = ((u1 >> 22u32) & 3u32) | (((u1 >> 24u32) & 1u32) << 2u32);
                let v19 = (u1 >> 25u32) & 7u32;
                let v20 = (u1 >> 28u32) & 7u32;
                let v21 = ((u1 >> 31u32) & 1u32) | ((u2 & 3u32) << 1u32);
                let v22 = (u2 >> 2u32) & 7u32;
                let v23 = (u2 >> 5u32) & 7u32;
                acc = acc
                    + (v16.cast::<f32>() * scale + bias) * load(input[xo + 16u32]).cast::<f32>();
                acc = acc
                    + (v17.cast::<f32>() * scale + bias) * load(input[xo + 17u32]).cast::<f32>();
                acc = acc
                    + (v18.cast::<f32>() * scale + bias) * load(input[xo + 18u32]).cast::<f32>();
                acc = acc
                    + (v19.cast::<f32>() * scale + bias) * load(input[xo + 19u32]).cast::<f32>();
                acc = acc
                    + (v20.cast::<f32>() * scale + bias) * load(input[xo + 20u32]).cast::<f32>();
                acc = acc
                    + (v21.cast::<f32>() * scale + bias) * load(input[xo + 21u32]).cast::<f32>();
                acc = acc
                    + (v22.cast::<f32>() * scale + bias) * load(input[xo + 22u32]).cast::<f32>();
                acc = acc
                    + (v23.cast::<f32>() * scale + bias) * load(input[xo + 23u32]).cast::<f32>();

                // Chunk 3 — bytes 1,2,3 of u2
                let v24 = (u2 >> 8u32) & 7u32;
                let v25 = (u2 >> 11u32) & 7u32;
                let v26 = ((u2 >> 14u32) & 3u32) | (((u2 >> 16u32) & 1u32) << 2u32);
                let v27 = (u2 >> 17u32) & 7u32;
                let v28 = (u2 >> 20u32) & 7u32;
                let v29 = ((u2 >> 23u32) & 1u32) | (((u2 >> 24u32) & 3u32) << 1u32);
                let v30 = (u2 >> 26u32) & 7u32;
                let v31 = (u2 >> 29u32) & 7u32;
                acc = acc
                    + (v24.cast::<f32>() * scale + bias) * load(input[xo + 24u32]).cast::<f32>();
                acc = acc
                    + (v25.cast::<f32>() * scale + bias) * load(input[xo + 25u32]).cast::<f32>();
                acc = acc
                    + (v26.cast::<f32>() * scale + bias) * load(input[xo + 26u32]).cast::<f32>();
                acc = acc
                    + (v27.cast::<f32>() * scale + bias) * load(input[xo + 27u32]).cast::<f32>();
                acc = acc
                    + (v28.cast::<f32>() * scale + bias) * load(input[xo + 28u32]).cast::<f32>();
                acc = acc
                    + (v29.cast::<f32>() * scale + bias) * load(input[xo + 29u32]).cast::<f32>();
                acc = acc
                    + (v30.cast::<f32>() * scale + bias) * load(input[xo + 30u32]).cast::<f32>();
                acc = acc
                    + (v31.cast::<f32>() * scale + bias) * load(input[xo + 31u32]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// MLX-format int3 dequantizing gather. Per output element: bit-extract
// the right 3-bit value within its 3-byte stream slot.
#[kernel]
fn dequant_gather_int3<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let u32_per_row = hidden * 3u32 / 32u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let row_u32_off = token_id * u32_per_row;

    let chunk_idx = d / 8u32; // which 8-value chunk
    let intra = d & 7u32; // which value within the chunk
    let byte_off = chunk_idx * 3u32; // 3 bytes per chunk

    let u_idx0 = byte_off / 4u32;
    let u0 = load(weight[row_u32_off + u_idx0]);
    let u1 = load(weight[row_u32_off + u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    // Extract value `intra` ∈ [0,8) from 3 bytes
    let v0 = b0 & 7u32;
    let v1 = (b0 >> 3u32) & 7u32;
    let v2 = ((b0 >> 6u32) & 3u32) | ((b1 & 1u32) << 2u32);
    let v3 = (b1 >> 1u32) & 7u32;
    let v4 = (b1 >> 4u32) & 7u32;
    let v5 = ((b1 >> 7u32) & 1u32) | ((b2 & 3u32) << 1u32);
    let v6 = (b2 >> 2u32) & 7u32;
    let v7 = (b2 >> 5u32) & 7u32;

    // Pick value by intra index using nested selects (4-deep)
    let s01 = select(intra == 0u32, v0, v1);
    let s23 = select(intra == 2u32, v2, v3);
    let s45 = select(intra == 4u32, v4, v5);
    let s67 = select(intra == 6u32, v6, v7);
    let s0123 = select(intra < 2u32, s01, s23);
    let s4567 = select(intra < 6u32, s45, s67);
    let q = select(intra < 4u32, s0123, s4567);

    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}

// MLX-format int5 dequantizing GEMV. 5-bit values: 8 values in 5 bytes
// (40 bits). uint32 cycle: 4 chunks span 5 uint32 (20 bytes = 32 vals).
//
//   chunk 0: u0 bytes 0-3 + u1 byte 0
//   chunk 1: u1 bytes 1-3 + u2 bytes 0-1
//   chunk 2: u2 bytes 2-3 + u3 bytes 0-2
//   chunk 3: u3 byte 3   + u4 bytes 0-3
#[kernel]
fn dequant_gemv_int5<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_groups = in_dim / group_size;
    let u32_per_row = in_dim * 5u32 / 32u32;
    let u32_per_group = group_size * 5u32 / 32u32;
    let row_u32_off = row * u32_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let g_iters = (n_groups + lsize - 1u32) / lsize;
    for g_iter in range(0u32, g_iters, 1u32) {
        let g = g_iter * lsize + tid;
        if g < n_groups {
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();
            let g_start = g * group_size;
            let g_u32_off = row_u32_off + g * u32_per_group;
            let cycles = group_size / 32u32;

            for c in range(0u32, cycles, 1u32) {
                let cy = g_u32_off + c * 5u32;
                let u0 = load(weight[cy]);
                let u1 = load(weight[cy + 1u32]);
                let u2 = load(weight[cy + 2u32]);
                let u3 = load(weight[cy + 3u32]);
                let u4 = load(weight[cy + 4u32]);
                let xo = g_start + c * 32u32;

                // Chunk 0 — u0 bytes 0-3 + u1 byte 0
                let v0 = u0 & 31u32;
                let v1 = ((u0 >> 5u32) & 7u32) | (((u0 >> 8u32) & 3u32) << 3u32);
                let v2 = (u0 >> 10u32) & 31u32;
                let v3 = ((u0 >> 15u32) & 1u32) | (((u0 >> 16u32) & 15u32) << 1u32);
                let v4 = ((u0 >> 20u32) & 15u32) | (((u0 >> 24u32) & 1u32) << 4u32);
                let v5 = (u0 >> 25u32) & 31u32;
                let v6 = ((u0 >> 30u32) & 3u32) | ((u1 & 7u32) << 2u32);
                let v7 = (u1 >> 3u32) & 31u32;
                acc =
                    acc + (v0.cast::<f32>() * scale + bias) * load(input[xo + 0u32]).cast::<f32>();
                acc =
                    acc + (v1.cast::<f32>() * scale + bias) * load(input[xo + 1u32]).cast::<f32>();
                acc =
                    acc + (v2.cast::<f32>() * scale + bias) * load(input[xo + 2u32]).cast::<f32>();
                acc =
                    acc + (v3.cast::<f32>() * scale + bias) * load(input[xo + 3u32]).cast::<f32>();
                acc =
                    acc + (v4.cast::<f32>() * scale + bias) * load(input[xo + 4u32]).cast::<f32>();
                acc =
                    acc + (v5.cast::<f32>() * scale + bias) * load(input[xo + 5u32]).cast::<f32>();
                acc =
                    acc + (v6.cast::<f32>() * scale + bias) * load(input[xo + 6u32]).cast::<f32>();
                acc =
                    acc + (v7.cast::<f32>() * scale + bias) * load(input[xo + 7u32]).cast::<f32>();

                // Chunk 1 — u1 bytes 1-3 + u2 bytes 0-1
                let w0 = (u1 >> 8u32) & 31u32;
                let w1 = ((u1 >> 13u32) & 7u32) | (((u1 >> 16u32) & 3u32) << 3u32);
                let w2 = (u1 >> 18u32) & 31u32;
                let w3 = ((u1 >> 23u32) & 1u32) | (((u1 >> 24u32) & 15u32) << 1u32);
                let w4 = ((u1 >> 28u32) & 15u32) | ((u2 & 1u32) << 4u32);
                let w5 = (u2 >> 1u32) & 31u32;
                let w6 = ((u2 >> 6u32) & 3u32) | (((u2 >> 8u32) & 7u32) << 2u32);
                let w7 = (u2 >> 11u32) & 31u32;
                acc =
                    acc + (w0.cast::<f32>() * scale + bias) * load(input[xo + 8u32]).cast::<f32>();
                acc =
                    acc + (w1.cast::<f32>() * scale + bias) * load(input[xo + 9u32]).cast::<f32>();
                acc =
                    acc + (w2.cast::<f32>() * scale + bias) * load(input[xo + 10u32]).cast::<f32>();
                acc =
                    acc + (w3.cast::<f32>() * scale + bias) * load(input[xo + 11u32]).cast::<f32>();
                acc =
                    acc + (w4.cast::<f32>() * scale + bias) * load(input[xo + 12u32]).cast::<f32>();
                acc =
                    acc + (w5.cast::<f32>() * scale + bias) * load(input[xo + 13u32]).cast::<f32>();
                acc =
                    acc + (w6.cast::<f32>() * scale + bias) * load(input[xo + 14u32]).cast::<f32>();
                acc =
                    acc + (w7.cast::<f32>() * scale + bias) * load(input[xo + 15u32]).cast::<f32>();

                // Chunk 2 — u2 bytes 2-3 + u3 bytes 0-2
                let x0 = (u2 >> 16u32) & 31u32;
                let x1 = ((u2 >> 21u32) & 7u32) | (((u2 >> 24u32) & 3u32) << 3u32);
                let x2 = (u2 >> 26u32) & 31u32;
                let x3 = ((u2 >> 31u32) & 1u32) | ((u3 & 15u32) << 1u32);
                let x4 = ((u3 >> 4u32) & 15u32) | (((u3 >> 8u32) & 1u32) << 4u32);
                let x5 = (u3 >> 9u32) & 31u32;
                let x6 = ((u3 >> 14u32) & 3u32) | (((u3 >> 16u32) & 7u32) << 2u32);
                let x7 = (u3 >> 19u32) & 31u32;
                acc =
                    acc + (x0.cast::<f32>() * scale + bias) * load(input[xo + 16u32]).cast::<f32>();
                acc =
                    acc + (x1.cast::<f32>() * scale + bias) * load(input[xo + 17u32]).cast::<f32>();
                acc =
                    acc + (x2.cast::<f32>() * scale + bias) * load(input[xo + 18u32]).cast::<f32>();
                acc =
                    acc + (x3.cast::<f32>() * scale + bias) * load(input[xo + 19u32]).cast::<f32>();
                acc =
                    acc + (x4.cast::<f32>() * scale + bias) * load(input[xo + 20u32]).cast::<f32>();
                acc =
                    acc + (x5.cast::<f32>() * scale + bias) * load(input[xo + 21u32]).cast::<f32>();
                acc =
                    acc + (x6.cast::<f32>() * scale + bias) * load(input[xo + 22u32]).cast::<f32>();
                acc =
                    acc + (x7.cast::<f32>() * scale + bias) * load(input[xo + 23u32]).cast::<f32>();

                // Chunk 3 — u3 byte 3 + u4 bytes 0-3
                let y0 = (u3 >> 24u32) & 31u32;
                let y1 = ((u3 >> 29u32) & 7u32) | ((u4 & 3u32) << 3u32);
                let y2 = (u4 >> 2u32) & 31u32;
                let y3 = ((u4 >> 7u32) & 1u32) | (((u4 >> 8u32) & 15u32) << 1u32);
                let y4 = ((u4 >> 12u32) & 15u32) | (((u4 >> 16u32) & 1u32) << 4u32);
                let y5 = (u4 >> 17u32) & 31u32;
                let y6 = ((u4 >> 22u32) & 3u32) | (((u4 >> 24u32) & 7u32) << 2u32);
                let y7 = (u4 >> 27u32) & 31u32;
                acc =
                    acc + (y0.cast::<f32>() * scale + bias) * load(input[xo + 24u32]).cast::<f32>();
                acc =
                    acc + (y1.cast::<f32>() * scale + bias) * load(input[xo + 25u32]).cast::<f32>();
                acc =
                    acc + (y2.cast::<f32>() * scale + bias) * load(input[xo + 26u32]).cast::<f32>();
                acc =
                    acc + (y3.cast::<f32>() * scale + bias) * load(input[xo + 27u32]).cast::<f32>();
                acc =
                    acc + (y4.cast::<f32>() * scale + bias) * load(input[xo + 28u32]).cast::<f32>();
                acc =
                    acc + (y5.cast::<f32>() * scale + bias) * load(input[xo + 29u32]).cast::<f32>();
                acc =
                    acc + (y6.cast::<f32>() * scale + bias) * load(input[xo + 30u32]).cast::<f32>();
                acc =
                    acc + (y7.cast::<f32>() * scale + bias) * load(input[xo + 31u32]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// MLX-format int5 dequantizing gather. Per output element: extract 5
// bytes spanning up to 2 uint32, then bit-extract value `intra`.
#[kernel]
fn dequant_gather_int5<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let u32_per_row = hidden * 5u32 / 32u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let row_u32_off = token_id * u32_per_row;

    let chunk_idx = d / 8u32;
    let intra = d & 7u32;
    let byte_off = chunk_idx * 5u32; // 5 bytes per chunk

    // 5 consecutive bytes can span at most 2 uint32. Read both.
    let u_idx0 = byte_off / 4u32;
    let u0 = load(weight[row_u32_off + u_idx0]);
    let u1 = load(weight[row_u32_off + u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let s3 = (byte_off + 3u32) & 3u32;
    let s4 = (byte_off + 4u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let in0_3 = (byte_off + 3u32) / 4u32 == u_idx0;
    let in0_4 = (byte_off + 4u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;
    let b3 = (select(in0_3, u0, u1) >> (s3 * 8u32)) & 255u32;
    let b4 = (select(in0_4, u0, u1) >> (s4 * 8u32)) & 255u32;

    let v0 = b0 & 31u32;
    let v1 = ((b0 >> 5u32) & 7u32) | ((b1 & 3u32) << 3u32);
    let v2 = (b1 >> 2u32) & 31u32;
    let v3 = ((b1 >> 7u32) & 1u32) | ((b2 & 15u32) << 1u32);
    let v4 = ((b2 >> 4u32) & 15u32) | ((b3 & 1u32) << 4u32);
    let v5 = (b3 >> 1u32) & 31u32;
    let v6 = ((b3 >> 6u32) & 3u32) | ((b4 & 7u32) << 2u32);
    let v7 = (b4 >> 3u32) & 31u32;

    let s01 = select(intra == 0u32, v0, v1);
    let s23 = select(intra == 2u32, v2, v3);
    let s45 = select(intra == 4u32, v4, v5);
    let s67 = select(intra == 6u32, v6, v7);
    let s0123 = select(intra < 2u32, s01, s23);
    let s4567 = select(intra < 6u32, s45, s67);
    let q = select(intra < 4u32, s0123, s4567);

    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}

// MLX-format int6 dequantizing gather. Per output element (token, d):
//   pack_idx_in_row = d / 4    (which 4-value pack)
//   intra_pack      = d & 3    (which value within the pack)
//
// Each pack is 3 bytes; 4 packs span 3 uint32. Compute the byte offset
// in the per-row byte stream, then read it from the right uint32 with
// the right shift. Bytes b0,b1,b2 of pack are at byte offsets
// pack_idx*3, pack_idx*3+1, pack_idx*3+2.
#[kernel]
fn dequant_gather_int6<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let u32_per_row = hidden * 3u32 / 16u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let row_u32_off = token_id * u32_per_row;

    let pack_idx = d / 4u32;
    let intra = d & 3u32;
    // Byte offsets in the per-row byte stream for the 3 bytes of this pack.
    let byte_off = pack_idx * 3u32;

    // Read up to 2 uint32 (a pack can straddle one boundary)
    let u_idx0 = byte_off / 4u32;
    let u0 = load(weight[row_u32_off + u_idx0]);
    let u1 = load(weight[row_u32_off + u_idx0 + 1u32]);

    // Read the 3 bytes (b0_byte_off..b0_byte_off+2 in stream).
    // Helper logic: byte at stream offset s is at uint32 index s/4, byte (s & 3) within.
    // Use select() to merge between u0/u1.
    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    // Extract value `intra` from the 3-byte pack
    let v0 = b0 & 63u32;
    let v1 = ((b0 >> 6u32) & 3u32) | ((b1 & 15u32) << 2u32);
    let v2 = ((b1 >> 4u32) & 15u32) | ((b2 & 3u32) << 4u32);
    let v3 = (b2 >> 2u32) & 63u32;

    let vsel0 = select(intra == 0u32, v0, v1);
    let vsel1 = select(intra == 2u32, v2, v3);
    let q = select(intra < 2u32, vsel0, vsel1);

    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}

// MLX-format int6 dequantizing GEMV. 6-bit values: 4 values fit in 3
// bytes (24 bits); a row of `in_dim` values is `in_dim * 3 / 4` bytes
// = `in_dim * 3 / 16` uint32s. Packs straddle uint32 boundaries with a
// 4-pack / 3-uint32 cycle:
//
//   pack 0: bytes 0,1,2 from u0
//   pack 1: byte 3 from u0, bytes 0,1 from u1
//   pack 2: bytes 2,3 from u1, byte 0 from u2
//   pack 3: bytes 1,2,3 from u2
//
// Inside each 3-byte pack:
//   val[0] = byte0 & 0x3F
//   val[1] = ((byte0 >> 6) & 0x3) | ((byte1 & 0xF) << 2)
//   val[2] = ((byte1 >> 4) & 0xF) | ((byte2 & 0x3) << 4)
//   val[3] = byte2 >> 2
//
// group_size must be a multiple of 16 (typical 32 / 64 / 128).
#[kernel]
fn dequant_gemv_int6<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_groups = in_dim / group_size;
    let u32_per_row = in_dim * 3u32 / 16u32;
    let u32_per_group = group_size * 3u32 / 16u32;
    let row_u32_off = row * u32_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let g_iters = (n_groups + lsize - 1u32) / lsize;
    for g_iter in range(0u32, g_iters, 1u32) {
        let g = g_iter * lsize + tid;
        if g < n_groups {
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();
            let g_start = g * group_size;
            let g_u32_off = row_u32_off + g * u32_per_group;
            let chunks = group_size / 16u32;

            for c in range(0u32, chunks, 1u32) {
                let chunk_off = g_u32_off + c * 3u32;
                let u0 = load(weight[chunk_off]);
                let u1 = load(weight[chunk_off + 1u32]);
                let u2 = load(weight[chunk_off + 2u32]);
                let xo = g_start + c * 16u32;

                // Pack 0 — bytes 0,1,2 of u0
                let p0v0 = u0 & 63u32;
                let p0v1 = ((u0 >> 6u32) & 3u32) | (((u0 >> 8u32) & 15u32) << 2u32);
                let p0v2 = ((u0 >> 12u32) & 15u32) | (((u0 >> 16u32) & 3u32) << 4u32);
                let p0v3 = (u0 >> 18u32) & 63u32;
                acc = acc
                    + (p0v0.cast::<f32>() * scale + bias) * load(input[xo + 0u32]).cast::<f32>();
                acc = acc
                    + (p0v1.cast::<f32>() * scale + bias) * load(input[xo + 1u32]).cast::<f32>();
                acc = acc
                    + (p0v2.cast::<f32>() * scale + bias) * load(input[xo + 2u32]).cast::<f32>();
                acc = acc
                    + (p0v3.cast::<f32>() * scale + bias) * load(input[xo + 3u32]).cast::<f32>();

                // Pack 1 — byte 3 of u0, bytes 0,1 of u1
                let p1v0 = (u0 >> 24u32) & 63u32;
                let p1v1 = ((u0 >> 30u32) & 3u32) | ((u1 & 15u32) << 2u32);
                let p1v2 = ((u1 >> 4u32) & 15u32) | (((u1 >> 8u32) & 3u32) << 4u32);
                let p1v3 = (u1 >> 10u32) & 63u32;
                acc = acc
                    + (p1v0.cast::<f32>() * scale + bias) * load(input[xo + 4u32]).cast::<f32>();
                acc = acc
                    + (p1v1.cast::<f32>() * scale + bias) * load(input[xo + 5u32]).cast::<f32>();
                acc = acc
                    + (p1v2.cast::<f32>() * scale + bias) * load(input[xo + 6u32]).cast::<f32>();
                acc = acc
                    + (p1v3.cast::<f32>() * scale + bias) * load(input[xo + 7u32]).cast::<f32>();

                // Pack 2 — bytes 2,3 of u1, byte 0 of u2
                let p2v0 = (u1 >> 16u32) & 63u32;
                let p2v1 = ((u1 >> 22u32) & 3u32) | (((u1 >> 24u32) & 15u32) << 2u32);
                let p2v2 = ((u1 >> 28u32) & 15u32) | ((u2 & 3u32) << 4u32);
                let p2v3 = (u2 >> 2u32) & 63u32;
                acc = acc
                    + (p2v0.cast::<f32>() * scale + bias) * load(input[xo + 8u32]).cast::<f32>();
                acc = acc
                    + (p2v1.cast::<f32>() * scale + bias) * load(input[xo + 9u32]).cast::<f32>();
                acc = acc
                    + (p2v2.cast::<f32>() * scale + bias) * load(input[xo + 10u32]).cast::<f32>();
                acc = acc
                    + (p2v3.cast::<f32>() * scale + bias) * load(input[xo + 11u32]).cast::<f32>();

                // Pack 3 — bytes 1,2,3 of u2
                let p3v0 = (u2 >> 8u32) & 63u32;
                let p3v1 = ((u2 >> 14u32) & 3u32) | (((u2 >> 16u32) & 15u32) << 2u32);
                let p3v2 = ((u2 >> 20u32) & 15u32) | (((u2 >> 24u32) & 3u32) << 4u32);
                let p3v3 = (u2 >> 26u32) & 63u32;
                acc = acc
                    + (p3v0.cast::<f32>() * scale + bias) * load(input[xo + 12u32]).cast::<f32>();
                acc = acc
                    + (p3v1.cast::<f32>() * scale + bias) * load(input[xo + 13u32]).cast::<f32>();
                acc = acc
                    + (p3v2.cast::<f32>() * scale + bias) * load(input[xo + 14u32]).cast::<f32>();
                acc = acc
                    + (p3v3.cast::<f32>() * scale + bias) * load(input[xo + 15u32]).cast::<f32>();
            }
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// MLX-format int8 dequantizing GEMV — sub-group cooperative version.
// One threadgroup per output row; threads stride across packs (in_dim/4
// packs per row), giving max parallelism within a row.
#[kernel]
fn dequant_gemv_int8<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    input: Tensor<T>,
    output: Tensor<T>,
    #[constexpr] in_dim: u32,
    #[constexpr] group_size: u32,
) {
    let row = program_id::<0>();
    let n_packs_per_row = in_dim / 4u32;
    let n_groups = in_dim / group_size;
    let packs_per_group = group_size / 4u32;
    let row_pack_off = row * n_packs_per_row;
    let row_group_off = row * n_groups;

    let mut acc = 0.0f32;

    let p_iters = (n_packs_per_row + lsize - 1u32) / lsize;
    for p_iter in range(0u32, p_iters, 1u32) {
        let pack_idx = p_iter * lsize + tid;
        if pack_idx < n_packs_per_row {
            let g = pack_idx / packs_per_group;
            let scale = load(scales[row_group_off + g]).cast::<f32>();
            let bias = load(biases[row_group_off + g]).cast::<f32>();

            let packed = load(weight[row_pack_off + pack_idx]);
            let p_off = pack_idx * 4u32;

            let q0 = (packed >> 0u32) & 255u32;
            let q1 = (packed >> 8u32) & 255u32;
            let q2 = (packed >> 16u32) & 255u32;
            let q3 = (packed >> 24u32) & 255u32;

            acc = acc + (q0.cast::<f32>() * scale + bias) * load(input[p_off + 0u32]).cast::<f32>();
            acc = acc + (q1.cast::<f32>() * scale + bias) * load(input[p_off + 1u32]).cast::<f32>();
            acc = acc + (q2.cast::<f32>() * scale + bias) * load(input[p_off + 2u32]).cast::<f32>();
            acc = acc + (q3.cast::<f32>() * scale + bias) * load(input[p_off + 3u32]).cast::<f32>();
        }
    }

    let total = reduce_sum(acc);
    if tid == 0u32 {
        store(output[row], total.cast::<T>());
    }
}

// MLX-format int8 dequantizing gather. One thread per output element.
#[kernel]
fn dequant_gather_int8<T>(
    weight: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    indices: Tensor<u32>,
    out: Tensor<T>,
    #[constexpr] hidden: u32,
    #[constexpr] group_size: u32,
) {
    let idx = program_id::<0>();
    let token = idx / hidden;
    let d = idx - token * hidden;
    let token_id = load(indices[token]);
    let packs_per_row = hidden / 4u32;
    let groups_per_row = hidden / group_size;
    let g = d / group_size;
    let pack_idx = token_id * packs_per_row + d / 4u32;
    let byte = d & 3u32;
    let packed = load(weight[pack_idx]);
    let q = (packed >> (byte * 8u32)) & 255u32;
    let scale = load(scales[token_id * groups_per_row + g]).cast::<f32>();
    let bias = load(biases[token_id * groups_per_row + g]).cast::<f32>();
    let w_real = q.cast::<f32>() * scale + bias;
    store(out[idx], w_real.cast::<T>());
}

// ─── Registry ─────────────────────────────────────────────────────────────

/// Build the list of kernels to emit. Each entry is a fully-named IR ready
/// for codegen.
fn register_kernels() -> Vec<Kernel> {
    let mut kernels: Vec<Kernel> = Vec::new();
    let dtypes = [DType::F32, DType::F16, DType::BF16];

    // ─── elementwise (Elementwise mode = default) ────────────────────
    // Naming convention: `vector_add` (legacy MLX name), `mt_*` for
    // metaltile-canonical ops, `ffai_*` for FFAI-specific row ops.
    for &dt in &dtypes {
        let mut k = vector_add::kernel_ir_for(dt);
        k.name = format!("vector_add_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = mt_mul::kernel_ir_for(dt);
        k.name = format!("mt_mul_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = mt_silu::kernel_ir_for(dt);
        k.name = format!("mt_silu_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = mt_softplus::kernel_ir_for(dt);
        k.name = format!("mt_softplus_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = ffai_gather::kernel_ir_for(dt);
        k.name = format!("ffai_gather_{}", dtype_suffix(dt));
        k.mode = KernelMode::Grid3D;
        kernels.push(k);

        let mut k = mt_gemv::kernel_ir_for(dt);
        k.name = format!("mt_gemv_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── rms_norm (Reduction mode) ───────────────────────────────────
    // Reduction mode is required so the codegen emits `lsize`/`tid`/`tgid`
    // aliases used inside the kernel body.
    for &dt in &dtypes {
        let mut k = mt_rms_norm::kernel_ir_for(dt);
        k.name = format!("mt_rms_norm_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── rope_llama (Grid3D — uses program_id<0> for head and
    //     program_id<1> for half-pair index)
    for &dt in &dtypes {
        let mut k = ffai_rope_llama::kernel_ir_for(dt);
        k.name = format!("ffai_rope_llama_{}", dtype_suffix(dt));
        k.mode = KernelMode::Grid3D;
        kernels.push(k);
    }

    // ─── ffai_sdpa_decode (Reduction, head_dim=128 default) ──────────
    // Canonical kernel body uses tgid_x / simd_id / simd_lane / n_simd
    // — needs Reduction mode for the codegen to emit those aliases.
    // Has sink_end / window_start constexprs for sliding-window-with-
    // sink attention (Qwen3-class SWA, Gemma sliding layers).
    for &dt in &dtypes {
        let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
        k.name = format!("ffai_sdpa_decode_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── kv cache update (Elementwise) ───────────────────────────────
    for &dt in &dtypes {
        let mut k = kv_cache_update::kernel_ir_for(dt);
        k.name = format!("kv_cache_update_{}", dtype_suffix(dt));
        kernels.push(k);
    }

    // ─── SSM step (Elementwise) — Mamba 2 single-token decode ────────
    // One thread per (head, channel). State `h` lives in fp32 across
    // every dtype variant.
    for &dt in &dtypes {
        let mut k = ssm_step::kernel_ir_for(dt);
        k.name = format!("ssm_step_{}", dtype_suffix(dt));
        kernels.push(k);
    }

    // ─── conv1d causal step (Elementwise) — Mamba 2 input-proj conv ─
    // One thread per channel. State holds the last K-1 inputs; shifts
    // in place after compute.
    for &dt in &dtypes {
        let mut k = conv1d_causal_step::kernel_ir_for(dt);
        k.name = format!("conv1d_causal_step_{}", dtype_suffix(dt));
        kernels.push(k);
    }

    // ─── affine-quantized KV cache (Elementwise) ─────────────────────
    // quantize_kv_int{4,8} : one thread per group
    // bulk_dequant_kv_int{4,8} : one thread per output element
    for &dt in &dtypes {
        let mut k = quantize_kv_int8::kernel_ir_for(dt);
        k.name = format!("quantize_kv_int8_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = bulk_dequant_kv_int8::kernel_ir_for(dt);
        k.name = format!("bulk_dequant_kv_int8_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = quantize_kv_int4::kernel_ir_for(dt);
        k.name = format!("quantize_kv_int4_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = bulk_dequant_kv_int4::kernel_ir_for(dt);
        k.name = format!("bulk_dequant_kv_int4_{}", dtype_suffix(dt));
        kernels.push(k);
    }

    // ─── ffai_argmax (Reduction) ─────────────────────────────────────
    for &dt in &dtypes {
        let mut k = ffai_argmax::kernel_ir_for(dt);
        k.name = format!("ffai_argmax_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_gelu / mt_relu / mt_sigmoid (Elementwise) ────────────────
    // Sourced from metaltile-std/mlx/unary.rs — canonical activations
    // FFAI consumers expect alongside silu / softplus.
    for &dt in &dtypes {
        let mut k = mt_gelu::kernel_ir_for(dt);
        k.name = format!("mt_gelu_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = mt_relu::kernel_ir_for(dt);
        k.name = format!("mt_relu_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = mt_sigmoid::kernel_ir_for(dt);
        k.name = format!("mt_sigmoid_{}", dtype_suffix(dt));
        kernels.push(k);
    }

    // ─── mt_cast_to_f32 (Elementwise) — T → fp32 in-place cast ───────
    // Used by the fused GDN prep path to bring bf16 / f16 model
    // activations up to fp32 so the kernel can run against the fp32
    // recurrence state without a host round-trip. One thread per
    // element. f32-source variant is a no-op identity (kept for
    // dispatch-table uniformity; callers should avoid calling it).
    for &dt in &dtypes {
        let mut k = mt_cast_to_f32::kernel_ir_for(dt);
        k.name = format!("mt_cast_to_f32_{}", dtype_suffix(dt));
        kernels.push(k);
    }

    // ─── mt_gated_delta_step (Reduction) — Mamba-style GDN decode step ─
    // Used by Qwen3.5 / Qwen3.6 hybrid GDN layers, one dispatch per token.
    for &dt in &dtypes {
        let mut k = mt_gated_delta_step::kernel_ir_for(dt);
        k.name = format!("mt_gated_delta_step_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_gated_delta_chunk (Reduction) — chunked-T GDN prefill step ─
    // Batched-T counterpart of mt_gated_delta_step. Used in
    // forwardMany / batched prefill paths.
    for &dt in &dtypes {
        let mut k = mt_gated_delta_chunk::kernel_ir_for(dt);
        k.name = format!("mt_gated_delta_chunk_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_sdpa_prefill_mma (SimdGroup2D) — batched-T causal SDPA ────
    // MMA-tiled prefill kernel used by FFAI's batched prefill across
    // Qwen / Mistral / Phi / Gemma. KernelMode::SimdGroup2D matches the
    // BenchSpec dispatch geometry (per crates/metaltile-std/src/ffai/
    // sdpa_decode_batched_prefill.rs).
    for &dt in &dtypes {
        let mut k = mt_sdpa_prefill_mma::kernel_ir_for(dt);
        k.name = format!("mt_sdpa_prefill_mma_{}", dtype_suffix(dt));
        k.mode = KernelMode::SimdGroup2D;
        kernels.push(k);
    }

    // ─── ffai_sdpa_decode_d{64,256,512} — head_dim variants ──────────
    // The kernel's head_dim is a runtime constexpr (setBytes at encode),
    // so all three variants share the same MSL source as the d=128
    // baseline above. The distinct Swift wrapper names let callers
    // self-document the head-dim shape they're dispatching against.
    for &dt in &dtypes {
        let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
        k.name = format!("ffai_sdpa_decode_d64_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
        k.name = format!("ffai_sdpa_decode_d256_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
        k.name = format!("ffai_sdpa_decode_d512_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── softmax_categorical_sample (Reduction) ──────────────────────
    for &dt in &dtypes {
        let mut k = softmax_categorical_sample::kernel_ir_for(dt);
        k.name = format!("softmax_categorical_sample_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── dequant gemv (Reduction) + gather (Elementwise) ─────────────
    // dequant_gemv_*: cooperative-thread (one threadgroup per output row).
    // dequant_gather_*: one thread per output element.
    for &dt in &dtypes {
        let mut k = dequant_gemv_int4::kernel_ir_for(dt);
        k.name = format!("dequant_gemv_int4_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = dequant_gather_int4::kernel_ir_for(dt);
        k.name = format!("dequant_gather_int4_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = dequant_gemv_int8::kernel_ir_for(dt);
        k.name = format!("dequant_gemv_int8_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = dequant_gather_int8::kernel_ir_for(dt);
        k.name = format!("dequant_gather_int8_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = dequant_gemv_int6::kernel_ir_for(dt);
        k.name = format!("dequant_gemv_int6_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = dequant_gather_int6::kernel_ir_for(dt);
        k.name = format!("dequant_gather_int6_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = dequant_gemv_int3::kernel_ir_for(dt);
        k.name = format!("dequant_gemv_int3_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = dequant_gather_int3::kernel_ir_for(dt);
        k.name = format!("dequant_gather_int3_{}", dtype_suffix(dt));
        kernels.push(k);

        let mut k = dequant_gemv_int5::kernel_ir_for(dt);
        k.name = format!("dequant_gemv_int5_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);

        let mut k = dequant_gather_int5::kernel_ir_for(dt);
        k.name = format!("dequant_gather_int5_{}", dtype_suffix(dt));
        kernels.push(k);
    }

    // ─── mt_qmm_mma (Reduction) — simdgroup-matrix int4 qmm ────────────────
    // High-perf quantized matmul B>=4, K>=32, N>=32, group_size=64. Used by
    // Linear/Dense layers at prefill / batched decode. bf16 added for the
    // FFAI Qwen3.6-A3B prefill (Ops.dequantGemmDynamicM dispatches mt_qmm_mma
    // through a host-side T-padding driver — the model is bf16, so the
    // bf16 specialization is the load-bearing one for production).
    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = mt_qmm_mma::kernel_ir_for(dt);
        k.name = format!("mt_qmm_mma_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_moe_gather_qmm_mma_int4_bm16 (Reduction) — MoE grouped int4 qmm
    // BM=16 variant matches MLX's affine_gather_qmm_rhs_nt geometry. Used by
    // SwitchGLU / MoE FFN at prefill.
    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = mt_moe_gather_qmm_mma_int4_bm16::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm16_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_moe_gather_qmm_mma_int4_bm16_mpp (Reduction) — MPP MoE BGEMM ───
    // MPP-backed counterpart of mt_moe_gather_qmm_mma_int4_bm16. Mirrors the
    // BM=16 row-partitioning + int4 dequant pipeline but routes the inner
    // 16×32×16 tile matmul through `mpp::tensor_ops::matmul2d` — the same
    // Apple-private API MLX uses to hit ~3000 GF on Qwen3.6-A3B `down_proj`.
    // TG = 32 lanes = 1 SG (matmul2d is `execution_simdgroup`).
    // Requires macOS 26+ / Metal 4. See
    // `crates/metaltile-std/src/ffai/moe_mpp.rs`.
    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = moe_mpp::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm16_mpp_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_moe_gather_qmm_mma_int4_bm64_mpp (Reduction) — BM=BN=64 MPP MoE ──
    // 4-SG WM=WN=2 scale-up of the BM=16 MPP MoE kernel. Mirrors MLX's
    // `affine_gather_qmm_rhs_nax` tile geometry (BM=BN=BK=64, WM=WN=2). Each
    // SG owns a 32×32 sub-tile; descriptor is (32, 32, 32) per SG. Required
    // to close the 1.55× gap vs MLX on Qwen3.6-35B-A3B prefill at T=32K.
    // See `crates/metaltile-std/src/ffai/moe_mpp_bm64.rs`.
    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = moe_mpp_bm64::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm64_mpp_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_moe_gather_qmm_mma_int4_bm8_mpp (Reduction) — BM=8 MPP MoE ───────
    // Half-height MoE BGEMM for topK=8 decode (`m_total = 8`). BM=16 wasted
    // 50% of the tile rows; BM=8 fills exactly. Uses the destination-only-
    // cooperative MPP path so the descriptor `(M=8, N=32, K=16)` clears the
    // simdgroup-scope `M%8 / N%16 / one-of-M,N ≡ 16` constraint that the
    // all-cooperative `matmul2d` would reject. See
    // `crates/metaltile-std/src/ffai/moe_mpp_bm8.rs` + the Constraints note
    // on PR #137 for the cooperative-tensor descriptor refinement.
    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = moe_mpp_bm8::kernel_ir_for(dt);
        k.name = format!("mt_moe_gather_qmm_mma_int4_bm8_mpp_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_gated_delta_prep_step (Reduction) — fused GDN prep + recurrence ──
    // One dispatch absorbs conv-split + per-head q/k RMSNorm + g + beta +
    // the existing GDN recurrence step, collapsing 3 host commit+wait pairs
    // per layer down to 1 in Qwen35GDNMixer.forward. Same dispatch geometry
    // as `mt_gated_delta_step` (`[Dv, B·Hv, 1]` × `[32, 1, 1]`). See
    // `crates/metaltile-std/src/ffai/gated_delta_prep.rs`.
    for &dt in &[DType::F32, DType::F16, DType::BF16] {
        let mut k = mt_gated_delta_prep_step::kernel_ir_for(dt);
        k.name = format!("mt_gated_delta_prep_step_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_qmm_mma_mpp (Reduction) — MPP `matmul2d` production int4 qmm ──
    // BM=BN=BK=32, TG=128 (4 SG × 32 lanes WM=WN=2), per-SG 16×16 MMA via
    // `mpp::tensor_ops::matmul2d<desc, execution_simdgroup>`. Same int4
    // dequant-into-TG-mem pattern as mt_qmm_mma; the matmul step swaps the
    // manual 8×8 `simdgroup_matmul` ladder for one cooperative `matmul2d`
    // per SG per K-block. This is the MPP/NAX path MLX uses for
    // `affine_qmm_t_nax` / `gather_qmm_rhs_nax`. Requires macOS 26+ /
    // Metal 4. See `crates/metaltile-std/src/mlx/quantized_mpp.rs`.
    for &dt in &[DType::F32, DType::F16] {
        let mut k = quantized_mpp::kernel_ir_for(dt);
        k.name = format!("mt_qmm_mma_mpp_{}", dtype_suffix(dt));
        k.mode = KernelMode::Reduction;
        kernels.push(k);
    }

    // ─── mt_mpp_matmul_smoke (Elementwise) — MPP `matmul2d` smoke kernel
    // Single-simdgroup 16×32 fp16 → 16×16 fp32 matmul. Requires macOS 26+ /
    // Metal 4 (header `<MetalPerformancePrimitives/...>` only available on
    // those toolchains). Pre-Metal-4 builds compile a stub fallback so the
    // metallib still links; correctness test fails as the intended signal.
    //
    // This is the foothold for the future MPP-backed `mt_qmm_mma` variant —
    // taps the NAX hardware path MLX uses for `down_proj` (~3000 GF on
    // Qwen3.6-A3B). See `crates/metaltile-std/src/probe/mpp_matmul_smoke.rs`.
    {
        let k = mpp_matmul_smoke::kernel_ir();
        kernels.push(k);
    }

    kernels
}

// ─── Manifest schema (v1) ─────────────────────────────────────────────────

#[derive(Serialize)]
struct Manifest {
    /// Manifest schema version. Bump on breaking changes.
    version: u32,
    metaltile_emit_version: String,
    kernels: Vec<KernelManifest>,
}

#[derive(Serialize)]
struct KernelManifest {
    /// Public kernel name (also the MSL function name).
    name: String,
    /// Path to the MSL source file relative to the manifest.
    source: String,
    /// Thread-indexing mode — informs default grid/threadgroup sizing.
    kernel_mode: String,
    /// Buffer-bound parameters in slot order.
    params: Vec<ParamManifest>,
    /// Constexpr scalars bound as `setBytes` after `params`.
    constexprs: Vec<ConstExprManifest>,
}

#[derive(Serialize)]
struct ParamManifest {
    name: String,
    /// "Tensor", "Strided", or "Scalar".
    kind: String,
    /// "f32", "f16", "bf16", "u32", "i32", etc.
    dtype: String,
    is_output: bool,
}

#[derive(Serialize)]
struct ConstExprManifest {
    name: String,
    dtype: String,
}

// ─── Main flow ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let resources_dir = cli.out.join("Resources");
    let kernels_dir = resources_dir.join("kernels");
    let generated_dir = cli.out.join("Generated");

    fs::create_dir_all(&kernels_dir).context("create Resources/kernels")?;
    fs::create_dir_all(&generated_dir).context("create Generated")?;

    let kernels = register_kernels();
    println!("metaltile-emit: registered {} kernels", kernels.len());

    let mut manifest_entries: Vec<KernelManifest> = Vec::new();
    let mut metal_files: Vec<PathBuf> = Vec::new();
    let generator = MslGenerator::default();

    for kernel in &kernels {
        let msl = generator
            .generate(kernel)
            .map_err(|e| anyhow::anyhow!("generate MSL for {}: {:?}", kernel.name, e))?;

        let metal_path = kernels_dir.join(format!("{}.metal", kernel.name));
        fs::write(&metal_path, &msl).with_context(|| format!("write {}", metal_path.display()))?;
        println!("  wrote {}", metal_path.display());

        manifest_entries.push(kernel_to_manifest(kernel));
        metal_files.push(metal_path);
    }

    // Manifest
    let manifest = Manifest {
        version: 1,
        metaltile_emit_version: env!("CARGO_PKG_VERSION").to_string(),
        kernels: manifest_entries,
    };
    let manifest_path = resources_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    println!("  wrote {}", manifest_path.display());

    // Generated Swift wrappers
    let swift = generate_swift_wrappers(&manifest);
    let swift_path = generated_dir.join("MetalTileKernels.swift");
    fs::write(&swift_path, swift).with_context(|| format!("write {}", swift_path.display()))?;
    println!("  wrote {}", swift_path.display());

    // Compile metallib (unless explicitly skipped)
    if cli.no_compile {
        println!("--no-compile: skipping metallib build");
    } else {
        let metallib_path = resources_dir.join("kernels.metallib");
        compile_metallib(&metal_files, &metallib_path, &cli.sdk)?;
        println!("  wrote {}", metallib_path.display());
    }

    println!("metaltile-emit: done");
    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn dtype_suffix(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I64 => "i64",
        DType::U64 => "u64",
        DType::I4 => "i4",
        DType::Bool => "bool",
    }
}

fn param_kind_str(k: &ParamKind) -> &'static str {
    match k {
        ParamKind::Tensor => "Tensor",
        ParamKind::Strided => "Strided",
        ParamKind::Scalar => "Scalar",
    }
}

fn kernel_mode_str(m: KernelMode) -> &'static str {
    match m {
        KernelMode::Elementwise => "Elementwise",
        KernelMode::Reduction => "Reduction",
        KernelMode::Grid3D => "Grid3D",
        KernelMode::Tile2D => "Tile2D",
        KernelMode::SimdGroup2D => "SimdGroup2D",
    }
}

fn kernel_to_manifest(k: &Kernel) -> KernelManifest {
    KernelManifest {
        name: k.name.clone(),
        source: format!("kernels/{}.metal", k.name),
        kernel_mode: kernel_mode_str(k.mode).to_string(),
        params: k
            .params
            .iter()
            .map(|p: &Param| ParamManifest {
                name: p.name.clone(),
                kind: param_kind_str(&p.kind).to_string(),
                dtype: dtype_suffix(p.dtype).to_string(),
                is_output: p.is_output,
            })
            .collect(),
        constexprs: k
            .constexprs
            .iter()
            .map(|c| ConstExprManifest {
                name: c.name.name().to_string(),
                dtype: dtype_suffix(c.dtype).to_string(),
            })
            .collect(),
    }
}

// ─── Swift wrapper generation ─────────────────────────────────────────────
//
// One static function per kernel. Caller supplies MTLBuffers (+ offsets),
// constexpr scalars, grid + threadgroup sizes, and a command buffer. The
// wrapper looks up the PSO from `PSOCache.shared`, encodes the dispatch,
// and ends the encoder. PSOCache lives in MetalTileSwift hand-written code.

fn generate_swift_wrappers(manifest: &Manifest) -> String {
    let mut out = String::new();
    out.push_str(
        "// AUTOGENERATED by metaltile-emit. DO NOT EDIT.\n\
         //\n\
         // Each function dispatches a single Metal kernel from kernels.metallib.\n\
         // Looks up the pre-compiled PSO from PSOCache.shared, encodes the\n\
         // dispatch on the supplied command buffer, ends the encoder.\n\n\
         import Metal\n\n\
         public enum MetalTileKernels {\n",
    );

    for k in &manifest.kernels {
        emit_swift_wrapper(&mut out, k);
    }

    out.push_str("}\n");
    out
}

fn emit_swift_wrapper(out: &mut String, k: &KernelManifest) {
    use std::fmt::Write as _;
    let fn_name = swift_safe_name(&k.name);

    writeln!(out, "    /// Dispatches `{}` from kernels.metallib.", k.name).ok();
    writeln!(out, "    public static func {fn_name}(").ok();

    // Buffer params (Tensor / Strided / Scalar all bind as buffers in Phase 0)
    for p in &k.params {
        let label = swift_safe_name(&p.name);
        writeln!(out, "        {label}: MTLBuffer, {label}Offset: Int = 0,").ok();
    }
    // Constexpr scalars (bound via setBytes after the param buffers)
    for c in &k.constexprs {
        let label = swift_safe_name(&c.name);
        let swift_ty = swift_scalar_type(&c.dtype);
        writeln!(out, "        {label}: {swift_ty},").ok();
    }
    // Grid + threadgroup sizing
    writeln!(out, "        gridSize: MTLSize,").ok();
    writeln!(out, "        threadgroupSize: MTLSize,").ok();
    writeln!(out, "        on commandBuffer: MTLCommandBuffer").ok();
    writeln!(out, "    ) {{").ok();
    writeln!(out, "        let pso = PSOCache.shared.pipelineState(for: \"{}\")", k.name).ok();
    writeln!(
        out,
        "        guard let enc = commandBuffer.makeComputeCommandEncoder() else {{ return }}"
    )
    .ok();
    writeln!(out, "        enc.setComputePipelineState(pso)").ok();

    let mut slot = 0usize;
    for p in &k.params {
        let label = swift_safe_name(&p.name);
        writeln!(out, "        enc.setBuffer({label}, offset: {label}Offset, index: {slot})").ok();
        slot += 1;
    }
    for c in &k.constexprs {
        let label = swift_safe_name(&c.name);
        let len = swift_scalar_size(&c.dtype);
        writeln!(out, "        var {label}_v = {label}").ok();
        writeln!(out, "        enc.setBytes(&{label}_v, length: {len}, index: {slot})").ok();
        slot += 1;
    }
    // dispatchThreads (in threads, not threadgroups) so out-of-bound
    // threads aren't created and the kernel doesn't need bounds checks.
    // Requires Metal 2.0 non-uniform threadgroup support (M-series ✓).
    writeln!(out, "        enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)")
        .ok();
    writeln!(out, "        enc.endEncoding()").ok();
    writeln!(out, "    }}\n").ok();
}

fn swift_safe_name(s: &str) -> String {
    // For Phase 0 just snake-case → snake-case. We may want camelCase later
    // for idiomatic Swift; revisit when we have more kernels.
    s.replace('-', "_")
}

fn swift_scalar_type(dtype: &str) -> &'static str {
    match dtype {
        "f32" => "Float",
        "f16" => "Float16",
        "bf16" => "Float", // no native Swift bfloat16; pass widened, kernel reads narrow
        "i32" => "Int32",
        "u32" => "UInt32",
        "i64" => "Int64",
        "u64" => "UInt64",
        "i8" => "Int8",
        "u8" => "UInt8",
        "bool" => "Bool",
        _ => "UInt32",
    }
}

fn swift_scalar_size(dtype: &str) -> usize {
    match dtype {
        "f32" | "i32" | "u32" => 4,
        "f16" | "bf16" | "i16" | "u16" => 2,
        "i8" | "u8" | "bool" => 1,
        "i64" | "u64" => 8,
        _ => 4,
    }
}

// ─── Metal toolchain invocation ───────────────────────────────────────────

fn compile_metallib(metal_files: &[PathBuf], output: &Path, sdk: &str) -> Result<()> {
    if metal_files.is_empty() {
        bail!("no .metal files to compile");
    }

    let air_dir = tempdir_in_target()?;
    let mut air_files: Vec<PathBuf> = Vec::new();

    println!("compiling {} .metal files...", metal_files.len());
    for metal in metal_files {
        let stem = metal
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("bad metal filename: {}", metal.display()))?;
        let air = air_dir.join(format!("{stem}.air"));

        let status = Command::new("xcrun")
            .args(["-sdk", sdk, "metal", "-c"])
            .arg(metal)
            .arg("-o")
            .arg(&air)
            .status()
            .with_context(|| format!("invoke xcrun metal for {}", metal.display()))?;
        if !status.success() {
            bail!("xcrun metal failed for {}", metal.display());
        }
        air_files.push(air);
    }

    println!("linking metallib {}", output.display());
    let status = Command::new("xcrun")
        .args(["-sdk", sdk, "metallib"])
        .args(&air_files)
        .arg("-o")
        .arg(output)
        .status()
        .context("invoke xcrun metallib")?;
    if !status.success() {
        bail!("xcrun metallib failed");
    }

    Ok(())
}

fn tempdir_in_target() -> Result<PathBuf> {
    // Use cargo's target/ so we don't pollute /tmp on every build.
    let dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"))
        .join("metaltile-emit-air");
    fs::create_dir_all(&dir).context("create air tempdir")?;
    Ok(dir)
}
