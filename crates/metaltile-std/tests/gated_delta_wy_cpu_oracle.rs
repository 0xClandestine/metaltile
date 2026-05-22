//! CPU-only oracle validation for the chunked-WY GDN algorithm.
//!
//! Two implementations:
//!   1. `sequential_gdn` — per-step recurrence, the source of truth
//!      (mirrors `gated_delta_sequential` in
//!      `/tmp/gdn_chunked_wy/gdn_wy_ref.py`, which in turn mirrors
//!      `_gated_delta_step_ops` from `mlx_lm/models/gated_delta.py`).
//!   2. `chunked_wy_gdn` — block-parallel WY form, also CPU-only.
//!      This is the algorithm the GPU kernel will implement.
//!
//! The test asserts the two agree to f32 within a tight tolerance across a
//! matrix of shapes. Validating the Rust port BEFORE touching the DSL keeps
//! the GPU-kernel work isolated to a known-good algorithm.
//!
//! Pure CPU — no `target_os` gate, runs anywhere in CI.

use std::ops::Range;

// ────────────────────────────────────────────────────────────────────
//  Tiny linear-algebra helpers (column-major-free, row-major flat Vecs)
// ────────────────────────────────────────────────────────────────────

/// Row-major matmul: `C[m,n] = A[m,k] * B[k,n]`. f64 internal.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f64;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Row-major `A^T`: input `[m,n]` → output `[n,m]`. f64.
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// Forward-substitution solve of `L X = rhs` where L is unit lower-triangular
/// `[c, c]` and rhs is `[c, w]`. Returns X with same shape as rhs. Diagonal
/// assumed = 1 (no division). f64.
fn solve_unit_lower_tri(l: &[f64], rhs: &[f64], c: usize, w: usize) -> Vec<f64> {
    let mut x = rhs.to_vec();
    for i in 0..c {
        if i > 0 {
            // X[i, :] -= L[i, 0..i] @ X[0..i, :]
            for j in 0..i {
                let l_ij = l[i * c + j];
                if l_ij == 0.0 {
                    continue;
                }
                for ww in 0..w {
                    x[i * w + ww] -= l_ij * x[j * w + ww];
                }
            }
        }
    }
    x
}

// ────────────────────────────────────────────────────────────────────
//  GDN reference: sequential (per-step) recurrence
// ────────────────────────────────────────────────────────────────────

/// Sequential GDN over a multi-token chunk/prefill. Mirrors
/// `gated_delta_sequential` from the Python reference.
///
/// Layouts (single batch slot, dropped leading B dim):
///   - q, k:   `[t_total, hk, dk]`
///   - v:      `[t_total, hv, dv]`
///   - g, beta: `[t_total, hv]`
///   - state:  `[hv, dv, dk]`  (modified in place; in fp32)
///
/// Output `y` has shape `[t_total, hv, dv]`. GQA: `hk_idx = hv_idx / (hv/hk)`.
#[allow(clippy::too_many_arguments)]
pub fn sequential_gdn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    t_total: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let hv_per_hk = hv / hk;
    let mut y = vec![0.0_f32; t_total * hv * dv];

    for t in 0..t_total {
        for h_v in 0..hv {
            let h_k = h_v / hv_per_hk;
            let gt = g[t * hv + h_v];
            let bt = beta[t * hv + h_v];
            for d_v in 0..dv {
                let v_val = v[(t * hv + h_v) * dv + d_v];
                let s_base = (h_v * dv + d_v) * dk;

                // Phase 1: decay + kv_mem
                let mut kv_mem = 0.0_f32;
                let mut decayed = vec![0.0_f32; dk];
                for s_idx in 0..dk {
                    let s = state[s_base + s_idx] * gt;
                    decayed[s_idx] = s;
                    kv_mem += s * k[(t * hk + h_k) * dk + s_idx];
                }
                let delta = (v_val - kv_mem) * bt;

                // Phase 2: rank-1 update + output projection
                let mut out = 0.0_f32;
                for s_idx in 0..dk {
                    let s_new = decayed[s_idx] + k[(t * hk + h_k) * dk + s_idx] * delta;
                    state[s_base + s_idx] = s_new;
                    out += s_new * q[(t * hk + h_k) * dk + s_idx];
                }
                y[(t * hv + h_v) * dv + d_v] = out;
            }
        }
    }
    y
}

// ────────────────────────────────────────────────────────────────────
//  GDN chunked-WY (block-parallel) — the algorithm the GPU kernel runs
// ────────────────────────────────────────────────────────────────────

/// Process one chunk `[t0..t1)` of length `c = t1 - t0` for a single
/// `(b=0, h_v)` slot. Reads K/Q/V/g/β for the chunk, applies the WY
/// identity, writes y_chunk and updates `state` in place.
///
/// Per the validated Python reference at
/// `/tmp/gdn_chunked_wy/gdn_wy_ref.py::_process_chunk`.
#[allow(clippy::too_many_arguments)]
fn process_chunk_one_head(
    q_full: &[f32],
    k_full: &[f32],
    v_full: &[f32],
    g_full: &[f32],
    beta_full: &[f32],
    state: &mut [f32], // [dv, dk] for this (b, hv) slot
    y_out: &mut [f32], // [t_total, hv, dv] full buffer; we write only this chunk
    t_total: usize,
    chunk: Range<usize>,
    h_v: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) {
    let c = chunk.end - chunk.start;
    let hv_per_hk = hv / hk;
    let h_k = h_v / hv_per_hk;

    // Gather chunk-local inputs into f64 [c, *] buffers (numerical stability).
    let mut q_c = vec![0.0_f64; c * dk];
    let mut k_c = vec![0.0_f64; c * dk];
    let mut v_c = vec![0.0_f64; c * dv];
    let mut g_c = vec![0.0_f64; c];
    let mut beta_c = vec![0.0_f64; c];
    for i in 0..c {
        let t = chunk.start + i;
        for d in 0..dk {
            q_c[i * dk + d] = q_full[(t * hk + h_k) * dk + d] as f64;
            k_c[i * dk + d] = k_full[(t * hk + h_k) * dk + d] as f64;
        }
        for d in 0..dv {
            v_c[i * dv + d] = v_full[(t * hv + h_v) * dv + d] as f64;
        }
        g_c[i] = g_full[t * hv + h_v] as f64;
        beta_c[i] = beta_full[t * hv + h_v] as f64;
    }

    // Upcast state to f64 for the chunk math, downcast at the end.
    let state_f64: Vec<f64> = state.iter().map(|&x| x as f64).collect();

    // 1. Prefix gates G_t and Γ[t,j] = G_t / G_j (masked j ≤ t).
    let mut big_g = vec![0.0_f64; c];
    big_g[0] = g_c[0];
    for t in 1..c {
        big_g[t] = big_g[t - 1] * g_c[t];
    }
    let mut gamma_masked = vec![0.0_f64; c * c];
    for t in 0..c {
        for j in 0..=t {
            gamma_masked[t * c + j] = big_g[t] / big_g[j];
        }
    }

    // 2. KKT[i,j] = k_i · k_j.
    let k_c_t = transpose(&k_c, c, dk);
    let kkt = matmul(&k_c, &k_c_t, c, dk, c);

    // 3. Passthrough p-rows: solve `(I+L) p = K`, L[j,i] = β_i KKT[j,i] for i<j.
    let mut one_plus_l = vec![0.0_f64; c * c];
    for t in 0..c {
        one_plus_l[t * c + t] = 1.0;
        for j in 0..t {
            one_plus_l[t * c + j] = beta_c[j] * kkt[t * c + j];
        }
    }
    let p_rows = solve_unit_lower_tri(&one_plus_l, &k_c, c, dk);

    // 4. Chunk-local A: solve `(I+A) u^v = β⊙V`, A[t,j] = β_t Γ[t,j] KKT[t,j] for t>j.
    let mut one_plus_a = vec![0.0_f64; c * c];
    for t in 0..c {
        one_plus_a[t * c + t] = 1.0;
        for j in 0..t {
            one_plus_a[t * c + j] = beta_c[t] * gamma_masked[t * c + j] * kkt[t * c + j];
        }
    }
    let mut b_v = vec![0.0_f64; c * dv];
    for t in 0..c {
        for d in 0..dv {
            b_v[t * dv + d] = beta_c[t] * v_c[t * dv + d];
        }
    }
    let u_v = solve_unit_lower_tri(&one_plus_a, &b_v, c, dv);

    // 5. y_local = (Γ_masked ⊙ QKT) @ u^v
    let q_kt = matmul(&q_c, &k_c_t, c, dk, c);
    let mut weights = vec![0.0_f64; c * c];
    for t in 0..c {
        for j in 0..=t {
            weights[t * c + j] = gamma_masked[t * c + j] * q_kt[t * c + j];
        }
    }
    let y_local = matmul(&weights, &u_v, c, c, dv);

    // 6. y_pass = G_t · (S_0 q_t − correction[t]).
    let s0_t = transpose(&state_f64, dv, dk);
    let s0_q = matmul(&q_c, &s0_t, c, dk, dv);

    let mut weight = vec![0.0_f64; c * c];
    for t in 0..c {
        for i in 0..=t {
            weight[t * c + i] = beta_c[i] * q_kt[t * c + i];
        }
    }
    let p_t = transpose(&p_rows, c, dk);
    let s0_p = matmul(&state_f64, &p_t, dv, dk, c);
    let s0_p_t = transpose(&s0_p, dv, c);
    let correction = matmul(&weight, &s0_p_t, c, c, dv);

    // 7. y_chunk = y_pass + y_local; write to y_out (downcast to f32).
    for t in 0..c {
        let t_abs = chunk.start + t;
        for d in 0..dv {
            let val = big_g[t] * (s0_q[t * dv + d] - correction[t * dv + d]) + y_local[t * dv + d];
            y_out[(t_abs * hv + h_v) * dv + d] = val as f32;
        }
    }
    let _ = t_total;

    // 8. End-of-chunk state: S_end = G_C·(S_0 − S_0·(β⊙p)^T·K) + Σ_j (G_C/G_j)·u^v_j⊗k_j.
    let big_g_c = big_g[c - 1];
    let mut bp = vec![0.0_f64; c * dk];
    for i in 0..c {
        for d in 0..dk {
            bp[i * dk + d] = beta_c[i] * p_rows[i * dk + d];
        }
    }
    let bp_t = transpose(&bp, c, dk);
    let s0_bp_t = matmul(&state_f64, &bp_t, dv, dk, c);
    let s0_bp_t_k = matmul(&s0_bp_t, &k_c, dv, c, dk);
    let mut s_through = vec![0.0_f64; dv * dk];
    for v_idx in 0..dv {
        for d in 0..dk {
            s_through[v_idx * dk + d] =
                big_g_c * (state_f64[v_idx * dk + d] - s0_bp_t_k[v_idx * dk + d]);
        }
    }

    let mut rw_uv = vec![0.0_f64; c * dv];
    for j in 0..c {
        let scale = big_g_c / big_g[j];
        for v_idx in 0..dv {
            rw_uv[j * dv + v_idx] = scale * u_v[j * dv + v_idx];
        }
    }
    let rw_uv_t = transpose(&rw_uv, c, dv);
    let u_end = matmul(&rw_uv_t, &k_c, dv, c, dk);

    for i in 0..(dv * dk) {
        state[i] = (s_through[i] + u_end[i]) as f32;
    }
}

/// Chunked-WY GDN over the full T sequence. Walks (B × Hv) slots,
/// processes each slot's T tokens in chunks of `chunk_size`. Pure CPU.
#[allow(clippy::too_many_arguments)]
pub fn chunked_wy_gdn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    t_total: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    chunk_size: usize,
) -> Vec<f32> {
    let mut y = vec![0.0_f32; t_total * hv * dv];

    for h_v in 0..hv {
        let slot_offset = h_v * dv * dk;
        let mut t = 0;
        while t < t_total {
            let chunk_end = (t + chunk_size).min(t_total);
            process_chunk_one_head(
                q,
                k,
                v,
                g,
                beta,
                &mut state[slot_offset..slot_offset + dv * dk],
                &mut y,
                t_total,
                t..chunk_end,
                h_v,
                hk,
                hv,
                dk,
                dv,
            );
            t = chunk_end;
        }
    }
    y
}

// ────────────────────────────────────────────────────────────────────
//  Equivalence tests — sequential vs chunked-WY (CPU only)
// ────────────────────────────────────────────────────────────────────

type SyntheticInputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn synthetic_inputs(t_total: usize, hk: usize, hv: usize, dk: usize, dv: usize) -> SyntheticInputs {
    // Scale q, k so that ‖k‖² ≈ 1 regardless of Dk — otherwise the
    // β·(k·k^T) Householder reflector grows with Dk and the recurrence
    // becomes unstable (state explodes), and any f32 comparison becomes
    // noise relative to f64. With sin² mean ≈ 0.5, k_scale = 1/√(Dk/2)
    // gives ‖k‖² ≈ 1.
    let kscale = (2.0 / dk as f32).sqrt();
    let q: Vec<f32> =
        (0..t_total * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * kscale).collect();
    let k: Vec<f32> =
        (0..t_total * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * kscale).collect();
    let v: Vec<f32> = (0..t_total * hv * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    // g ∈ (0, 1) — realistic gating distribution.
    let g: Vec<f32> = (0..t_total * hv).map(|i| 0.8 + 0.15 * ((i as f32) * 0.013).sin()).collect();
    // β ∈ (0, 1) — realistic learning rate (sigmoid output).
    let beta: Vec<f32> =
        (0..t_total * hv).map(|i| 0.4 + 0.3 * ((i as f32) * 0.017).cos()).collect();
    let state: Vec<f32> = (0..hv * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();
    (q, k, v, g, beta, state)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

#[test]
fn wy_debug_t2_minimal() {
    // Bug hunt: simplest non-trivial multi-token chunk.
    // T=2, single head (Hk=Hv=1), Dk=Dv=4, g=1.0 (no decay), β=0.5.
    let (t, hk, hv, dk, dv) = (2, 1, 1, 4, 4);
    let n_total = t * hv;
    let q: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.1).sin()).collect();
    let k: Vec<f32> = (0..t * hk * dk).map(|i| ((i as f32) * 0.13).cos()).collect();
    let v: Vec<f32> = (0..t * hv * dv).map(|i| ((i as f32) * 0.07).sin() * 0.3).collect();
    let g = vec![1.0_f32; n_total];
    let beta = vec![0.5_f32; n_total];
    let state = vec![0.0_f32; hv * dv * dk]; // start from zero state

    let mut s1 = state.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut s1, t, hk, hv, dk, dv);
    let mut s2 = state.clone();
    let y_wy = chunked_wy_gdn(&q, &k, &v, &g, &beta, &mut s2, t, hk, hv, dk, dv, 32);

    eprintln!("y_seq = {:?}", y_seq);
    eprintln!("y_wy  = {:?}", y_wy);
    eprintln!("y_diff = {:?}", y_seq.iter().zip(&y_wy).map(|(a, b)| a - b).collect::<Vec<_>>());
    eprintln!("s_seq = {:?}", s1);
    eprintln!("s_wy  = {:?}", s2);

    let dy = max_abs_diff(&y_seq, &y_wy);
    let ds = max_abs_diff(&s1, &s2);
    assert!(dy < 1e-5, "y diff = {dy:.2e}");
    assert!(ds < 1e-5, "state diff = {ds:.2e}");
}

/// Pure-f64 sequential reference, mirrors `gated_delta_sequential` in
/// `/tmp/gdn_chunked_wy/gdn_wy_ref.py`. Used to factor out f32 precision
/// noise when validating the chunked-WY algorithm port.
#[allow(clippy::too_many_arguments)]
fn sequential_gdn_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    g: &[f64],
    beta: &[f64],
    state: &mut [f64],
    t_total: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Vec<f64> {
    let hv_per_hk = hv / hk;
    let mut y = vec![0.0_f64; t_total * hv * dv];
    for t in 0..t_total {
        for h_v in 0..hv {
            let h_k = h_v / hv_per_hk;
            let gt = g[t * hv + h_v];
            let bt = beta[t * hv + h_v];
            for d_v in 0..dv {
                let v_val = v[(t * hv + h_v) * dv + d_v];
                let s_base = (h_v * dv + d_v) * dk;
                let mut kv_mem = 0.0_f64;
                let mut decayed = vec![0.0_f64; dk];
                for s_idx in 0..dk {
                    let s = state[s_base + s_idx] * gt;
                    decayed[s_idx] = s;
                    kv_mem += s * k[(t * hk + h_k) * dk + s_idx];
                }
                let delta = (v_val - kv_mem) * bt;
                let mut out = 0.0_f64;
                for s_idx in 0..dk {
                    let s_new = decayed[s_idx] + k[(t * hk + h_k) * dk + s_idx] * delta;
                    state[s_base + s_idx] = s_new;
                    out += s_new * q[(t * hk + h_k) * dk + s_idx];
                }
                y[(t * hv + h_v) * dv + d_v] = out;
            }
        }
    }
    y
}

#[test]
fn wy_debug_scaling_t() {
    // Walk T from 2 -> 32 with one chunk to find where blowup begins.
    // Use f64 EVERYWHERE so f32 precision noise is factored out.
    for t in [2usize, 4, 8, 16, 32, 64, 128] {
        let (hk, hv, dk, dv) = (1, 1, 8, 8);
        let n_total = t * hv;
        let q: Vec<f64> = (0..t * hk * dk).map(|i| ((i as f64) * 0.1).sin()).collect();
        let k: Vec<f64> = (0..t * hk * dk).map(|i| ((i as f64) * 0.13).cos()).collect();
        let v: Vec<f64> = (0..t * hv * dv).map(|i| ((i as f64) * 0.07).sin() * 0.3).collect();
        let g: Vec<f64> = (0..n_total).map(|i| 0.95 + 0.04 * ((i as f64) * 0.01).sin()).collect();
        let beta: Vec<f64> = (0..n_total).map(|i| 0.5 + 0.2 * ((i as f64) * 0.02).cos()).collect();
        let state = vec![0.0_f64; hv * dv * dk];

        let mut s1 = state.clone();
        let y_seq = sequential_gdn_f64(&q, &k, &v, &g, &beta, &mut s1, t, hk, hv, dk, dv);

        // Reuse the chunked-WY single-head process; need f32 → f64 trampoline.
        // Convert inputs to f32 and pass through the existing f32 entry,
        // upcasting internally. Diff then is f64 seq vs f64-internal wy.
        let q32: Vec<f32> = q.iter().map(|x| *x as f32).collect();
        let k32: Vec<f32> = k.iter().map(|x| *x as f32).collect();
        let v32: Vec<f32> = v.iter().map(|x| *x as f32).collect();
        let g32: Vec<f32> = g.iter().map(|x| *x as f32).collect();
        let b32: Vec<f32> = beta.iter().map(|x| *x as f32).collect();
        let mut s2 = vec![0.0_f32; hv * dv * dk];
        let y_wy_f32 = chunked_wy_gdn(&q32, &k32, &v32, &g32, &b32, &mut s2, t, hk, hv, dk, dv, t);

        let y_seq_f32: Vec<f32> = y_seq.iter().map(|x| *x as f32).collect();
        let s1_f32: Vec<f32> = s1.iter().map(|x| *x as f32).collect();
        let dy = max_abs_diff(&y_seq_f32, &y_wy_f32);
        let ds = max_abs_diff(&s1_f32, &s2);
        eprintln!("T={t:3}: y_diff={dy:.2e}  state_diff={ds:.2e}");
    }
}

#[test]
fn wy_matches_sequential_t1() {
    // Edge case: chunk_size > T, T=1 → one chunk of 1 token.
    let (t, hk, hv, dk, dv, c) = (1, 2, 4, 32, 16, 32);
    let (q, k, v, g, beta, state) = synthetic_inputs(t, hk, hv, dk, dv);

    let mut state_seq = state.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut state_seq, t, hk, hv, dk, dv);

    let mut state_wy = state.clone();
    let y_wy = chunked_wy_gdn(&q, &k, &v, &g, &beta, &mut state_wy, t, hk, hv, dk, dv, c);

    let dy = max_abs_diff(&y_seq, &y_wy);
    let ds = max_abs_diff(&state_seq, &state_wy);
    assert!(dy < 1e-5, "y diff = {dy:.2e}");
    assert!(ds < 1e-5, "state diff = {ds:.2e}");
}

#[test]
fn wy_matches_sequential_chunk_aligned() {
    // T exactly divisible by chunk_size.
    let (t, hk, hv, dk, dv, c) = (64, 2, 4, 32, 16, 32);
    let (q, k, v, g, beta, state) = synthetic_inputs(t, hk, hv, dk, dv);

    let mut state_seq = state.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut state_seq, t, hk, hv, dk, dv);

    let mut state_wy = state.clone();
    let y_wy = chunked_wy_gdn(&q, &k, &v, &g, &beta, &mut state_wy, t, hk, hv, dk, dv, c);

    let dy = max_abs_diff(&y_seq, &y_wy);
    let ds = max_abs_diff(&state_seq, &state_wy);
    assert!(dy < 5e-5, "y diff = {dy:.2e}");
    assert!(ds < 5e-5, "state diff = {ds:.2e}");
}

#[test]
fn wy_matches_sequential_uneven_chunk() {
    // T not divisible by chunk_size — last chunk shorter.
    let (t, hk, hv, dk, dv, c) = (65, 2, 4, 32, 16, 32);
    let (q, k, v, g, beta, state) = synthetic_inputs(t, hk, hv, dk, dv);

    let mut state_seq = state.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut state_seq, t, hk, hv, dk, dv);

    let mut state_wy = state.clone();
    let y_wy = chunked_wy_gdn(&q, &k, &v, &g, &beta, &mut state_wy, t, hk, hv, dk, dv, c);

    let dy = max_abs_diff(&y_seq, &y_wy);
    let ds = max_abs_diff(&state_seq, &state_wy);
    assert!(dy < 5e-5, "y diff = {dy:.2e}");
    assert!(ds < 5e-5, "state diff = {ds:.2e}");
}

#[test]
fn wy_matches_sequential_long_ctx() {
    // T=256 with c=64 → 4 chunks. Realistic prefill-scale validation.
    let (t, hk, hv, dk, dv, c) = (256, 2, 4, 64, 32, 64);
    let (q, k, v, g, beta, state) = synthetic_inputs(t, hk, hv, dk, dv);

    let mut state_seq = state.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut state_seq, t, hk, hv, dk, dv);

    let mut state_wy = state.clone();
    let y_wy = chunked_wy_gdn(&q, &k, &v, &g, &beta, &mut state_wy, t, hk, hv, dk, dv, c);

    let dy = max_abs_diff(&y_seq, &y_wy);
    let ds = max_abs_diff(&state_seq, &state_wy);
    assert!(dy < 5e-4, "long-ctx y diff = {dy:.2e}");
    assert!(ds < 5e-4, "long-ctx state diff = {ds:.2e}");
}

#[test]
fn wy_matches_sequential_qwen36_dims() {
    // Qwen3.6-35B-A3B linear-attention dims: Hk=2, Hv=4, Dk=Dv=128.
    // Smaller T=128 to keep the f32 test fast on CI.
    let (t, hk, hv, dk, dv, c) = (128, 2, 4, 128, 128, 64);
    let (q, k, v, g, beta, state) = synthetic_inputs(t, hk, hv, dk, dv);

    let mut state_seq = state.clone();
    let y_seq = sequential_gdn(&q, &k, &v, &g, &beta, &mut state_seq, t, hk, hv, dk, dv);

    let mut state_wy = state.clone();
    let y_wy = chunked_wy_gdn(&q, &k, &v, &g, &beta, &mut state_wy, t, hk, hv, dk, dv, c);

    let dy = max_abs_diff(&y_seq, &y_wy);
    let ds = max_abs_diff(&state_seq, &state_wy);
    assert!(dy < 1e-3, "qwen36 y diff = {dy:.2e}");
    assert!(ds < 1e-3, "qwen36 state diff = {ds:.2e}");
}
