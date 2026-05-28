//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GatedDeltaNet innovation-tape capture + replay — port of
//! `gated_delta_replay.metal` (spec 020 phase 2). Companion to
//! `gated_delta.rs`; the speculative-decode rollback path for
//! GDN-bearing models (Qwen 3.5 / 3.6).
//!
//! Two kernels:
//!   - `gated_delta_step_record` — the standard GatedDelta forward step
//!     that *also* writes each step's `delta_t` to a `delta_log` tape.
//!   - `state_replay` — re-folds the accepted prefix `[0, accepted)` of
//!     an innovation tape onto a pre-record state snapshot:
//!     `state ← select(do_step, state·g_t + k_t·delta_t, state)`,
//!     branchless via `select` (good SIMD occupancy when the timestep
//!     mask is non-uniform within a simdgroup).
//!
//! Tape layout: `delta_log` [B, T, Hv, Dv], `k_log` [B, T, Hv, Dk]
//! (GQA-expanded by the cache), `g_log` [B, T, Hv].
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, Dv, batch*Hv]`, `tg = [32, 1, 1]`.
//! - `Dk` a multiple of 32.
//!
//! Codegen-only; correctness pinned by
//! `tests/gated_delta_replay_gpu_correctness.rs`.

use metaltile::kernel;

mod tests_support {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

    use metaltile::test_kernel;
    use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn u32_le(v: u32) -> Vec<u8> { v.to_le_bytes().to_vec() }

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    fn xorshift(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s % 20_000) as f32 / 20_000.0 - 0.5
    }

    fn src(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| xorshift(&mut s) * scale).collect()
    }

    const DK: usize = 64;
    const DV: usize = 32;
    const HK: usize = 2;
    const HV: usize = 2;

    /// Branchless tape re-fold reference (state_replay).
    fn naive_replay(
        delta_log: &[f32],
        k_log: &[f32],
        g_log: &[f32],
        state_in: &[f32],
        mask: &[u32],
        batch: usize,
        t_log: usize,
        accepted: usize,
        has_mask: bool,
    ) -> Vec<f32> {
        let mut state = state_in.to_vec();
        for n in 0..batch * HV {
            let b = n / HV;
            let hvh = n % HV;
            for t in 0..t_log {
                let do_step = t < accepted && (!has_mask || mask[b * t_log + t] != 0);
                if !do_step {
                    continue;
                }
                let dr = (b * t_log + t) * HV * DV + hvh * DV;
                let kr = (b * t_log + t) * HV * DK + hvh * DK;
                let g = g_log[(b * t_log + t) * HV + hvh];
                for dv in 0..DV {
                    let s0 = (n * DV + dv) * DK;
                    for dk in 0..DK {
                        state[s0 + dk] = state[s0 + dk] * g + k_log[kr + dk] * delta_log[dr + dv];
                    }
                }
            }
        }
        state
    }

    #[test_kernel(name = "ffai/gated_delta/replay", dtypes = [f32], tol = 2e-3)]
    fn test_state_replay(dt: DType) -> TestSetup {
        use super::state_replay_d64_32_2_2;
        let batch = 1usize;
        let t_log = 5usize;
        let accepted = 3usize;

        let delta_log = src(batch * t_log * HV * DV, 0x21, 0.5);
        let k_log = src(batch * t_log * HV * DK, 0x22, 0.4);
        let g_log: Vec<f32> = src(batch * t_log * HV, 0x23, 0.1).iter().map(|v| 0.9 + v).collect();
        let state_in = src(batch * HV * DV * DK, 0x24, 0.3);
        let mask: Vec<u32> = vec![1; batch * t_log];
        let expected = naive_replay(
            &delta_log, &k_log, &g_log, &state_in, &mask, batch, t_log, accepted, false,
        );

        let mut kernel_ir = state_replay_d64_32_2_2::kernel_ir_for(dt);
        kernel_ir.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(kernel_ir)
            .input(TestBuffer::from_vec("delta_log", pack(&delta_log, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("k_log", pack(&k_log, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("g_log", pack(&g_log, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("state_in", pack(&state_in, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("mask", u32_bytes(&mask), DType::U32))
            .input(TestBuffer::from_vec("t_log", u32_le(t_log as u32), DType::U32))
            .input(TestBuffer::from_vec("accepted", u32_le(accepted as u32), DType::U32))
            .input(TestBuffer::from_vec("has_mask", u32_le(0u32), DType::U32))
            .expect(TestBuffer::from_vec("state_out", pack(&expected, DType::F32), DType::F32))
            .grid_3d(1, DV as u32, (batch * HV) as u32, [32, 1, 1])
    }
}

// ── Forward GatedDelta step with per-step delta-tape capture ────────────────
#[rustfmt::skip]
macro_rules! gated_delta_record {
    ($name:ident, $dk:literal, $dv:literal, $hk:literal, $hv:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            g: Tensor<T>,
            beta: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut y: Tensor<T>,
            mut state_out: Tensor<T>,
            mut delta_log: Tensor<T>,
            #[constexpr] t_val: u32,
            #[constexpr] has_mask: u32,
        ) {
            let lane = program_id::<0>();
            let dv_idx = program_id::<1>();
            let n = program_id::<2>();
            let b_idx = n / $hv;
            let hv_idx = n - b_idx * $hv;
            let hk_idx = hv_idx / ($hv / $hk);
            let i_state_base = (n * $dv + dv_idx) * $dk;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[i_state_base + $n_per_t * lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            for t in range(0u32, t_val, 1u32) {
                let m = select(has_mask == 0u32, 1u32, load(mask[b_idx * t_val + t]));
                if m > 0u32 {
                    let qk_base = (b_idx * t_val + t) * $hk * $dk + hk_idx * $dk;
                    let v_base = (b_idx * t_val + t) * $hv * $dv + hv_idx * $dv;
                    let gb_idx = (b_idx * t_val + t) * $hv + hv_idx;
                    let g_val = load(g[gb_idx]).cast::<f32>();
                    let beta_val = load(beta[gb_idx]).cast::<f32>();

                    let mut kv_mem = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let s_idx = $n_per_t * lane + i;
                        let st = stack_load("state", i) * g_val;
                        stack_store("state", i, st);
                        kv_mem = kv_mem + st * load(k[qk_base + s_idx]).cast::<f32>();
                    }
                    let kv = simd_sum(kv_mem);
                    let delta = (load(v[v_base + dv_idx]).cast::<f32>() - kv) * beta_val;

                    // Tape write: surface delta_t for the replay kernel.
                    if lane == 0u32 {
                        store(delta_log[v_base + dv_idx], delta.cast::<T>());
                    }

                    let mut out_acc = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let s_idx = $n_per_t * lane + i;
                        let st =
                            stack_load("state", i) + load(k[qk_base + s_idx]).cast::<f32>() * delta;
                        stack_store("state", i, st);
                        out_acc = out_acc + st * load(q[qk_base + s_idx]).cast::<f32>();
                    }
                    let out_red = simd_sum(out_acc);
                    if lane == 0u32 {
                        store(y[v_base + dv_idx], out_red.cast::<T>());
                    }
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[i_state_base + $n_per_t * lane + i], st.cast::<T>());
            }
        }
    };
}

// ── Tape replay: re-fold the accepted prefix onto a snapshot ────────────────
#[rustfmt::skip]
macro_rules! state_replay {
    ($name:ident, $dk:literal, $dv:literal, $hv:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            delta_log: Tensor<T>,
            k_log: Tensor<T>,
            g_log: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut state_out: Tensor<T>,
            #[constexpr] t_log: u32,
            #[constexpr] accepted: u32,
            #[constexpr] has_mask: u32,
        ) {
            let lane = program_id::<0>();
            let dv_idx = program_id::<1>();
            let n = program_id::<2>();
            let b_idx = n / $hv;
            let hv_idx = n - b_idx * $hv;
            let i_state_base = (n * $dv + dv_idx) * $dk;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[i_state_base + $n_per_t * lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            for t in range(0u32, t_log, 1u32) {
                let mask_v = select(has_mask == 0u32, 1u32, load(mask[b_idx * t_log + t]));
                // do_step = (t < accepted) && mask_passes — branchless.
                let do_step = select(t < accepted, mask_v, 0u32);

                let delta_row = (b_idx * t_log + t) * $hv * $dv + hv_idx * $dv;
                let k_row = (b_idx * t_log + t) * $hv * $dk + hv_idx * $dk;
                let g_idx = (b_idx * t_log + t) * $hv + hv_idx;
                let g_val = load(g_log[g_idx]).cast::<f32>();
                let d_val = load(delta_log[delta_row + dv_idx]).cast::<f32>();

                for i in range(0u32, $n_per_t, 1u32) {
                    let s_idx = $n_per_t * lane + i;
                    let old = stack_load("state", i);
                    let new_val = old * g_val + load(k_log[k_row + s_idx]).cast::<f32>() * d_val;
                    stack_store("state", i, select(do_step > 0u32, new_val, old));
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[i_state_base + $n_per_t * lane + i], st.cast::<T>());
            }
        }
    };
}

// Qwen 3.5/3.6 A3B: Dk=192, Dv=128, Hk=4, Hv=4.
gated_delta_record!(
    gated_delta_step_record_d192_128_4_4,
    192u32,
    128u32,
    4u32,
    4u32,
    6u32,
    "record_d192_128_4_4"
);
state_replay!(state_replay_d192_128_4_4, 192u32, 128u32, 4u32, 6u32, "replay_d192_128_4_4");
// Small unit-test cell: Dk=64, Dv=32, Hk=2, Hv=2.
gated_delta_record!(
    gated_delta_step_record_d64_32_2_2,
    64u32,
    32u32,
    2u32,
    2u32,
    2u32,
    "record_d64_32_2_2"
);
state_replay!(state_replay_d64_32_2_2, 64u32, 32u32, 2u32, 2u32, "replay_d64_32_2_2");

#[cfg(target_os = "macos")]
pub mod tests_support_ctx {
    //! GPU correctness tests for gated_delta_step_record and state_replay.
    #![allow(clippy::too_many_arguments)]

    use std::collections::BTreeMap;

    use metaltile_core::{dtype::DType, ir::KernelMode};
    use metaltile_runtime::Context;

    use super::{gated_delta_step_record_d64_32_2_2, state_replay_d64_32_2_2};

    const DK: usize = 64;
    const DV: usize = 32;
    const HK: usize = 2;
    const HV: usize = 2;

    fn pack_f32(vals: &[f32]) -> Vec<u8> { bytemuck::cast_slice::<f32, u8>(vals).to_vec() }
    fn unpack_f32(bytes: &[u8]) -> Vec<f32> { bytemuck::cast_slice::<u8, f32>(bytes).to_vec() }
    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }

    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
    }

    fn src(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 20_000) as f32 / 20_000.0 * scale - scale * 0.5
            })
            .collect()
    }

    fn naive_record(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        batch: usize,
        t_val: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; batch * t_val * HV * DV];
        let mut delta_log = vec![0.0_f32; batch * t_val * HV * DV];
        let mut state = state_in.to_vec();
        for n in 0..batch * HV {
            let b = n / HV;
            let hvh = n % HV;
            let hkh = hvh / (HV / HK);
            for t in 0..t_val {
                let qk = (b * t_val + t) * HK * DK + hkh * DK;
                let vb = (b * t_val + t) * HV * DV + hvh * DV;
                let gb = (b * t_val + t) * HV + hvh;
                for dv in 0..DV {
                    let s0 = (n * DV + dv) * DK;
                    let mut kv = 0.0_f32;
                    for dk in 0..DK {
                        state[s0 + dk] *= g[gb];
                        kv += state[s0 + dk] * k[qk + dk];
                    }
                    let delta = (v[vb + dv] - kv) * beta[gb];
                    delta_log[vb + dv] = delta;
                    let mut out = 0.0_f32;
                    for dk in 0..DK {
                        state[s0 + dk] += k[qk + dk] * delta;
                        out += state[s0 + dk] * q[qk + dk];
                    }
                    y[vb + dv] = out;
                }
            }
        }
        (y, state, delta_log)
    }

    fn naive_replay(
        delta_log: &[f32],
        k_log: &[f32],
        g_log: &[f32],
        state_in: &[f32],
        mask: &[u32],
        batch: usize,
        t_log: usize,
        accepted: usize,
        has_mask: bool,
    ) -> Vec<f32> {
        let mut state = state_in.to_vec();
        for n in 0..batch * HV {
            let b = n / HV;
            let hvh = n % HV;
            for t in 0..t_log {
                let do_step = t < accepted && (!has_mask || mask[b * t_log + t] != 0);
                if !do_step {
                    continue;
                }
                let dr = (b * t_log + t) * HV * DV + hvh * DV;
                let kr = (b * t_log + t) * HV * DK + hvh * DK;
                let g = g_log[(b * t_log + t) * HV + hvh];
                for dv in 0..DV {
                    let s0 = (n * DV + dv) * DK;
                    for dk in 0..DK {
                        state[s0 + dk] = state[s0 + dk] * g + k_log[kr + dk] * delta_log[dr + dv];
                    }
                }
            }
        }
        state
    }

    fn dispatch(
        kernel_ir: fn(metaltile_core::dtype::DType) -> metaltile_core::ir::Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        batch: usize,
        want: &[&str],
    ) -> Vec<Vec<f32>> {
        let ctx = Context::new().expect("Context::new on macOS");
        let mut kernel = kernel_ir(DType::F32);
        kernel.mode = KernelMode::Grid3D;
        let result = ctx
            .dispatch_with_grid(&kernel, buffers, &BTreeMap::new(), [1, DV, batch * HV], [32, 1, 1])
            .expect("gated_delta_replay dispatch");
        want.iter().map(|w| unpack_f32(result.outputs.get(*w).expect(w))).collect()
    }

    #[test]
    fn gated_delta_record_captures_tape_f32() {
        let _g = gpu_lock();
        let (batch, t_val) = (1usize, 3usize);
        let q = src(batch * t_val * HK * DK, 0x1, 0.4);
        let k = src(batch * t_val * HK * DK, 0x2, 0.4);
        let v = src(batch * t_val * HV * DV, 0x3, 1.0);
        let g: Vec<f32> = src(batch * t_val * HV, 0x4, 0.1).iter().map(|x| 0.9 + x).collect();
        let beta: Vec<f32> = src(batch * t_val * HV, 0x5, 0.1).iter().map(|x| 0.5 + x).collect();
        let state_in = src(batch * HV * DV * DK, 0x6, 0.2);
        let (exp_y, exp_s, exp_d) = naive_record(&q, &k, &v, &g, &beta, &state_in, batch, t_val);
        let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        b.insert("q".into(), pack_f32(&q));
        b.insert("k".into(), pack_f32(&k));
        b.insert("v".into(), pack_f32(&v));
        b.insert("g".into(), pack_f32(&g));
        b.insert("beta".into(), pack_f32(&beta));
        b.insert("state_in".into(), pack_f32(&state_in));
        b.insert("mask".into(), u32_bytes(&vec![1; batch * t_val]));
        b.insert("y".into(), pack_f32(&vec![0.0; exp_y.len()]));
        b.insert("state_out".into(), pack_f32(&vec![0.0; state_in.len()]));
        b.insert("delta_log".into(), pack_f32(&vec![0.0; exp_d.len()]));
        b.insert("t_val".into(), (t_val as u32).to_le_bytes().to_vec());
        b.insert("has_mask".into(), 0u32.to_le_bytes().to_vec());
        let got = dispatch(gated_delta_step_record_d64_32_2_2::kernel_ir_for, &b, batch, &[
            "y",
            "state_out",
            "delta_log",
        ]);
        assert!(got[2].iter().any(|&x| x != 0.0), "tape is all zeros");
        assert!(max_abs_diff(&got[0], &exp_y) < 2e-3, "y mismatch");
        assert!(max_abs_diff(&got[1], &exp_s) < 2e-3, "state mismatch");
        assert!(max_abs_diff(&got[2], &exp_d) < 2e-3, "delta tape mismatch");
    }

    fn run_replay(accepted: usize, has_mask: bool, mask: &[u32]) {
        let _g = gpu_lock();
        let (batch, t_log) = (1usize, 5usize);
        let delta_log = src(batch * t_log * HV * DV, 0x21, 0.5);
        let k_log = src(batch * t_log * HV * DK, 0x22, 0.4);
        let g_log: Vec<f32> = src(batch * t_log * HV, 0x23, 0.1).iter().map(|x| 0.9 + x).collect();
        let state_in = src(batch * HV * DV * DK, 0x24, 0.3);
        let expected = naive_replay(
            &delta_log, &k_log, &g_log, &state_in, mask, batch, t_log, accepted, has_mask,
        );
        let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        b.insert("delta_log".into(), pack_f32(&delta_log));
        b.insert("k_log".into(), pack_f32(&k_log));
        b.insert("g_log".into(), pack_f32(&g_log));
        b.insert("state_in".into(), pack_f32(&state_in));
        b.insert("mask".into(), u32_bytes(mask));
        b.insert("state_out".into(), pack_f32(&vec![0.0; state_in.len()]));
        b.insert("t_log".into(), (t_log as u32).to_le_bytes().to_vec());
        b.insert("accepted".into(), (accepted as u32).to_le_bytes().to_vec());
        b.insert("has_mask".into(), u32::from(has_mask).to_le_bytes().to_vec());
        let got = dispatch(state_replay_d64_32_2_2::kernel_ir_for, &b, batch, &["state_out"]);
        assert!(
            max_abs_diff(&got[0], &expected) < 2e-3,
            "replay accepted={accepted} mask={has_mask}: state mismatch"
        );
    }

    #[test]
    fn state_replay_full_prefix_f32() { run_replay(5, false, &[1; 5]); }

    #[test]
    fn state_replay_partial_prefix_f32() { run_replay(3, false, &[1; 5]); }

    #[test]
    fn state_replay_masked_steps_f32() { run_replay(5, true, &[1, 0, 1, 0, 1]); }
}
