//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Mamba / Mamba 2 state replay — port of `ssm_replay.metal`
//! (spec 040). The speculative-decode rollback companion to
//! `ssm.rs`'s `ssm_step`. Two kernels:
//!
//!   - `ssm_step_record` — the sequential SSD forward over `t_total`
//!     steps, capturing each step's `(dA, dBx)` into delta logs
//!     alongside the standard `(y, state_out)`.
//!   - `ssm_replay` — re-folds the first `k` log entries onto a
//!     recurrent-state snapshot to recover state-after-k.
//!
//! Threading (matches `ssm.metal`): a 32-lane simdgroup splits the
//! `Ds` state axis (`n_per_t = Ds/32` per lane); `program_id::<1>()`
//! = `Dh` index, `program_id::<2>()` = `batch*H + h`. `simd_sum`
//! reduces `y = C·state` across the `Ds` lanes.
//!
//! Layouts: `x` / `y` / `dt` [B,T,H,Dh|H]; `B` / `C` [B,T,G,Ds];
//! `state` [B,H,Dh,Ds]; `dA_log` [B,T,H,Ds]; `dBx_log` [B,T,H,Dh,Ds].
//! `mask` (u32, 0/1) makes a masked timestep identity (`dA=1, dBx=0`)
//! so rollback past it is order-preserving.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, Dh, batch*H]`, `tg = [32, 1, 1]`.
//! - `Ds` a multiple of 32.
//!
//! Codegen-only; correctness pinned by
//! `tests/ssm_replay_gpu_correctness.rs`.

use metaltile::kernel;

// ── SSD forward step with (dA, dBx) tape capture ────────────────────────────
#[rustfmt::skip]
macro_rules! ssm_step_record {
    ($name:ident, $dh:literal, $ds:literal, $h:literal, $g:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            x: Tensor<T>,
            a_log: Tensor<T>,
            b: Tensor<T>,
            c: Tensor<T>,
            d: Tensor<T>,
            dt: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut y: Tensor<T>,
            mut state_out: Tensor<T>,
            mut da_log: Tensor<T>,
            mut dbx_log: Tensor<T>,
            #[constexpr] t_total: u32,
            #[constexpr] has_mask: u32,
        ) {
            let ds_lane = program_id::<0>();
            let d_idx = program_id::<1>();
            let n = program_id::<2>();
            let h_idx = n - (n / $h) * $h;
            let b_idx = n / $h;
            let g_idx = h_idx / ($h / $g);
            let state_base = (n * $dh + d_idx) * $ds;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[state_base + $n_per_t * ds_lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            // A = -exp(A_log[h]).
            let a_neg = 0.0f32 - exp(load(a_log[h_idx]).cast::<f32>());

            for t in range(0u32, t_total, 1u32) {
                let bt = b_idx * t_total + t;
                let bt_h = bt * $h + h_idx;
                let bt_g = bt * $g + g_idx;
                let active = select(has_mask == 0u32, 1u32, load(mask[bt]));

                let dt_raw = load(dt[bt_h]).cast::<f32>();
                // Masked step: dA=1, dt_eff=0 → identity recurrence.
                let dt_eff = select(active > 0u32, dt_raw, 0.0f32);
                let d_a = select(active > 0u32, exp(a_neg * dt_raw), 1.0f32);

                // Capture dA (same scalar in every Ds slot for this lane).
                for i in range(0u32, $n_per_t, 1u32) {
                    store(da_log[bt_h * $ds + $n_per_t * ds_lane + i], d_a.cast::<T>());
                }

                let x_v = load(x[bt_h * $dh + d_idx]).cast::<f32>();
                let dbx_base = (bt_h * $dh + d_idx) * $ds;
                let mut y_acc = 0.0f32;
                for i in range(0u32, $n_per_t, 1u32) {
                    let s_idx = $n_per_t * ds_lane + i;
                    let b_v = load(b[bt_g * $ds + s_idx]).cast::<f32>();
                    let dbx = x_v * dt_eff * b_v;
                    store(dbx_log[dbx_base + s_idx], dbx.cast::<T>());
                    let st = d_a * stack_load("state", i) + dbx;
                    stack_store("state", i, st);
                    y_acc = y_acc + st * load(c[bt_g * $ds + s_idx]).cast::<f32>();
                }
                let y_sum = simd_sum(y_acc);
                if ds_lane == 0u32 {
                    let y_d = y_sum + x_v * load(d[h_idx]).cast::<f32>();
                    store(y[bt_h * $dh + d_idx], y_d.cast::<T>());
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[state_base + $n_per_t * ds_lane + i], st.cast::<T>());
            }
        }
    };
}

// ── Tape replay: re-fold the first k log entries onto a snapshot ────────────
#[rustfmt::skip]
macro_rules! ssm_replay {
    ($name:ident, $dh:literal, $ds:literal, $h:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            state_snapshot: Tensor<T>,
            da_log: Tensor<T>,
            dbx_log: Tensor<T>,
            mask: Tensor<u32>,
            mut state_after_k: Tensor<T>,
            #[constexpr] k_steps: u32,
            #[constexpr] t_total: u32,
            #[constexpr] has_mask: u32,
        ) {
            let ds_lane = program_id::<0>();
            let d_idx = program_id::<1>();
            let n = program_id::<2>();
            let h_idx = n - (n / $h) * $h;
            let b_idx = n / $h;
            let state_base = (n * $dh + d_idx) * $ds;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_snapshot[state_base + $n_per_t * ds_lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            for t in range(0u32, k_steps, 1u32) {
                let bt = b_idx * t_total + t;
                let bt_h = bt * $h + h_idx;
                let active = select(has_mask == 0u32, 1u32, load(mask[bt]));
                let dbx_base = (bt_h * $dh + d_idx) * $ds;
                for i in range(0u32, $n_per_t, 1u32) {
                    let s_idx = $n_per_t * ds_lane + i;
                    let old = stack_load("state", i);
                    let d_a = load(da_log[bt_h * $ds + s_idx]).cast::<f32>();
                    let dbx = load(dbx_log[dbx_base + s_idx]).cast::<f32>();
                    let new_val = d_a * old + dbx;
                    // Masked steps were recorded as dA=1, dBx=0 (identity),
                    // but guard anyway so a stale tape entry can't perturb.
                    stack_store("state", i, select(active > 0u32, new_val, old));
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_after_k[state_base + $n_per_t * ds_lane + i], st.cast::<T>());
            }
        }
    };
}

// Small unit-test cell: Dh=16, Ds=64, H=4, G=2.
ssm_step_record!(ssm_step_record_d16_64_4_2, 16u32, 64u32, 4u32, 2u32, 2u32, "record_d16_64_4_2");
ssm_replay!(ssm_replay_d16_64_4, 16u32, 64u32, 4u32, 2u32, "replay_d16_64_4");
// Production cell: Dh=128, Ds=128, H=32, G=2 (Jamba / Nemotron class).
ssm_step_record!(
    ssm_step_record_d128_128_32_2,
    128u32,
    128u32,
    32u32,
    2u32,
    4u32,
    "record_d128_128_32_2"
);
ssm_replay!(ssm_replay_d128_128_32, 128u32, 128u32, 32u32, 4u32, "replay_d128_128_32");

#[cfg(target_os = "macos")]
pub mod tests_support {
    use std::collections::BTreeMap;

    use metaltile_core::{dtype::DType, ir::KernelMode};
    use metaltile_runtime::Context;

    use super::{ssm_replay_d16_64_4, ssm_step_record_d16_64_4_2};

    const DH: usize = 16;
    const DS: usize = 64;
    const H: usize = 4;
    const G: usize = 2;

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

    struct Tape {
        y: Vec<f32>,
        state_out: Vec<f32>,
        da_log: Vec<f32>,
        dbx_log: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_record(
        x: &[f32],
        a_log: &[f32],
        bmat: &[f32],
        cmat: &[f32],
        dvec: &[f32],
        dt: &[f32],
        state_in: &[f32],
        mask: &[u32],
        batch: usize,
        t_total: usize,
        has_mask: bool,
    ) -> Tape {
        let mut y = vec![0.0_f32; batch * t_total * H * DH];
        let mut da_log = vec![0.0_f32; batch * t_total * H * DS];
        let mut dbx_log = vec![0.0_f32; batch * t_total * H * DH * DS];
        let mut state = state_in.to_vec();
        for n in 0..batch * H {
            let b = n / H;
            let h = n % H;
            let g = h / (H / G);
            let a_neg = -a_log[h].exp();
            for t in 0..t_total {
                let bt = b * t_total + t;
                let bt_h = bt * H + h;
                let bt_g = bt * G + g;
                let active = !has_mask || mask[bt] != 0;
                let dt_v = dt[bt_h];
                let dt_eff = if active { dt_v } else { 0.0 };
                let d_a = if active { (a_neg * dt_v).exp() } else { 1.0 };
                for ds in 0..DS {
                    da_log[bt_h * DS + ds] = d_a;
                }
                for dh in 0..DH {
                    let x_v = x[bt_h * DH + dh];
                    let mut y_acc = 0.0_f32;
                    for ds in 0..DS {
                        let dbx = x_v * dt_eff * bmat[bt_g * DS + ds];
                        dbx_log[(bt_h * DH + dh) * DS + ds] = dbx;
                        let s0 = (n * DH + dh) * DS + ds;
                        state[s0] = d_a * state[s0] + dbx;
                        y_acc += state[s0] * cmat[bt_g * DS + ds];
                    }
                    y[bt_h * DH + dh] = y_acc + x_v * dvec[h];
                }
            }
        }
        Tape { y, state_out: state, da_log, dbx_log }
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_replay(
        snapshot: &[f32],
        da_log: &[f32],
        dbx_log: &[f32],
        mask: &[u32],
        batch: usize,
        t_total: usize,
        k: usize,
        has_mask: bool,
    ) -> Vec<f32> {
        let mut state = snapshot.to_vec();
        for n in 0..batch * H {
            let b = n / H;
            let h = n % H;
            for t in 0..k {
                let bt = b * t_total + t;
                if has_mask && mask[bt] == 0 {
                    continue;
                }
                let bt_h = bt * H + h;
                for dh in 0..DH {
                    for ds in 0..DS {
                        let s0 = (n * DH + dh) * DS + ds;
                        state[s0] = da_log[bt_h * DS + ds] * state[s0]
                            + dbx_log[(bt_h * DH + dh) * DS + ds];
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
            .dispatch_with_grid(
                &kernel,
                buffers,
                &BTreeMap::new(),
                [1, DH, batch * H],
                [32, 1, 1],
            )
            .expect("ssm_replay dispatch");
        want.iter().map(|w| unpack_f32(result.outputs.get(*w).expect(w))).collect()
    }

    #[test]
    fn ssm_step_record_captures_tape_f32() {
        let _g = gpu_lock();
        let (batch, t) = (1usize, 4usize);
        let x = src(batch * t * H * DH, 0x1, 1.0);
        let a_log = src(H, 0x2, 1.0);
        let bmat = src(batch * t * G * DS, 0x3, 1.0);
        let cmat = src(batch * t * G * DS, 0x4, 1.0);
        let dvec = src(H, 0x5, 0.5);
        let dt: Vec<f32> = src(batch * t * H, 0x6, 0.1).iter().map(|v| 0.2 + v).collect();
        let state_in = src(batch * H * DH * DS, 0x7, 0.3);
        let exp =
            naive_record(&x, &a_log, &bmat, &cmat, &dvec, &dt, &state_in, &[], batch, t, false);
        let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        b.insert("x".into(), pack_f32(&x));
        b.insert("a_log".into(), pack_f32(&a_log));
        b.insert("b".into(), pack_f32(&bmat));
        b.insert("c".into(), pack_f32(&cmat));
        b.insert("d".into(), pack_f32(&dvec));
        b.insert("dt".into(), pack_f32(&dt));
        b.insert("state_in".into(), pack_f32(&state_in));
        b.insert("mask".into(), u32_bytes(&vec![1; batch * t]));
        b.insert("y".into(), pack_f32(&vec![0.0; exp.y.len()]));
        b.insert("state_out".into(), pack_f32(&vec![0.0; state_in.len()]));
        b.insert("da_log".into(), pack_f32(&vec![0.0; exp.da_log.len()]));
        b.insert("dbx_log".into(), pack_f32(&vec![0.0; exp.dbx_log.len()]));
        b.insert("t_total".into(), (t as u32).to_le_bytes().to_vec());
        b.insert("has_mask".into(), 0u32.to_le_bytes().to_vec());
        let got = dispatch(
            ssm_step_record_d16_64_4_2::kernel_ir_for,
            &b,
            batch,
            &["y", "state_out", "da_log", "dbx_log"],
        );
        assert!(got[3].iter().any(|&v| v != 0.0), "dBx tape all zeros");
        assert!(max_abs_diff(&got[0], &exp.y) < 2e-3, "y mismatch");
        assert!(max_abs_diff(&got[1], &exp.state_out) < 2e-3, "state mismatch");
        assert!(max_abs_diff(&got[2], &exp.da_log) < 2e-3, "dA tape mismatch");
        assert!(max_abs_diff(&got[3], &exp.dbx_log) < 2e-3, "dBx tape mismatch");
    }

    fn run_replay(k: usize, has_mask: bool, mask: &[u32]) {
        let _g = gpu_lock();
        let (batch, t) = (1usize, 5usize);
        let snapshot = src(batch * H * DH * DS, 0x21, 0.3);
        let da_log: Vec<f32> =
            src(batch * t * H * DS, 0x22, 0.1).iter().map(|v| 0.9 + v).collect();
        let dbx_log = src(batch * t * H * DH * DS, 0x23, 0.4);
        let expected = naive_replay(&snapshot, &da_log, &dbx_log, mask, batch, t, k, has_mask);
        let mut b: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        b.insert("state_snapshot".into(), pack_f32(&snapshot));
        b.insert("da_log".into(), pack_f32(&da_log));
        b.insert("dbx_log".into(), pack_f32(&dbx_log));
        b.insert("mask".into(), u32_bytes(mask));
        b.insert("state_after_k".into(), pack_f32(&vec![0.0; snapshot.len()]));
        b.insert("k_steps".into(), (k as u32).to_le_bytes().to_vec());
        b.insert("t_total".into(), (t as u32).to_le_bytes().to_vec());
        b.insert("has_mask".into(), u32::from(has_mask).to_le_bytes().to_vec());
        let got = dispatch(ssm_replay_d16_64_4::kernel_ir_for, &b, batch, &["state_after_k"]);
        assert!(
            max_abs_diff(&got[0], &expected) < 2e-3,
            "replay k={k} mask={has_mask}: state mismatch",
        );
    }

    #[test]
    fn ssm_replay_full_prefix_f32() { run_replay(5, false, &[1; 5]); }

    #[test]
    fn ssm_replay_partial_prefix_f32() { run_replay(2, false, &[1; 5]); }

    #[test]
    fn ssm_replay_masked_steps_f32() { run_replay(5, true, &[1, 0, 1, 1, 0]); }
}
