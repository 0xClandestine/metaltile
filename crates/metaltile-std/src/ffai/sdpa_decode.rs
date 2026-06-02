//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! All single-token SDPA decode variants in one file.
//!
//! Five variants are generated from one `sdpa_decode_kernel!` macro:
//!
//! | Kernel                  | elems | Phases | Sink | Window |
//! |-------------------------|-------|--------|------|--------|
//! | `ffai_sdpa_decode_d64`  |   2   |   1×2  |  ✓  |        |
//! | `ffai_sdpa_decode_d96`  |   3   |   1×3  |      |        |
//! | `ffai_sdpa_decode`      |   4   |   1×4  |  ✓  |   ✓    |
//! | `ffai_sdpa_decode_d256` |   8   |   2×4  |  ✓  |        |
//! | `ffai_sdpa_decode_d512` |  16   |   4×4  |      |        |
//!
//! The `$elems:literal` macro parameter drives every inner loop
//! (`range(0u32, $elems, 1u32)`) so the DSL's `UnrollPass` unrolls
//! them to constant-indexed `stack_alloc` accesses — identical to
//! hand-written named variable sequences in the generated MSL.
//!
//! The output reduction always reuses 4 `tg_out` buffers of 1056
//! floats (≈16 KB), phased for variants with >4 elements per lane.
//! d64 uses 2 slots and d96 uses 3 slots (single phase, no padding)
//! to avoid the extra threadgroup-memory overhead.
//!
//! See `sdpa_decode.rs` (now replaced by this file) for the full
//! algorithm walkthrough and dispatch invariant documentation.

use metaltile::kernel;

macro_rules! sdpa_decode_kernel {
    // ── Arm 1: 2-slot single phase, sink-aware reduction ────────────
    // Used by: ffai_sdpa_decode_d64 (head_dim=64, 2 elems/lane)
    (sink_2, $name:ident, $elems:literal) => {
        #[kernel]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] n_kv: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] heads_per_group: u32,
            #[constexpr] has_sink: u32,
            #[constexpr] sink_logit: f32,
            #[constexpr] scale: f32,
        ) {
            let q_head = tgid_x;
            let kv_head = q_head / heads_per_group;
            let sg = simd_id;
            let lane = simd_lane;
            let ns = n_simd;
            threadgroup_alloc("tg_max", 32);
            threadgroup_alloc("tg_sum", 32);
            threadgroup_alloc("tg_out0", 1056);
            threadgroup_alloc("tg_out1", 1056);
            let q_off = q_head * head_dim;
            let kv_head_base = kv_head * kv_stride * head_dim;
            let d0 = lane * $elems;
            stack_alloc("qs", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("qs", _i, load(q[q_off + d0 + _i]).cast::<f32>() * scale);
            }
            let mut run_max = neg_infinity();
            let mut run_sum = 0.0f32;
            stack_alloc("os", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("os", _i, 0.0f32);
            }
            for _t in range(sg, n_kv, ns) {
                let base = kv_head_base + _t * head_dim;
                let kv0 = base + d0;
                let mut partial = 0.0f32;
                for _i in range(0u32, $elems, 1u32) {
                    partial = partial + stack_load("qs", _i) * load(k[kv0 + _i]).cast::<f32>();
                }
                let score = simd_sum(partial);
                let new_max = select(score > run_max, score, run_max);
                let factor = exp(run_max - new_max);
                let weight = exp(score - new_max);
                run_sum = run_sum * factor + weight;
                run_max = new_max;
                for _i in range(0u32, $elems, 1u32) {
                    stack_store(
                        "os",
                        _i,
                        stack_load("os", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
                    );
                }
            }
            if lane == 0 {
                threadgroup_store("tg_max", sg, run_max);
                threadgroup_store("tg_sum", sg, run_sum);
            }
            threadgroup_barrier();
            if sg == 0 {
                let g_max_raw = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
                let sink_max = select(has_sink > 0u32, sink_logit, neg_infinity());
                let g_max_in = select(
                    lane == 0u32,
                    select(g_max_raw > sink_max, g_max_raw, sink_max),
                    g_max_raw,
                );
                let g_max = simd_max(g_max_in);
                let g_sum_in = select(
                    lane < ns,
                    threadgroup_load("tg_sum", lane) * exp(g_max_raw - g_max),
                    0.0f32,
                );
                let sink_sum = select(has_sink > 0u32, exp(sink_logit - g_max), 0.0f32);
                let g_sum = simd_sum(g_sum_in + select(lane == 0u32, sink_sum, 0.0f32));
                if lane == 0 {
                    threadgroup_store("tg_max", 0, g_max);
                    threadgroup_store("tg_sum", 0, g_sum);
                }
            }
            threadgroup_barrier();
            let g_max = threadgroup_load("tg_max", 0);
            let g_sum = threadgroup_load("tg_sum", 0);
            let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
            let stride = ns + 1u32;
            let idx = lane * stride + sg;
            threadgroup_store("tg_out0", idx, stack_load("os", 0u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 1u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so0 = 0.0f32;
                let mut so1 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so0 = so0 + threadgroup_load("tg_out0", ri);
                    so1 = so1 + threadgroup_load("tg_out1", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off], so0.cast::<T>());
                store(out[out_off + 1u32], so1.cast::<T>());
            }
        }
    };

    // ── Arm 2: 3-slot single phase, simple reduction ─────────────────
    // Used by: ffai_sdpa_decode_d96 (head_dim=96, 3 elems/lane)
    (simple_3, $name:ident, $elems:literal) => {
        #[kernel]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] n_kv: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] heads_per_group: u32,
            #[constexpr] scale: f32,
        ) {
            let q_head = tgid_x;
            let kv_head = q_head / heads_per_group;
            let sg = simd_id;
            let lane = simd_lane;
            let ns = n_simd;
            threadgroup_alloc("tg_max", 32);
            threadgroup_alloc("tg_sum", 32);
            threadgroup_alloc("tg_out0", 1056);
            threadgroup_alloc("tg_out1", 1056);
            threadgroup_alloc("tg_out2", 1056);
            let q_off = q_head * head_dim;
            let kv_head_base = kv_head * kv_stride * head_dim;
            let d0 = lane * $elems;
            stack_alloc("qs", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("qs", _i, load(q[q_off + d0 + _i]).cast::<f32>() * scale);
            }
            let mut run_max = neg_infinity();
            let mut run_sum = 0.0f32;
            stack_alloc("os", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("os", _i, 0.0f32);
            }
            for _t in range(sg, n_kv, ns) {
                let base = kv_head_base + _t * head_dim;
                let kv0 = base + d0;
                let mut partial = 0.0f32;
                for _i in range(0u32, $elems, 1u32) {
                    partial = partial + stack_load("qs", _i) * load(k[kv0 + _i]).cast::<f32>();
                }
                let score = simd_sum(partial);
                let new_max = select(score > run_max, score, run_max);
                let factor = exp(run_max - new_max);
                let weight = exp(score - new_max);
                run_sum = run_sum * factor + weight;
                run_max = new_max;
                for _i in range(0u32, $elems, 1u32) {
                    stack_store(
                        "os",
                        _i,
                        stack_load("os", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
                    );
                }
            }
            if lane == 0 {
                threadgroup_store("tg_max", sg, run_max);
                threadgroup_store("tg_sum", sg, run_sum);
            }
            threadgroup_barrier();
            if sg == 0 {
                let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
                let g_max = simd_max(g_max_in);
                let g_sum_in = select(
                    lane < ns,
                    threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max),
                    0.0f32,
                );
                let g_sum = simd_sum(g_sum_in);
                if lane == 0 {
                    threadgroup_store("tg_max", 0, g_max);
                    threadgroup_store("tg_sum", 0, g_sum);
                }
            }
            threadgroup_barrier();
            let g_max = threadgroup_load("tg_max", 0);
            let g_sum = threadgroup_load("tg_sum", 0);
            let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
            let stride = ns + 1u32;
            let idx = lane * stride + sg;
            threadgroup_store("tg_out0", idx, stack_load("os", 0u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 1u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 2u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so0 = 0.0f32;
                let mut so1 = 0.0f32;
                let mut so2 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so0 = so0 + threadgroup_load("tg_out0", ri);
                    so1 = so1 + threadgroup_load("tg_out1", ri);
                    so2 = so2 + threadgroup_load("tg_out2", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off], so0.cast::<T>());
                store(out[out_off + 1u32], so1.cast::<T>());
                store(out[out_off + 2u32], so2.cast::<T>());
            }
        }
    };

    // ── Arm 3: 4-slot single phase, sink+window, two KV passes ──────
    // Used by: ffai_sdpa_decode (head_dim=128, 4 elems/lane)
    // Two passes: [0, sink_end) then [window_start, n_kv).
    // Dense path: sink_end=0, window_start=0 → only second pass fires.
    (sink_window_4, $name:ident, $elems:literal) => {
        #[kernel]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] n_kv: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] heads_per_group: u32,
            #[constexpr] sink_end: u32,
            #[constexpr] window_start: u32,
            #[constexpr] has_sink: u32,
            #[constexpr] sink_logit: f32,
            #[constexpr] scale: f32,
        ) {
            let q_head = tgid_x;
            let kv_head = q_head / heads_per_group;
            let sg = simd_id;
            let lane = simd_lane;
            let ns = n_simd;
            threadgroup_alloc("tg_max", 32);
            threadgroup_alloc("tg_sum", 32);
            threadgroup_alloc("tg_out0", 1056);
            threadgroup_alloc("tg_out1", 1056);
            threadgroup_alloc("tg_out2", 1056);
            threadgroup_alloc("tg_out3", 1056);
            let q_off = q_head * head_dim;
            let kv_head_base = kv_head * kv_stride * head_dim;
            let d0 = lane * $elems;
            stack_alloc("qs", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("qs", _i, load(q[q_off + d0 + _i]).cast::<f32>() * scale);
            }
            let mut run_max = neg_infinity();
            let mut run_sum = 0.0f32;
            stack_alloc("os", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("os", _i, 0.0f32);
            }
            // Sink-token pass [0, sink_end). When sink_end=0 no iterations fire.
            for _t in range(sg, sink_end, ns) {
                let base = kv_head_base + _t * head_dim;
                let kv0 = base + d0;
                let mut partial = 0.0f32;
                for _i in range(0u32, $elems, 1u32) {
                    partial = partial + stack_load("qs", _i) * load(k[kv0 + _i]).cast::<f32>();
                }
                let score = simd_sum(partial);
                let new_max = select(score > run_max, score, run_max);
                let factor = exp(run_max - new_max);
                let weight = exp(score - new_max);
                run_sum = run_sum * factor + weight;
                run_max = new_max;
                for _i in range(0u32, $elems, 1u32) {
                    stack_store(
                        "os",
                        _i,
                        stack_load("os", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
                    );
                }
            }
            // Window pass [window_start, n_kv). Dense: window_start=0 → full range.
            for _t in range(sg + window_start, n_kv, ns) {
                let base = kv_head_base + _t * head_dim;
                let kv0 = base + d0;
                let mut partial = 0.0f32;
                for _i in range(0u32, $elems, 1u32) {
                    partial = partial + stack_load("qs", _i) * load(k[kv0 + _i]).cast::<f32>();
                }
                let score = simd_sum(partial);
                let new_max = select(score > run_max, score, run_max);
                let factor = exp(run_max - new_max);
                let weight = exp(score - new_max);
                run_sum = run_sum * factor + weight;
                run_max = new_max;
                for _i in range(0u32, $elems, 1u32) {
                    stack_store(
                        "os",
                        _i,
                        stack_load("os", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
                    );
                }
            }
            if lane == 0 {
                threadgroup_store("tg_max", sg, run_max);
                threadgroup_store("tg_sum", sg, run_sum);
            }
            threadgroup_barrier();
            if sg == 0 {
                let g_max_raw = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
                let sink_max = select(has_sink > 0u32, sink_logit, neg_infinity());
                let g_max_in = select(
                    lane == 0u32,
                    select(g_max_raw > sink_max, g_max_raw, sink_max),
                    g_max_raw,
                );
                let g_max = simd_max(g_max_in);
                let g_sum_in = select(
                    lane < ns,
                    threadgroup_load("tg_sum", lane) * exp(g_max_raw - g_max),
                    0.0f32,
                );
                let sink_sum = select(has_sink > 0u32, exp(sink_logit - g_max), 0.0f32);
                let g_sum = simd_sum(g_sum_in + select(lane == 0u32, sink_sum, 0.0f32));
                if lane == 0 {
                    threadgroup_store("tg_max", 0, g_max);
                    threadgroup_store("tg_sum", 0, g_sum);
                }
            }
            threadgroup_barrier();
            let g_max = threadgroup_load("tg_max", 0);
            let g_sum = threadgroup_load("tg_sum", 0);
            let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
            let stride = ns + 1u32;
            let idx = lane * stride + sg;
            threadgroup_store("tg_out0", idx, stack_load("os", 0u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 1u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 2u32) * rescale);
            threadgroup_store("tg_out3", idx, stack_load("os", 3u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so0 = 0.0f32;
                let mut so1 = 0.0f32;
                let mut so2 = 0.0f32;
                let mut so3 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so0 = so0 + threadgroup_load("tg_out0", ri);
                    so1 = so1 + threadgroup_load("tg_out1", ri);
                    so2 = so2 + threadgroup_load("tg_out2", ri);
                    so3 = so3 + threadgroup_load("tg_out3", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off], so0.cast::<T>());
                store(out[out_off + 1u32], so1.cast::<T>());
                store(out[out_off + 2u32], so2.cast::<T>());
                store(out[out_off + 3u32], so3.cast::<T>());
            }
        }
    };

    // ── Arm 4: 4-slot × 2 phases, sink-aware reduction ───────────────
    // Used by: ffai_sdpa_decode_d256 (head_dim=256, 8 elems/lane)
    (sink_4x2, $name:ident, $elems:literal) => {
        #[kernel]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] n_kv: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] heads_per_group: u32,
            #[constexpr] has_sink: u32,
            #[constexpr] sink_logit: f32,
            #[constexpr] scale: f32,
        ) {
            let q_head = tgid_x;
            let kv_head = q_head / heads_per_group;
            let sg = simd_id;
            let lane = simd_lane;
            let ns = n_simd;
            threadgroup_alloc("tg_max", 32);
            threadgroup_alloc("tg_sum", 32);
            threadgroup_alloc("tg_out0", 1056);
            threadgroup_alloc("tg_out1", 1056);
            threadgroup_alloc("tg_out2", 1056);
            threadgroup_alloc("tg_out3", 1056);
            let q_off = q_head * head_dim;
            let kv_head_base = kv_head * kv_stride * head_dim;
            let d0 = lane * $elems;
            stack_alloc("qs", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("qs", _i, load(q[q_off + d0 + _i]).cast::<f32>() * scale);
            }
            let mut run_max = neg_infinity();
            let mut run_sum = 0.0f32;
            stack_alloc("os", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("os", _i, 0.0f32);
            }
            for _t in range(sg, n_kv, ns) {
                let base = kv_head_base + _t * head_dim;
                let kv0 = base + d0;
                let mut partial = 0.0f32;
                for _i in range(0u32, $elems, 1u32) {
                    partial = partial + stack_load("qs", _i) * load(k[kv0 + _i]).cast::<f32>();
                }
                let score = simd_sum(partial);
                let new_max = select(score > run_max, score, run_max);
                let factor = exp(run_max - new_max);
                let weight = exp(score - new_max);
                run_sum = run_sum * factor + weight;
                run_max = new_max;
                for _i in range(0u32, $elems, 1u32) {
                    stack_store(
                        "os",
                        _i,
                        stack_load("os", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
                    );
                }
            }
            if lane == 0 {
                threadgroup_store("tg_max", sg, run_max);
                threadgroup_store("tg_sum", sg, run_sum);
            }
            threadgroup_barrier();
            if sg == 0 {
                let g_max_raw = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
                let sink_max = select(has_sink > 0u32, sink_logit, neg_infinity());
                let g_max_in = select(
                    lane == 0u32,
                    select(g_max_raw > sink_max, g_max_raw, sink_max),
                    g_max_raw,
                );
                let g_max = simd_max(g_max_in);
                let g_sum_in = select(
                    lane < ns,
                    threadgroup_load("tg_sum", lane) * exp(g_max_raw - g_max),
                    0.0f32,
                );
                let sink_sum = select(has_sink > 0u32, exp(sink_logit - g_max), 0.0f32);
                let g_sum = simd_sum(g_sum_in + select(lane == 0u32, sink_sum, 0.0f32));
                if lane == 0 {
                    threadgroup_store("tg_max", 0, g_max);
                    threadgroup_store("tg_sum", 0, g_sum);
                }
            }
            threadgroup_barrier();
            let g_max = threadgroup_load("tg_max", 0);
            let g_sum = threadgroup_load("tg_sum", 0);
            let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
            let stride = ns + 1u32;
            let idx = lane * stride + sg;
            // Phase 1 (dims 0..3)
            threadgroup_store("tg_out0", idx, stack_load("os", 0u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 1u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 2u32) * rescale);
            threadgroup_store("tg_out3", idx, stack_load("os", 3u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so0 = 0.0f32;
                let mut so1 = 0.0f32;
                let mut so2 = 0.0f32;
                let mut so3 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so0 = so0 + threadgroup_load("tg_out0", ri);
                    so1 = so1 + threadgroup_load("tg_out1", ri);
                    so2 = so2 + threadgroup_load("tg_out2", ri);
                    so3 = so3 + threadgroup_load("tg_out3", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off], so0.cast::<T>());
                store(out[out_off + 1u32], so1.cast::<T>());
                store(out[out_off + 2u32], so2.cast::<T>());
                store(out[out_off + 3u32], so3.cast::<T>());
            }
            threadgroup_barrier();
            // Phase 2 (dims 4..7)
            threadgroup_store("tg_out0", idx, stack_load("os", 4u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 5u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 6u32) * rescale);
            threadgroup_store("tg_out3", idx, stack_load("os", 7u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so4 = 0.0f32;
                let mut so5 = 0.0f32;
                let mut so6 = 0.0f32;
                let mut so7 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so4 = so4 + threadgroup_load("tg_out0", ri);
                    so5 = so5 + threadgroup_load("tg_out1", ri);
                    so6 = so6 + threadgroup_load("tg_out2", ri);
                    so7 = so7 + threadgroup_load("tg_out3", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off + 4u32], so4.cast::<T>());
                store(out[out_off + 5u32], so5.cast::<T>());
                store(out[out_off + 6u32], so6.cast::<T>());
                store(out[out_off + 7u32], so7.cast::<T>());
            }
        }
    };

    // ── Arm 5: 4-slot × 4 phases, simple reduction ───────────────────
    // Used by: ffai_sdpa_decode_d512 (head_dim=512, 16 elems/lane)
    // TPG=512 (16 simdgroups): 16 live Q + 16 accumulators exceeds
    // the 1024-thread pipeline cap.
    (simple_4x4, $name:ident, $elems:literal) => {
        #[kernel]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] n_kv: u32,
            #[constexpr] kv_stride: u32,
            #[constexpr] heads_per_group: u32,
            #[constexpr] scale: f32,
        ) {
            let q_head = tgid_x;
            let kv_head = q_head / heads_per_group;
            let sg = simd_id;
            let lane = simd_lane;
            let ns = n_simd;
            threadgroup_alloc("tg_max", 32);
            threadgroup_alloc("tg_sum", 32);
            threadgroup_alloc("tg_out0", 1056);
            threadgroup_alloc("tg_out1", 1056);
            threadgroup_alloc("tg_out2", 1056);
            threadgroup_alloc("tg_out3", 1056);
            let q_off = q_head * head_dim;
            let kv_head_base = kv_head * kv_stride * head_dim;
            let d0 = lane * $elems;
            stack_alloc("qs", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("qs", _i, load(q[q_off + d0 + _i]).cast::<f32>() * scale);
            }
            let mut run_max = neg_infinity();
            let mut run_sum = 0.0f32;
            stack_alloc("os", $elems, "f32");
            for _i in range(0u32, $elems, 1u32) {
                stack_store("os", _i, 0.0f32);
            }
            for _t in range(sg, n_kv, ns) {
                let base = kv_head_base + _t * head_dim;
                let kv0 = base + d0;
                let mut partial = 0.0f32;
                for _i in range(0u32, $elems, 1u32) {
                    partial = partial + stack_load("qs", _i) * load(k[kv0 + _i]).cast::<f32>();
                }
                let score = simd_sum(partial);
                let new_max = select(score > run_max, score, run_max);
                let factor = exp(run_max - new_max);
                let weight = exp(score - new_max);
                run_sum = run_sum * factor + weight;
                run_max = new_max;
                for _i in range(0u32, $elems, 1u32) {
                    stack_store(
                        "os",
                        _i,
                        stack_load("os", _i) * factor + weight * load(v[kv0 + _i]).cast::<f32>(),
                    );
                }
            }
            if lane == 0 {
                threadgroup_store("tg_max", sg, run_max);
                threadgroup_store("tg_sum", sg, run_sum);
            }
            threadgroup_barrier();
            if sg == 0 {
                let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
                let g_max = simd_max(g_max_in);
                let g_sum_in = select(
                    lane < ns,
                    threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max),
                    0.0f32,
                );
                let g_sum = simd_sum(g_sum_in);
                if lane == 0 {
                    threadgroup_store("tg_max", 0, g_max);
                    threadgroup_store("tg_sum", 0, g_sum);
                }
            }
            threadgroup_barrier();
            let g_max = threadgroup_load("tg_max", 0);
            let g_sum = threadgroup_load("tg_sum", 0);
            let rescale = select(g_sum > 0.0f32, exp(run_max - g_max) / g_sum, 0.0f32);
            let stride = ns + 1u32;
            let idx = lane * stride + sg;
            // Phase 1 (dims 0..3)
            threadgroup_store("tg_out0", idx, stack_load("os", 0u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 1u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 2u32) * rescale);
            threadgroup_store("tg_out3", idx, stack_load("os", 3u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so0 = 0.0f32;
                let mut so1 = 0.0f32;
                let mut so2 = 0.0f32;
                let mut so3 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so0 = so0 + threadgroup_load("tg_out0", ri);
                    so1 = so1 + threadgroup_load("tg_out1", ri);
                    so2 = so2 + threadgroup_load("tg_out2", ri);
                    so3 = so3 + threadgroup_load("tg_out3", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off], so0.cast::<T>());
                store(out[out_off + 1u32], so1.cast::<T>());
                store(out[out_off + 2u32], so2.cast::<T>());
                store(out[out_off + 3u32], so3.cast::<T>());
            }
            threadgroup_barrier();
            // Phase 2 (dims 4..7)
            threadgroup_store("tg_out0", idx, stack_load("os", 4u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 5u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 6u32) * rescale);
            threadgroup_store("tg_out3", idx, stack_load("os", 7u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so4 = 0.0f32;
                let mut so5 = 0.0f32;
                let mut so6 = 0.0f32;
                let mut so7 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so4 = so4 + threadgroup_load("tg_out0", ri);
                    so5 = so5 + threadgroup_load("tg_out1", ri);
                    so6 = so6 + threadgroup_load("tg_out2", ri);
                    so7 = so7 + threadgroup_load("tg_out3", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off + 4u32], so4.cast::<T>());
                store(out[out_off + 5u32], so5.cast::<T>());
                store(out[out_off + 6u32], so6.cast::<T>());
                store(out[out_off + 7u32], so7.cast::<T>());
            }
            threadgroup_barrier();
            // Phase 3 (dims 8..11)
            threadgroup_store("tg_out0", idx, stack_load("os", 8u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 9u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 10u32) * rescale);
            threadgroup_store("tg_out3", idx, stack_load("os", 11u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so8 = 0.0f32;
                let mut so9 = 0.0f32;
                let mut so10 = 0.0f32;
                let mut so11 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so8 = so8 + threadgroup_load("tg_out0", ri);
                    so9 = so9 + threadgroup_load("tg_out1", ri);
                    so10 = so10 + threadgroup_load("tg_out2", ri);
                    so11 = so11 + threadgroup_load("tg_out3", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off + 8u32], so8.cast::<T>());
                store(out[out_off + 9u32], so9.cast::<T>());
                store(out[out_off + 10u32], so10.cast::<T>());
                store(out[out_off + 11u32], so11.cast::<T>());
            }
            threadgroup_barrier();
            // Phase 4 (dims 12..15)
            threadgroup_store("tg_out0", idx, stack_load("os", 12u32) * rescale);
            threadgroup_store("tg_out1", idx, stack_load("os", 13u32) * rescale);
            threadgroup_store("tg_out2", idx, stack_load("os", 14u32) * rescale);
            threadgroup_store("tg_out3", idx, stack_load("os", 15u32) * rescale);
            threadgroup_barrier();
            if sg == 0 {
                let mut so12 = 0.0f32;
                let mut so13 = 0.0f32;
                let mut so14 = 0.0f32;
                let mut so15 = 0.0f32;
                for _g in range(0u32, ns, 1u32) {
                    let ri = lane * stride + _g;
                    so12 = so12 + threadgroup_load("tg_out0", ri);
                    so13 = so13 + threadgroup_load("tg_out1", ri);
                    so14 = so14 + threadgroup_load("tg_out2", ri);
                    so15 = so15 + threadgroup_load("tg_out3", ri);
                }
                let out_off = q_off + d0;
                store(out[out_off + 12u32], so12.cast::<T>());
                store(out[out_off + 13u32], so13.cast::<T>());
                store(out[out_off + 14u32], so14.cast::<T>());
                store(out[out_off + 15u32], so15.cast::<T>());
            }
        }
    };
}

// ── Kernel instantiations ────────────────────────────────────────────────

sdpa_decode_kernel!(sink_2, ffai_sdpa_decode_d64, 2u32);
sdpa_decode_kernel!(simple_3, ffai_sdpa_decode_d96, 3u32);
sdpa_decode_kernel!(sink_window_4, ffai_sdpa_decode, 4u32);
sdpa_decode_kernel!(sink_4x2, ffai_sdpa_decode_d256, 8u32);
sdpa_decode_kernel!(simple_4x4, ffai_sdpa_decode_d512, 16u32);

// ── Codegen smoke tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use metaltile_codegen::msl::MslGenerator;
    use metaltile_core::ir::KernelMode;

    use super::{
        ffai_sdpa_decode,
        ffai_sdpa_decode_d64,
        ffai_sdpa_decode_d96,
        ffai_sdpa_decode_d256,
        ffai_sdpa_decode_d512,
    };
    use crate::bench_types::DType;

    fn check(name: &str, src: &str) {
        assert!(!src.trim().is_empty(), "MSL for {name} should not be empty");
        assert!(src.contains(&format!("kernel void {name}")), "MSL should declare {name}:\n{src}",);
    }

    #[test]
    fn codegen_d64() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let mut k = ffai_sdpa_decode_d64::kernel_ir_for(dt);
            k.mode = KernelMode::Reduction;
            let src = MslGenerator::default().generate(&k).unwrap();
            check("ffai_sdpa_decode_d64", &src);
        }
    }

    #[test]
    fn codegen_d96() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let mut k = ffai_sdpa_decode_d96::kernel_ir_for(dt);
            k.mode = KernelMode::Reduction;
            let src = MslGenerator::default().generate(&k).unwrap();
            check("ffai_sdpa_decode_d96", &src);
        }
    }

    #[test]
    fn codegen_d128() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let mut k = ffai_sdpa_decode::kernel_ir_for(dt);
            k.mode = KernelMode::Reduction;
            let src = MslGenerator::default().generate(&k).unwrap();
            check("ffai_sdpa_decode", &src);
            for tok in &["simd_group", "simd_lane", "threadgroup_barrier", "simd_sum", "simd_max"] {
                assert!(src.contains(tok), "d128 MSL missing `{tok}`:\n{src}");
            }
        }
    }

    #[test]
    fn codegen_d256() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let mut k = ffai_sdpa_decode_d256::kernel_ir_for(dt);
            k.mode = KernelMode::Reduction;
            let src = MslGenerator::default().generate(&k).unwrap();
            check("ffai_sdpa_decode_d256", &src);
        }
    }

    #[test]
    fn codegen_d512() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let mut k = ffai_sdpa_decode_d512::kernel_ir_for(dt);
            k.mode = KernelMode::Reduction;
            let src = MslGenerator::default().generate(&k).unwrap();
            check("ffai_sdpa_decode_d512", &src);
        }
    }
}

// ── GPU correctness tests ────────────────────────────────────────────────

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{
        ffai_sdpa_decode,
        ffai_sdpa_decode_d64,
        ffai_sdpa_decode_d96,
        ffai_sdpa_decode_d256,
        ffai_sdpa_decode_d512,
    };
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, step: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((start + i as f32 * step) % 2.0) - 1.0).collect()
    }

    /// Full-featured oracle. `sink_end`/`window_start` drive the
    /// sliding-window mask (both 0 = dense); `has_sink`/`sink_logit`
    /// drive the learned attention sink (virtual key, value 0).
    #[allow(clippy::too_many_arguments)]
    fn naive_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_q_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        n_kv: usize,
        kv_stride: usize,
        sink_end: usize,
        window_start: usize,
        has_sink: bool,
        sink_logit: f32,
        scale: f32,
    ) -> Vec<f32> {
        let gqa = n_q_heads / n_kv_heads;
        let attended: Vec<usize> = (0..sink_end).chain(window_start..n_kv).collect();
        let mut out = vec![0.0f32; n_q_heads * head_dim];
        for qh in 0..n_q_heads {
            let kvh = qh / gqa;
            let q_off = qh * head_dim;
            let kv_slab = kvh * kv_stride * head_dim;
            let mut scores: Vec<(usize, f32)> = attended
                .iter()
                .map(|&t| {
                    let dot: f32 =
                        (0..head_dim).map(|d| q[q_off + d] * k[kv_slab + t * head_dim + d]).sum();
                    (t, dot * scale)
                })
                .collect();
            let mut m = scores.iter().map(|&(_, s)| s).fold(f32::NEG_INFINITY, f32::max);
            if has_sink {
                m = m.max(sink_logit);
            }
            let mut sum = if has_sink { (sink_logit - m).exp() } else { 0.0f32 };
            for (_, s) in scores.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for d in 0..head_dim {
                out[q_off + d] =
                    scores.iter().map(|&(t, s)| s * inv * v[kv_slab + t * head_dim + d]).sum();
            }
        }
        out
    }

    // ── d64 ──────────────────────────────────────────────────────────

    fn setup_d64(dt: DType, has_sink: bool) -> TestSetup {
        let (nqh, nkh, hd) = (8usize, 4usize, 64usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let hpg = nqh / nkh;
        let scale = 1.0 / (hd as f32).sqrt();
        let sink_logit = 0.5f32;
        let q = unpack_f32(&pack_f32(&ramp(nqh * hd, 0.013, -0.4), dt), dt);
        let k = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.011, -0.5), dt), dt);
        let v = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.007, -0.3), dt), dt);
        let expected = naive_sdpa(
            &q, &k, &v, nqh, nkh, hd, n_kv, kv_stride, 0, 0, has_sink, sink_logit, scale,
        );
        TestSetup::new(ffai_sdpa_decode_d64::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", nqh * hd, dt))
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", hpg as u32)
            .constexpr("has_sink", u32::from(has_sink))
            .constexpr("sink_logit", sink_logit)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_d64(dt: DType) -> TestSetup { setup_d64(dt, false) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_d64_sink(dt: DType) -> TestSetup { setup_d64(dt, true) }

    // ── d96 ──────────────────────────────────────────────────────────

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_d96(dt: DType) -> TestSetup {
        let (nqh, nkh, hd) = (8usize, 4usize, 96usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let hpg = nqh / nkh;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = unpack_f32(&pack_f32(&ramp(nqh * hd, 0.013, -0.4), dt), dt);
        let k = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.011, -0.5), dt), dt);
        let v = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.007, -0.3), dt), dt);
        let expected =
            naive_sdpa(&q, &k, &v, nqh, nkh, hd, n_kv, kv_stride, 0, 0, false, 0.0, scale);
        TestSetup::new(ffai_sdpa_decode_d96::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", nqh * hd, dt))
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", hpg as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
    }

    // ── d128 ─────────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn setup_d128(
        dt: DType,
        n_kv: usize,
        kv_stride: usize,
        sink_end: usize,
        window_start: usize,
        has_sink: bool,
        sink_logit: f32,
    ) -> TestSetup {
        let (nqh, nkh, hd) = (8usize, 4usize, 128usize);
        let hpg = nqh / nkh;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = unpack_f32(&pack_f32(&ramp(nqh * hd, 0.013, -0.4), dt), dt);
        let k = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.011, -0.5), dt), dt);
        let v = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.007, -0.3), dt), dt);
        let expected = naive_sdpa(
            &q,
            &k,
            &v,
            nqh,
            nkh,
            hd,
            n_kv,
            kv_stride,
            sink_end,
            window_start,
            has_sink,
            sink_logit,
            scale,
        );
        TestSetup::new(ffai_sdpa_decode::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", nqh * hd, dt))
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", hpg as u32)
            .constexpr("sink_end", sink_end as u32)
            .constexpr("window_start", window_start as u32)
            .constexpr("has_sink", u32::from(has_sink))
            .constexpr("sink_logit", sink_logit)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode(dt: DType) -> TestSetup { setup_d128(dt, 64, 64, 0, 0, false, 0.0) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_swa_sink(dt: DType) -> TestSetup {
        setup_d128(dt, 16, 16, 2, 8, false, 0.0)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_swa(dt: DType) -> TestSetup {
        setup_d128(dt, 16, 16, 0, 8, false, 0.0)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_learned_sink(dt: DType) -> TestSetup {
        setup_d128(dt, 64, 64, 0, 0, true, 2.5)
    }

    // ── d256 ─────────────────────────────────────────────────────────

    fn setup_d256(dt: DType, has_sink: bool) -> TestSetup {
        let (nqh, nkh, hd) = (8usize, 4usize, 256usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let hpg = nqh / nkh;
        let scale = 1.0 / (hd as f32).sqrt();
        let sink_logit = 0.5f32;
        let q = unpack_f32(&pack_f32(&ramp(nqh * hd, 0.013, -0.4), dt), dt);
        let k = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.011, -0.5), dt), dt);
        let v = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.007, -0.3), dt), dt);
        let expected = naive_sdpa(
            &q, &k, &v, nqh, nkh, hd, n_kv, kv_stride, 0, 0, has_sink, sink_logit, scale,
        );
        TestSetup::new(ffai_sdpa_decode_d256::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", nqh * hd, dt))
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", hpg as u32)
            .constexpr("has_sink", u32::from(has_sink))
            .constexpr("sink_logit", sink_logit)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_d256(dt: DType) -> TestSetup { setup_d256(dt, false) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_ffai_sdpa_decode_d256_sink(dt: DType) -> TestSetup { setup_d256(dt, true) }

    // ── d512 ─────────────────────────────────────────────────────────

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 3e-3, 1.5e-2])]
    fn test_ffai_sdpa_decode_d512(dt: DType) -> TestSetup {
        let (nqh, nkh, hd) = (8usize, 4usize, 512usize);
        let (n_kv, kv_stride) = (64usize, 64usize);
        let hpg = nqh / nkh;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = unpack_f32(&pack_f32(&ramp(nqh * hd, 0.013, -0.4), dt), dt);
        let k = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.011, -0.5), dt), dt);
        let v = unpack_f32(&pack_f32(&ramp(nkh * kv_stride * hd, 0.007, -0.3), dt), dt);
        let expected =
            naive_sdpa(&q, &k, &v, nqh, nkh, hd, n_kv, kv_stride, 0, 0, false, 0.0, scale);
        TestSetup::new(ffai_sdpa_decode_d512::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::zeros("out", nqh * hd, dt))
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", hpg as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(nqh as u32, 1, 1, [512, 1, 1])
    }
}

// ── Bench specs ──────────────────────────────────────────────────────────

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        ffai_sdpa_decode,
        ffai_sdpa_decode_d64,
        ffai_sdpa_decode_d96,
        ffai_sdpa_decode_d256,
        ffai_sdpa_decode_d512,
    };

    #[bench(name = "ffai/sdpa_decode_d64", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_d64(dt: DType) -> BenchSetup {
        let (nqh, nkh, hd) = (32usize, 8usize, 64usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let bytes = (2 * nqh * hd + 2 * nkh * n_kv * hd) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode_d64::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", nqh * hd, dt))
            .buffer(BenchBuffer::random("k", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::random("v", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::zeros("out", nqh * hd, dt).output())
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", (nqh / nkh) as u32)
            .constexpr("has_sink", 0u32)
            .constexpr("sink_logit", 0.0f32)
            .constexpr("scale", 1.0f32 / (hd as f32).sqrt())
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/sdpa_decode_d96", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_d96(dt: DType) -> BenchSetup {
        let (nqh, nkh, hd) = (32usize, 8usize, 96usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let bytes = (2 * nqh * hd + 2 * nkh * n_kv * hd) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode_d96::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", nqh * hd, dt))
            .buffer(BenchBuffer::random("k", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::random("v", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::zeros("out", nqh * hd, dt).output())
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", (nqh / nkh) as u32)
            .constexpr("scale", 1.0f32 / (hd as f32).sqrt())
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/sdpa_decode", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode(dt: DType) -> BenchSetup {
        let (nqh, nkh, hd) = (32usize, 8usize, 128usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let bytes = (nqh * hd + 2 * nkh * n_kv * hd + nqh * hd) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", nqh * hd, dt))
            .buffer(BenchBuffer::random("k", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::random("v", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::zeros("out", nqh * hd, dt).output())
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", (nqh / nkh) as u32)
            .constexpr("sink_end", 0u32)
            .constexpr("window_start", 0u32)
            .constexpr("has_sink", 0u32)
            .constexpr("sink_logit", 0.0f32)
            .constexpr("scale", 1.0f32 / (hd as f32).sqrt())
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/sdpa_decode_d256", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_d256(dt: DType) -> BenchSetup {
        let (nqh, nkh, hd) = (32usize, 8usize, 256usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let bytes = (2 * nqh * hd + 2 * nkh * n_kv * hd) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode_d256::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", nqh * hd, dt))
            .buffer(BenchBuffer::random("k", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::random("v", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::zeros("out", nqh * hd, dt).output())
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", (nqh / nkh) as u32)
            .constexpr("has_sink", 0u32)
            .constexpr("sink_logit", 0.0f32)
            .constexpr("scale", 1.0f32 / (hd as f32).sqrt())
            .grid_3d(nqh as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/sdpa_decode_d512", dtypes = [f32, f16, bf16])]
    fn bench_sdpa_decode_d512(dt: DType) -> BenchSetup {
        let (nqh, nkh, hd) = (32usize, 8usize, 512usize);
        let (n_kv, kv_stride) = (4096usize, 4096usize);
        let bytes = (2 * nqh * hd + 2 * nkh * n_kv * hd) * dt.size_bytes();
        BenchSetup::new(ffai_sdpa_decode_d512::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", nqh * hd, dt))
            .buffer(BenchBuffer::random("k", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::random("v", nkh * kv_stride * hd, dt))
            .buffer(BenchBuffer::zeros("out", nqh * hd, dt).output())
            .constexpr("head_dim", hd as u32)
            .constexpr("n_kv", n_kv as u32)
            .constexpr("kv_stride", kv_stride as u32)
            .constexpr("heads_per_group", (nqh / nkh) as u32)
            .constexpr("scale", 1.0f32 / (hd as f32).sqrt())
            .grid_3d(nqh as u32, 1, 1, [512, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
