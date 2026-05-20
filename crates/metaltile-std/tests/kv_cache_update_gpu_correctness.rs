//! End-to-end correctness test for `ffai::kv_cache_update` on real Metal.
//!
//! Grid3D kernel — one thread per `(head, d)` output element. Writes
//! `out[h, position, d] = src[h, d]`. No simdgroup arithmetic, so this
//! test is more about pinning the indexing math than the dispatch
//! invariants (which are trivial here). Still worth: a wrong index
//! formula would silently smear the cache.
//!
//! Coverage rationale: `kv_cache_update` had its body silently emptied
//! by PR #19's macro refactor (restored in this PR). It has no
//! `BenchDispatch` variant, so `tile bench` can't exercise it — this
//! GPU correctness test is the only thing standing between a future
//! regression and a silently-broken KV cache.
//!
//! macOS-gated.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{gpu_lock, pack_bytes, unpack_bytes, Dt};
use metaltile_core::{dtype::DType, ir::KernelMode};
use metaltile_runtime::Context;
use metaltile_std::ffai::kv_cache::kv_cache_update;

fn f32_slice_to_bytes(vals: &[f32]) -> Vec<u8> { pack_bytes(vals, Dt::F32) }
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> { unpack_bytes(bytes, Dt::F32) }

#[test]
fn kv_cache_update_writes_to_correct_slot_f32() {
    let _g = gpu_lock();

    // 4 heads × 8 capacity × 16 head_dim. Write src into position=3.
    let n_kv_heads = 4usize;
    let head_dim = 16usize;
    let max_seq = 8usize;
    let position = 3usize;

    // Pre-fill the cache with a recognizable sentinel so we can tell
    // exactly which slots were written. The kernel must touch ONLY
    // `[*, position, *]`, leaving every other slot at the sentinel.
    let sentinel = 999.0_f32;
    let cache = vec![sentinel; n_kv_heads * max_seq * head_dim];

    // src: each (head, d) has a recognizable value.
    let src: Vec<f32> = (0..n_kv_heads * head_dim).map(|i| 10.0 + i as f32).collect();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), f32_slice_to_bytes(&src));
    buffers.insert("out".into(), f32_slice_to_bytes(&cache));
    buffers.insert("head_dim".into(), (head_dim as u32).to_le_bytes().to_vec());
    buffers.insert("max_seq".into(), (max_seq as u32).to_le_bytes().to_vec());
    buffers.insert("position".into(), (position as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new should succeed on macOS");
    let mut kernel = kv_cache_update::kernel_ir_for(DType::F32);
    kernel.mode = KernelMode::Grid3D;

    // Grid3D: total threads = grid_groups.x × tg.x (dispatchThreadgroups
    // semantics). For N legitimate threads we want grid=[1,…] tg=[N,…].
    let total_threads = n_kv_heads * head_dim;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [total_threads, 1, 1])
        .expect("dispatch_with_grid should succeed");

    let out_bytes = result.outputs.get("out").expect("`out` buffer in dispatch result");
    let actual = bytes_to_f32_vec(out_bytes);

    // Verify written slots match src.
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let cache_idx = h * max_seq * head_dim + position * head_dim + d;
            let expected_val = src[h * head_dim + d];
            assert!(
                (actual[cache_idx] - expected_val).abs() < 1e-6,
                "cache[h={h}, pos={position}, d={d}] = {} (expected {})",
                actual[cache_idx],
                expected_val,
            );
        }
    }

    // Verify untouched slots still hold the sentinel.
    for h in 0..n_kv_heads {
        for p in 0..max_seq {
            if p == position {
                continue;
            }
            for d in 0..head_dim {
                let cache_idx = h * max_seq * head_dim + p * head_dim + d;
                assert!(
                    (actual[cache_idx] - sentinel).abs() < 1e-6,
                    "cache[h={h}, pos={p}, d={d}] = {} (should be sentinel {})",
                    actual[cache_idx],
                    sentinel,
                );
            }
        }
    }
}
