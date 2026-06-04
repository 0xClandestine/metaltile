//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GPU smoke test for the `Op::SimdgroupLoad` HW intrinsic.
//!
//! The kernel under test (`mt_sgload_smoke` in `mlx::sgload_smoke`)
//! stages a flat-64 `src` into TG memory, runs **one** MSL
//! `simdgroup_load(...)` instruction to land it into a
//! `simdgroup_matrix<T, 8, 8>` fragment, then scatters the fragment
//! back into `dst`. With no math in between, the round-trip should be
//! byte-exact for `f32` and `f16`.
//!
//! What this file pins:
//!
//!   1. **MSL emit** — the generated MSL contains an actual
//!      `simdgroup_load(` call site. If the codegen for
//!      `Op::SimdgroupLoad` ever silently drops to a no-op or to a
//!      different MSL primitive, this guard catches it without
//!      needing to read the disassembly.
//!
//!   2. **GPU round-trip correctness** — the kernel runs on real
//!      Metal and the output equals the input. Pins parser → IR →
//!      passes → codegen → runtime for the HW intrinsic end-to-end.
//!
//! macOS-gated (Metal-only).

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile::{
    codegen::msl::{MslConfig, MslGenerator},
    core::dtype::DType,
};
use metaltile_runtime::Context;
use metaltile_std::mlx::sgload_smoke::mt_sgload_smoke;

/// Sanity-check that the MSL the codegen produces actually contains
/// a `simdgroup_load(...)` call — this is the whole point of having
/// a smoke kernel for the primitive.
#[test]
fn mt_sgload_smoke_emits_simdgroup_load_instruction_f32() {
    let kernel = mt_sgload_smoke::kernel_ir_for(DType::F32);
    let msl_gen = MslGenerator::new(MslConfig::default());
    let msl = msl_gen.generate(&kernel).expect("MSL emit should succeed");

    // 1× the HW intrinsic call site — anything more would mean
    // codegen accidentally cloned the op; anything less means the
    // emit path was bypassed.
    let n_calls = msl.matches("simdgroup_load(").count();
    assert_eq!(
        n_calls, 1,
        "expected exactly one `simdgroup_load(` call site in emitted MSL, got {n_calls}.\n\
         --- MSL dump ---\n{msl}\n--- end MSL ---",
    );

    // Spot-check the call site has the expected argument shape:
    //   simdgroup_load(<frag>, &tg_tile[<off>], 8, ulong2(0, 0), false);
    // We don't pin the value-name (`frag` is rewritten by codegen)
    // but the TG-name + stride + ulong2 + transpose flag are stable.
    assert!(
        msl.contains("&tg_tile["),
        "expected `&tg_tile[<off>]` in simdgroup_load call.\n--- MSL ---\n{msl}",
    );
    assert!(
        msl.contains(", 8, ulong2(0, 0), false);"),
        "expected `, 8, ulong2(0, 0), false);` (stride=8 row, origin (0,0), transpose=false) \
         in simdgroup_load call.\n--- MSL ---\n{msl}",
    );
}

/// Same MSL guard, for f16. The dtype only changes the
/// `simdgroup_matrix<half, 8, 8>` declaration; the `simdgroup_load(...)`
/// call itself is dtype-agnostic. Keeping the assertion ensures the
/// dtype-specialisation path of `kernel_ir_for` doesn't accidentally
/// drop the primitive.
#[test]
fn mt_sgload_smoke_emits_simdgroup_load_instruction_f16() {
    let kernel = mt_sgload_smoke::kernel_ir_for(DType::F16);
    let msl_gen = MslGenerator::new(MslConfig::default());
    let msl = msl_gen.generate(&kernel).expect("MSL emit should succeed");

    assert_eq!(
        msl.matches("simdgroup_load(").count(),
        1,
        "expected exactly one `simdgroup_load(` call site in f16 MSL:\n{msl}",
    );
}

/// Round-trip an 8×8 tile of `src` values through the HW intrinsic
/// and back. With no math in between, the output must equal the
/// input exactly — bit-exact for f32, bit-exact for f16 since the
/// values themselves never leave their native dtype.
fn run_round_trip(dtype: Dt) {
    let _g = gpu_lock();

    // Use distinct, non-trivial values so any layout-permutation bug
    // surfaces immediately (a zero buffer would happily pass even
    // if the frag-load wrote nothing). Magnitudes ≈ [0, 8) — well
    // inside both f32 and f16 finite range with no rounding loss.
    let src_f32: Vec<f32> = (0..64).map(|i| 0.125 * (i as f32) + 0.5).collect();
    // Round-trip through the dtype so the CPU-side oracle matches
    // what the GPU loads — no-op for f32, lossless quantisation
    // here for f16 (the chosen magnitudes are representable).
    let src: Vec<f32> = src_f32.iter().map(|&v| dtype.round(v)).collect();

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("src".into(), pack_bytes(&src, dtype));
    buffers.insert("dst".into(), vec![0u8; 64 * dtype.bytes()]);

    let ctx = Context::new().expect("Context::new on macOS");
    let kernel = mt_sgload_smoke::kernel_ir_for(dtype.to_dtype());

    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [1, 1, 1], [32, 1, 1])
        .expect("dispatch_with_grid");

    let dst = unpack_bytes(result.outputs.get("dst").expect("dst buffer"), dtype);
    assert_eq!(dst.len(), 64);

    for i in 0..64 {
        assert_eq!(
            dst[i].to_bits(),
            src[i].to_bits(),
            "round-trip mismatch at i={i} (dtype={:?}): src={} dst={}",
            dtype.to_dtype(),
            src[i],
            dst[i],
        );
    }
}

#[test]
fn mt_sgload_smoke_round_trips_8x8_tile_f32() { run_round_trip(Dt::F32); }

#[test]
fn mt_sgload_smoke_round_trips_8x8_tile_f16() { run_round_trip(Dt::F16); }
