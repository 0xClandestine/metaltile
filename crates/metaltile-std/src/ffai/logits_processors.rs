//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Logits-processor kernels for the sampling pipeline.
//!
//! Decode-form samplers (other than the bare-softmax fused
//! `softmax_categorical_sample`) compose a small set of in-place
//! transforms on the logits vector before the final categorical
//! draw. Pipeline shape: temperature → repetition penalty → top-k →
//! top-p (nucleus) → categorical sample. This file ships kernels 1
//! and 2 (temperature, repetition penalty); top-k / top-p require a
//! sort or quickselect pass and live in a follow-up.
//!
//! Semantic contracts:
//!
//!   - **temperature**: `logits[i] /= temperature` (no-op at 1.0;
//!     small T sharpens toward argmax). Caller clamps to a positive
//!     floor before dispatch.
//!   - **repetition penalty**: for each token id in `token_ids`,
//!     `v > 0 → v /= penalty`, `v ≤ 0 → v *= penalty`. Matches the
//!     HuggingFace `transformers.LogitsProcessorList` and vLLM
//!     conventions. `penalty == 1.0` is a no-op.
//!
//! Top-k and top-p require a sort or quickselect pass — they live
//! in a follow-up kernel since the sort dispatch geometry doesn't
//! fit the simple one-thread-per-element shape these two use.
//!
//! Generic over T; all values are upcast to f32 internally so f16/bf16
//! logits accumulate cleanly across the scale and don't drift on the
//! repeated-token gather. Output dtype matches input dtype.

use metaltile::kernel;

// ── Temperature scaling ───────────────────────────────────────────────────
//
// Pure elementwise `logits[i] /= temperature`. Generic-T, one thread per
// vocab position. At `temperature == 1.0` this is a copy; at very small
// temperature it sharpens the distribution toward greedy argmax (the
// downstream `softmax_categorical_sample` handles the softmax itself).
//
// Caller contract: `temperature > 0`. A zero or negative temperature
// produces inf / sign-flipped logits — callers should clamp before
// dispatch (`max(temperature, 1e-5)` is the standard guard).
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D.** One thread per vocab position.
// - **Grid: `[ceil(n / TPG), 1, 1]`, TG: `[TPG, 1, 1]`** (TPG = 256 is the
//   tested geometry; any value works since the kernel is pure elementwise
//   and uses no `threadgroup_*` / `simd_*` cooperation).
// - **`n = grid.x * tg.x`** — the caller is responsible for `n` covering
//   the full logits length. Threads with `program_id::<0>() >= n` would
//   read/write out of bounds; the runtime should size the dispatch so the
//   total thread count exactly matches the logits length.
#[kernel]
pub fn logits_temperature<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] temperature: f32) {
    let i = program_id::<0>();
    let inv_t = 1.0f32 / temperature;
    let v = load(inp[i]).cast::<f32>();
    store(out[i], (v * inv_t).cast::<T>());
}

// ── Repetition penalty ────────────────────────────────────────────────────
//
// In-place mutate the logits at every position appearing in `token_ids`,
// scaling toward 0 to discourage repeats. Convention matches HuggingFace
// `transformers.LogitsProcessorList`:
//
//   for tok in token_ids:
//       if logits[tok] > 0: logits[tok] /= penalty
//       else:               logits[tok] *= penalty
//
// `penalty == 1.0` is a no-op; `penalty > 1.0` discourages repeats;
// `penalty < 1.0` encourages repeats (rare).
//
// Dispatch: one thread per `token_ids` entry. The kernel reads
// `logits[token_ids[i]]`, updates, and writes back. With duplicate
// token ids the operation is **idempotent in expectation but
// non-deterministic in order** — multiple threads racing on the same
// vocab slot pick a write order. Callers MUST dedupe `token_ids` before
// dispatch (or accept the last-writer-wins semantics, which matches
// what a sequential CPU pass produces *only* on a deduped input).
//
// ## DISPATCH INVARIANTS
//
// - **Mode: Grid3D.** One thread per `token_ids` entry.
// - **Grid / TG: `grid.x * tg.x == token_ids.len()`** — caller must size
//   the dispatch to exactly the token-id count. TPG = 256 (or smaller for
//   small contexts) is the tested geometry.
// - **No `threadgroup_*` / `simd_*` cooperation** — every thread is
//   independent. The only invariant is the dedupe contract above.
#[kernel]
pub fn logits_repetition_penalty<T>(
    mut logits: Tensor<T>,
    token_ids: Tensor<u32>,
    #[constexpr] penalty: f32,
) {
    let i = program_id::<0>();
    let tok = load(token_ids[i]);
    let v = load(logits[tok]).cast::<f32>();
    let scaled = select(v > 0.0f32, v / penalty, v * penalty);
    store(logits[tok], scaled.cast::<T>());
}

pub mod kernel_tests {
    #![allow(unused, dead_code, clippy::too_many_arguments)]

use metaltile_core::{
        DType,
        bench::{TestBuffer, TestSetup},
    };
    use metaltile_macros::test_kernel;

    use super::*;

    fn pack(vals: &[f32], dt: DType) -> Vec<u8> {
        match dt {
            DType::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
            DType::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
            DType::BF16 =>
                vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
            _ => panic!("unsupported dtype {dt:?}"),
        }
    }

    fn pack_u32(vals: &[u32]) -> Vec<u8> { bytemuck::cast_slice::<u32, u8>(vals).to_vec() }

    // ── logits_temperature ────────────────────────────────────────────────

    /// CPU oracle for temperature: `logit /= temperature`.
    fn cpu_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        logits.iter().map(|&v| v / temperature).collect()
    }

    #[test_kernel(name = "logits/temperature_identity", dtypes = [f32, f16, bf16], tol = 1e-4)]
    fn test_temperature_identity(dt: DType) -> TestSetup {
        let n = 1024usize;
        let temperature = 1.0f32;
        let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() - 0.5).collect();
        let expected = cpu_temperature(&logits, temperature);

        let mut k = logits_temperature::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec(
                "temperature",
                temperature.to_le_bytes().to_vec(),
                DType::F32,
            ))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    #[test_kernel(name = "logits/temperature_half", dtypes = [f32], tol = 1e-4)]
    fn test_temperature_half(dt: DType) -> TestSetup {
        let n = 256usize;
        let temperature = 2.0f32;
        let logits: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 64.0).collect();
        let expected = cpu_temperature(&logits, temperature);

        let mut k = logits_temperature::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("inp", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec(
                "temperature",
                temperature.to_le_bytes().to_vec(),
                DType::F32,
            ))
            .expect(TestBuffer::from_vec("out", pack(&expected, dt), dt))
            .grid_1d(n, 256)
    }

    // ── logits_repetition_penalty ─────────────────────────────────────────

    /// CPU oracle for repetition penalty.
    fn cpu_repetition_penalty(logits: &[f32], token_ids: &[u32], penalty: f32) -> Vec<f32> {
        let mut out = logits.to_vec();
        for &tok in token_ids {
            let i = tok as usize;
            let v = out[i];
            out[i] = if v > 0.0 { v / penalty } else { v * penalty };
        }
        out
    }

    #[test_kernel(name = "logits/repetition_penalty_noop", dtypes = [f32], tol = 1e-5)]
    fn test_repetition_penalty_noop(dt: DType) -> TestSetup {
        let n = 256usize;
        let penalty = 1.0f32;
        let logits: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 12.0).collect();
        let token_ids: Vec<u32> = vec![3, 7, 11, 137, 200];
        let expected = cpu_repetition_penalty(&logits, &token_ids, penalty);
        let n_tokens = token_ids.len();

        let mut k = logits_repetition_penalty::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("logits", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("token_ids", pack_u32(&token_ids), DType::U32))
            .input(TestBuffer::from_vec("penalty", penalty.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("logits", pack(&expected, dt), dt))
            .grid_1d(n_tokens, 256)
    }

    #[test_kernel(name = "logits/repetition_penalty_positive", dtypes = [f32], tol = 1e-4)]
    fn test_repetition_penalty_positive(dt: DType) -> TestSetup {
        let n = 256usize;
        let penalty = 2.0f32;
        let logits: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let token_ids: Vec<u32> = vec![0, 5, 100, 255];
        let expected = cpu_repetition_penalty(&logits, &token_ids, penalty);
        let n_tokens = token_ids.len();

        let mut k = logits_repetition_penalty::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("logits", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("token_ids", pack_u32(&token_ids), DType::U32))
            .input(TestBuffer::from_vec("penalty", penalty.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("logits", pack(&expected, dt), dt))
            .grid_1d(n_tokens, 256)
    }

    #[test_kernel(name = "logits/repetition_penalty_mixed_signs", dtypes = [f32], tol = 1e-4)]
    fn test_repetition_penalty_mixed_signs(dt: DType) -> TestSetup {
        let n = 1024usize;
        let penalty = 1.3f32;
        let logits: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin() * 5.0 - 0.5).collect();
        let token_ids: Vec<u32> = vec![7, 42, 137, 251, 513, 999];
        let expected = cpu_repetition_penalty(&logits, &token_ids, penalty);
        let n_tokens = token_ids.len();

        let mut k = logits_repetition_penalty::kernel_ir_for(dt);
        k.mode = metaltile_core::ir::KernelMode::Grid3D;

        TestSetup::new(k)
            .input(TestBuffer::from_vec("logits", pack(&logits, dt), dt))
            .input(TestBuffer::from_vec("token_ids", pack_u32(&token_ids), DType::U32))
            .input(TestBuffer::from_vec("penalty", penalty.to_le_bytes().to_vec(), DType::F32))
            .expect(TestBuffer::from_vec("logits", pack(&expected, dt), dt))
            .grid_1d(n_tokens, 256)
    }
}
