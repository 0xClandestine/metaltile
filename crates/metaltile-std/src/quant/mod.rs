//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Quantization formats — the registry of every precision metaltile supports.
//!
//! [`codec`] holds the bit-level element + scale encodings — the **single source
//! of truth** shared by the host-side weight packer, the CPU correctness oracle,
//! and the math the generated Metal kernels emit. Every format family below
//! decodes through it, so a kernel that disagrees with its oracle is a real bug,
//! not an oracle re-derivation that could share a mistake.
//!
//! The families are split by *layout provenance* — each owns the layout its
//! ecosystem dictates, but they share `codec`:
//!
//! | family | module | provenance | layout |
//! |--------|--------|------------|--------|
//! | OCP / NVIDIA / legacy / symmetric-int / MXINT / fp16-twin | [`format`] ([`QFormat`](format::QFormat)) | metaltile's own block-scaled formats (nvfp4, mxfp4, mxfp8, nvfp8, int2–8, mxint*, …) | planar, symmetric (`element·block_scale·global`) |
//! | GGUF (`ggml`) | [`gguf`] ([`GgufFormat`](gguf::GgufFormat)) | llama.cpp super-block k-quants (q8_0, q2_k, …) | super-block, asymmetric, host-decomposed |
//! | MLX affine | `mlx/quantized.rs` | MLX checkpoint interop (asymmetric int w/ zero-point) | group, affine |
//!
//! See `specs/BENCH_METRICS_SPEC.md` Appendix B for the format roadmap and
//! `specs/KERNEL_AUDIT.md` for per-op precision coverage.

pub mod codec;
pub mod format;
pub mod gguf;
