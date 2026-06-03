//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Block-scaled quantization formats (nvfp4 / mxfp4 / mxfp8 / nvfp8).
//!
//! [`codec`] holds the bit-level element + scale encodings — the single source
//! of truth shared by the host-side weight packer, the CPU correctness oracle,
//! and the math the generated Metal kernels emit. See
//! `specs/BENCH_METRICS_SPEC.md` Appendix B for the format roadmap.

pub mod codec;
pub mod format;
