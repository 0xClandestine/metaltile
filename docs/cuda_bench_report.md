# metaltile CUDA — kernel latency profile (GB10 / sm_121)

Generated from the registered `#[test_kernel]` corpus, timed on a real DGX-Spark-class GB10 via CUDA events (`cuEvent`), 5 warmup + 50 timed launches each.

## Summary

| | |
|---|---|
| Device | NVIDIA GB10, compute capability **sm_121** |
| Backend | metaltile CUDA (NVRTC → PTX → driver JIT) |
| Kernels timed | **4164** (kernel × dtype), 0 skipped |
| Correctness | 100% — all 4164 bit-accurate vs the CPU oracle (separate run) |
| Timing | GPU-side `cuEvent` wall-clock, min = steady state |
| Launch-overhead floor | ~2.1 µs (lightest elementwise kernel) |

> **Read this as latency, not throughput.** The corpus uses correctness-sized tensors (a few KB), so every kernel floors at the ~2.4 µs launch overhead and the GB/s figure is launch-bound noise. The honest signal is **min µs** (per-launch latency). A true throughput sweep needs the large-input bench harness (Phase 6 CLI wiring); this run proves the cuEvent timing path end-to-end on the full corpus.

> **MMA caveat.** Cooperative-matmul ops (qmm / moe / attention) are still *software-emulated* on CUDA — bit-accurate but slow. Their latency below is the emulation cost and marks the targets for the real `mma.sync` path.

## Heaviest 30 kernels by latency

Worst dtype per kernel shown (latency is dtype-insensitive here — the emulated paths dominate).

| # | kernel | dtype | min µs | cv % |
|---:|---|:---:|---:|---:|
| 1 | `test_ffai_sdpa_bidirectional_d64_vision_tower` | f32 | 1599.7 | 3.3 |
| 2 | `test_indexer_score_dsv4` | f16 | 866.5 | 0.1 |
| 3 | `test_nvfp4_moe_gather_qmm_bm64_mpp` | f32 | 268.4 | 0.2 |
| 4 | `test_mxfp4_moe_gather_qmm_bm64_mpp` | f32 | 268.3 | 0.1 |
| 5 | `test_fp4_f16_moe_gather_qmm_bm64_mpp` | f32 | 266.3 | 0.2 |
| 6 | `test_fp4_moe_gather_qmm_bm64_mpp` | f32 | 266.3 | 0.1 |
| 7 | `test_moe_gather_qmm_mma_int8_bm64_mpp` | f32 | 258.1 | 0.1 |
| 8 | `test_int8_f16_moe_gather_qmm_bm64_mpp` | f32 | 254.3 | 0.3 |
| 9 | `test_moe_gather_qmm_mma_int4_bm64_mpp` | f32 | 254.1 | 0.1 |
| 10 | `test_mxint8_moe_gather_qmm_bm64_mpp` | f32 | 254.0 | 0.1 |
| 11 | `test_int8_moe_gather_qmm_bm64_mpp` | f32 | 253.9 | 0.2 |
| 12 | `test_fp8_e5m2_f16_moe_gather_qmm_bm64_mpp` | f32 | 252.0 | 0.1 |
| 13 | `test_nvfp8_f16_moe_gather_qmm_bm64_mpp` | f32 | 252.0 | 0.1 |
| 14 | `test_nvfp8_moe_gather_qmm_bm64_mpp` | f32 | 252.0 | 0.1 |
| 15 | `test_fp8_e4m3_f16_moe_gather_qmm_bm64_mpp` | f32 | 251.9 | 1.2 |
| 16 | `test_fp8_e4m3_moe_gather_qmm_bm64_mpp` | f32 | 251.9 | 0.2 |
| 17 | `test_fp8_e5m2_moe_gather_qmm_bm64_mpp` | f32 | 251.9 | 0.2 |
| 18 | `test_mxfp8_e5m2_moe_gather_qmm_bm64_mpp` | f32 | 251.6 | 0.1 |
| 19 | `test_mxfp8_e4m3_moe_gather_qmm_bm64_mpp` | f32 | 250.6 | 0.1 |
| 20 | `test_int2_f16_moe_gather_qmm_bm64_mpp` | f32 | 250.1 | 0.3 |
| 21 | `test_int2_moe_gather_qmm_bm64_mpp` | f32 | 250.1 | 0.2 |
| 22 | `test_int4_f16_moe_gather_qmm_bm64_mpp` | f32 | 250.1 | 0.2 |
| 23 | `test_mxint2_moe_gather_qmm_bm64_mpp` | f32 | 250.0 | 0.3 |
| 24 | `test_mxint4_moe_gather_qmm_bm64_mpp` | f32 | 250.0 | 0.7 |
| 25 | `test_int4_moe_gather_qmm_bm64_mpp` | f32 | 249.9 | 0.1 |
| 26 | `test_mxint6_moe_gather_qmm_bm64_mpp` | f32 | 249.4 | 0.7 |
| 27 | `test_int6_moe_gather_qmm_bm64_mpp` | f32 | 249.2 | 0.1 |
| 28 | `test_mxint5_moe_gather_qmm_bm64_mpp` | f32 | 248.4 | 0.1 |
| 29 | `test_int5_f16_moe_gather_qmm_bm64_mpp` | f32 | 248.3 | 0.2 |
| 30 | `test_int5_moe_gather_qmm_bm64_mpp` | f32 | 248.3 | 0.3 |

## Interpretation

- **Vision-tower SDPA (~1600 µs)** and **`indexer_score_dsv4` (~866 µs)** are the latency outliers — large attention/scoring kernels run as full software-emulated cooperative matmul.
- The **`moe_gather_qmm_*_bm64_mpp` family (~255 µs)** is the bulk of the tail: every quant format (fp4/nvfp4/mxfp4/int8/int4) emulated through the MPP cooperative-tile path. Single biggest win from a real `mma.sync` lowering.
- Everything below ~20 µs is launch/elementwise-bound and already fine.
- cv% is mostly <3% → timing is stable and trustworthy.

## Reproduce

```sh
cargo test -p metaltile-std --features cuda --test cuda_bench_corpus \
    -- --ignored --nocapture
```

_Full 4164-row table available on request; this report shows the actionable tail._
