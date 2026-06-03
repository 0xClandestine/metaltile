# Bench Metrics & Kernel-Optimization Spec

**Status:** ✅ Implemented (Phases 1–4: latency µs, GFLOP/s, roofline %-of-peak +
arithmetic intensity, and the bottleneck verdict). Precision roadmap (Appendix B)
is tracked separately.
**Captured:** 2026-05-31
**Context:** follow-up to the MLX A/B comparison work on `ek/delete-legacy-gpu-tests` (PR #240)

---

## 1. Motivation

`tile bench` today reports **GB/s** (each kernel's `bytes_moved ÷ time`) as its headline metric. That is a *bandwidth-efficiency* number, and it is **not sufficient to optimize kernels or to compare precisions**:

- **No wall-clock latency.** GB/s is bytes÷time; two kernels doing the *same* logical work but moving *different* bytes (e.g. int4 vs int8 vs f16 weights) have non-comparable GB/s. You cannot read "which precision is fastest" off the table.
- **No compute throughput.** There is no GFLOP/s, so compute-bound kernels (matmul, attention, conv) can't be judged against the hardware's compute ceiling.
- **No utilization / roofline.** Nothing tells you whether a kernel is *saturating* memory bandwidth or compute — i.e. whether it's already optimal or leaving the GPU idle.
- **Bottleneck is hidden by default.** A coarse bottleneck label + occupancy/registers exist but only under `-vv`.

Concrete example that motivated this (decode/gemv, N=K=4096, M1 Max):

| kernel | weights | MT GB/s |
|---|---|---|
| `gemv` (f16 dense) | 33.6 MB | 530 |
| `qmv` (int4) | 8.4 MB | 380 |
| `qmv_b8` (int8) | 16.8 MB | ~60 ⚠️ |

GB/s alone can't tell you the int4 path is *fastest* (¼ the bytes), nor that the int8 path is **badly under-utilized** (~60 GB/s ≈ 15 % of peak) — both require latency + % -of-peak.

## 2. Goals / Non-goals

**Goals**
- Add the measurements needed to (a) find a kernel's bottleneck and (b) tell whether the GPU is being used efficiently.
- Make "which precision/variant is fastest" directly readable from a bench run.
- **Additive only** — do not remove or change the existing GB/s / ref-vs-MT / correctness columns or the JSON fields already consumed by `baselines/*.json` diffing.

**Non-goals (this spec)**
- Implementing new precisions/kernels (see Appendix B roadmap).
- Changing the A/B correctness mechanism (only adding perf metrics).

## 3. Current state (what already exists)

- **`BenchStats`** (`crates/metaltile-std/src/stats.rs`) already captures per-kernel timing: `min_us`, `mean_us`, `median_us`, `p95_us`, `p99_us`, `stddev_us`, `cv_pct`. **Latency is measured but not surfaced** in the default table or JSON.
- **`OpResult`** (`crates/metaltile-std/src/bench_types.rs`) carries `ref_perf`/`mt_perf` (GB/s), `equiv`, and optional `mt_timing`/`ref_timing` (`BenchStats`).
- **`run_kernel_bench`** (`crates/metaltile-std/src/run_kernel.rs`) computes `gbps` from `bench_gbps(...)` and **discards the `BenchStats`** (`let (gbps, _stats) = ...`). `bytes_moved` comes from `BenchSetup::compute_bytes_moved()` (sum of buffer sizes unless overridden via `.bytes_moved()`).
- **`-vv` profile** (`compute_profiles` in `crates/metaltile-cli/src/cmd/bench.rs`) already shows `occ%`, `regs`, and a coarse `bottleneck` label (e.g. "register-limited"), plus `p95/p99/cv%`.
- **JSON** (`tile bench --json`) emits only `{op, shape, metric, ref, mt}` — consumed by baseline diffing; must stay backward-compatible.

## 4. Proposed additions

All additive. New fields default to `None`/absent so untouched kernels and the baseline-diff schema are unaffected.

### 4.1 Wall-clock latency (µs) — *Phase 1, low risk*
- Surface `BenchStats.min_us` (primary) in the default table and JSON; keep `mean/median/p95/p99/cv` available under `-v`/`-vv`.
- `run_kernel_bench` must **stop discarding** the `BenchStats` and thread it into the `OpResult` (the `mt_timing`/`ref_timing` slots already exist on `result_sub_timed`).
- Display: add a `MT(µs)` column (and `Ref(µs)` when a reference is present). This is the metric that makes cross-precision "fastest" readable.

### 4.2 Compute throughput (GFLOP/s) — *Phase 2*
- Add `flops: Option<u64>` to `BenchSetup` + a `.flops(n)` builder (mirrors `.bytes_moved()`). Default `None`.
- `to_gflops(stats, flops)` already exists in `runner.rs` — wire it through.
- Annotate the compute-heavy kernels with their FLOP counts: matmul/gemv (`2·M·N·K`), attention (QKᵀ + softmax + ·V), conv. Memory-bound elementwise/reduction can leave it unset (GFLOP/s blank).
- Display: `GFLOP/s` column, shown when `flops` is set.

### 4.3 Roofline / utilization — *Phase 3*
- **Device spec table**: `DeviceSpecs { peak_bw_gbps, peak_f32_tflops, peak_f16_tflops, na_f16_tflops?, na_int8_tops?, ane_tops? }` looked up by `runner.device_name`. Seed with M1/M2/M3/M4 Max and **M5/M5 Max** (the M5 adds per-GPU-core Neural Accelerators — see Appendix C). Unknown device → utilization columns blank (don't fail).
- **% peak bandwidth** = `GB/s ÷ peak_bw_gbps` — tells you if a memory-bound kernel is saturating DRAM.
- **% peak compute** = `GFLOP/s ÷ peak_*_tflops` (pick the dtype/engine: SIMD f16/f32, or the M5 NA path) — tells you if a compute-bound kernel is saturating the ALUs/accelerators.
- **Arithmetic intensity** = `flops ÷ bytes_moved` (FLOPs/byte) — places the kernel on the roofline (left of the ridge ⇒ memory-bound; right ⇒ compute-bound).

### 4.4 Bottleneck classification surfaced by default — *Phase 4*
- Combine roofline position (AI vs ridge point) with the existing occupancy/register signals into one **`bottleneck`** verdict shown by default: `memory-bound` / `compute-bound` / `occupancy-limited` / `register-limited` / `latency-bound`.
- Keep the finer `-vv` breakdown (occ%, regs, cv%).

## 5. Data-model & display summary

| Layer | Change |
|---|---|
| `BenchSetup` | `+ flops: Option<u64>`, `+ .flops()` builder |
| `run_kernel_bench` | thread `BenchStats` into `OpResult` (stop discarding); compute gflops/util when data present |
| `OpResult` | `+ mt_us`/`ref_us` (from `BenchStats.min_us`), `+ gflops`, `+ pct_peak_bw`, `+ pct_peak_flops`, `+ arith_intensity`, `+ bottleneck` |
| Device specs | new `DeviceSpecs` lookup by device name |
| Table | default: add `MT(µs)`, `GFLOP/s`, `%BW`/`%FLOP`, `bottleneck`; keep GB/s + `ok`. `-v`/`-vv` keep the existing profile detail |
| JSON | **add** `latency_us`, `gflops`, `pct_peak_bw`, `pct_peak_flops`, `arith_intensity`, `bottleneck`; **keep** `ref`/`mt` (GB/s) for baseline-diff compatibility |

## 6. Implementation plan (phased, each independently shippable)

1. **Phase 1 — Latency.** Thread `BenchStats` through `run_kernel_bench` → `OpResult`; add `µs` column + JSON `latency_us`. (Data already measured; smallest, highest-value change.)
2. **Phase 2 — GFLOP/s.** `BenchSetup.flops` + `.flops()`; annotate matmul/gemv/attention/conv; add column + JSON.
3. **Phase 3 — Roofline.** `DeviceSpecs` table; `%peak BW`, `%peak compute`, arithmetic intensity.
4. **Phase 4 — Bottleneck verdict** surfaced by default; fold in occ/regs.

Test each phase: unit-test the metric math (latency→µs, gflops, %peak, AI) with synthetic `BenchStats`; snapshot the table/JSON; verify a known memory-bound kernel reads `memory-bound` and a known compute-bound one reads `compute-bound`.

## 7. Open questions

- Peak-spec source of truth: hard-code per device, or probe at runtime where Metal exposes it? (M5 NA TFLOPS aren't in a Metal API — likely a maintained table.)
- For the NA path on M5, which "peak compute" do we divide by — SIMD f16 or NA f16? Probably both, labeled.
- Default table width: which columns are default vs `-v`? (Proposal: µs + GB/s + GFLOP/s + bottleneck default; %peak + AI under `-v`; occ/regs/cv under `-vv`.)
- JSON consumers beyond `baselines/*.json`? Confirm before finalizing the schema.

---

## Appendix A — Related findings / bugs to fix (surfaced during the MLX A/B work)

1. **Flaky quantized A/B correctness** — the quantized matmul benches seed **random** packed weights, so `qmm_*` A/B comparisons pass/fail nondeterministically (this is the **PR #240 CI "Bench" failure**). Fix: deterministic packed weights (or a justified, dtype-aware tol). *Being fixed alongside this spec.*
2. **int8 gemv under-optimization** — `qmv_b8` hits ~60 GB/s vs int4 `qmv`'s ~380 GB/s on M1 Max. Likely a kernel/codegen inefficiency, not a precision law. Investigate once the latency/%-peak metrics (Phase 1/3) make it measurable.
3. **Widen f32-only tests** — several `#[test_kernel(dtypes=[f32])]` are on *generic `<T>`* kernels (`fft`, `softmax`, `logsumexp`, …) whose kernels already support f16/bf16; widen the tests. Genuinely-typed exceptions (`mt_fp4_quant_dequant` f32, `mt_random_hash` u32) stay as-is.

## Appendix B — Precision-support roadmap (separate, larger effort)

Goal: **support all precisions in every weight-bearing kernel** (matmul/gemv/attention/moe) so a bench reveals the fastest, and **properly support nvfp4 / mxfp4 / mxfp8** (today we have E2M1 fp4 with a *float* scale — good accuracy, but not spec-conformant and not interoperable with real checkpoints).

| format | element | block | block scale | status |
|---|---|---|---|---|
| nvfp4 | E2M1 | 16 | E4M3 + global FP32 | ✅ spec-conformant (`quant::format::Nvfp4`) |
| mxfp4 | E2M1 | 32 | E8M0 (pow-2) | ✅ (`Mxfp4`) |
| mxfp8 | E4M3/E5M2 | 32 | E8M0 | ✅ (`Mxfp8E4`/`Mxfp8E5`) |
| nvfp8 | E4M3 | 16 | per-block FP32 | ✅ (`Nvfp8`) |
| fp4 (legacy) | E2M1 | 32 | per-group FP32 | ✅ (`Fp4`) |
| fp8 (legacy) | E4M3/E5M2 | 32 | per-group FP32 | ✅ (`Fp8E4m3`/`Fp8E5m2`) |
| int8 (symmetric) | int8 | group 64 | per-group FP32 | ✅ (`Int8`) |
| int2–8 affine | int | group 64 | per-group scale+bias | ✅ (qmv/qmm/gather — current, **not** legacy: MLX-checkpoint + KV-cache interop) |

**Status (✅ FULL MATRIX implemented):** spec-conformant block-scaled codecs (`crates/metaltile-std/src/quant/{codec,format}.rs`) — the single source of truth shared by the host packer, the CPU correctness oracle, and the kernels via first-class DSL decode intrinsics (`e2m1_decode`/`e4m3_decode`/`e5m2_decode`/`int8_decode`). **All 9 quant formats** (int8, legacy fp4, legacy fp8 e4m3/e5m2, mxfp4, nvfp4, mxfp8 e4m3/e5m2, nvfp8) are wired across **every weight-bearing family** in fp16/bf16/fp32 activation:

| family | file |
|---|---|
| dequant (standalone) | `mlx/block_scaled_dequant.rs` |
| qgemv | `mlx/block_scaled_matmul.rs` |
| qmm | `mlx/block_scaled_qmm.rs` |
| qmm-MMA (simdgroup) | `mlx/block_scaled_mma.rs` |
| MoE gather-qmm | `mlx/block_scaled_moe.rs` |
| fused RMSNorm+GEMV | `ffai/rms_norm_block_scaled_qgemv.rs` |
| fused gated-RMSNorm+GEMV | `ffai/gated_rms_norm_block_scaled_qgemv.rs` |
| batched-4 qgemv / qmm | `ffai/batched_4_block_scaled_{qgemv,qmm}.rs` |
| batched-Q/K/V qgemv / qmm | `ffai/batched_qkv_block_scaled_{qgemv,qmm}.rs` |
| flash SDPA (block-scaled KV, d64/96/128/256/512) | `ffai/flash_block_scaled_sdpa.rs` |
| embedding gather | `ffai/dequant_gather_block_scaled.rs` |
| qmm via MPP (tensor engine) | `mlx/block_scaled_qmm_mpp.rs` |
| qmm via NAX | `mlx/block_scaled_qmm_nax.rs` |
| MoE gather-qmm via MPP (bm8/bm16/bm64) | `ffai/moe_mpp{,_bm8,_bm64}_block_scaled.rs` |
| expert-indexed GEMV | `ffai/dequant_gemv_expert_indexed_block_scaled.rs` |
| patch embedding (linear projection) | `ffai/patch_embed_block_scaled.rs` |
| patch embedding (simdgroup-MMA) | `ffai/patch_embed_mma_block_scaled.rs` |
| conv2d / conv3d (direct) | `ffai/{conv2d,conv3d}_block_scaled.rs` |
| conv2d / conv3d (im2col simdgroup-MMA) | `ffai/{conv2d,conv3d}_mma_block_scaled.rs` |
| depthwise conv2d | `ffai/depthwise_conv2d_block_scaled.rs` |
| audio conv1d (STT) / fishspeech conv1d (TTS) | `ffai/{audio_conv1d,fishspeech_conv1d}_block_scaled.rs` |

Each (family × format) ships a `#[test_kernel]` CPU-oracle correctness check (1:1, GPU-verified vs `quant::format::dequant`) and a `#[bench]` with `.flops()` so the PR-#1 latency/GFLOP/roofline columns rank precisions side-by-side. `fp8_e4m3` reuses each family's `nvfp8` kernel (identical 8-bit-E4M3 + f32-scale shape). The MPP/NAX/MoE-MPP cooperative-matmul variants dequant W to `coop_stage(T)` during threadgroup staging and reuse the proven int4/int8 `mpp::tensor_ops::matmul2d` dispatch geometry byte-for-byte (no new freeze surface). Block-scaled coverage spans **every quantized weight-bearing op + backend + MoE tile** — all **30 Track-1 formats** (8 spec/legacy float formats + the full symmetric integer matrix `int2-6`/`int8` FP32-group and `mxint2-8` E8M0-block + an **FP16-scale twin** `*_f16` of every FP32-scaled format), GPU-verified on f32/f16/bf16, each 1:1 tested + benched. The integer formats are emitted by a parameterized `(bit-width × scale-kind)` decode macro per family (straddle-aware sub-byte bit-stream extract + float sign-extend) that reuses each family's proven geometry verbatim; the fp16-scale twins clone the FP32-scaled kernels with a `Tensor<f16>` scale read. Flash KV covers every production head dim (d64/96/128/256/512 × all 30 formats) with **no holes** — int8 @ d96 (where the group size 64 doesn't divide 96) tiles with a ragged trailing block (64 + 32), the host packer and kernel rounding up `n_blocks` identically. The MMA patch-embed (`patch_embed_mma_block_scaled.rs`) reuses the dense `patch_embed_mma` geometry + the `conv2d_mma` block-scaled W-dequant.

**The full symmetric integer matrix is in every family above** — `int2/3/4/5/6/8` (FP32 group) + `mxint2/3/4/5/6/8` (E8M0 block) — including the fast tensor-engine paths (simdgroup-MMA, MPP, NAX, MoE-MPP), where integer throughput is highest on Apple GPUs / the ANE and the `mxint*` E8M0 layout maps to tensor-core block-scaling on future NVIDIA / AMD targets. The core matmul / MoE / fused-norm / batched-QKV / KV-cache / attention families *also* carry the pre-existing asymmetric affine (scale+bias) integers for MLX-checkpoint interop. No weight-bearing family lacks an integer path.

### Where block-scaled quantization does **not** apply

Quantization compresses a large persistent *weight/parameter* tensor, so it is only added to ops that read one. The following are intentionally left as activation-precision-only (they already support fp16/bf16/fp32 via the generic `<T>`), because they carry no quantizable weight:

| op class | why no quantization |
|---|---|
| RoPE (`rope_*`) | rotates Q/K *activations* by position; no weight matrix (cos/sin are tiny + precision-sensitive) |
| Gated-DeltaNet / SSM (`gated_delta*`, `ssm_replay`) | state recurrence over activations; the Q/K/V/gate *projections* are separate matmuls (already covered) |
| RMSNorm / gated-RMSNorm (standalone) | the norm-weight is a tiny per-channel vector, precision-sensitive |
| dense SDPA / attention (`sdpa_*`, `scaled_dot_product_attention`, `steel_attention*`) | operate on Q/K/V activations; the quantized-KV path is `flash_block_scaled_sdpa` (covered) |
| dense GEMM (`gemm`, `steel_gemm_*`) | the quantized path is the qmm family (covered); dense stays dense |
| KV-cache *write* (`kv_cache*`) | the dynamic cache is quantized **per decode step** → affine int4/int8 (cheap min/max encode) is the right scheme; block-scaled is a static-weight format whose per-step *encode* is costly + needs GPU encode intrinsics. The block-scaled KV *read* is covered (`flash_block_scaled_sdpa`) for pre-quantized caches. |
| Winograd conv (`winograd_conv`) | the filter is pre-transformed into the Winograd domain (`GgGᵀ`), which strongly amplifies quantization error — quantized Winograd is non-standard and counterproductive |
| elementwise / reduction / softmax / sort / scan / fft / rope / gather-axis / scatter | no persistent parameter tensor |

Quantized **conv + patch-embed** are covered across the family — direct
(`patch_embed`, `conv2d`, `conv3d`, `depthwise_conv2d`, `audio_conv1d`,
`fishspeech_conv1d`) and the implicit-im2col simdgroup-MMA (`patch_embed_mma`,
`conv2d_mma`, `conv3d_mma`) — quantizing the filter / projection weight block-wise
along the contraction (all 30 formats, GPU-verified on f32/f16/bf16).

**Flash KV (closed):** block-scaled flash-SDPA KV now covers every production head
dim — d ∈ {64, 96, 128, 256, 512} × all 30 formats — matching the affine path, with
**no (format × dim) holes**. int8's group size (64) doesn't divide d96, so that case
tiles with a ragged trailing block (`n_blocks = ceil(dim/block_size)`: a 64-block + a
32-block); the host packer and kernel round up identically, so codes + scales stay
self-consistent. The geometry is one simdgroup per query, identical across dims (only
the per-lane dim count changes), so the extension added no freeze surface.

> **Test-gate note:** the `#[test_kernel]` harness (`tests/kernel_tests_harness.rs`)
> must enumerate the registry via `metaltile_std::all_tests()`, not
> `metaltile::harness::registry::all_tests()` — the latter leaves metaltile-std's
> inventory statics dead-code-eliminated from the integration-test link, so the
> harness silently runs **zero** checks (a vacuous gate). With the correct
> accessor it runs ~1910 GPU-correctness checks, all green on f32/f16/bf16.

> **fp4 simdgroup-MMA f32 fix (this PR):** exposing the real harness surfaced two
> intertwined fp4-MMA defects, both now fixed (every fp4 MMA test runs f32/f16/bf16):
> (1) the original `fp_quantized_mma::mt_fp4_qmm_mma` hand-rolled the E2M1 magnitude
> as `(mantissa + 2) << (exp − 1)`, an **undefined shift when `exp == 0`** (subnormal
> codes) that miscompiled on the f32 path → unwritten output (zeros / stale garbage);
> replaced with the `e2m1_decode` intrinsic. (2) the block-scaled fp4 MMA kernel was
> **also named `mt_fp4_qmm_mma`**, colliding with the original in the MSL/pipeline
> cache → order-dependent wrong pipelines (the source of the ~0.46 f32 anomaly and the
> apparent `ffai_sdpa_multi_d256_causal` failure, which was a *victim* of the shared-
> state contamination, not itself buggy); renamed to `mt_fp4_float_qmm_mma`.

**Two parallel, both-current quant tracks (by design — neither supersedes the other):**
the **affine** int2–8 (scale+bias) track lives in `dequant_gemv`/`quantized`/`quantized_{mpp,nax}`/`kv_cache` and is *not* legacy — it is the on-disk format for MLX-quantized checkpoints (asymmetric, carries a per-group **bias**) and the right scheme for per-decode-step KV-cache quant (cheap min/max encode; block-scaled would need GPU encode intrinsics). The new symmetric `Int8` is a parallel spec-family member (scale-only), not a replacement; there is no block-scaled int4 at all, so affine int4 remains required to load MLX 4-bit models.

**Remaining (follow-ups, non-blocking):** the `winograd_conv` filter-transform variant (quantizing Winograd amplifies error in the transform domain; every other weight-bearing conv — direct + im2col-MMA — is covered); and an audit of whether the `ekryski/mlx@alpha` reference kernels are themselves spec-correct.

## Appendix C — M5 Neural Accelerator hardware context

The M5/A19 added a per-GPU-core **Neural Accelerator** — Apple's first dedicated *GPU* matmul hardware (~1024 FP16 FMACs/cycle/accelerator), **embedded in the GPU cores** (unlike the standalone Apple Neural Engine that held the matrix block on M1–M4). Published FP16 matmul ceilings: **base 16.8 / Pro 33.2 / Max 66.4 / Ultra 132.8 TFLOPS** (Ultra a projected 2× Max) — `device_specs::na_f16_tflops`. Accelerated formats:

- ✅ **FP16** (FP16 or FP32 accumulate) — fast path
- ✅ **INT8** (INT32 accumulate) — accelerated but **lags FP16** (opposite of NVIDIA)
- ✅ FP32 (general SIMD pipe, not the NA)
- ❌ **bfloat16 — not accelerated** on first-gen NA
- ❌ **fp8 / fp4 — not supported**

Implications for the roadmap: on M5, **fp16 is the fastest compute precision**; int8 helps memory-bound decode (bandwidth) but not compute; **bf16 may be slower than fp16** for matmul; fp4/fp8 elements get no native acceleration (dequantize-to-fp16 for the NA). The roofline device-spec table (§4.3) carries the SIMD (`peak_f32`/`peak_f16`), the M5+ GPU-Neural-Accelerator (`na_f16_tflops`), and the standalone-ANE (`ane_tops`, M1–M4 — for the upcoming ANE kernels/benches) ceilings.

Sources: [Apple Developer — Accelerate ML with M5 & A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/) · [Investigating the GPU Neural Accelerators on A19/M5 (tzakharko)](https://tzakharko.github.io/apple-neural-accelerators-benchmark/) · [TechBoards — A19/M5 GPU Neural Accelerators](https://techboards.net/threads/apple-a19-m5-gpu-neural-accelerators.5297/)
