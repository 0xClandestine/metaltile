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
| nvfp4 | E2M1 | 16 | E4M3 + global FP32 | ❌ (we use gs32 + float scale) |
| mxfp4 | E2M1 | 32 | E8M0 (pow-2) | ❌ (we use float scale) |
| mxfp8 | E4M3/E5M2 | 32 | E8M0 | ❌ |
| int2/4/8 affine | int | group 64 | per-group | ✅ (qmv/qmm variants) |
| fp8 E4M3/E5M2 | fp8 | — | — | ✅ (kv-cache, dequant) |

Work items: implement block-scaling (gs16+E4M3 for nvfp4, gs32+E8M0 for mxfp4/mxfp8); add int4/int8/fp4/fp8 weight variants across the weight-bearing kernel families; audit whether the `ekryski/mlx@alpha` reference kernels are themselves spec-correct before trusting them as oracles. The Phase-1/2 latency+GFLOP metrics are a **prerequisite** so "fastest precision" is observable.

## Appendix C — M5 Neural Accelerator hardware context

The M5/A19 added a per-GPU-core **Neural Accelerator** — Apple's first dedicated *GPU* matmul hardware (~1024 FP16 FMACs/cycle/accelerator), **embedded in the GPU cores** (unlike the standalone Apple Neural Engine that held the matrix block on M1–M4). Published FP16 matmul ceilings: **base 16.8 / Pro 33.2 / Max 66.4 / Ultra 132.8 TFLOPS** (Ultra a projected 2× Max) — `device_specs::na_f16_tflops`. Accelerated formats:

- ✅ **FP16** (FP16 or FP32 accumulate) — fast path
- ✅ **INT8** (INT32 accumulate) — accelerated but **lags FP16** (opposite of NVIDIA)
- ✅ FP32 (general SIMD pipe, not the NA)
- ❌ **bfloat16 — not accelerated** on first-gen NA
- ❌ **fp8 / fp4 — not supported**

Implications for the roadmap: on M5, **fp16 is the fastest compute precision**; int8 helps memory-bound decode (bandwidth) but not compute; **bf16 may be slower than fp16** for matmul; fp4/fp8 elements get no native acceleration (dequantize-to-fp16 for the NA). The roofline device-spec table (§4.3) carries the SIMD (`peak_f32`/`peak_f16`), the M5+ GPU-Neural-Accelerator (`na_f16_tflops`), and the standalone-ANE (`ane_tops`, M1–M4 — for the upcoming ANE kernels/benches) ceilings.

Sources: [Apple Developer — Accelerate ML with M5 & A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/) · [Investigating the GPU Neural Accelerators on A19/M5 (tzakharko)](https://tzakharko.github.io/apple-neural-accelerators-benchmark/) · [TechBoards — A19/M5 GPU Neural Accelerators](https://techboards.net/threads/apple-a19-m5-gpu-neural-accelerators.5297/)
