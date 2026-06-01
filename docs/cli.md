# CLI

`tile` is the command-line driver for benchmarking, building, and inspecting kernels. Install it, or run it through `cargo` from a checkout.

```bash
cargo install --path crates/metaltile-cli      # installs the `tile` binary
# or, from a checkout, without installing:
cargo run -p metaltile-cli -- <command> …
```

`make bench` wraps `tile bench`; for the other subcommands run `tile` (or the `cargo run` form) directly.

## `tile bench` — benchmark vs MLX

Runs every kernel against its MLX reference and reports throughput, wall-clock
latency, compute throughput, and a correctness check.

```
tile bench [-f <substr>] [-v|-vv] [-o <file.json>] [--allow-dirty]
           [--diff] [--baseline-ref <git-ref>]
```

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only run kernels whose name contains `<substr>` |
| `-v` / `-vv` | `-v` adds the reference latency, roofline (`%BW` / `%FLOP` / arithmetic intensity), occupancy/registers, and a bottleneck verdict; `-vv` adds the GPU timing distribution (`p95` / `p99` / `cv%`) |
| `-o, --json <file>` | also write results as JSON |
| `--allow-dirty` | run on a dirty working tree (default: refuses, so numbers tie to a clean SHA) |
| `--diff` | opt into the post-bench diff against the target-branch baseline |
| `--baseline-ref <ref>` | git ref whose `baselines/<chip>.json` to diff against (default: first of `origin/dev`, `upstream/dev`, `dev`) |

### Metrics

The default table shows, per kernel/dtype: `MT(µs)` (wall-clock latency, the
`min` sample — the metric that makes "which precision is fastest" directly
readable), `Ref`/`MT` (GB/s bandwidth), `MT%` (MT-vs-reference ratio), `GFLOP/s`
(compute throughput, blank for memory-bound kernels), and `ok` (correctness).

`-v` adds the roofline view: `%BW` (achieved ÷ the device's peak DRAM bandwidth),
`%FLOP` (achieved ÷ peak compute — the M5 Neural-Accelerator FP16 ceiling where
applicable, the SIMD pipe otherwise), `AI` (arithmetic intensity, FLOPs/byte),
the estimated `occ%`/`regs`, and a combined `bottleneck` verdict
(`memory-bound` / `compute-bound` / `occupancy-limited` / `register-limited` /
`latency-bound`). Peak ceilings come from a per-device table
(`crates/metaltile-std/src/device_specs.rs`); an unknown GPU leaves the roofline
columns blank rather than failing.

GFLOP/s, latency, and the roofline figures only appear for kernels that declared
a FLOP count (`#[bench(flops = …)]` or `BenchSetup::flops`) — matmul, attention,
and convolution; memory-bound elementwise/reduction kernels leave them blank. The
JSON (`-o`) is **additive**: it keeps the `ref`/`mt` (GB/s) keys baseline diffing
consumes and adds `latency_us`, `gflops`, `pct_peak_bw`, `pct_peak_flops`, and
`arith_intensity`.

## `tile build` — compile kernels to MSL

Compiles every kernel and reports errors; with `--emit`, writes artifacts.

```
tile build [-f <substr>] [--dtypes f32,f16,bf16] [-v]
           [--emit msl,metallib,swift,ir,all] [-o <dir>] [--sdk <sdk>] [-t]
```

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only build matching kernels |
| `--dtypes <list>` | comma-separated dtypes to build (`f32,f16,bf16`) |
| `-v` | print the generated MSL for each kernel |
| `--emit <list>` | emit artifacts — `msl`, `metallib`, `swift`, `ir`, or `all` |
| `-o, --out <dir>` | output directory (required when `--emit` is set) |
| `--sdk <sdk>` | `xcrun` SDK for the Metal toolchain (default: `macosx`) |
| `-t, --time-passes` | run the pass pipeline 25× per kernel, print per-pass median wall time instead of emitting |

Codegen smoke check — emit everything and confirm `xcrun metal` accepts it: `tile build --emit all -o /tmp/mt-smoke`.

The output layout matches a SwiftPM `Sources/<Target>/` convention so `--out` can point directly at a target directory:

```
<out>/Resources/kernels/<name>.metal
<out>/Resources/kernels.metallib
<out>/Resources/manifest.json
<out>/Generated/MetalTileKernels.swift
```

## `tile inspect` — IR and MSL for one kernel

```
tile inspect [<kernel>] [--filter <substr>] [--all] [--ir] [--stats]
             [--pass <name>] [--dtype <f32|f16|bf16|i32|u32>] [-o <dir>]
```

| Flag | Effect |
|---|---|
| *(no flag)* | print the final generated MSL |
| `--ir` | print the raw IR before any passes |
| `--pass <name>` | print the IR after a specific pass (`--pass all` for every stage) |
| `--stats` | print the per-pass op-count reduction table |
| `--dtype <d>` | dtype override for monomorphisation |
| `--filter <substr>` / `--all` | inspect many kernels at once |
| `-o, --dir <dir>` | write output files instead of printing to stdout |

Omit the kernel name to list every registered kernel. See [Developing → debugging a kernel](developing.md#debugging-a-kernel).

## `tile device` — GPU info

Prints the Metal device name, Metal version, Apple GPU family, and the supported feature flags (native `bfloat`, simdgroup matrix, etc.). Add `--json` for machine-readable output.

## `tile snap` — save a perf regression baseline

```
tile snap [-o <file>] [--from <file.json>] [--note <text>] [-f <substr>]
```

| Flag | Effect |
|---|---|
| `-o, --out <file>` | write the snapshot here (default: `.tile-snapshots/<sha>.json`) |
| `--from <file.json>` | promote an existing bench JSON instead of re-running the bench |
| `--note <text>` | attach a note to the snapshot |
| `-f, --filter <substr>` | only include kernels whose name contains `<substr>` |

## `tile diff` — compare against a baseline

```
tile diff <baseline> [<current>] [-f <substr>] [--threshold <pct>]
          [--sort name|delta|pct] [--only-regressions] [--only-improvements]
```

`<baseline>` is a saved snapshot JSON; `<current>` is an optional bench JSON — omit it and `diff` runs the bench itself.

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only show kernels whose name contains `<substr>` |
| `--threshold <pct>` | highlight regressions larger than this percentage (default: `5`) |
| `--sort <key>` | sort rows by `name`, `delta`, or `pct` (default: `name`) |
| `--only-regressions` | show only regressed kernels |
| `--only-improvements` | show only improved kernels |
