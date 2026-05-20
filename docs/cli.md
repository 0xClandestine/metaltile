# CLI

`tile` is the command-line driver for benchmarking, building, and inspecting
kernels. Install it, or run it through `cargo` from a checkout.

```bash
cargo install --path crates/metaltile-cli      # installs the `tile` binary
# or, from a checkout, without installing:
cargo run -p metaltile-cli -- <command> …
```

`make bench` wraps `tile bench`; for the other subcommands run `tile` (or the
`cargo run` form) directly.

## `tile bench` — benchmark vs MLX

Runs every kernel against its MLX reference and reports throughput + a
correctness check.

```
tile bench [-f <substr>] [-v|-vv] [-o <file.json>] [--allow-dirty]
           [--no-diff] [--baseline-ref <git-ref>]
```

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only run kernels whose name contains `<substr>` |
| `-v` / `-vv` | `-v` adds occupancy + register profile; `-vv` adds GPU timing (min µs + bandwidth) |
| `-o, --json <file>` | also write results as JSON |
| `--allow-dirty` | run on a dirty working tree (default: refuses, so numbers tie to a clean SHA) |
| `--no-diff` | skip the post-bench diff against the target-branch baseline |
| `--baseline-ref <ref>` | git ref whose `baselines/<chip>.json` to diff against (default: first of `origin/dev`, `upstream/dev`, `dev`) |

## `tile build` — compile kernels to MSL

Compiles every kernel and reports errors; with `--emit`, writes artifacts.

```
tile build [-f <substr>] [--dtypes f32,f16,bf16] [-v]
           [--emit msl,metallib,swift,ir,all] [-o <dir>] [-t]
```

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only build matching kernels |
| `--dtypes <list>` | comma-separated dtypes to build (`f32,f16,bf16`) |
| `-v` | print the generated MSL for each kernel |
| `--emit <list>` | emit artifacts — `msl`, `metallib`, `swift`, `ir`, or `all` |
| `-o, --out <dir>` | output directory (required when `--emit` is set) |
| `-t, --time-passes` | run the pass pipeline 25× per kernel, print per-pass median wall time instead of emitting |

Codegen smoke check — emit everything and confirm `xcrun metal` accepts it:
`tile build --emit all -o /tmp/mt-smoke`.

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

Omit the kernel name to list every registered kernel. See
[Developing → debugging a kernel](developing.md#debugging-a-kernel).

## `tile device` — GPU info

Prints the Metal device name, Metal version, Apple GPU family, and the
supported feature flags (native `bfloat`, simdgroup matrix, etc.).

## `tile snap` / `tile diff` — perf regression baselines

```
tile snap [-o <file>] [--from <file.json>] [--note <text>] [-f <substr>]
tile diff <baseline> [<current>]
```

`snap` saves bench results as a baseline (default `.tile-snapshots/<sha>.json`);
`--from` promotes an existing JSON instead of re-running the bench. `diff`
compares current results — or a saved `<current>` JSON — against `<baseline>`
and reports per-kernel deltas.
