# CLI — `tile` Reference

`tile` is the command-line driver for benchmarking, building, testing, and inspecting MetalTile kernels.

```bash
cargo install --path crates/metaltile-cli      # installs the `tile` binary
# or, from a checkout:
cargo run -p metaltile-cli -- <command> …
```

In the `metaltile` repository, `make bench` wraps `tile bench`.

---

## Common filter flags

All commands that operate on kernels accept these filter flags:

| Flag | Matches against | Example |
|---|---|---|
| `-f, --filter <substr>` | kernel name (substring, existing) | `--filter exp` |
| `--match-kernel <regex>` | kernel symbol name (regex) | `--match-kernel "exp\|log"` |
| `--match-module <regex>` | op group (`op` field) | `--match-module unary` |
| `--no-match-kernel <regex>` | kernel symbol name (exclude) | `--no-match-kernel sort` |
| `--no-match-module <regex>` | op group (exclude) | `--no-match-module attention` |

All active filters are AND-ed: a kernel must pass every filter provided. `--no-match-*`
exclusions are applied last.

**Module** is the `op` string declared in `#[bench_kernel(op = "...")]` — the logical
group name (`"unary"`, `"softmax"`, `"attention"`).

---

## `tile bench` — benchmark vs MLX reference

Runs every kernel against its MLX reference and reports throughput + correctness.

```
tile bench [filter flags] [-v|-vv] [-o <file.json>] [--allow-dirty]
           [--diff] [--baseline-ref <git-ref>]
```

| Flag | Effect |
|---|---|
| `-v` / `-vv` | `-v` adds occupancy + register profile; `-vv` adds GPU timing (min µs + bandwidth) |
| `-o, --json <file>` | also write results as JSON |
| `--allow-dirty` | run on a dirty working tree (default: refuses) |
| `--diff` | opt into post-bench diff against target-branch baseline |
| `--baseline-ref <ref>` | git ref whose `baselines/<chip>.json` to diff against |

Detects `Tile.toml` in the working directory. When present, compiles the project's
harness binary and invokes it via the JSONL subprocess protocol. When absent, uses
the built-in kernel registry (`metaltile_std::bench_specs()`).

---

## `tile build` — compile kernels to MSL

Compiles every kernel and reports errors; with `--emit`, writes artifacts.

```
tile build [filter flags] [--dtypes f32,f16,bf16] [-v|-vv|-vvv]
           [--emit msl,metallib,swift,ir,all] [-o <dir>] [--sdk <sdk>]
           [--watch] [-t]
```

| Flag | Effect |
|---|---|
| `--dtypes <list>` | comma-separated dtypes to build (`f32,f16,bf16`) |
| `-v` | print the generated MSL for each kernel |
| `-vv` | print IR before passes |
| `-vvv` | print IR after each pass stage |
| `--emit <list>` | emit artifacts — `msl`, `metallib`, `swift`, `ir`, or `all` |
| `-o, --out <dir>` | output directory (required when `--emit` is set) |
| `--sdk <sdk>` | `xcrun` SDK for the Metal toolchain (default: `macosx`) |
| `--watch` | rebuild when source files change |
| `-t, --time-passes` | run pass pipeline 25× per kernel, print per-pass median wall time |

---

## `tile test` — GPU correctness checks

Runs GPU correctness tests without MLX. Separate from `tile bench` for CI use.

```
tile test [filter flags] [--dtypes f32,f16,bf16] [-v|-vv] [--no-gpu]
```

| Flag | Effect |
|---|---|
| `--dtypes <list>` | comma-separated dtypes to test |
| `-v` | show per-element error on failure |
| `-vv` | show full output diff on failure |
| `--no-gpu` | skip GPU dispatch; only check that kernels compile |

Output mirrors `forge test`:

```
tile test · Apple M4 Max
  mt_vector_add f32   ✓ (max_err=0.00e0)
  mt_vector_add f16   ✓ (max_err=3.81e-6)
  mt_vector_add bf16  ✓ (max_err=7.81e-3)

3 passed, 0 failed
```

Exits non-zero if any test fails.

---

## `tile init` — project bootstrapping

Creates a new MetalTile project that builds and benches immediately.

```
tile init [<name>] [--template <template>] [--no-git] [--vscode]
```

| Argument | Effect |
|---|---|
| `<name>` | project name (default: `my-kernels`) |
| `--template <name>` | `kernel` (default), `library`, `swift` |
| `--no-git` | skip git repository initialisation |
| `--vscode` | generate `.vscode` settings |

Generates:

| File | Contents |
|---|---|
| `Tile.toml` | profile config with CI and release profiles |
| `Cargo.toml` | MetalTile dependency and bin target |
| `kernels/lib.rs` | `mt_vector_add` example kernel |
| `benches/kernels.rs` | bench harness entry point (`tile_harness!`) |
| `tests/vector_add.t.rs` | GPU correctness tests |
| `.gitignore` | `target/` and `tile-out/` |

After `tile init`:

```sh
cd my-kernels
tile build    # compiles immediately
tile test     # GPU correctness check passes
tile bench    # benchmark table with ref vs MT throughput
```

---

## `tile config` — resolved configuration

Prints the fully resolved `Tile.toml` configuration for the active profile.

```
tile config [--profile <name>]
```

```
tile config · profile=default (from Tile.toml)
  src              kernels
  test             tests
  bench            benches
  out              tile-out
  baselines        baselines
  dtypes           f32, f16, bf16
  bench.n          67108864
  bench.iters      10
  tol.f32          0.0001
  tol.f16          0.015
  tol.bf16         0.13
```

The active profile is `default`, or the value of `TILE_PROFILE` env var.

---

## `tile clean` — remove artifacts

Removes build artifacts (output directory) and the Metal compile cache.

```
tile clean [--cache-only]
```

| Flag | Effect |
|---|---|
| `--cache-only` | keep emitted artifacts; clear intermediate `.air` files and Metal compile cache |

Removes the `out` directory (default: `tile-out/`). With `--cache-only`, leaves the
emitted artifacts but purges the intermediate compile cache (`target/tile-build-air/`).

---

## `tile inspect` — IR and MSL for one kernel

```
tile inspect [<kernel>] [filter flags] [--all] [--ir] [--stats]
             [--pass <name>] [--dtype <f32|f16|bf16|i32|u32>] [-o <dir>]
```

| Flag | Effect |
|---|---|
| *(no flag)* | print the final generated MSL |
| `--ir` | print the raw IR before any passes |
| `--pass <name>` | print IR after a specific pass (`--pass all` for every stage) |
| `--stats` | print per-pass op-count reduction table |
| `--dtype <d>` | dtype override for monomorphisation |
| `--all` | inspect every kernel |
| `-o, --dir <dir>` | write output files instead of stdout |

Omit the kernel name to list every registered kernel.

---

## `tile device` — GPU info

Prints the Metal device name, Metal version, Apple GPU family, and supported features
(native `bfloat`, simdgroup matrix, etc.). Add `--json` for machine-readable output.

---

## `tile snap` — save a perf baseline

```
tile snap [filter flags] [-o <file>] [--from <file.json>] [--note <text>]
```

| Flag | Effect |
|---|---|
| `-o, --out <file>` | snapshot path (default: `.tile-snapshots/<sha>.json`) |
| `--from <file.json>` | promote an existing bench JSON instead of re-running |
| `--note <text>` | attach a note to the snapshot |

---

## `tile diff` — compare against a baseline

```
tile diff <baseline> [<current>] [filter flags] [--threshold <pct>]
          [--sort name|delta|pct] [--only-regressions] [--only-improvements]
```

| Flag | Effect |
|---|---|
| `--threshold <pct>` | highlight regressions larger than this % (default: `5`) |
| `--sort <key>` | sort by `name`, `delta`, or `pct` (default: `name`) |
| `--only-regressions` | show only regressed kernels |
| `--only-improvements` | show only improved kernels |

---

## `tile update` — self-update

```
tile update [--check] [--pr <N>] [--commit <SHA>]
```

| Flag | Effect |
|---|---|
| `--check` | print what would be installed without modifying anything |
| `--pr <N>` | build and install from a PR |
| `--commit <SHA>` | build and install from a commit |

---

## Verbosity ladder

All commands use a consistent `-v` ladder:

| Flag | `tile build` | `tile test` | `tile bench` |
|---|---|---|---|
| (none) | pass/fail table | pass/fail summary | throughput table |
| `-v` | + generated MSL | + per-element max error | + occupancy / register profile |
| `-vv` | + IR before passes | + full output diff on failure | + GPU timing (min µs + bandwidth) |
| `-vvv` | + IR after each pass | — | — |
