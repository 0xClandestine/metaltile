# External Developer UX Spec

A Foundry-inspired developer experience for MetalTile: any project can `tile init`,
write kernels, and immediately get `tile build`, `tile test`, and `tile bench` working
against their own library — with no changes to the `metaltile` repo required.

---

## Philosophy

Foundry's success comes from a few principles MetalTile should mirror:

- **Works out of the box.** `forge init` produces a project where `forge build` and
  `forge test` run immediately, with working examples, before the user writes a single
  line of their own code. `tile init` should do the same.
- **Convention over configuration.** File suffixes and directory layout communicate
  intent. You don't configure what's a test vs. a benchmark — you name it `.bench.rs`.
- **The config file is a first-class citizen.** `foundry.toml` with
  `[profile.default]` / `[profile.ci]` / `[profile.production]` teaches users one
  concept (profiles) that covers local dev, CI strictness, and release tuning. `Tile.toml`
  should work identically.
- **Verbosity is a ladder, not a mode.** `-v` / `-vv` / `-vvv` in Foundry each reveal
  another layer. MetalTile already has this for bench; it should extend to build and test.
- **The stdlib is just a dep.** `forge-std` is a git submodule the template includes.
  `metaltile` (the facade crate) plays this role — it's the thing you import in your
  kernels and it comes pre-wired in the generated template.

---

## 1. Current CLI Audit — External-Repo Gaps

### What works today (no kernel registry needed)

| Command | External status | Notes |
|---|---|---|
| `tile device` | Works | Pure Metal introspection |
| `tile update` | Works | Binary self-update |

### What is broken for external repos

| Command | Blocker |
|---|---|
| `tile build` | Iterates `inventory::iter::<BenchSpec>` — link-time only, cannot reach external crates |
| `tile bench` | Same, plus MLX reference runner |
| `tile snap` | Depends on `tile bench` output |
| `tile diff` | Depends on snapshots/bench JSON |
| `tile inspect` (list) | Same inventory iter |

**Root cause:** `BenchSpec` uses `inventory::collect!` — a linker-section trick. A
pre-built `tile` binary can never see kernels compiled in a separate crate. There is no
plugin ABI and no external manifest the CLI reads today.

---

## 2. The Target Workflow

This is what a developer using MetalTile from outside the repo should experience:

```sh
# Bootstrap a new project
tile init my-kernels
cd my-kernels

# Immediately works — no edits required yet
tile build
tile test
tile bench

# Inspect the generated MSL for the example kernel
tile inspect mt_vector_add

# Write your kernel, then iterate
tile build -v                              # print generated MSL as you go
tile test --match-kernel vector_add        # run only matching tests
tile bench --match-kernel vector_add       # bench only matching kernels
tile bench --match-module softmax          # bench all kernels in the softmax op group

# Save a baseline before optimizing
tile snap

# Optimize, then see what changed
tile bench --diff

# Emit artifacts for a Swift app
tile build --emit all -o Sources/MyTarget

# Stay up to date
tile update
```

Zero `cargo run -p ...` gymnastics. No knowledge of the `metaltile` repo internals
required.

---

## 3. `Tile.toml` — Project Configuration

Modelled directly on `foundry.toml`. The `[profile.default]` section is always active;
other profiles inherit from it and override only what differs. Sub-tables (`[bench]`,
`[tol]`) mirror how Foundry namespaces `[fuzz]` and `[invariant]` — short top-level keys,
no prefixes.

```toml
[profile.default]
src       = "kernels"       # kernel source directory
test      = "tests"         # GPU correctness tests
bench     = "benches"       # bench harness binaries
out       = "tile-out"      # artifact output (tile build --emit)
baselines = "baselines"     # tile snap / tile diff snapshots
dtypes    = ["f32", "f16", "bf16"]

[bench]
n     = 67108864   # element count for throughput runs (64M)
iters = 10         # warm repetitions before timing

[tol]
f32  = 1e-4
f16  = 1.5e-2
bf16 = 1.3e-1

# ── Profiles ────────────────────────────────────────────────────────────────
# Only override what differs; everything else inherits from [profile.default].

[profile.ci.bench]
# Fewer elements so CI finishes faster; correctness is still fully checked.
n     = 4194304    # 4M
iters = 3

[profile.release.bench]
# More iterations for stable perf numbers in release snapshots.
iters = 20
```

Switch profiles with `TILE_PROFILE=ci tile bench`, exactly like `FOUNDRY_PROFILE`.

Run `tile config` (new command) to print the fully-resolved configuration for the active
profile, including defaults for any field not specified.

---

## 4. Project Layout

```
my-kernels/
  Tile.toml                   # project manifest and profile config
  Cargo.toml                  # standard Rust workspace / package
  kernels/                    # kernel source (= src/ in Foundry)
    lib.rs
    vector_add.rs
    softmax.rs
  benches/                    # bench harness (= script/ in Foundry)
    kernels.rs                # entry point: tile_harness!(crate::bench_specs)
  tests/                      # GPU correctness tests (= test/ in Foundry)
    vector_add.t.rs           # .t.rs suffix = test file convention
  baselines/                  # tile snap output; committed to git
    <chip>-<sha>.json
  tile-out/                   # build artifacts; in .gitignore
    Resources/kernels/
    Resources/kernels.metallib
    Generated/MetalTileKernels.swift
```

**File conventions:**

| Suffix | Meaning | Analogy |
|---|---|---|
| `kernels/*.rs` | Kernel definitions with `#[kernel]` | `src/*.sol` |
| `tests/*.t.rs` | GPU correctness tests | `test/*.t.sol` |
| `benches/*.rs` | Bench harness entry points | `script/*.s.sol` |

---

## 5. `tile init` — Project Bootstrapping

```sh
tile init [<name>] [--template <template>] [--no-git] [--vscode]
```

Creates a project that **builds and benches immediately** — just like `forge init`
produces a working Counter contract with tests before the user writes anything.

**Templates:**

| Template | Description |
|---|---|
| `kernel` (default) | Single kernel, bench entry point, one correctness test |
| `library` | Multi-kernel library layout with a dedicated bench crate |
| `swift` | kernel + `--emit all` wiring for a SwiftPM target |

**What `tile init my-kernels` generates:**

`kernels/lib.rs` — a real, runnable example kernel:
```rust
use metaltile::prelude::*;

#[bench_kernel(
    op    = "vector_add",
    class = Binary,
    input = Signed,
    tol   = 1e-4,
)]
#[kernel]
pub fn mt_vector_add<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], load(a[idx]) + load(b[idx]));
}

pub fn bench_specs() -> &'static [&'static metaltile::BenchSpec] {
    &[&MT_VECTOR_ADD_SPEC]
}
```

`benches/kernels.rs` — the harness entry point:
```rust
tile_harness!(my_kernels::bench_specs);
```

`tests/vector_add.t.rs` — a GPU correctness test:
```rust
#[cfg(test)]
mod tests {
    use metaltile::testing::*;
    use my_kernels::bench_specs;

    #[test]
    fn vector_add_f32_correctness() {
        let ctx = GpuContext::new().unwrap();
        let result = run_correctness_check(&ctx, bench_specs(), "vector_add", DType::F32);
        assert!(result.passed, "vector_add f32: {result}");
    }
}
```

`Tile.toml` — ready-to-use config with a CI profile pre-filled.

After `tile init`:
```sh
tile build          # ✓ compiles immediately
tile test           # ✓ GPU correctness check passes
tile bench          # ✓ benchmark table with ref vs MT throughput
tile inspect mt_vector_add   # ✓ prints the generated MSL
```

---

## 6. Command Specifications

### `tile build`

**Unchanged interface.** When a `Tile.toml` is present, it compiles the project's bench
binary and uses it as the kernel source instead of the built-in registry. Falls back to
the built-in registry when no `Tile.toml` exists (in-tree contributors see no difference).

```sh
tile build [-f <substr>] [--match-kernel <regex>] [--match-module <regex>]
           [--no-match-kernel <regex>] [--no-match-module <regex>]
           [--dtypes f32,f16,bf16] [-v|-vv|-vvv]
           [--emit msl,metallib,swift,ir,all] [-o <dir>] [--watch]
```

| Flag | Effect |
|---|---|
| `--match-kernel <regex>` | only build kernels whose name matches |
| `--match-module <regex>` | only build kernels whose op group matches (e.g. `softmax`, `unary`) |
| `--no-match-kernel <regex>` | exclude kernels whose name matches |
| `--no-match-module <regex>` | exclude kernels whose op group matches |

New flags:
- `--watch` — rebuild whenever a source file changes (like `forge build --watch`)
- `-vvv` — also print the IR at each pass stage (currently only `tile inspect --pass all`)

### `tile test`

**New command.** Runs GPU correctness tests. Separate from `tile bench` so CI can
verify correctness without MLX installed.

```sh
tile test [-f <substr>] [--match-kernel <regex>] [--match-module <regex>]
          [--no-match-kernel <regex>] [--no-match-module <regex>]
          [--dtypes f32,f16,bf16] [-v|-vv] [--no-gpu]
```

| Flag | Effect |
|---|---|
| `--match-kernel <regex>` | only run tests whose kernel name matches |
| `--match-module <regex>` | only run tests whose op group matches |
| `--no-match-kernel <regex>` | exclude tests whose kernel name matches |
| `--no-match-module <regex>` | exclude tests whose op group matches |
| `--no-gpu` | skip GPU dispatch; only check that kernels compile |
| `-v` | show per-element error on failure |
| `-vv` | show full diff between ref and MT outputs on failure |

Output mirrors `forge test`:
```
tile test · Apple M4 Max
  mt_vector_add f32   ✓ (max_err=0.00e0)
  mt_vector_add f16   ✓ (max_err=3.81e-6)
  mt_vector_add bf16  ✓ (max_err=7.81e-3)

3 passed, 0 failed
```

Exits non-zero if any test fails — CI-safe by design.

### `tile bench`

**Unchanged interface.** Uses the same Tile.toml detection as `tile build`.

```sh
tile bench [-f <substr>] [--match-kernel <regex>] [--match-module <regex>]
           [--no-match-kernel <regex>] [--no-match-module <regex>]
           [-v|-vv] [-o <file.json>] [--allow-dirty]
           [--diff] [--baseline-ref <git-ref>]
```

### `tile inspect`

**Unchanged interface for in-tree use.** When `Tile.toml` is present, lists and
inspects kernels from the project.

```sh
tile inspect [<kernel>] [-f <substr>] [--match-kernel <regex>] [--all] [--ir] [--stats]
             [--pass <name>] [--dtype <f32|f16|bf16|i32|u32>] [-o <dir>]
```

### `tile config` (new)

Prints the resolved configuration for the active profile — all fields including
defaulted values. Useful for debugging why a bench run is using unexpected parameters.

```sh
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

Equivalent to `forge config`.

### `tile clean` (new)

Removes build artifacts and cache.

```sh
tile clean [--cache-only]
```

Removes the directory named by `out` (default `tile-out/`) and the tile build cache.
With `--cache-only`, leaves emitted artifacts but clears intermediate `.air` files and
the Metal compile cache.

### `tile snap`, `tile diff`

**Unchanged interface, updated filter flags.** Both already operate on JSON snapshot
files and are project-agnostic. They adopt `--match-kernel` / `--match-module` for
consistency; the `Tile.toml` `baselines` field changes where `tile snap` writes by
default.

### `tile device`, `tile update`

**Unchanged.**

---

## 7. Filtering

All commands that operate on kernels accept the same filter flags:

| Flag | Matches against | Example |
|---|---|---|
| `-f, --filter <substr>` | kernel name (substring, existing) | `--filter exp` |
| `--match-kernel <regex>` | kernel symbol name (regex) | `--match-kernel "exp\|log"` |
| `--match-module <regex>` | op group (`op` field on `BenchSpec`) | `--match-module unary` |
| `--no-match-kernel <regex>` | kernel symbol name (exclude) | `--no-match-kernel sort` |
| `--no-match-module <regex>` | op group (exclude) | `--no-match-module attention` |

`--filter` is kept for backwards compatibility and quick ad-hoc use. `--match-kernel`
and `--match-module` are AND-ed with each other and with `--filter`: a kernel must
satisfy all active filters to run. `--no-match-*` exclusions are applied last.

This mirrors Foundry's `--match-test` / `--match-contract` /
`--no-match-test` / `--no-match-contract` quad, with `--filter` as a familiar shorthand.

**Module** is the `op` string declared in `#[bench_kernel(op = "...")]` — the logical
group name (`"unary"`, `"softmax"`, `"attention"`). **Kernel** is the Rust symbol
(`mt_exp`, `mt_softmax_f32`). Filtering by module lets you run an entire op family
without knowing the individual kernel names.

---

## 8. Verbosity Ladder

MetalTile commands use a consistent `-v` ladder modelled on `forge test`'s levels:

| Flag | `tile build` | `tile test` | `tile bench` |
|---|---|---|---|
| (none) | pass/fail table | pass/fail summary | throughput table |
| `-v` | + generated MSL | + per-element max error | + occupancy / register profile |
| `-vv` | + IR before passes | + full output diff on failure | + GPU timing (min µs + bandwidth) |
| `-vvv` | + IR after each pass | + ref vs MT tensors | — |

---

## 9. `metaltile` as the Standard Library

`forge-std` is the test/assertion library every Foundry project depends on — it's a
git submodule in `lib/forge-std`, pre-wired by `forge init`.

The `metaltile` facade crate plays this role:
- It re-exports `#[kernel]`, `#[bench_kernel]`, all IR primitives, and all stdlib ops
- `tile init` adds it as a dependency automatically
- Users import `use metaltile::prelude::*;` — they don't need to know which sub-crate
  anything lives in

`metaltile-bench` (the harness library) is the second piece — it provides
`tile_harness!`, `run_correctness_check`, `GpuContext`, and the `OpBench` / `OpResult`
types that were previously internal to `metaltile-cli`.

---

## 10. The Inventory Problem and Its Solution

**Current state:** `#[bench_kernel]` emits `inventory::submit!` which deposits a
`BenchSpec` record into a linker section. `tile bench` calls
`inventory::iter::<BenchSpec>` to collect them. External crates cannot inject into a
pre-built binary's linker sections.

**Solution:** replace implicit linker magic with an explicit `bench_specs()` function
convention.

```rust
// Before (in-tree only)
inventory::submit!(MY_KERNEL_SPEC);   // generated by #[bench_kernel]

// After (works anywhere)
pub fn bench_specs() -> &'static [&'static BenchSpec] {
    &[&MY_KERNEL_SPEC]                // generated by #[bench_kernel] into a named static
}
```

The `#[bench_kernel]` macro still generates the `BenchSpec` static — the only change
is it no longer emits `inventory::submit!`. The harness binary collects specs by calling
`bench_specs()` directly, then speaks a lightweight JSON protocol to `tile` over stdout.

This removes `inventory` from the workspace entirely.

### Subprocess protocol (implementation detail)

`tile` detects `Tile.toml`, compiles the harness binary if stale, then invokes it:

```sh
./<harness_bin> --tile-protocol=jsonl --action=bench --filter=vector_add
```

The harness writes JSON-Lines to stdout:

```jsonc
{"type":"result","op":"vector_add","dtype":"f32","shape":"N=64M","ref_gbps":850.2,"mt_gbps":912.1,"passed":true}
{"type":"done","ok":3,"errors":0}
```

`tile` renders the table from these records. The harness binary is the kernel library;
`tile` is the display layer. This is how Foundry separates the test runner (forge) from
the EVM executor (built-in revm) — one compiles contracts, the other renders results.

---

## 11. `metaltile-std` Migration

`metaltile-std` switches from `inventory` to explicit export — it becomes the reference
implementation of the pattern external users follow:

1. Remove `inventory::collect!(BenchSpec)` from `spec.rs`
2. Add `pub fn bench_specs() -> &'static [&'static BenchSpec]` listing all op specs
3. `metaltile-cli`'s in-tree build/bench path calls `metaltile_std::bench_specs()`
   directly (still compiled-in, no subprocess overhead for maintainers)
4. External-project path: `Tile.toml` detected → compile harness → subprocess

No behaviour change for anyone running `make bench` in the `metaltile` repo.

---

## 12. Implementation Phases

### Phase 1 — Harness library + protocol
- Create `metaltile-bench` library crate (extract runner from `metaltile-cli`)
- Define JSONL subprocess protocol
- Update `#[bench_kernel]` macro: generate named static, drop `inventory::submit!`
- Add `bench_specs()` to `metaltile-std`
- Update `tile build` and `tile bench` to detect `Tile.toml` and invoke harness
- Remove `inventory` from workspace

### Phase 2 — New commands
- `tile init` with `kernel` and `library` templates
- `tile test` subcommand
- `tile config` subcommand
- `tile clean` subcommand
- `tile inspect` harness path

### Phase 3 — Polish
- `--watch` flag for `tile build`
- Extended verbosity levels (`-vvv` IR-per-pass in `tile build`)
- `Tile.toml` schema validation with actionable error messages
- Update `docs/getting-started.md` to cover the external-project path
- New `docs/external-kernels.md` cookbook
