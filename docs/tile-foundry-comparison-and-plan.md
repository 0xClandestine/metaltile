# Tile CLI vs Foundry CLI: Comparison & Rewrite Plan

**Status:** Initial draft  
**Date:** 2026-05-29  
**Sources:**
- MetalTile `crates/metaltile-cli` source (current monolith)
- MetalTile `docs/TOOLCHAIN_DESIGN.md` (proposed subprocess architecture)
- Foundry repo at `github.com/foundry-rs/foundry` (v1.7.2, crates `forge`, `cast`, `anvil`, `foundry-cli`, `foundry-config`)

---

## 1. High-Level Architecture Comparison

| Dimension | Foundry (`forge`/`cast`/`anvil`) | MetalTile (`tile`) | Target |
|---|---|---|---|
| **Binary model** | Multi-binary: `forge`, `cast`, `anvil`, `chisel` — separate crates, separate binaries | Single `tile` binary + auto-generated `__tile_runner` sub-binary | Single `tile` binary, project-specific runner spawned as subprocess |
| **Project concept** | Implicit: CWD is the project. `foundry.toml` discovered walking up from CWD | None currently. Runs from CWD, no manifest | `tile.toml` discovered walking up from CWD |
| **Config hierarchy** | `foundry.toml` → CLI args → env vars → defaults via `figment` | CLI args only. `--dtypes`, `--sdk`, `--filter` etc. | `tile.toml` → CLI args → env vars → defaults |
| **Subprocess model** | ❌ No subprocesses. Compiler, EVM, runner all in-process library calls | ❌ Currently in-process: all kernel logic linked into CLI | ✅ CLI spawns auto-generated `__tile_runner` binary per project |
| **GPU/execution coupling** | N/A (EVM execution is a library crate) | GPU runner (`GpuRunner`) linked directly into CLI | GPU code lives only in the project binary, never in the CLI |
| **Watch mode** | ✅ `forge test --watch`, `forge build --watch` via `watchexec` | ❌ Not implemented | ❌ Phase 2 |
| **Rendering** | `yansi` + `comfy-table` + `indicatif` for progress bars | `anstyle`/`anstream` custom table code in `suite_printer.rs` + `term.rs` | Keep custom printer, add progress bars for compilation |
| **Test filter** | Regex-based: `--match-test REGEX`, `--match-contract REGEX`, `--match-path GLOB` | Substring `--filter` only | Add regex + glob support |
| **Error handling** | `color-eyre` with backtraces, `eyre::Result` everywhere | `thiserror` custom `CliError` enum | Keep `CliError` but add backtrace support |
| **Machine output** | `--machine` flag → structured JSON envelope for agent use | Not present | Phase 2 |

---

## 2. Foundry CLI Architecture Deep-Dive

### 2.1 Binary structure

```
Cargo workspace root/
├── crates/
│   ├── forge/            # Forge CLI binary + all subcommands
│   │   ├── bin/main.rs   # fn main → args::run() → match on subcommand
│   │   └── src/
│   │       ├── args.rs   # run() dispatcher
│   │       ├── opts.rs   # Forge + ForgeSubcommand clap enums
│   │       ├── cmd/      # Per-subcommand modules (build.rs, test/mod.rs, ...)
│   │       └── ...
│   ├── cast/             # Cast binary (separate binary, same pattern)
│   ├── anvil/            # Anvil binary
│   ├── chisel/           # Chisel REPL binary
│   ├── cli/              # Shared CLI infrastructure
│   │   ├── src/opts/     # GlobalArgs, BuildOpts, EvmArgs, etc.
│   │   └── src/utils/    # LoadConfig trait, allocator, cmd helpers
│   └── config/           # foundry.toml parsing via figment
└── ...
```

Every binary follows the same pattern:

```rust
// bin/main.rs
fn main() {
    if let Err(err) = run() {
        // machine-mode exit or error display
        std::process::exit(1);
    }
}

// src/args.rs
pub fn run() -> Result<()> {
    // 1. Check for machine/introspect/markdown flags (pre-parse)
    foundry_cli::machine::check_machine();
    foundry_cli::opts::GlobalArgs::check_introspect_with(Forge::command, &REGISTRY);
    foundry_cli::opts::GlobalArgs::check_markdown_help::<Forge>();
    // 2. Setup logging/tracing
    setup()?;
    // 3. Parse CLI args
    let args = foundry_cli::parse_or_exit::<Forge>();
    args.global.init()?;
    // 4. Dispatch to subcommand
    run_command(args)
}
```

### 2.2 Config system (figment)

Foundry's config is a **hierarchical layer** system built on `figment`:

```
1. Defaults            (Config::default())
2. foundry.toml        (profile-aware TOML file, walked up from CWD)
3. CLI args            (each Command's Args implements figment::Provider)
4. Env vars            (FOUNDRY_ prefixed)
5. Inline config       (solc remappings embedded in Solidity files)
```

Each subcommand (`BuildArgs`, `TestArgs`, etc.) implements `figment::Provider` via the `merge_impl_figment_convert!` macro:

```rust
// Merges CLI args into the figment so they override foundry.toml values.
foundry_config::merge_impl_figment_convert!(TestArgs, build, evm);
//                                    ^         ^      ^
//                                    |         |      additional providers
//                                    target type      to merge
```

`Config::from_provider(figment)` then extracts a fully resolved `Config`. This means:
- `forge build --optimizer-runs 999` overrides the `optimizer_runs` in `foundry.toml`
- Profile-specific settings: `[profile.ci]` vs `[profile.default]`
- Global config cached at `~/.foundry/foundry.toml`

### 2.3 Subprocess handling in Foundry

Foundry does **not** use subprocesses for its core commands. Everything runs in-process:
- Solidity compilation → `foundry-compilers` library crate
- EVM execution → `revm` / `alloy-evm` library crate
- Test runner → `ForgeRunner` runs in-process

The only subprocess usage is:
- `forge watch` → spawns own binary with args (to restart on file changes)
- `forge script` → may spawn `anvil` as a child process
- External solc binaries → managed by `svm-rs`

This is a key difference: MetalTile's planned subprocess model (CLI spawns user project's runner) is a **deliberate departure** from Foundry's approach, driven by the need to avoid re-compiling the CLI for every kernel change.

### 2.4 Watch mode

```rust
// crates/forge/src/cmd/watch.rs
// Uses watchexec crate:
use watchexec::{Watchexec, action::ActionHandler, command::Program};

// Forge watches source/test directories and re-runs the command.
// On changes:
//   1. Recompiles (in-process)
//   2. Re-runs only changed tests (by default) or full suite (--run-all)
//   3. Rerun previously failed tests first (--rerun-failed)
```

Watch detects changes via `watchexec-events` file system events and re-triggers the build/test pipeline.

### 2.5 Test filtering

Foundry supports **regex test filtering**:

```rust
--match-test REGEX       # Only run test functions matching this regex
--no-match-test REGEX    # Exclude test functions matching this regex
--match-contract REGEX   # Only test contracts matching this regex  
--no-match-contract REGEX
--match-path GLOB        # Only test source files matching this glob
--no-match-path GLOB
```

This is more flexible than MetalTile's current substring `--filter` approach. Filters compose AND-wise.

---

## 3. Current MetalTile CLI Architecture

### 3.1 Binary structure

```
crates/metaltile-cli/
├── Cargo.toml              # [[bin]] name = "tile"
├── src/
│   ├── main.rs             # Cli { command: Command } → dispatch
│   ├── error.rs            # CliError enum (Io, Json, MetalCompile, GpuInit, Subprocess, Other)
│   ├── git.rs              # Git wrappers for dirty-tree check, baseline resolution
│   ├── suite_printer.rs    # OpResult table rendering
│   ├── term.rs             # anstyle/anstream terminal styling
│   └── cmd/
│       ├── mod.rs
│       ├── bench.rs        # ~500 lines — runs benchmarks via in-process GpuRunner
│       ├── test.rs         # ~100 lines — runs #[test_kernel] tests in-process
│       ├── build.rs        # ~400 lines — compiles kernels, emits artifacts
│       ├── inspect.rs      # ~250 lines — prints IR/MSL for a kernel
│       ├── device.rs       # GPU info display
│       ├── snap.rs         # Save baseline snapshots
│       ├── diff.rs         # Compare bench results
│       └── update.rs       # Self-update
```

### 3.2 Key coupling problem

The CLI currently links **all** of `metaltile-codegen`, `metaltile-core`, `metaltile-std`, and `metaltile-runtime` directly:

```toml
# crates/metaltile-cli/Cargo.toml (current)
[dependencies]
metaltile.workspace = true        # re-export crate
metaltile-core.workspace = true    # IR types, inventory
metaltile-runtime.workspace = true  # GPU dispatch
metaltile-codegen.workspace = true  # MSL generation, passes
metaltile-std.workspace = true     # ALL kernel definitions + bench specs
```

This means:
1. Adding a kernel → rebuilds the entire CLI
2. Changing bench parameters → rebuilds the entire CLI
3. GPU code runs in the CLI process → crash in GPU code takes down the CLI

### 3.3 Current data flow

```
User input → main.rs → dispatch → bench.rs
                                     │
                                     ▼
                                  GpuRunner::new()  ← Metal device init
                                     │
                                     ▼
                                  all_specs()  ← inventory! from metaltile-std
                                     │
                                     ▼
                                  run_spec()  ← CPU oracle, GPU dispatch, timing
                                     │
                                     ▼
                                  OpResult → SuitePrinter → terminal table
                                             ↓
                                          JSON file (optional)
```

---

## 4. Comparison Summary — Key Differences

| Aspect | Foundry | Current MetalTile | Target State |
|---|---|---|---|
| **Config** | `foundry.toml` + CLI + env via figment | CLI args only | `tile.toml` + CLI + env via figment |
| **Project root** | Walk up from CWD for `foundry.toml` | CWD, no project concept | Walk up from CWD for `tile.toml` |
| **GPU coupling** | N/A | In-process CLI | Subprocess only |
| **Runner** | Library crate (in-process) | Library crates (in-process) | Project-generated subprocess |
| **Recompile on kernel change** | N/A | Yes — CLI must be rebuilt | No — only the project runner rebuilds |
| **Test filter** | Regex (match-test, match-contract, match-path) | Case-insensitive substring | Regex + substring + glob |
| **Watch** | watchexec-based | None | watchexec-based (phase 2) |
| **Progress bars** | indicatif (compilation steps) | None (only terminal tables) | indicatif for compilation |
| **Machine output** | `--machine` flag | None | Phase 2 |
| **Error display** | color-eyre with suggestions | Custom term styling | Keep custom, add backtrace |
| **Version check** | Automatic nightly check | None | Phase 2 |

---

## 5. Rewrite Plan — Phased

### Phase 1: Project Manifest + Config System

**Goal:** CLI discovers `tile.toml`, loads project config, minimal surface area change.

**Add:**

1. **`crates/metaltile-config/`** — new crate (or extend `metaltile-core`):
   - `TileConfig` struct with sections:
     ```rust
     pub struct TileConfig {
         pub project: ProjectConfig,
         pub runner: RunnerConfig,
         pub bench: BenchConfig,
         pub test: TestConfig,
         pub build: BuildConfig,
     }
     ```
   - `tile.toml` deserialization via `serde` + `toml`
   - Walk-up discovery from CWD
   - Environment variable overrides: `TILE_*` prefixed

2. **`tile.toml` schema** (defined from TOOLCHAIN_DESIGN.md):
   ```toml
   [project]
   name = "metaltile-std"

   [runner]
   cargo_args = ["--release"]

   [bench]
   warmup_iters = 5
   bench_iters = 20

   [test]
   default_tol = 1e-4

   [build]
   sdk = "macosx"
   default_dtypes = ["f32", "f16"]
   ```

3. **CLI updates in `metaltile-cli`**:
   - `tile bench`, `tile test`, `tile build` load `TileConfig` on startup
   - Config values merge with CLI args (CLI wins)
   - `tile init` — scaffold a `tile.toml` in CWD

**Non-goals:** No subprocess model yet. CLI still links everything in-process.

### Phase 2: Runner Traits + Inventory

**Goal:** Move bench/test/build logic behind traits so it can be abstracted behind a protocol.

**Add in `metaltile-core` / `metaltile`:**

4. **`BenchHarness` trait** — abstract over "run bench + return JSON":

   ```rust
   pub trait BenchHarness {
       fn run(&self, filter: Option<&str>) -> Result<Vec<serde_json::Value>>;
   }
   ```

5. **`InProcessRunner`** — wraps current `GpuRunner` + `run_spec` behind `BenchHarness`:

   ```rust
   pub struct InProcessRunner(GpuRunner);
   impl BenchHarness for InProcessRunner { ... }
   ```

6. **CLI** switches from calling `cmd::bench::run()` to calling `config.load_runner()?.run(filter)`.

**Non-goals:** No subprocess spawning yet. This is the refactoring step that makes the subprocess a drop-in replacement.

### Phase 3: Subprocess Runner (The Big Change)

**Goal:** `tile bench`/`test`/`build` spawns a project-compiled binary via JSON Lines protocol.

**Implementation (from TOOLCHAIN_DESIGN.md):**

7. **`crates/metaltile/runner/`** — the protocol loop library:
   ```rust
   pub fn run(args: RunnerArgs) -> Result<()> {
       match args.command {
           Command::Bench { filter } => run_benches(filter),
           Command::Test { filter } => run_tests(filter),
           Command::Build { filter, dtypes } => run_build(filter, dtypes),
       }
   }
   ```

8. **Harness generation** — when CLI detects no `target/tile/__runner.rs` or stale:

   ```rust
   fn ensure_harness(project_root: &Path) -> PathBuf {
       let harness_dir = project_root.join("target").join("tile");
       let harness_path = harness_dir.join("__runner.rs");
       // Write auto-generated main that calls metaltile::runner::run()
       std::fs::write(&harness_path, HARNESS_TEMPLATE)?;
       harness_path
   }
   ```

   The generated file:
   ```rust
   // auto-generated by tile
   fn main() {
       metaltile::runner::run(metaltile::runner::Args::from_env());
   }
   ```

9. **Subprocess spawning in CLI:**

   ```rust
   fn run_via_subprocess(filter: Option<&str>) -> Result<Vec<serde_json::Value>> {
       ensure_harness(&project_root)?;
       let status = std::process::Command::new("cargo")
           .args(["run", "--bin", "__tile_runner", "--", "bench"])
           .stdout(Stdio::piped())
           .spawn()?;
       // Stream JSON lines → render
   }
   ```

10. **JSON Lines protocol** (from TOOLCHAIN_DESIGN.md):
    ```json
    {"type":"start","runner_version":"0","total_benches":42}
    {"type":"bench","name":"unary/exp","dtype":"f16","mt_gbps":1234.5,...}
    {"type":"test","name":"unary/exp","dtype":"f16","passed":true,...}
    {"type":"done","bench_passed":41,"bench_failed":1,...}
    ```

11. **Remove GPU dependencies from CLI Cargo.toml:** After this phase, `metaltile-cli` depends only on `metaltile-config` (or `metaltile-core` for types) — no `metaltile-codegen`, `metaltile-runtime`, `metaltile-std`.

12. **`tile build`** now spawns `__tile_runner -- build` to compile-check. The `--emit` path generates files to disk from the project side. The CLI just orchestrates.

### Phase 4: CLI Ergonomics + Filtering

**Goal:** Match Foundry's CLI UX quality.

13. **Regex + glob test filtering:**
    ```rust
    // New args in crate::TestArgs / BenchArgs:
    --match-test REGEX        # Only run tests matching this regex
    --no-match-test REGEX     # Exclude
    --match-kernel REGEX      # Only run kernels matching this regex
    --no-match-kernel REGEX
    ```
    Current `--filter` (substring) retained as shorthand.

14. **`tile build --watch`** — watchexec-based file watching:
    - Default watch paths: `src/` and `tile.toml`
    - On change: recompiles project, re-runs build/bench/test
    - Passes through subprocess model (spawns project runner)

15. **Progress bars** — `indicatif` for compilation steps:
    ```
    Compiling tile-runner ████████░░░░░░ 12/42 kernels
    Compiling metallib     ██████████████ done
    ```

16. **Friendly error messages:**
    - `tile bench` with no `tile.toml` found → suggest `tile init`
    - Compilation failure in subprocess → display stderr snippet
    - GPU init failure → specific recovery suggestion

### Phase 5 (Post-Rewrite)

- `tile install` / `tile update` improvements
- `tile new` — scaffold a new kernel project
- `tile check` — validate project configuration
- Auto-generated shell completions
- `--machine` structured output for CI/agent use

---

## 6. File-by-File Migration Plan

### Phase 1 (Config — minimal risk)

| File | Action |
|---|---|
| `crates/metaltile-config/Cargo.toml` | **Create** — `serde`, `toml`, `thiserror` |
| `crates/metaltile-config/src/lib.rs` | **Create** — `TileConfig` struct, `ConfigLoader` |
| `crates/metaltile-config/src/discover.rs` | **Create** — walk-up discovery logic |
| `crates/metaltile-cli/Cargo.toml` | **Add** `metaltile-config` dep |
| `crates/metaltile-cli/src/main.rs` | **Modify** — load config at startup, pass to commands |
| `crates/metaltile-cli/src/cmd/*.rs` | **Modify** — accept `&TileConfig`, merge with CLI args |
| `Cargo.toml` (workspace) | **Add** `"crates/metaltile-config"` |

### Phase 2 (Trait abstraction — medium risk)

| File | Action |
|---|---|
| `crates/metaltile-core/src/runner.rs` | **Create** — `BenchHarness`, `TestHarness` traits |
| `crates/metaltile/src/runner.rs` | **Create** — `InProcessRunner` impl delegating to current logic |
| `crates/metaltile-cli/src/cmd/bench.rs` | **Refactor** — use `BenchHarness` trait instead of direct calls |
| `crates/metaltile-cli/src/cmd/test.rs` | **Refactor** — use `TestHarness` trait |
| `crates/metaltile-cli/src/cmd/build.rs` | **Refactor** — use `BuildHarness` trait |

### Phase 3 (Subprocess — highest risk, biggest payoff)

| File | Action |
|---|---|
| `crates/metaltile/src/runner/mod.rs` | **Create** — protocol library (`run()`, `Args`, JSON schema) |
| `crates/metaltile/src/runner/harness_gen.rs` | **Create** — harness source generation |
| `crates/metaltile-cli/src/subprocess.rs` | **Create** — `SubprocessRunner` impl of `BenchHarness` |
| `crates/metaltile-cli/src/cmd/bench.rs` | **Replace** body — spawn subprocess, stream JSON |
| `crates/metaltile-cli/src/cmd/test.rs` | **Replace** body — same pattern |
| `crates/metaltile-cli/src/cmd/build.rs` | **Replace** body — same pattern |
| `crates/metaltile-cli/src/suite_printer.rs` | **Refactor** — consume JSON protocol instead of `OpResult` |
| `crates/metaltile-cli/Cargo.toml` | **Remove** `metaltile-codegen`, `metaltile-runtime`, `metaltile-std` deps |

### Phase 4 (Ergonomics)

| File | Action |
|---|---|
| `crates/metaltile-cli/src/cmd/test/filter.rs` | **Create** — Regex + glob filter types |
| `crates/metaltile-cli/src/cmd/watch.rs` | **Create** — watchexec watch loop |
| `crates/metaltile-cli/src/cmd/init.rs` | **Create** — `tile init` scaffold |
| `crates/metaltile-cli/src/progress.rs` | **Create** — indicatif progress helpers |

---

## 7. Risk Assessment

| Risk | Mitigation |
|---|---|
| **Subprocess protocol version mismatch** (CLI vs runner from different tile versions) | Versioned JSON protocol with graceful degradation. CLI checks `runner_version` field in `start` message. |
| **Harness generation race** (two `tile` invocations in same project) | Lock file in `target/tile/.lock` during generation. Or generate once, check staleness by comparing source hash. |
| **Performance regression** (subprocess overhead vs in-process) | Subprocess startup overhead ~50-200ms for `cargo run`. Add `--persist` to keep subprocess alive across multiple invocations. For `make bench-vv` workflows, the GPU timing dominates anyway. |
| **Cross-compilation complexity** (running on non-Mac CI for Metal projects) | CLI gracefully reports "Metal not available" when spawning runner on non-macOS. `tile build --emit` and `tile inspect` still work. |
| **Kernel author confusion** ("why is there a `__tile_runner` binary in my cargo output?") | `__tile_runner` is gitignored via `target/`. Doc note: "this is auto-generated, never check it in." |
| **Breaking the current workflow during migration** | Ship phase 1 (config) independently. Phase 2 (trait abstraction) is internal refactoring that preserves behavior. Phase 3 (subprocess) is the only user-facing change. |

---

## 8. Success Criteria

- [ ] `tile bench` works without GPU deps linked in CLI
- [ ] `tile test` spawns project runner subprocess
- [ ] `tile build --emit all -o <dir>` generates artifacts via subprocess
- [ ] `tile.toml` controls bench iterations, dtypes, SDK
- [ ] Changing a kernel and running `tile bench` does NOT rebuild the CLI
- [ ] All current CLI flags preserved and forward-compatible
- [ ] JSON protocol renders identical terminal output to current `SuitePrinter`
- [ ] `tile bench --diff` and `tile diff` continue to work (they read files, not GPU)
- [ ] `tile inspect` continues to work (uses inventory directly)
- [ ] `tile device` continues to work (GPU query, no subprocess needed)
- [ ] Filtering (`--filter`, `--match-test`, `--match-kernel`) works through subprocess

---

## 9. What Stays the Same

- **`tile device`** — GPU info query. No subprocess needed (it's a simple Metal query).
- **`tile snap`** — file I/O only. No GPU, no subprocess change.
- **`tile diff`** — pure data comparison of JSON files. No change needed.
- **`tile inspect`** — uses inventory. Could either link in-process or spawn subprocess. Leaving in-process for now keeps it fast for debugging.
- **`tile update`** — self-update. No change.
- **`suite_printer.rs`** — the rendering logic is good. It just needs to consume JSON protocol events instead of `OpResult` directly.
- **`term.rs`** — anstyle/anstream terminal handling. No change needed.
- **`git.rs`** — dirty-tree check still works. No change needed.
- **`error.rs`** — `CliError` enum is fine, just needs fewer variants as GPU deps move out.
- **`Makefile`** — `make tile` passthrough still works (it calls `cargo run -p metaltile-cli`).
