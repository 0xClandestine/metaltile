# `tile` CLI vs `forge` CLI — Feature Gap Analysis

> **Purpose:** Identify every feature, flag, output convention, and UX pattern in
> Foundry's `forge` that is missing from our `tile` CLI. The goal is to make the
> `tile` developer experience as close to `forge` as is practically meaningful
> for a GPU-kernel toolchain (metal shaders vs. Solidity contracts).
>
> Generated: 2026-06-02
> Source: `metaltile-cli/src/main.rs` + Foundry `crates/forge/src/` (HEAD @ `/tmp/foundry`)

---

## Ground Truth: What We Have Today

### `tile` subcommands (9)

| Subcommand   | Status      | Description                                            |
|-------------|-------------|--------------------------------------------------------|
| `bench`     | Implemented | Benchmark kernels vs MLX reference (GB/s)              |
| `test`      | Implemented | Run `#[test_kernel]` correctness against CPU oracle    |
| `build`     | Implemented | Compile kernels → MSL, emit metallib/Swift/manifest    |
| `inspect`   | Implemented | Print IR, MSL, per-pass stats for registered kernels   |
| `device`    | Implemented | Show GPU device info and feature flags                 |
| `snap`      | Implemented | Save bench results as regression baseline JSON         |
| `diff`      | Implemented | Compare bench results against saved baseline           |
| `update`    | Implemented | Self-update the `tile` binary from GitHub release/source |
| `init`      | Implemented | Scaffold a new MetalTile kernel project                |

### Shared filter flags (our `FilterArgs`)

- `--filter` / `-f` — substring (case-insensitive)
- `--match-name` / `--mn` — regex against kernel name
- `--no-match-name` / `--nmn` — exclude by regex (name)
- `--match-group` / `--mg` — regex against op group (path component before `/`)
- `--no-match-group` / `--nmg` — exclude by regex (group)
- `--match-path` / `--mp` — glob against source file
- `--no-match-path` / `--nmp` — exclude by glob (source file)

All per-command flags are documented in [`crates/metaltile-cli/src/main.rs`](../crates/metaltile-cli/src/main.rs).

### Our output format convention

- Colored terminal output using `anstyle` with custom `SuitePrinter`
- Banner line per subcommand (e.g. `tile bench · Apple M5 Max  warmup=1 runs=3`)
- Table rows with `✓` / `✗` checkmarks
- Summary line at end with counts
- JSON output via `--json` flag (selected subcommands)
- Stderr for diagnostics, stdout for structured output
- Tracing subscriber via `METALTILE_DEBUG` env var

---

## `forge` subcommands — Complete Canonical List

From `/tmp/foundry/crates/forge/src/opts.rs`:

| Subcommand       | Alias    | Foundry Category     | Description                                    |
|-----------------|----------|----------------------|------------------------------------------------|
| `test`          | `t`      | Core                 | Run project tests (fuzz, invariant, debugger)  |
| `script`        | —        | Core                 | Run smart contract as a script / deploy         |
| `coverage`      | —        | Core                 | Generate coverage reports                       |
| `bind`          | `bi`     | Code Generation      | Generate Rust bindings for contracts            |
| `build`         | `b`, `compile` | Core          | Build smart contracts                           |
| `clone`         | —        | Source Mgmt          | Clone a contract from Etherscan                 |
| `update`        | `u`      | Dependencies         | Update one or multiple dependencies             |
| `install`       | `i`, `add` | Dependencies       | Install one or multiple dependencies            |
| `remove`        | `rm`     | Dependencies         | Remove one or multiple dependencies             |
| `remappings`    | `re`     | Config               | Show inferred remappings                        |
| `verify-contract` | `v`    | Verification         | Verify smart contract on Etherscan              |
| `verify-check`  | `vc`     | Verification         | Check verification status                       |
| `verify-bytecode` | `vb`   | Verification         | Verify deployed bytecode vs source              |
| `create`        | `c`      | Deployment           | Deploy a smart contract                         |
| `init`          | —        | Project              | Create a new Forge project                      |
| `completions`   | `com`    | Shell                | Generate shell completions                      |
| `clean`         | `cl`     | Build                | Remove build artifacts and cache                |
| `cache`         | —        | Build                | Manage Foundry cache (clean / ls)               |
| `snapshot`      | `s`      | Testing              | Gas snapshot of each test's gas usage           |
| `config`        | `co`     | Config               | Display current config                          |
| `flatten`       | `f`      | Source Mgmt          | Flatten a source file + imports into one file   |
| `fmt`           | —        | Formatting           | Format Solidity source files                    |
| `lint`          | `l`      | Linting              | Lint Solidity source files                      |
| `inspect`       | `in`     | Introspection        | Get specialized info about a smart contract     |
| `tree`          | `tr`     | Dependencies         | Display dependency graph as a tree              |
| `geiger`        | —        | *(deprecated)*       | Detects unsafe cheat codes (alias for lint)     |
| `doc`           | —        | Documentation        | Generate documentation                          |
| `selectors`     | `se`     | Utilities            | Function selector utilities (collision/upload/list/find/cache) |
| `generate`      | —        | Scaffolding          | Generate scaffold files (e.g. test)             |
| `compiler`      | —        | Compiler             | Compiler utilities                              |
| `soldeer`       | —        | Dependencies         | Soldeer dependency manager                      |
| `eip712`        | —        | Utilities            | Generate EIP-712 struct encodings               |
| `bind-json`     | —        | Code Generation      | JSON serialization bindings for structs         |

**Total: 31 subcommands** (including `completions` and non-Core utilities).

---

## Gap Analysis

### Tier 1 — High priority (directly analogous to MetalTile concerns)

| Missing Feature | `forge` Equivalent | Rationale for `tile` |
|----------------|-------------------|---------------------|
| **Watch mode** (`--watch`) on build | `forge build --watch`, `forge test --watch`, etc. | Iterating on GPU kernels benefits enormously from auto-rebuild on file change. Currently user must re-run `tile build` manually. Our `WatchArgs` already exists in forge's pattern. |
| **Shell completions** | `forge completions <shell>` | Standard developer UX. Users expect `tile <TAB>` to work after install. |
| **Clean command** | `forge clean` | `target/` and build artifacts accumulate. A `tile clean` that removes `.metallib`, air files, or snapshots would be expected. |
| **Tree / dependency graph** | `forge tree` | If tile projects grow multi-crate, a dependency tree visualization would be useful. Lower priority today but worth noting. |
| **Config display** | `forge config` | Show the effective merged config (tile.toml + env vars + defaults). Users debugging `tile.toml` issues need this. |

### Tier 2 — Medium priority (improves ergonomics)

| Missing Feature | `forge` Equivalent | Rationale |
|----------------|-------------------|-----------|
| **`--help` / manpage-quality docs** | `forge <cmd> --help` (good clap docs) | Our clap docs are minimal. Should match forge's verbosity. |
| **Fmt subcommand** | `forge fmt` | If there's a kernel DSL or `#[kernel]` representation that gets printed/serialized, a canonical formatter would help (pre-commit hooks, CI). |
| **Global `--json` flag** (not per-command) | `forge --json` via `GlobalArgs` | Forge sets JSON output mode globally, available to every subcommand. We only have `--json` on `bench` and `device`. |
| **Global `--quiet` / `-q`** | `forge -q` | Suppress all log output. |
| **Global `--color`** | `forge --color <always|auto|never>` | Useful for CI and piping. |
| **Global `--threads` / `-j`** | `forge -j <N>` | Control parallelism for multi-kernel compilation. |

### Tier 3 — Not applicable (domain-specific to Solidity/contracts)

These forge features have no MetalTile analogue and should **not** be emulated:

| `forge` Feature | Reason Not Applicable |
|----------------|----------------------|
| `script` | On-chain Solidity script deployment |
| `coverage` | Code coverage for Solidity; not meaningful for GPU shaders |
| `bind` / `bind-json` | Rust ABI generation for EVM contracts |
| `clone` | Clone contract from Etherscan |
| `install` / `remove` / `update` | Solidity dependency management via git submodules |
| `verify-*` | Contract verification on Etherscan |
| `create` | Deploy contract to a blockchain |
| `snapshot` (gas) | Gas cost snapshot per test — our `snap` is for benchmark regression |
| `selectors` | EVM function selector collision detection |
| `eip712` | EIP-712 struct encoding |
| `soldeer` | Solidity-specific package manager |
| `geiger` | Detects unsafe cheat codes |
| `doc` | Solidity NatSpec doc generation |
| `cache` | Etherscan/ABI cache (not relevant to Metal compilation) |
| `compiler` | Solc compiler version mgmt |

### Output Format Differences

| Aspect | `forge` | `tile` | Gap |
|--------|---------|--------|-----|
| **Verbosity levels** | `-v`, `-vv`, `-vvv`, `-vvvv`, `-vvvvv` (global) | `-v` only on `bench` and `build` (per-command) | Should make verbosity global and consistent |
| **JSON output** | `--json` global flag | `--json` on `bench` and `device` only | Missing JSON on `test`, `build`, `inspect` |
| **Machine-mode** | `--machine` structured JSON envelope for agents | None | Would be useful for CI/automation |
| **Progress display** | Comfy-table gas reports, progress bars for compilation | `SuitePrinter` with custom terminal tables | Forge has richer progress for long operations |
| **Exit codes** | Canonical exit codes defined in `foundry_cli::ExitCode` | None explicit (just `Ok(())` / `Err(...)`) | Should define proper exit codes |
| **Error presentation** | Structured diagnostics with source spans | Simple error string + `eprintln!` | Could improve with colored source-context |

### Filter Flag Differences

| Concept | `forge test` | `tile` (all commands) | Gap |
|---------|-------------|----------------------|-----|
| Test name filter | `--match-test REGEX` / `--mt` | `--filter STR` (substring) + `--match-name REGEX` / `--mn` | Forge uses regex directly on test functions; tile has both substring and regex on kernel name. Feature-comparable. |
| Contract / group filter | `--match-contract REGEX` / `--mc` | `--match-group REGEX` / `--mg` | Equivalent concept, different naming |
| Source file filter | `--match-path GLOB` / `--mp` | `--match-path GLOB` / `--mp` | **Same** ✓ |
| Negative source filter | `--no-match-path GLOB` / `--nmp` | `--no-match-path GLOB` / `--nmp` | **Same** ✓ |
| Negative test filter | `--no-match-test REGEX` | `--no-match-name REGEX` | Equivalent |
| Negative contract filter | `--no-match-contract REGEX` | `--no-match-group REGEX` | Equivalent |
| Coverage inverse | `--no-match-coverage` | N/A | Not relevant (no coverage) |
| **Test path positional** | `forge test <path>` (glob shortcut for `--match-path`) | N/A | Missing — convenient `tile test <path>` would be nice |

**Key difference in naming:** Forge uses `--match-test`, `--match-contract` domain-specific terminology. Tile uses `--match-name`, `--match-group` which are more generic. These are semantically equivalent but forge's naming is more discoverable for their domain.

---

## Recommended Action Items

### Do Now (quick wins, high impact)

1. **Add global flags** (`-q`, `--color`, `--json`, `-j`)
2. **Add `tile clean`** — remove build artifacts (air files, metallibs, snapshots)
3. **Add `tile completions <shell>`** — shell completion generation (clap built-in)
4. **Standardize exit codes** — define `TileExitCode` enum
5. **Improve `--help` text** — match forge's verbosity per command

### Do Next (moderate effort, high polish)

6. **Add `--watch` mode to `tile build`** — auto-rebuild on file change
7. **Add implicit positional path to `tile test` / `tile bench`** — `tile bench softmax` equivalent to `tile bench --filter softmax`
8. **Make verbosity global** — `-v`/`-vv`/`-vvv` works on every subcommand, not just `bench`
9. **Add JSON output to `tile test`, `tile build`, `tile inspect`**
10. **Add `tile config`** — print effective merged config

### Future (lower priority)

11. **Add `tile tree`** — dependency graph visualization (when multi-crate projects materialize)
12. **Add `tile fmt`** — canonical formatting for `#[kernel]` or emitted MSL
13. **Add `--machine` mode** — structured JSON envelope for automation
14. **Consider renaming filter flags to match forge naming** — `--match-contract` → `--match-group`, etc. (only if domain user research shows confusion)

### Alignment with forge filter flags

Forge's `test` filter args have an implicit prefix (`match-contract`, `match-test`, `match-path`). Our filter args are named for a generic kernel context (`match-name`, `match-group`, `match-path`). These are semantically equivalent. **Recommendation:** Keep tile's naming — it's more accurate for the GPU kernel domain. The only missing flag is the positional path shortcut (`tile test <path>` → `--match-path`).

---

## How to Add Each Feature

### 1. Global flags (`-q`, `--color`, `--json`, `-j`)

Add a `GlobalArgs` struct (like forge's `GlobalArgs` in `foundry_cli::opts`) flattened into our `Cli` struct:

```rust
#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Parser)]
struct GlobalArgs {
    #[arg(short, long, global = true)]
    quiet: bool,
    #[arg(long, global = true)]
    json: bool,
    #[arg(long, global = true, value_enum)]
    color: Option<ColorChoice>,
    #[arg(short = 'j', long, global = true)]
    threads: Option<usize>,
}
```

Then thread `global` state through to each subcommand handler.

### 2. `tile clean`

```rust
#[derive(Args)]
struct CleanArgs {
    /// Also remove regression baselines
    #[arg(long)]
    snapshots: bool,
    /// Remove all build artifacts
    #[arg(long)]
    all: bool,
}
```

Delete `target/tile-build-air/`, `.tile-snapshots/`, and any emit output.

### 3. `tile completions`

Add to `Command` enum (forge uses it in dispatch, clap built-in):

```rust
Completions {
    #[arg(value_enum)]
    shell: clap_complete::Shell,
}
```

Dispatch: `clap_complete::generate(shell, &mut Cli::command(), "tile", &mut std::io::stdout());`

### 4. Standardized exit codes

```rust
#[repr(i32)]
pub enum TileExitCode {
    Success = 0,
    TestFailure = 1,
    BuildFailure = 2,
    Regression = 3,
    ConfigError = 10,
}
```

### 5. Positional path shortcut

In `TestArgs` and `BenchArgs`:

```rust
/// Path filter (shortcut for --match-path)
#[arg(value_hint = ValueHint::FilePath)]
pub path: Option<GlobMatcher>,
```

If present, synthesize a `--match-path` filter.

### 6. Watch mode

Use `watchexec` (same as forge) or `notify` crate:

```rust
#[command(flatten)]
pub watch: WatchArgs,
```

Where `WatchArgs` has `--watch`, `--no-restart`, `--run-all`.

---

## Appendix: Filter Flag Cross-Reference

| tile flag | forge flag | Notes |
|-----------|-----------|-------|
| `--filter <str>` | — | tile-only substring convenience filter |
| `--match-name <regex>` | `--match-test <regex>` | tile: kernel name; forge: test function name |
| `--no-match-name <regex>` | `--no-match-test <regex>` | |
| `--match-group <regex>` | `--match-contract <regex>` | tile: op group; forge: contract name |
| `--no-match-group <regex>` | `--no-match-contract <regex>` | |
| `--match-path <glob>` | `--match-path <glob>` | **Identical** |
| `--no-match-path <glob>` | `--no-match-path <glob>` | **Identical** |
| — | `--no-match-coverage <regex>` | N/A (no coverage) |
| — | `forge test <path>` (positional) | Missing from tile |

---

## Appendix: Verbosity Level Comparison

| Level | forge meaning (test context) | tile meaning (bench context) |
|-------|---------------------------|------------------------------|
| `-v`  | Print logs for all tests  | Show occupancy/register profile columns |
| `-vv` | Print logs + execution traces for failing tests | Show GPU timing stats columns |
| `-vvv`| Execution traces for all tests + setup traces for failing tests | — |
| `-vvvv` | All traces + storage changes | — |
| `-vvvvv` | All traces + backtraces with line numbers | — |

**Gap:** tile only uses 2 levels (`-v`, `-vv`). forge uses 5. While GPU benchmarking doesn't need EVM-level verbosity depth, standardizing the `-v` count and making `-vvvv` mean "dump all intermediate IR + MSL" could be useful for debugging.