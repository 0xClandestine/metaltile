# `tile` CLI + Config — Implementation Plan

> Synthesized from the CLI gap analysis and the config gap analysis. Orders work
> by dependency, impact, and effort. Each phase is self-contained and shippable.

---

## Phase 0 — Foundation (config + project structure)

**Theme:** Fix the config system so everything else builds on a solid base.

### 0a. Parent-directory config walk

**Files:** `crates/metaltile-cli/src/config.rs`

Walk up from CWD looking for `tile.toml`. Set `project_path` to the containing
directory when found. This is what lets `tile bench` work from any subdirectory
of a project.

```rust
// In ConfigLoader::load():
fn find_tile_toml() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut dir: &Path = cwd.as_path();
    loop {
        let candidate = dir.join("tile.toml");
        if candidate.exists() {
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
}
```

**Dependencies:** None. Self-contained change to `ConfigLoader`.
**Files touched:** 1
**Tests:** Find/create a project root with `tile.toml`, spawn subdir, verify load picks it up.

---

### 0b. CLI args as figment Providers

**Files:** `crates/metaltile-cli/src/config.rs`, every `*Args` struct in `cmd/*.rs`

Implement `figment::Provider` (or at minimum a merge function) for each
subcommand's args so CLI flags properly override tile.toml and env vars.

Current state:
```
tile.toml verbose=1  +  tile build --verbose 2
  → CLI reads verbose from tile.toml (1)
  → bench's run() reads args.verbose separately (2)
  → confusion
```

Target state (figment merge):
```
Default (verbose=0)
  → tile.toml (verbose=1)
  → TILE_VERBOSE env var
  → CLI --verbose (2)    ← wins
```

```
  Harness.config.verbose = 2  (single source of truth)
```

**Approach:** Implement `Provider` on each args struct that serialises the
relevant subset of its fields. Then in `main()`:

```rust
let figment = ConfigLoader::build_figment()  // defaults + tile.toml + env
    .merge(&cli.global)?
    .merge(&cli.command)?;  // subcommand-specific overrides
let config: TileConfig = figment.extract()?;
```

Forge uses proc macros (`merge_impl_figment_convert!`) for this. We can start
with manual `Provider` impls on the 3 most-used commands and add a macro later.

**Dependencies:** Phase 0a (so the figment knows where tile.toml lives).
**Files touched:** 10+ (config + each args struct)
**Tests:** Write a tile.toml, pass conflicting CLI flag, verify config value is correct.

---

### 0c. Environment variable interpolation

**Files:** `crates/metaltile-cli/src/config.rs` (new `resolve` module or inline)

Allow config values to reference env vars:

```toml
runner_binary = "${HOME}/bin/__tile_runner"
project_path = "${TILE_PROJECT_DIR:-./kernels}"
```

Simple regex-based replacement:

```rust
fn interpolate(input: &str) -> Result<String, InterpolationError> {
    let re = Regex::new(r"\$\{([^}:]+)(?::-(.+?))?\}")?;
    // Replace ${VAR} with env var value
    // Support ${VAR:-default} syntax
}
```

Apply at the figment level via a custom `Provider` wrapper that transforms
string values before extraction.

**Dependencies:** Phase 0a. Can be a standalone wrapper.
**Files touched:** 1–2
**Tests:** Set env var, write config referencing it, verify extracted value.

---

### 0d. Warnings on config load

**Files:** `crates/metaltile-cli/src/config.rs`

Collect warnings (unknown keys, deprecated fields, env vars that failed to parse)
into a `warnings: Vec<String>` field. Print them at the end of each command if
non-empty.

```rust
pub struct TileConfig {
    // ... existing fields ...
    #[serde(skip)]
    pub warnings: Vec<String>,
}
```

Figment can report unknown keys via `Metadata` + custom error handling. Forge
uses a `WarningsProvider` that collects them.

**Dependencies:** Phase 0a, 0b.
**Files touched:** 1
**Tests:** Write tile.toml with an unknown key, verify warning is collected.

---

## Phase 1 — CLI Polishing (quick wins, high visibility)

**Theme:** Add the forge-quality-of-life features that don't require the deep
config changes from Phase 0.

### 1a. Global flags

**Files:** `crates/metaltile-cli/src/main.rs`, new `global.rs`

Add `GlobalArgs` struct flattened into `Cli`:

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

Thread `global` state through `Harness` so every subcommand can access it.
Replace per-command `--json` flags with the global one (keep backward compat
with alias).

**Dependencies:** None (can land before Phase 0).
**Files touched:** 3–4 (main.rs, harness.rs, new global.rs)
**Tests:** Parse CLI with each flag, verify `Harness.global` values.

---

### 1b. `tile clean`

**Files:** `crates/metaltile-cli/src/cmd/clean.rs` (new), `cmd/mod.rs`, `main.rs`

```rust
#[derive(Args)]
struct CleanArgs {
    #[arg(long)]
    snapshots: bool,      // remove .tile-snapshots/
    #[arg(long)]
    all: bool,            // remove everything
}
```

Remove `target/tile-build-air/`, `.tile-snapshots/`, any emitted artifacts in
`out` dirs.

**Dependencies:** None.
**Files touched:** 3
**Tests:** Create temp dir with known artifacts, run clean, verify removal.

---

### 1c. `tile completions <shell>`

**Files:** `crates/metaltile-cli/src/main.rs`, `Cargo.toml` (add `clap_complete`)

Add to `Command` enum:

```rust
/// Generate shell completions
#[command(visible_alias = "com")]
Completions {
    #[arg(value_enum)]
    shell: clap_complete::Shell,
}
```

Dispatch:
```rust
clap_complete::generate(shell, &mut Cli::command(), "tile", &mut std::io::stdout());
```

**Dependencies:** None.
**Files touched:** 1–2
**Tests:** Run `tile completions bash`, verify output has `_tile()` function.

---

### 1d. Standardized exit codes

**Files:** `crates/metaltile-cli/src/error.rs`

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

Return these from `main()` instead of `Ok(())` / `Err(...)`.

**Dependencies:** None.
**Files touched:** 2 (error.rs, main.rs)
**Tests:** Run failing bench, verify exit code 3. Run JSON output with error, verify envelope.

---

### 1e. Improve `--help` text

**Files:** Every `*Args` struct + `main.rs`

Forge's help text is comprehensive per command. Our text is minimal. Add:
- `about` / `long_about` annotations on every subcommand
- `after_help` with links to docs
- `next_help_heading` groupings for args (like forge does with "Test filtering",
  "Build options", "Display options")

**Dependencies:** None.
**Files touched:** 9 (all args structs)
**Tests:** Visual review of `tile bench --help`, `tile build --help`, etc.

---

## Phase 2 — Ergonomics (power-user features)

**Theme:** Features that change how users interact with tile day-to-day.

### 2a. Positional path shortcut

**Files:** `crates/metaltile-cli/src/cmd/test.rs`, `cmd/bench.rs`, `FilterArgs`

```rust
/// Source file path filter (shortcut for --match-path)
#[arg(value_hint = ValueHint::FilePath)]
pub path: Option<GlobMatcher>,
```

If present, merge into the existing filter chain:

```rust
if let Some(path) = &args.path {
    // Synthesise a new FilterSpec that ANDs the existing filter with
    // this path glob (or synthesise a match_path entry)
}
```

This lets users write `tile bench softmax` or `tile test src/kernels/` directly.

**Dependencies:** Phase 0b (so path CLI flag merges correctly).
**Files touched:** 2–3
**Tests:** `tile bench softmax` → only runs kernels containing "softmax".

---

### 2b. Watch mode (`--watch` on `tile build`)

**Files:** `crates/metaltile-cli/src/cmd/watch.rs` (new), `cmd/build.rs`, `cmd/test.rs`, `Cargo.toml` (add `watchexec`)

```
tile build --watch
  → watches src/ for changes
  → re-runs tile build on every file change
```

Use `watchexec` (same as forge). Template:

```rust
#[derive(Clone, Debug, Default, Parser)]
pub struct WatchArgs {
    #[arg(short, long, num_args(0..))]
    pub watch: Option<Vec<PathBuf>>,
    #[arg(long)]
    pub no_restart: bool,
    #[arg(long)]
    pub run_all: bool,
}
```

Flatten into `BuildArgs` and `TestArgs`.

**Dependencies:** Phase 0b (so watch args participate in config merge).
**Files touched:** 4–5 + Cargo.toml
**Tests:** Manual — start watch, touch file, verify rebuild triggers.

---

### 2c. Global verbosity

**Files:** `crates/metaltile-cli/src/main.rs`, `GlobalArgs`, `Harness`

Make `-v` / `-vv` / `-vvv` a global flag (like forge) instead of per-command.
Remove the per-command `verbose` fields on `BenchArgs` and `BuildArgs`.

```rust
#[arg(global = true, short, long, action = ArgAction::Count)]
verbose: u8,
```

Store in `Harness.global.verbose`. Each command reads from there.

**Dependencies:** Phase 1a (GlobalArgs).
**Files touched:** 4–5
**Tests:** `tile -vv bench` and `tile bench -vv` both set verbose=2.

---

### 2d. JSON output on all subcommands

**Files:** `crates/metaltile-cli/src/main.rs`, `cmd/test.rs`, `cmd/build.rs`, `cmd/inspect.rs`

Replace per-command `--json` flags with the global `--json` from Phase 1a (with
backward-compat aliases). Add JSON serializers for `test`, `build`, and
`inspect` output.

Test JSON schema (following forge's `--machine` pattern):
```json
{
  "command": "test",
  "passed": 42,
  "failed": 0,
  "total": 42,
  "results": [
    { "name": "mt_relu [f32]", "status": "passed", "max_abs_err": 3.2e-7 }
  ]
}
```

Build JSON schema:
```json
{
  "command": "build",
  "kernels": 120,
  "ok": 120,
  "errors": 0,
  "elapsed_ms": 4520
}
```

**Dependencies:** Phase 1a (global `--json` flag).
**Files touched:** 5–6
**Tests:** `tile --json test` outputs valid JSON. `tile --json build` outputs valid JSON.

---

## Phase 3 — Advanced UX (config display, machine mode)

**Theme:** Diagnostic and automation polish.

### 3a. `tile config`

**Files:** `crates/metaltile-cli/src/cmd/config.rs` (new), `cmd/mod.rs`, `main.rs`

```rust
#[derive(Args)]
struct ConfigArgs {
    #[arg(long)]
    basic: bool,       // simplified output
}
```

Dispatches `Harness.config.to_string_pretty()` or JSON output.

```rust
pub fn run(args: &ConfigArgs) -> Result<()> {
    let config = Harness::from_config().config;
    if json_output() {
        println!("{}", serde_json::to_string_pretty(&config)?);
    } else {
        println!("{}", config.to_string_pretty()?);
    }
    Ok(())
}
```

Add `to_string_pretty()` to `TileConfig`:

```rust
impl TileConfig {
    pub fn to_string_pretty(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}
```

**Dependencies:** Phase 0b (so config is the canonical merged view), Phase 1a (global --json).
**Files touched:** 3–4
**Tests:** `tile config` → valid TOML. `tile --json config` → valid JSON.

---

### 3b. `--machine` mode (structured JSON envelope)

**Files:** `crates/metaltile-cli/src/main.rs`, `GlobalArgs`, error handling

Structured JSON envelope for every command + exit, following forge's `--machine`
spec:

```json
{
  "schema_id": "https://metaltile.dev/machine/1",
  "binary": { "name": "tile", "version": "0.1.0" },
  "command_id": "tile.bench",
  "status": "success",
  "data": { ... },
  "warnings": [...],
  "errors": [...]
}
```

Inspired by forge's `JsonEnvelope`. Implement as a wrapper layer in `main()`:

```rust
fn main() {
    let machine = pre_parse_machine_flag();
    let result = run();
    if machine {
        print_machine_envelope(result);
    } else {
        // current path
    }
}
```

**Dependencies:** Phase 1a (GlobalArgs), Phase 1d (exit codes).
**Files touched:** 3–4
**Tests:** `tile --machine bench --json` outputs machine envelope.

---

### 3c. `tile tree`

**Files:** `crates/metaltile-cli/src/cmd/tree.rs` (new), `cmd/mod.rs`, `main.rs`

Low priority, only useful when multi-crate projects exist. Would visualize
crate dependency graph for `metaltile-std` / kernel crates.

```rust
#[derive(Args)]
struct TreeArgs {
    #[arg(long)]
    no_dedupe: bool,
    #[arg(long, default_value = "utf8")]
    charset: Charset,
}
```

**Dependencies:** None (standalone command).
**Files touched:** 3

---

## Phase 4 — Config Evolution (profiles, sub-tables)

**Theme:** Make tile.toml structured enough for real projects.

### 4a. Sub-tables (`[bench]`, `[build]`, `[runner]`)

**Files:** `crates/metaltile-cli/src/config.rs`

Move top-level `runs`, `warmup_runs` into `[bench]` table, and `runner_binary`
into `[runner]`. Use `#[serde(flatten)]` to keep the Rust struct access flat
while exposing TOML nesting.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileConfig {
    pub project_path: Option<String>,
    pub verbose: u8,

    #[serde(flatten)]
    pub bench: BenchConfig,

    #[serde(flatten)]
    pub build: BuildConfig,

    #[serde(flatten)]
    pub runner: RunnerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub runs: usize,
    pub warmup_runs: usize,
    pub target_device: Option<String>,
    pub upload_baseline: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    pub default_dtypes: Vec<String>,
    pub sdk: String,
    pub time_passes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    pub binary: String,
    pub extra_args: Vec<String>,
}
```

**Backward compat:** Accept both top-level `runs` and `[bench] runs` during a
transition period with a warning on the deprecated path.

**Dependencies:** Phase 0d (load warnings for deprecation notices).
**Files touched:** 1 (config.rs) + tile.toml docs
**Tests:** Write both old flat and new nested config, verify same extracted values.

---

### 4b. Profile system

**Files:** `crates/metaltile-cli/src/config.rs`

Use figment's built-in profile support:

```toml
[profile.default]
verbose = 0
[profile.default.bench]
runs = 3
warmup_runs = 1

[profile.ci]
verbose = 0
[profile.ci.bench]
runs = 15
warmup_runs = 3
```

Select via `TILE_PROFILE=ci` env var or `--profile ci` CLI flag.

```rust
impl ConfigLoader {
    pub fn load_with_profile(profile: Option<&str>) -> Result<TileConfig, ...> {
        let selected = profile
            .or_else(|| std::env::var("TILE_PROFILE").ok())
            .unwrap_or_else(|| "default".into());

        Figment::from(Serialized::defaults(TileConfig::default()))
            .merge(Toml::file("tile.toml"))
            .select(Profile::new(&selected))
            .merge(Env::prefixed("TILE_"))
            .extract()
    }
}
```

**Dependencies:** Phase 4a (sub-tables), Phase 0b (CLI merge), Phase 0d (load warnings).
**Files touched:** 2 (config.rs, main.rs for --profile flag)
**Tests:** Write multi-profile tile.toml, load with TILE_PROFILE=ci, verify bench.runs=15.

---

### 4c. Extends / inheritance

**Files:** `crates/metaltile-cli/src/config.rs`

```toml
extends = "~/.metaltile/global.toml"
```

or with strategy:

```toml
extends = { path = "../team-base.toml", strategy = "no-collision" }
```

Forge's strategies: `extend-arrays` (concat), `replace-arrays` (replace),
`no-collision` (error on overlap). Tile would mostly use `extend-arrays`.

Implement as a figment `Provider` that loads the base file first, then merges
the local config on top.

**Dependencies:** Phase 4a (sub-tables so merging doesn't collide flat fields).
**Files touched:** 1–2
**Tests:** Write base.toml + local tile.toml with extends, verify merged values.

---

## Dependency Graph

```
Phase 0a (parent walk)  ──────────┐
                                   ├── Phase 0b (CLI merge) ───┐
Phase 0c (env interp)  ───────────┘                            │
                                                                ├── Phase 2a (pos path)
Phase 0d (warnings)  ──────────────────────────────────────────┤
                                                                ├── Phase 2b (watch)
Phase 1a (global flags) ──┐                                     │
                           ├── Phase 1e (help text)             │
                           ├── Phase 2c (global verbose)        │
                           ├── Phase 2d (json all) ─────────────┤
                           │                                    ├── Phase 3a (config cmd)
                           ├── Phase 3b (machine mode)          │
                           │                                    │
Phase 1b (clean) ─────────┤                                    │
Phase 1c (completions) ───┤                                    │
Phase 1d (exit codes) ────┴── Phase 3b (machine mode) ─────────┘
                                                      │
                                                      └── Phase 4a (sub-tables) ── Phase 4b (profiles) ── Phase 4c (extends)
```

**Parallelisable tracks:**
- Track A: 0a → 0b → 0c → 0d → 2a → 2b → 3a → 4a → 4b → 4c
- Track B: 1a → 1b → 1c → 1d → 1e (all independent)
- Track C: 2c → 2d (after 1a)
- Track D: 3b (after 1a + 1d)
- Track E: 3c (independent, low priority)

Tracks A, B can run in parallel. Track C blocks on Track B. Track D blocks on
Track B + A.

---

## Summary: Effort Estimate

| Phase | Items | Files | Estimated effort |
|-------|-------|-------|-----------------|
| 0 — Foundation | 4 (0a–0d) | ~15 | Medium (config touches many things) |
| 1 — Polish | 5 (1a–1e) | ~10 | Small (mostly boilerplate) |
| 2 — Ergonomics | 4 (2a–2d) | ~12 | Medium (watcher is new dep + async) |
| 3 — Advanced UX | 3 (3a–3c) | ~8 | Small–Medium (mostly new cmd modules) |
| 4 — Config evolution | 3 (4a–4c) | ~3 | Medium (backward compat + profiles) |
| **Total** | **19** | | **Medium-Large** |

Items ready for parallel implementation on day 1:
- `tile clean` (1b)
- `tile completions` (1c)
- Exit codes (1d)
- `--help` text (1e)
- Parent-directory walk (0a)
- `tile tree` (3c)

These have zero dependencies on each other or on the figment Provider pattern.