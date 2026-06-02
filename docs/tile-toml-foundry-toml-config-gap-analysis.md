# `tile.toml` vs `foundry.toml` — Config Feature Gap Analysis

> **Purpose:** Identify every config capability in Foundry's `foundry.toml` that
> is missing from our `tile.toml`. Goal is not to blindly copy Solidity-specific
> fields, but to identify structural config patterns that would make the tile
> user experience more professional.
>
> Generated: 2026-06-02
> Source: `crates/metaltile-cli/src/config.rs` + `foundry-config` crate
> (`/tmp/foundry/crates/config/src/lib.rs`, ~7400 lines)

---

## Ground Truth: What We Have Today

### `TileConfig` fields (5 fields, 37 LOC)

```rust
pub struct TileConfig {
    pub runner_binary: String,     // default: "__tile_runner"
    pub project_path: Option<String>,  // default: None
    pub verbose: u8,               // default: 0
    pub runs: usize,               // default: 3
    pub warmup_runs: usize,        // default: 1
}
```

### Layering strategy

```
Defaults (Serialized::defaults)
    ↓
tile.toml (current dir, optional)
    ↓
TILE_* env vars (e.g. TILE_VERBOSE=1)
```

See [`crates/metaltile-cli/src/config.rs`](../crates/metaltile-cli/src/config.rs).

### Current `tile.toml`

```toml
runner_binary = "__tile_runner"
# project_path = "."
verbose = 0
runs = 3
warmup_runs = 1
```

---

## Foundry Config: Scale and Structure

`foundry-config` has **~90 fields** across ~15 categories, spanning 7,400 LOC in
`lib.rs` + 30+ module files. The `Config` struct is the central configuration
object, loaded once per command via `Config::load()` (which walks parent
directories for `foundry.toml`).

### Config categories

| Category | Fields | TOML table | Purpose |
|----------|--------|-----------|---------|
| **Project paths** | `src`, `test`, `script`, `out`, `libs`, `cache_path`, `broadcast` | top-level | Where source, artifacts, caches live |
| **Remappings** | `remappings`, `auto_detect_remappings` | top-level | Solidity import aliases |
| **Compiler** | `solc`, `evm_version`, `via_ir`, `optimizer`, `optimizer_runs`, `optimizer_details`, `bytecode_hash`, `cbor_metadata`, `revert_strings`, `sparse_mode`, `extra_output`, `extra_output_files`, `model_checker`, `ast`, `deny`, `ignored_error_codes` | top-level | Solc compiler behavior |
| **EVM (test/script)** | `sender`, `tx_origin`, `initial_balance`, `gas_limit`, `gas_price`, `block_*` (7 fields), `memory_limit`, `ffi`, `isolate`, `assertions_revert` | top-level | EVM execution environment |
| **RPC** | `eth_rpc_url`, `eth_rpc_*` (7 fields), `rpc_endpoints` | top-level + `[rpc_endpoints]` | Chain connectivity |
| **Etherscan** | `etherscan_api_key`, `etherscan` | top-level + `[etherscan]` | Verification API |
| **Fuzz testing** | `runs`, `seed`, `dictionary_weight`, `max_test_rejects`, `failure_persist_dir`, etc. | `[fuzz]` | Fuzz test parameters |
| **Invariant testing** | `runs`, `depth`, `workers`, `call_override`, `shrink_run_limit`, etc. | `[invariant]` | Invariant test parameters |
| **Linting** | `severity`, `exclude_lints`, `ignore`, `lint_on_build`, per-lint config | `[lint]` | Solidity linter rules |
| **Formatting** | `line_length`, `tab_width`, `style`, `bracket_spacing`, `quote_style`, `wrap_comments`, `sort_imports`, `ignore`, etc. (19 fields) | `[fmt]` | Code formatter settings |
| **Documentation** | `out`, `title`, `book`, `homepage`, `repository`, `path`, `ignore` | `[doc]` | Doc generation config |
| **File system permissions** | read/write/allow list | `[fs_permissions]` | Security for cheat codes |
| **Dependency management** | `install_lib_dir`, `dependencies`, `soldeer` | various | Package management |
| **Storage caching** | `chains`, `endpoints` | `[rpc_storage_caching]` | RPC response caching |
| **Extends / inheritance** | `extends` | top-level | Include another toml |
| **Profiles** | Multi-profile support (`[profile.default]`, `[profile.ci]`, etc.) | TOML profiles | Environment-specific config overrides |

---

## Structural Config Differences

### 1. Profile System (multi-environment overrides)

**Forge:** Config has a full profile system. Foundry's `foundry.toml` can contain:

```toml
[profile.default]
src = "src"
out = "out"
solc = "0.8.28"

[profile.ci]
solc = "0.8.28"
optimizer = true
fuzz = { runs = 1000 }

[profile.intense]
optimizer_runs = 1000000
via_ir = true
```

Profiles are chosen via `FOUNDRY_PROFILE` env var or `--profile` CLI flag.
Each profile inherits from `default` and overrides specific fields.

**Tile:** No profile support. Single flat config.

**Gap:** ⚠️ Profiles are useful for switching between `debug` (unoptimized,
verbose) and `release` (optimized, quiet) bench configurations, or between
CI vs local workstation.

### 2. Extends / inheritance

**Forge:** `extends = "base.toml"` or `extends = { path = "base.toml", strategy = "no-collision" }`.
Strategies: `extend-arrays` (default, concat arrays), `replace-arrays` (replace
arrays), `no-collision` (error on overlap). Base files cannot extend further.

```toml
extends = "~/.foundry/global.toml"
```

**Tile:** None. No way to share config across projects.

**Gap:** ⚠️ Useful for teams that share a common runner binary path or project layout.

### 3. Environment variable interpolation

**Forge:** Config values can reference env vars:

```toml
eth_rpc_url = "${ETH_RPC_MAINNET}"
etherscan_api_key = "${ETHERSCAN_API_KEY}"
```

The `resolve` module replaces `${VAR}` placeholders at load time.
Errors if a referenced var is unset.

**Tile:** No interpolation. You must use the `TILE_*` env var system instead
(which only overrides known fields, not arbitrary substrings in values).

**Gap:** ⚠️ Useful for sensitive values (API keys) and CI differentiation.

### 4. Auto-discovery of config file (parent directory walk)

**Forge:** `Config::load()` walks up from CWD to find `foundry.toml` in any
parent directory. The `root` field is set to the directory containing the
`foundry.toml`. This means `forge test` works from any subdirectory of a
project.

**Tile:** Only looks for `tile.toml` in CWD. No parent-directory walk.

```rust
TomlFileProvider::new(None, PathBuf::from("foundry.toml"))
// walks parents looking for the file
```

**Gap:** ⚠️ Moderate. Users must be in the project root. Can be fixed by
walking parents and setting `project_path` automatically.

### 5. Sub-table configurations (TOML sections)

**Forge:** Uses many TOML sub-tables:

| TOML table | Config struct |
|---|---|
| `[fuzz]` | `FuzzConfig` |
| `[invariant]` | `InvariantConfig` |
| `[fmt]` | `FormatterConfig` |
| `[lint]` | `LinterConfig` |
| `[doc]` | `DocConfig` |
| `[rpc_storage_caching]` | `StorageCachingConfig` |
| `[etherscan]` | `EtherscanConfigs` |
| `[rpc_endpoints]` | `RpcEndpoints` |
| `[fs_permissions]` | `FsPermissions` |
| `[vyper]` | `VyperConfig` |
| `[soldeer]` | `SoldeerConfig` |
| `[bind_json]` | `BindJsonConfig` |

**Tile:** Flat config, no sub-tables.

**Gap:** ⚠️ Many of these are Solidity-specific and don't apply, but the
**pattern** of sub-tables is worth adopting for our own domain:
- `[bench]` — default runs, warmup, binary path, device pinning
- `[build]` — emit defaults, optimization flags, compiler selection
- `[device]` — preferred GPU, feature toggles

### 6. CLI args as figment Providers

**Forge:** Every CLI args struct implements `figment::Provider`, so CLI flags
override both config file and env vars seamlessly. Figment merges all layers:

```
  defaults → foundry.toml → env vars → CLI args (--flag)
```

```rust
// BuildArgs implements Provider:
foundry_config::merge_impl_figment_convert!(BuildArgs, build);
// The Provider serializes relevant fields and merges them into the config Figment.
```

**Tile:** Config is loaded independently. CLI flags override via separate
logic in each `run()` function. No figment Provider pattern is used.

**Gap:** ⚠️ Direct override: `tile build --verbose=2` would not merge with
`tile.toml verbose=1` the way forge does. Currently our CLI args and config
are independently accessed.

### 7. Config as serializable/displayable

**Forge:** `Config` implements `Serialize` and can print itself via:
- `config.to_string_pretty()` — dump as TOML (supports `forge config`)
- `config.into_basic().to_string_pretty()` — simplified summary
- `serde_json::to_string_pretty(&config)` — JSON output for `--json`

**Tile:** `TileConfig` derives `Serialize`/`Deserialize` but has no
`to_string_pretty()` or display command. No `tile config` subcommand exists
(noted in the CLI gap analysis).

**Gap:** ⚠️ Minor. Easy to add once `tile config` is implemented.

---

## Gap Categorization: What to Backport vs What to Skip

### Backport: High Priority (directly applicable to MetalTile)

| Feature | Foundry equivalent | Why |
|---------|-------------------|-----|
| **Parent-directory config walk** | `Config::load()` walks parents for `foundry.toml` | Users shouldn't need to be in project root. Set `project_path` automatically. |
| **Config display (`tile config`)** | `forge config` → `config.to_string_pretty()` | Needed for debugging config merging. |
| **CLI args merge into config** | `merge_impl_figment_convert!` macros | `tile build --runs 10` should merge with tile.toml `runs`. Currently they're independent. |
| **Env var interpolation** | `${VAR}` in config values | Useful for `project_path = "${HOME}/kernels"`, runner paths, etc. |

### Backport: Medium Priority

| Feature | Foundry equivalent | Why |
|---------|-------------------|-----|
| **Profiles** | `[profile.default]`, `[profile.ci]`, `FOUNDRY_PROFILE` | Switch between `debug` (verbose, single-run) and `release` (quiet, many runs) bench configurations |
| **`[bench]` sub-table** | `[fuzz]` pattern | Keep bench-specific settings (runs, warmup, target GB/s) in their own section |
| **`[build]` sub-table** | compiler config pattern | Build-specific settings (default dtypes, emit kind, SDK) |
| **Warnings on config load** | `Config.warnings: Vec<Warning>` | Collect deprecation/unknown-key warnings during load, show at end |

### Skip: Not Applicable to MetalTile

| Feature | Reason |
|---------|--------|
| `[fuzz]`, `[invariant]` | Solidity test framework concepts |
| `[fmt]` | Solidity code formatter; no tile equivalent unless we add MSL formatter |
| `[lint]` | Solidity linter; GPU kernel linter could exist but doesn't yet |
| `[doc]` | Solidity NatSpec doc generation |
| `[etherscan]` | Contract verification API keys |
| `[rpc_storage_caching]` | RPC endpoint caching |
| `[rpc_endpoints]` | RPC connection config |
| `[fs_permissions]` | Cheat code filesystem security |
| `solc`, `evm_version`, optimizer | Solc compiler options |
| EVM env: `sender`, `tx_origin`, `gas_price`, `block_*` | EVM execution context |
| `eth_rpc_url`, `eth_rpc_*` | RPC connection settings |
| `verify-*` integration | Etherscan verification |
| `[vyper]` | Vyper compiler config |
| `[soldeer]` / `dependencies` | Solidity dependency manager |
| `gas_reports`, `gas_snapshot_*` | EVM gas accounting |

---

## Intended `tile.toml` Evolution (Phased)

### Phase 1: Immediate (add alongside CLI gaps)

```toml
# Phase 1 — structural improvements
project_path = "."           # already exists
verbose = 0                  # already exists

# Parent-directory walk: find nearest tile.toml
#   if none in CWD, walk up to $HOME
```

### Phase 2: Sub-tables for organization

```toml
[bench]
runs = 3                     # moved from top-level
warmup_runs = 1              # moved from top-level
# Target GPU device name (substring match, e.g. "M5 Max")
target_device = ""
# Upload results to baseline server on completion
upload_baseline = false

[build]
default_dtypes = ["f32", "f16"]   # which dtypes to compile by default
emit = ""                          # default emit kind
sdk = "macosx"                     # xcrun SDK
time_passes = false                # toggle pass timing

[runner]
# Path to __tile_runner binary
binary = "__tile_runner"           # moved from top-level
# Extra args passed to runner subprocess
extra_args = []
```

### Phase 3: Advanced (when needed)

```toml
[profile.default]
extends = "~/.metaltile/global.toml"

[profile.ci]
bench.runs = 10
bench.warmup_runs = 3
verbose = 0

[profile.debug]
verbose = 2
bench.runs = 1
bench.warmup_runs = 0
build.emit = "msl"           # emit MSL to disk for inspection

[device]
# Override GPU selection for multi-GPU Macs
preferred = "M5 Max"
# Feature override (debug/testing)
force_native_bfloat = false
force_simdgroup_hw = true
```

---

## Concrete Implementation Notes

### Parent-directory config walk

The `ConfigLoader::load()` currently uses `Toml::file("tile.toml")` which only
looks in CWD. To walk parents:

```rust
fn find_tile_toml() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut dir: &Path = cwd.as_path();
    loop {
        let candidate = dir.join("tile.toml");
        if candidate.exists() {
            // Set project_path if not explicitly set
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
}

fn load() -> Result<TileConfig, ...> {
    let toml_path = find_tile_toml().unwrap_or_else(|| PathBuf::from("tile.toml"));
    Figment::from(Serialized::defaults(TileConfig::default()))
        .merge(Toml::file(toml_path))
        .merge(Env::prefixed("TILE_"))
        .extract()
}
```

### CLI args as figment providers

Forge uses proc macros (`merge_impl_figment_convert!`) to auto-derive Provider
implementations for CLI args. For tile, a simpler version:

```rust
impl figment::Provider for &BenchArgs {
    fn metadata(&self) -> figment::Metadata {
        figment::Metadata::named("Bench CLI args")
    }

    fn data(&self) -> Result<figment::value::Map<Profile, Dict>, figment::Error> {
        let mut dict = Dict::new();
        if let Some(v) = self.verbose { dict.insert("verbose".into(), v.into()); }
        // ... more fields
        Ok(Map::from([(Profile::Default, dict)]))
    }
}
```

Then in `main()`:
```rust
let figment = ConfigLoader::build_figment() // defaults + tile.toml + env
    .merge(&args.global)?
    .merge(&args.cmd)?; // subcommand-specific overrides
let config: TileConfig = figment.extract()?;
```

### Profiles

Simplest approach: use figment's built-in profile support:

```rust
// In ConfigLoader:
fn build_figment(profile: Option<&str>) -> Figment {
    let selected_profile = profile
        .or_else(|| std::env::var("TILE_PROFILE").ok())
        .unwrap_or_else(|| "default".into());

    Figment::from(Serialized::defaults(TileConfig::default()))
        .merge(Toml::file("tile.toml"))
        .select(Profile::new(&selected_profile))
        .merge(Env::prefixed("TILE_"))
}
```

A `tile.toml` with profiles:
```toml
[profile.default]
verbose = 0
bench = { runs = 3, warmup_runs = 1 }

[profile.ci]
verbose = 0
bench = { runs = 15, warmup_runs = 3 }
```

---

## Summary: Key Gaps

| Gap | Impact | Effort | Notes |
|-----|--------|--------|-------|
| 1. No parent-directory config walk | **High** — must be in project root | Small | ~20 lines + `find_tile_toml()` |
| 2. No config display command | **Medium** — no feedback on config | Small | Add `tile config` |
| 3. CLI args don't merge into config | **Medium** — flags override conf separately | Medium | Wire figment Provider pattern |
| 4. No env var interpolation | **Low** — `TILE_*` works but ${} is more flexible | Small | ~30-line regex resolve |
| 5. No profiles | **Low-Medium** — nice for CI vs local | Medium | figment profiles + env var |
| 6. No sub-tables for bench/build | **Low** — flat config works fine | Small | `#[serde(flatten)]` pattern |
| 7. No load warnings | **Low** — silent ignore of unknown keys | Small | figment warning collector |

**Recommendation:** Implement gaps 1, 2, 3, and 4 before moving to sub-tables
and profiles. Gaps 5–7 are polish for when the user base demands it.

---

## Appendix: field-by-field cross reference

| `tile.toml` | Type | `foundry.toml` analogue | Notes |
|---|---|---|---|
| `runner_binary` | String | — | Unique to tile (subprocess runner path) |
| `project_path` | Option\<String\> | `Config.root` | Forge auto-detects; we're explicit |
| `verbose` | u8 | `Config.verbosity` | Same concept |
| `runs` | usize | `Config.fuzz.runs` (default 256) | Tile default 3 vs forge default 256 |
| `warmup_runs` | usize | — | Unique to tile (GPU warmup) |