//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Tile.toml configuration parsing and project detection.
//!
//! See §3 of the spec for the full schema.

use std::path::Path;

use serde::Deserialize;

/// Full Tile.toml configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TileConfig {
    #[serde(default)]
    pub profile: ProfileTable,
    #[serde(default)]
    pub bench: Option<BenchConfig>,
    #[serde(default)]
    pub tol: Option<TolConfig>,
}

/// Profile section — supports [profile.default], [profile.ci], etc.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ProfileTable {
    pub default: Option<Profile>,
    pub ci: Option<Profile>,
    pub release: Option<Profile>,
}

/// A single profile configuration.
#[derive(Debug, Clone)]
pub struct Profile {
    pub src: String,
    pub test: String,
    /// Bench directory path (from `[profile.default].bench = "benches"`).
    pub bench: String,
    pub out: String,
    pub baselines: String,
    pub dtypes: Vec<String>,
    /// Profile-specific bench override (from `[profile.<name>.bench]` sub-table).
    pub bench_config: Option<BenchConfig>,
}

impl<'de> Deserialize<'de> for Profile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        // Parse fields from a map. The bench key can be either a string
        // (directory path) or a table (n, iters bench config).
        use std::fmt;

        use serde::de::{MapAccess, Visitor};

        struct ProfileVisitor;
        impl<'de> Visitor<'de> for ProfileVisitor {
            type Value = Profile;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a profile map")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where A: MapAccess<'de> {
                let mut src = None::<String>;
                let mut test = None::<String>;
                let mut bench_str = None::<String>;
                let mut bench_config = None::<BenchConfig>;
                let mut out = None::<String>;
                let mut baselines = None::<String>;
                let mut dtypes = None::<Vec<String>>;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "src" => src = Some(map.next_value()?),
                        "test" => test = Some(map.next_value()?),
                        "out" => out = Some(map.next_value()?),
                        "baselines" => baselines = Some(map.next_value()?),
                        "dtypes" => dtypes = Some(map.next_value()?),
                        "bench" => {
                            // bench can be a string or a table
                            let val: serde_json::Value = map.next_value()?;
                            match val {
                                serde_json::Value::String(s) => bench_str = Some(s),
                                serde_json::Value::Object(obj) => {
                                    let n = obj
                                        .get("n")
                                        .and_then(|v| v.as_i64())
                                        .map(|i| i as usize)
                                        .unwrap_or_else(default_bench_n);
                                    let iters = obj
                                        .get("iters")
                                        .and_then(|v| v.as_i64())
                                        .map(|i| i as usize)
                                        .unwrap_or_else(default_bench_iters);
                                    bench_config = Some(BenchConfig { n, iters });
                                },
                                _ => {},
                            }
                        },
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        },
                    }
                }

                Ok(Profile {
                    src: src.unwrap_or_else(default_src),
                    test: test.unwrap_or_else(default_test),
                    bench: bench_str.unwrap_or_else(default_bench),
                    out: out.unwrap_or_else(default_out),
                    baselines: baselines.unwrap_or_else(default_baselines),
                    dtypes: dtypes.unwrap_or_else(default_dtypes),
                    bench_config,
                })
            }
        }

        deserializer.deserialize_map(ProfileVisitor)
    }
}

/// [bench] sub-table.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BenchConfig {
    #[serde(default = "default_bench_n")]
    pub n: usize,
    #[serde(default = "default_bench_iters")]
    pub iters: usize,
}

impl Default for BenchConfig {
    fn default() -> Self { Self { n: default_bench_n(), iters: default_bench_iters() } }
}

/// [tol] sub-table — dtype tolerances.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct TolConfig {
    #[serde(default = "default_tol_f32")]
    pub f32: f32,
    #[serde(default = "default_tol_f16")]
    pub f16: f32,
    #[serde(default = "default_tol_bf16")]
    pub bf16: f32,
}

impl Default for TolConfig {
    fn default() -> Self {
        Self { f32: default_tol_f32(), f16: default_tol_f16(), bf16: default_tol_bf16() }
    }
}

// ── Defaults ────────────────────────────────────────────────────────────

fn default_src() -> String { "kernels".to_string() }
fn default_test() -> String { "tests".to_string() }
fn default_bench() -> String { "benches".to_string() }
fn default_out() -> String { "tile-out".to_string() }
fn default_baselines() -> String { "baselines".to_string() }
fn default_dtypes() -> Vec<String> { vec!["f32".into(), "f16".into(), "bf16".into()] }
fn default_bench_n() -> usize { 67_108_864 }
fn default_bench_iters() -> usize { 10 }
fn default_tol_f32() -> f32 { 1e-4 }
fn default_tol_f16() -> f32 { 1.5e-2 }
fn default_tol_bf16() -> f32 { 1.3e-1 }

impl TileConfig {
    /// Try to load `Tile.toml` from `dir`.
    /// Returns `None` if the file does not exist.
    /// Returns validation errors for unknown keys, wrong types, or missing required fields.
    pub fn load(dir: &Path) -> Result<Option<TileConfig>, String> {
        let path = dir.join("Tile.toml");
        if !path.exists() {
            return Ok(None);
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("failed to read Tile.toml: {e}"))?;

        // First pass: parse with unknown field detection for the top-level table.
        let raw: toml::Value =
            toml::from_str(&text).map_err(|e| format!("Tile.toml parse error: {e}"))?;

        // Validate top-level keys
        let allowed_top = ["profile", "bench", "tol"];
        if let toml::Value::Table(table) = &raw {
            for key in table.keys() {
                if !allowed_top.contains(&key.as_str()) {
                    let line = find_key_line(&text, key);
                    return Err(format!(
                        "Tile.toml:{}: unknown key `{key}` — expected one of: {}",
                        line,
                        allowed_top.join(", "),
                    ));
                }
            }
        }

        let cfg: TileConfig = toml::from_str(&text).map_err(|e| format!("Tile.toml: {e}"))?;

        Ok(Some(cfg))
    }

    /// Get the resolved profile for `profile_name`.
    pub fn resolved_profile(&self, profile_name: &str) -> Profile {
        let fallback = Profile {
            src: default_src(),
            test: default_test(),
            bench: default_bench(),
            out: default_out(),
            baselines: default_baselines(),
            dtypes: default_dtypes(),
            bench_config: None,
        };
        match profile_name {
            "default" => self.profile.default.clone(),
            "ci" => self.profile.ci.clone(),
            "release" => self.profile.release.clone(),
            _ => None,
        }
        .unwrap_or(fallback)
    }

    /// Resolve bench config: profile-specific bench_config > top-level [bench] > defaults.
    pub fn resolved_bench(&self, profile_name: &str) -> BenchConfig {
        let profile = self.resolved_profile(profile_name);
        profile.bench_config.or(self.bench).unwrap_or_default()
    }

    /// Resolve tol config: top-level [tol] > defaults.
    pub fn resolved_tol(&self, _profile_name: &str) -> TolConfig { self.tol.unwrap_or_default() }
}

/// Find the line number of a key in TOML text (1-indexed).
fn find_key_line(text: &str, key: &str) -> usize {
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with(&format!("{key} =")) || trimmed.starts_with(&format!("{key}=")) {
            return i + 1;
        }
    }
    0
}
