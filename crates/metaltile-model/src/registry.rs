//! Kernel registry: op-name → `BenchSpec` lookup via `inventory`.
//!
//! Wraps the compile-time `inventory::iter::<BenchSpec>` registry that
//! `metaltile-std` populates via `inventory::submit!` in each kernel
//! module. Returns `&'static BenchSpec` references — the registry is
//! built once, O(1) lookup, no per-token overhead.
//!
//! ## Lookup keys
//!
//! - Primary: `"op"` — matches `BenchSpec.op`.
//! - Secondary: `"op/subop"` — matches `BenchSpec.op` + `/` + `BenchSpec.subop`.
//!   Only populated when `subop` is non-empty.
//!
//! Lookup first tries the secondary key, then falls back to primary.
//! This lets TOML files use either `op = "dequant_gemv"` (unique op)
//! or `op = "dequant_gemv/q_proj"` (disambiguated subop).

use std::collections::HashMap;

use metaltile_std::spec::BenchSpec;

/// Static registry of all `BenchSpec` entries known at compile time.
pub struct KernelRegistry {
    /// `op/subop` → `&'static BenchSpec` (secondary key, preferred).
    qualified: HashMap<String, &'static BenchSpec>,
    /// `op` → `&'static BenchSpec` (primary key, fallback).
    simple: HashMap<String, &'static BenchSpec>,
}

impl KernelRegistry {
    /// Build the registry from `inventory::iter::<BenchSpec>`.
    ///
    /// Each spec is indexed by both `op` and (when subop is non-empty)
    /// `op/subop`. Collisions: first registered wins (deterministic by
    /// link order; all metaltile-std specs have unique `op`/`subop`
    /// pairs, so collisions shouldn't occur in practice).
    pub fn build() -> Self {
        let mut qualified = HashMap::new();
        let mut simple = HashMap::new();
        for spec in inventory::iter::<BenchSpec> {
            simple.entry(spec.op.to_string()).or_insert(spec);
            if !spec.subop.is_empty() {
                let key = format!("{}/{}", spec.op, spec.subop);
                qualified.entry(key).or_insert(spec);
            }
        }
        Self { qualified, simple }
    }

    /// Look up a `BenchSpec` by op name.
    ///
    /// Tries `qualified` (op/subop) first, then `simple` (op only).
    pub fn get(&self, op: &str) -> Option<&'static BenchSpec> {
        self.qualified
            .get(op)
            .or_else(|| self.simple.get(op))
            .copied()
    }

    /// Number of registered kernel ops (simple keys).
    pub fn len(&self) -> usize { self.simple.len() }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool { self.simple.is_empty() }

    /// Iterate over all unique `BenchSpec` references (by simple key).
    pub fn iter(&self) -> impl Iterator<Item = &&'static BenchSpec> { self.simple.values() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_builds_without_panic() {
        let reg = KernelRegistry::build();
        // At minimum, we should have some specs from metaltile-std.
        assert!(!reg.is_empty(), "registry should contain built-in kernels");

        // Spot-check a few known ops exist.
        assert!(reg.get("rms_norm").is_some(), "rms_norm must be registered");
        assert!(reg.get("gather").is_some(), "gather must be registered");
        assert!(reg.get("binary/add").is_some(), "binary/add must be registered");
    }

    #[test]
    fn lookup_unknown_op_returns_none() {
        let reg = KernelRegistry::build();
        assert!(reg.get("nonexistent_kernel").is_none());
    }
}
