//! Kernel registry: op-name → `BenchSpec` lookup via `inventory`.
//!
//! Maps op names like `"rms_norm"` or `"gemv/smm"` to `&'static BenchSpec`
//! references registered at compile time via `inventory::submit!`.

use std::collections::{HashMap, hash_map::Entry};

use metaltile_std::spec::BenchSpec;

/// Static registry of all `BenchSpec` entries known at compile time.
pub struct KernelRegistry {
    map: HashMap<String, &'static BenchSpec>,
    n_unique: usize,
}

impl KernelRegistry {
    /// Build the registry from `inventory::iter::<BenchSpec>`.
    ///
    /// Each spec is indexed by both `op` and (when subop is non-empty)
    /// `op/subop`. Lookup tries the qualified key first, then falls back
    /// to the simple key.
    pub fn build() -> Self {
        let mut map = HashMap::new();
        let mut n_unique = 0;
        for spec in inventory::iter::<BenchSpec> {
            // Simple key: op only.
            match map.entry(spec.op.to_string()) {
                Entry::Vacant(e) => { e.insert(spec); n_unique += 1; },
                Entry::Occupied(_) => {},
            }
            // Qualified key: op/subop (when subop is non-empty).
            if !spec.subop.is_empty() {
                map.entry(format!("{}/{}", spec.op, spec.subop)).or_insert(spec);
            }
        }
        Self { map, n_unique }
    }

    /// Look up a `BenchSpec` by op name.
    ///
    /// Tries `op/subop` first (qualified key), then `op` only.
    pub fn get(&self, op: &str) -> Option<&'static BenchSpec> {
        self.map.get(op).copied()
    }

    /// Number of unique kernel ops.
    pub fn len(&self) -> usize { self.n_unique }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool { self.n_unique == 0 }

    /// Iterate over all unique `BenchSpec` references.
    pub fn iter(&self) -> impl Iterator<Item = &&'static BenchSpec> {
        self.map.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_builds_without_panic() {
        let reg = KernelRegistry::build();
        assert!(!reg.is_empty(), "registry should contain built-in kernels");
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