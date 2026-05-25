//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Unified kernel filter: compile-once regexes with subprocess forwarding.

/// Pre-compiled kernel filter. Holds the raw flag strings (for subprocess
/// forwarding) alongside the compiled regexes (for fast per-kernel matching).
///
/// Constructed once at command startup; used in hot loops over `BenchSpec`
/// slices and when forwarding flags to harness subprocesses.
pub struct Filters {
    raw_filter: Option<String>,
    raw_match_kernel: Option<String>,
    raw_match_module: Option<String>,
    raw_no_match_kernel: Option<String>,
    raw_no_match_module: Option<String>,
    filter: Option<regex::Regex>,
    match_kernel: Option<regex::Regex>,
    match_module: Option<regex::Regex>,
    no_match_kernel: Option<regex::Regex>,
    no_match_module: Option<regex::Regex>,
}

impl Filters {
    /// Compile filter flags from raw strings. Returns `Err` for any invalid regex.
    ///
    /// `filter` is a case-insensitive substring match on kernel name; the
    /// remaining flags are full regexes.
    pub fn build(
        filter: Option<&str>,
        match_kernel: Option<&str>,
        match_module: Option<&str>,
        no_match_kernel: Option<&str>,
        no_match_module: Option<&str>,
    ) -> Result<Self, String> {
        fn compile(p: Option<&str>) -> Result<Option<regex::Regex>, String> {
            match p {
                None => Ok(None),
                Some(s) =>
                    regex::Regex::new(s).map(Some).map_err(|e| format!("invalid regex `{s}`: {e}")),
            }
        }

        // --filter is a substring match; escape it to prevent regex metachar issues.
        let filter_re =
            filter.map(|f| regex::Regex::new(&format!("(?i){}", regex::escape(f))).ok()).flatten();

        Ok(Filters {
            raw_filter: filter.map(String::from),
            raw_match_kernel: match_kernel.map(String::from),
            raw_match_module: match_module.map(String::from),
            raw_no_match_kernel: no_match_kernel.map(String::from),
            raw_no_match_module: no_match_module.map(String::from),
            filter: filter_re,
            match_kernel: compile(match_kernel)?,
            match_module: compile(match_module)?,
            no_match_kernel: compile(no_match_kernel)?,
            no_match_module: compile(no_match_module)?,
        })
    }

    /// Returns `true` if the kernel passes all active filter flags.
    ///
    /// Rules (all active filters AND-ed):
    /// - `--filter`: case-insensitive substring match on `kernel_name`
    /// - `--match-kernel`: regex must match `kernel_name`
    /// - `--match-module`: regex must match `op_group`
    /// - `--no-match-kernel`: regex must NOT match `kernel_name`
    /// - `--no-match-module`: regex must NOT match `op_group`
    pub fn matches_kernel(&self, kernel_name: &str, op_group: &str) -> bool {
        if let Some(re) = &self.filter {
            if !re.is_match(kernel_name) {
                return false;
            }
        }
        if let Some(re) = &self.match_kernel {
            if !re.is_match(kernel_name) {
                return false;
            }
        }
        if let Some(re) = &self.match_module {
            if !re.is_match(op_group) {
                return false;
            }
        }
        if let Some(re) = &self.no_match_kernel {
            if re.is_match(kernel_name) {
                return false;
            }
        }
        if let Some(re) = &self.no_match_module {
            if re.is_match(op_group) {
                return false;
            }
        }
        true
    }

    /// Forward all active filter flags to a subprocess `Command` using
    /// the canonical `--flag value` form understood by `tile_harness!`.
    pub fn forward_to_cmd(&self, cmd: &mut std::process::Command) {
        if let Some(f) = &self.raw_filter {
            cmd.arg("--filter").arg(f);
        }
        if let Some(p) = &self.raw_match_kernel {
            cmd.arg("--match-kernel").arg(p);
        }
        if let Some(p) = &self.raw_match_module {
            cmd.arg("--match-module").arg(p);
        }
        if let Some(p) = &self.raw_no_match_kernel {
            cmd.arg("--no-match-kernel").arg(p);
        }
        if let Some(p) = &self.raw_no_match_module {
            cmd.arg("--no-match-module").arg(p);
        }
    }

    /// The raw `--filter` substring, if set. Used for diff/display purposes.
    pub fn raw_filter(&self) -> Option<&str> { self.raw_filter.as_deref() }
}

impl Default for Filters {
    fn default() -> Self {
        Filters {
            raw_filter: None,
            raw_match_kernel: None,
            raw_match_module: None,
            raw_no_match_kernel: None,
            raw_no_match_module: None,
            filter: None,
            match_kernel: None,
            match_module: None,
            no_match_kernel: None,
            no_match_module: None,
        }
    }
}
