//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Git provider abstraction and `RealGit` implementation.
//!
//! `GitProvider` trait lets callers inject a mock for testing.  The
//! default `RealGit` delegates to the `git` CLI.

use std::{path::Path, process::Command};

// ── Trait ────────────────────────────────────────────────────────────────

/// Abstraction over git operations used by `tile bench` and `tile diff`.
///
/// Each method returns `None` when git is unavailable, cwd isn't a git
/// repo, or the requested ref / path doesn't exist — callers treat that
/// as "skip the git-aware behaviour" rather than error.
pub trait GitProvider: Send + Sync {
    fn working_tree_dirty(&self) -> Option<bool>;
    fn list_dirty_files(&self) -> Vec<String>;
    fn resolve_baseline_ref(&self, candidates: &[&str]) -> Option<String>;
    fn merge_base_with(&self, reference: &str) -> Option<String>;
    fn show_file_at(&self, rev: &str, path: &str) -> Option<String>;
    fn short_sha(&self) -> Option<String>;
}

// ── Real implementation ──────────────────────────────────────────────────

/// Delegates to `git(1)` via `std::process::Command`.
#[derive(Default)]
pub struct RealGit;

impl GitProvider for RealGit {
    fn working_tree_dirty(&self) -> Option<bool> {
        git(&["rev-parse", "--is-inside-work-tree"])?;
        let out = Command::new("git").args(["diff", "HEAD", "--quiet"]).status().ok()?;
        Some(!out.success())
    }

    fn list_dirty_files(&self) -> Vec<String> {
        let Some(s) = git(&["diff", "HEAD", "--name-only"]) else {
            return Vec::new();
        };
        s.lines().map(|l| l.to_string()).filter(|l| !l.is_empty()).collect()
    }

    fn resolve_baseline_ref(&self, candidates: &[&str]) -> Option<String> {
        for &r in candidates {
            if git(&["rev-parse", "--verify", "--quiet", r]).is_some() {
                return Some(r.to_string());
            }
        }
        None
    }

    fn merge_base_with(&self, reference: &str) -> Option<String> {
        git(&["merge-base", "HEAD", reference])
    }

    fn show_file_at(&self, rev: &str, path: &str) -> Option<String> {
        let spec = format!("{rev}:{path}");
        let bytes = git_raw(&["show", &spec])?;
        String::from_utf8(bytes).ok()
    }

    fn short_sha(&self) -> Option<String> {
        let out = Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok()?;
        if !out.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

// ── Private CLI wrappers ─────────────────────────────────────────────────

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_raw(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

// ── Convenience ──────────────────────────────────────────────────────────

/// Check whether `path` is inside a git work tree (has a `.git` parent).
pub fn is_inside_work_tree(path: &Path) -> bool {
    let out = Command::new("git")
        .args(["-C", path.to_str().unwrap_or("."), "rev-parse", "--is-inside-work-tree"])
        .output()
        .ok();
    match out {
        Some(o) => String::from_utf8_lossy(&o.stdout).trim() == "true",
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_baseline_ref_returns_none_for_garbage() {
        let git = RealGit;
        assert!(git.resolve_baseline_ref(&["this/ref/does/not/exist"]).is_none());
    }
}
