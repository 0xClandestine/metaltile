//! Thin wrappers around the `git` CLI used by `tile bench` and `tile diff`.
//!
//! Each helper returns `None`/empty when git is unavailable, the cwd
//! isn't a git repo, or the requested ref/path doesn't exist — callers
//! treat that as "skip the git-aware behavior" rather than as an error.

use std::process::Command;

/// Run `git` with the given args, returning trimmed stdout on success.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Run `git` and return raw stdout (no trim) so callers can preserve
/// trailing newlines in file content.
fn git_raw(args: &[&str]) -> Option<Vec<u8>> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout)
}

/// True if the working tree has tracked-file modifications (modified or
/// staged). Untracked files are intentionally ignored — they don't
/// affect the compiled binary the bench just ran.
///
/// Returns `None` when not inside a git repo / git isn't available, in
/// which case callers should skip the dirty-tree check rather than
/// fail.
pub fn working_tree_dirty() -> Option<bool> {
    // `git rev-parse --is-inside-work-tree` is the canonical "are we in
    // a repo" probe; if it fails, return None so the caller silently
    // skips git-aware behavior.
    git(&["rev-parse", "--is-inside-work-tree"])?;
    // `--quiet` suppresses output; exit code 0 = clean, 1 = dirty.
    let out = Command::new("git").args(["diff", "HEAD", "--quiet"]).status().ok()?;
    Some(!out.success())
}

/// List the paths git considers dirty (modified or staged). Used to
/// give a useful error message when the dirty guard fires.
pub fn list_dirty_files() -> Vec<String> {
    let Some(s) = git(&["diff", "HEAD", "--name-only"]) else {
        return Vec::new();
    };
    s.lines().map(|l| l.to_string()).filter(|l| !l.is_empty()).collect()
}

/// Return the first ref in `candidates` that resolves to a commit, or
/// `None` if none do. The ref is returned verbatim (e.g. `origin/dev`)
/// so it can be fed back into git commands.
pub fn resolve_baseline_ref(candidates: &[&str]) -> Option<String> {
    for &r in candidates {
        if git(&["rev-parse", "--verify", "--quiet", r]).is_some() {
            return Some(r.to_string());
        }
    }
    None
}

/// `git merge-base HEAD <ref>` — common ancestor SHA, or None.
pub fn merge_base_with(reference: &str) -> Option<String> {
    git(&["merge-base", "HEAD", reference])
}

/// `git show <rev>:<path>` — file contents at a revision, or None if
/// the path doesn't exist at that rev.
pub fn show_file_at(rev: &str, path: &str) -> Option<String> {
    let spec = format!("{rev}:{path}");
    let bytes = git_raw(&["show", &spec])?;
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These probes can't be sandbox-mocked, but they should at least
    /// be safe to call: in CI / a non-repo cwd they must return None
    /// rather than panic. The build cwd is the repo, so we get Some
    /// here — assert only that the call type-checks and returns
    /// something representable.
    #[test]
    fn working_tree_dirty_smoke() {
        // Either Some(_) (inside repo) or None (not a repo) is fine.
        let _ = working_tree_dirty();
    }

    #[test]
    fn resolve_baseline_ref_returns_none_for_garbage() {
        assert!(resolve_baseline_ref(&["this/ref/does/not/exist"]).is_none());
    }
}
