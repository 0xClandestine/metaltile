//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! `tile update` — self-update the tile binary.

use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, process::Command};

use crate::{
    CliError,
    term::{Color, Style, paint_stderr},
};

const REPO_SLUG: &str = "metaltile/metaltile";
const REPO_URL: &str = "https://github.com/metaltile/metaltile.git";
const ASSET_NAME: &str = "tile-aarch64-apple-darwin.tar.gz";

#[derive(clap::Args, Debug)]
pub struct UpdateArgs {
    /// Print what would be installed without modifying anything.
    #[arg(long = "check")]
    pub check: bool,
    /// Build and install from the head of this PR number (requires git + cargo).
    #[arg(long = "pr", value_name = "N", conflicts_with = "commit")]
    pub pr: Option<u32>,
    /// Build and install from this commit SHA (requires git + cargo).
    #[arg(long = "commit", value_name = "SHA", conflicts_with = "pr")]
    pub commit: Option<String>,
}

impl UpdateArgs {
    pub fn run(&self) -> Result<(), CliError> {
        let args = self;
        match (&args.pr, &args.commit) {
            (Some(pr), _) => update_from_source(SourceRef::Pr(*pr), args.check),
            (_, Some(sha)) => update_from_source(SourceRef::Commit(sha.clone()), args.check),
            _ => update_from_release(args.check),
        }
    }
}

enum SourceRef {
    Pr(u32),
    Commit(String),
}

impl SourceRef {
    fn fetch_refspec(&self) -> String {
        match self {
            SourceRef::Pr(n) => format!("pull/{n}/head"),
            SourceRef::Commit(sha) => sha.clone(),
        }
    }

    fn display(&self) -> String {
        match self {
            SourceRef::Pr(n) => format!("PR #{n}"),
            SourceRef::Commit(sha) => format!("commit {sha}"),
        }
    }
}

fn install_path() -> Result<PathBuf, CliError> { std::env::current_exe().map_err(CliError::Io) }

// ── Release update (default) ─────────────────────────────────────────────

fn update_from_release(check_only: bool) -> Result<(), CliError> {
    header("tile update");

    let tag = fetch_latest_release_tag()?;
    let current = env!("CARGO_PKG_VERSION");
    let current_tagged = format!("v{current}");

    eprintln!(
        "  {}  {}",
        paint_stderr(format!("{:<10}", "current"), Style::new().fg(Color::BrightBlack).bold()),
        paint_stderr(current, Style::new().fg(Color::BrightWhite)),
    );
    eprintln!(
        "  {}  {}",
        paint_stderr(format!("{:<10}", "latest"), Style::new().fg(Color::BrightBlack).bold()),
        paint_stderr(&tag, Style::new().fg(Color::Green)),
    );
    eprintln!();

    if tag == current_tagged || tag == current {
        eprintln!("Already up to date.");
        return Ok(());
    }

    if check_only {
        eprintln!(
            "{}  Run without {} to install.",
            paint_stderr("note:", Style::new().fg(Color::Cyan).bold()),
            paint_stderr("--check", Style::new().fg(Color::Green)),
        );
        return Ok(());
    }

    let dest = install_path()?;
    eprintln!(
        "  {}  {}",
        paint_stderr(format!("{:<10}", "dest"), Style::new().fg(Color::BrightBlack).bold()),
        paint_stderr(dest.display().to_string(), Style::new().fg(Color::BrightWhite)),
    );
    eprintln!();

    eprint!("  downloading {}... ", tag);
    download_release_binary(&tag, &dest)?;
    eprintln!("done");
    eprintln!();

    eprintln!(
        "{}  tile {} installed.",
        paint_stderr("ok", Style::new().fg(Color::Green).bold()),
        tag,
    );
    Ok(())
}

fn fetch_latest_release_tag() -> Result<String, CliError> {
    if let Some(tag) = try_gh_latest_tag() {
        return Ok(tag);
    }
    curl_latest_tag()
}

fn try_gh_latest_tag() -> Option<String> {
    let out = Command::new("gh")
        .args(["release", "view", "--repo", REPO_SLUG, "--json", "tagName", "--jq", ".tagName"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let tag = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if tag.is_empty() { None } else { Some(tag) }
}

fn curl_latest_tag() -> Result<String, CliError> {
    let url = format!("https://api.github.com/repos/{REPO_SLUG}/releases/latest");
    let out = Command::new("curl")
        .args([
            "--silent",
            "--fail-with-body",
            "--header",
            "Accept: application/vnd.github+json",
            "--header",
            "User-Agent: tile-update/1",
            &url,
        ])
        .output()
        .map_err(|e| CliError::Subprocess(format!("curl not found: {e}")))?;

    if !out.status.success() {
        let body = String::from_utf8_lossy(&out.stdout);
        if body.contains("Not Found") || out.status.code() == Some(22) {
            return Err(CliError::Config(
                "No releases published yet.\nUse --pr <n> or --commit <sha> to build from source."
                    .into(),
            ));
        }
        return Err(CliError::Subprocess(format!(
            "GitHub API request failed (exit {}).",
            out.status.code().unwrap_or(-1),
        )));
    }

    let json: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let tag = json["tag_name"]
        .as_str()
        .ok_or_else(|| {
            CliError::Config("Unexpected GitHub API response (missing tag_name).".into())
        })?
        .to_string();

    if tag.is_empty() {
        return Err(CliError::Config("No releases found.".into()));
    }
    Ok(tag)
}

fn download_release_binary(tag: &str, dest: &PathBuf) -> Result<(), CliError> {
    let asset_url = format!("https://github.com/{REPO_SLUG}/releases/download/{tag}/{ASSET_NAME}");
    let tar_path = std::env::temp_dir().join("tile-update.tar.gz");
    let extract_dir = std::env::temp_dir().join("tile-update-extract");

    let status = Command::new("curl")
        .args([
            "--silent",
            "--fail",
            "--location",
            "--output",
            tar_path.to_str().unwrap(),
            &asset_url,
        ])
        .status()
        .map_err(|e| CliError::Subprocess(format!("curl not found: {e}")))?;

    if !status.success() {
        return Err(CliError::Subprocess(format!(
            "Failed to download release asset:\n  {asset_url}"
        )));
    }

    let _ = fs::remove_dir_all(&extract_dir);
    fs::create_dir_all(&extract_dir).map_err(CliError::Io)?;

    let status = Command::new("tar")
        .args(["xzf", tar_path.to_str().unwrap(), "-C", extract_dir.to_str().unwrap()])
        .status()
        .map_err(|e| CliError::Subprocess(format!("tar not found: {e}")))?;

    if !status.success() {
        return Err(CliError::Subprocess("Failed to extract release archive.".into()));
    }

    install_binary(&extract_dir.join("tile"), dest)?;

    let _ = fs::remove_dir_all(&extract_dir);
    let _ = fs::remove_file(&tar_path);
    Ok(())
}

// ── Source build (--pr / --commit) ───────────────────────────────────────

fn update_from_source(src: SourceRef, check_only: bool) -> Result<(), CliError> {
    header("tile update");

    eprintln!(
        "  {}  {}",
        paint_stderr(format!("{:<10}", "source"), Style::new().fg(Color::BrightBlack).bold()),
        paint_stderr(src.display(), Style::new().fg(Color::BrightWhite)),
    );
    eprintln!();

    if check_only {
        eprintln!(
            "{}  Run without {} to build and install.",
            paint_stderr("note:", Style::new().fg(Color::Cyan).bold()),
            paint_stderr("--check", Style::new().fg(Color::Green)),
        );
        return Ok(());
    }

    let dest = install_path()?;
    let tmp_dir = std::env::temp_dir().join("tile-update-src");
    let _ = fs::remove_dir_all(&tmp_dir);

    eprint!("  cloning repository... ");
    let status = Command::new("git")
        .args(["clone", "--quiet", REPO_URL, tmp_dir.to_str().unwrap()])
        .status()
        .map_err(|e| CliError::Subprocess(format!("git not found: {e}")))?;
    if !status.success() {
        return Err(CliError::Subprocess("git clone failed.".into()));
    }
    eprintln!("done");

    let refspec = src.fetch_refspec();
    eprint!("  fetching {}... ", src.display());
    let status = Command::new("git")
        .current_dir(&tmp_dir)
        .args(["fetch", "origin", &refspec])
        .status()
        .map_err(|e| CliError::Subprocess(format!("git not found: {e}")))?;
    if !status.success() {
        return Err(CliError::Git(format!(
            "git fetch origin {refspec} failed.\nMake sure the {} exists in the repository.",
            src.display(),
        )));
    }
    let status = Command::new("git")
        .current_dir(&tmp_dir)
        .args(["checkout", "FETCH_HEAD"])
        .status()
        .map_err(|e| CliError::Subprocess(format!("git not found: {e}")))?;
    if !status.success() {
        return Err(CliError::Git("git checkout FETCH_HEAD failed.".into()));
    }
    eprintln!("done");

    eprintln!("  compiling (this may take a few minutes)...");
    let status = Command::new("cargo")
        .current_dir(&tmp_dir)
        .args(["build", "--release", "-p", "metaltile-cli"])
        .status()
        .map_err(|e| CliError::Subprocess(format!("cargo not found: {e}")))?;
    if !status.success() {
        return Err(CliError::Subprocess("cargo build failed.".into()));
    }

    eprint!("  installing to {}... ", dest.display());
    install_binary(&tmp_dir.join("target/release/tile"), &dest)?;
    eprintln!("done");

    let _ = fs::remove_dir_all(&tmp_dir);
    eprintln!();

    eprintln!(
        "{}  tile installed from {}.",
        paint_stderr("ok", Style::new().fg(Color::Green).bold()),
        src.display(),
    );
    Ok(())
}

// ── Shared install helper ─────────────────────────────────────────────────

fn install_binary(src: &PathBuf, dest: &PathBuf) -> Result<(), CliError> {
    let tmp = dest.with_extension("update-tmp");

    fs::copy(src, &tmp).map_err(|e| CliError::Subprocess(format!("failed to copy binary: {e}")))?;

    fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755)).map_err(CliError::Io)?;

    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            CliError::Subprocess(format!(
                "permission denied writing to {}.\n  Try: sudo tile update",
                dest.display(),
            ))
        } else {
            CliError::Io(e)
        }
    })?;

    Ok(())
}

fn header(title: &str) {
    eprintln!("{}", paint_stderr(title, Style::new().fg(Color::Cyan).bold()));
    eprintln!();
}
