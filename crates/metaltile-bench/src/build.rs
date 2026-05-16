use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Pinned MLX commit. Update this to pull newer MLX kernels.
const MLX_COMMIT: &str = "80a1c206f963f713b8f1f2ce71bac039a3d3baa7";
const MLX_URL: &str = "https://github.com/ml-explore/mlx.git";

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/metaltile-bench → crates/ → repo root
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let cache_dir = repo_root.join(".cache/mlx");

    ensure_mlx(&cache_dir);

    let mlx_root = &cache_dir;
    let kernels_dir = cache_dir.join("mlx/backend/metal/kernels");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let out_metal = out_dir.join("metal");

    // Only rerun if build.rs or the cache marker changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", cache_dir.join(".commit").display());

    preprocess_dir(&kernels_dir, &kernels_dir, &out_metal, mlx_root);
}

/// Ensure `.cache/mlx` exists and contains the pinned MLX commit, fetching it
/// if absent or stale.
fn ensure_mlx(cache_dir: &Path) {
    let marker = cache_dir.join(".commit");

    if cache_dir.exists() {
        if std::fs::read_to_string(&marker).ok().map(|s| s.trim().to_string()).as_deref() == Some(MLX_COMMIT) {
            return; // cache is valid
        }
        // Stale or corrupt cache — start fresh.
        std::fs::remove_dir_all(cache_dir).unwrap();
    }

    println!("cargo:warning=Fetching MLX kernels @ {}…", &MLX_COMMIT[..8]);

    // Shallow blobless sparse clone (downloads no file blobs yet).
    run("git", &[
        "clone", "--filter=blob:none", "--sparse", "--depth=1",
        MLX_URL, cache_dir.to_str().unwrap(),
    ]);

    // Restrict working tree to only the Metal kernels directory.
    run_in("git", &["sparse-checkout", "set", "--cone", "mlx/backend/metal/kernels"], cache_dir);

    // If latest HEAD isn't our pinned commit, fetch and checkout the exact SHA.
    let head = git_head(cache_dir);
    if head != MLX_COMMIT {
        run_in("git", &["fetch", "--depth=1", "origin", MLX_COMMIT], cache_dir);
        run_in("git", &["checkout", "FETCH_HEAD"], cache_dir);
    }

    std::fs::write(&marker, MLX_COMMIT).unwrap();
}

fn git_head(dir: &Path) -> String {
    let out = Command::new("git")
        .args(["-C", dir.to_str().unwrap(), "rev-parse", "HEAD"])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn run(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd).args(args).status()
        .unwrap_or_else(|e| panic!("failed to run `{cmd}`: {e}"));
    assert!(status.success(), "`{cmd} {}` failed", args.join(" "));
}

fn run_in(cmd: &str, args: &[&str], dir: &Path) {
    let status = Command::new(cmd).args(args).current_dir(dir).status()
        .unwrap_or_else(|e| panic!("failed to run `{cmd}`: {e}"));
    assert!(status.success(), "`{cmd} {}` failed", args.join(" "));
}

fn preprocess_dir(dir: &Path, kernels_dir: &Path, out_metal: &Path, mlx_root: &Path) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            preprocess_dir(&path, kernels_dir, out_metal, mlx_root);
        } else if path.extension().map_or(false, |e| e == "metal") {
            let relative = path.strip_prefix(kernels_dir).unwrap();
            // MLX steel files live under steel/*/kernels/foo.metal; strip the inner
            // `kernels/` component to match the layout the ops/*.rs files expect.
            let out_relative = strip_inner_kernels(relative);
            let out_path = out_metal.join(&out_relative);
            std::fs::create_dir_all(out_path.parent().unwrap()).unwrap();

            let output = Command::new("xcrun")
                .args(["-sdk", "macosx", "metal", "-E",
                       "-fno-modules",  // force textual expansion; runtime compiler can't handle #pragma clang module import
                       "-I", mlx_root.to_str().unwrap(),
                       path.to_str().unwrap()])
                .output()
                .unwrap_or_else(|e| panic!("failed to run xcrun metal: {e}"));

            if !output.status.success() {
                panic!("metal -E failed for {}:\n{}",
                       path.display(), String::from_utf8_lossy(&output.stderr));
            }

            std::fs::write(&out_path, &output.stdout).unwrap();
        }
    }
}

/// Remove any `kernels/` path component that appears after the first segment.
/// e.g. `steel/gemm/kernels/foo.metal` → `steel/gemm/foo.metal`
fn strip_inner_kernels(path: &Path) -> PathBuf {
    path.components()
        .enumerate()
        .filter(|(i, c)| !(*i > 0 && c.as_os_str() == "kernels"))
        .map(|(_, c)| c)
        .collect()
}
