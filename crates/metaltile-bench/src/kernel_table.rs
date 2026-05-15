//! MetalTile vs MLX reference kernel coverage report.
//!
//! Replaces `kernel_table.py`. Derives the spec table from each op module's
//! `kernel_specs()` rather than a manually-maintained Python list.
//!
//! Requires macOS + Xcode command-line tools (`xcrun`/`clang -E`).
//!
//! Usage:
//!   cargo run --bin kernel_table [-- --metal-dir <path>]
//!   cargo run --bin kernel_table -- --metal-dir crates/metaltile-bench/src/metal

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use metaltile_bench::ops::{KernelSpec, RefSpec, all_kernel_specs};

const DEFAULT_METAL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let metal_dir = flag_val(&args, "--metal-dir")
        .unwrap_or_else(|| DEFAULT_METAL_DIR.to_string());

    eprintln!("Preprocessing Metal files from {metal_dir}…");
    let metal_files = collect_metal_files(&metal_dir);

    let mut kernel_map: HashMap<String, HashSet<String>> = HashMap::new();
    for (rel, abs) in &metal_files {
        let names = extract_kernels(abs);
        kernel_map.insert(rel.clone(), names);
    }

    let specs = all_kernel_specs();
    print_report(&specs, &kernel_map);
}

// ── Metal file discovery ──────────────────────────────────────────────────────

fn collect_metal_files(metal_dir: &str) -> Vec<(String, String)> {
    let mut files = Vec::new();
    collect_recursive(Path::new(metal_dir), Path::new(metal_dir), &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

fn collect_recursive(root: &Path, dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_recursive(root, &path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("metal") {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            out.push((
                rel.to_string_lossy().into_owned(),
                path.to_string_lossy().into_owned(),
            ));
        }
    }
}

// ── Kernel name extraction via clang -E ──────────────────────────────────────

fn extract_kernels(metal_file: &str) -> HashSet<String> {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "clang", "-E", "-x", "c", metal_file])
        .output();

    let text = match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(e) => {
            eprintln!("[warn] xcrun failed for {metal_file}: {e}");
            return HashSet::new();
        }
    };

    let mut names = HashSet::new();

    // host_name("part1" "part2" ...) — template instantiation pattern
    let mut pos = 0;
    while let Some(idx) = text[pos..].find("host_name(") {
        let start = pos + idx + "host_name(".len();
        if let Some(end) = text[start..].find(')') {
            let inner = &text[start..start + end];
            // String literals separated by whitespace: "a" "b" → "ab"
            let name: String = inner
                .split('"')
                .enumerate()
                .filter(|(i, _)| i % 2 == 1)
                .map(|(_, s)| s)
                .collect();
            if !name.is_empty() {
                names.insert(name);
            }
            pos = start + end + 1;
        } else {
            break;
        }
    }

    // [[kernel]] void name( — direct kernel declaration
    let mut pos = 0;
    while let Some(idx) = text[pos..].find("[[kernel]]") {
        let after = pos + idx + "[[kernel]]".len();
        let rest = text[after..].trim_start();
        if rest.starts_with("void ") {
            let name_start = "void ".len();
            let name_end = rest[name_start..]
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(rest.len() - name_start);
            let name = &rest[name_start..name_start + name_end];
            if !name.is_empty() {
                names.insert(name.to_string());
            }
        }
        pos = after;
    }

    names
}

// ── Markdown report ───────────────────────────────────────────────────────────

fn print_report(specs: &[KernelSpec], kernel_map: &HashMap<String, HashSet<String>>) {
    let mut lines: Vec<String> = Vec::new();
    let mut benched: HashMap<String, HashSet<String>> = HashMap::new();

    lines.push("# MetalTile vs MLX Reference Kernel Coverage".into());
    lines.push(String::new());
    lines.push("## Per-Op Reference Status".into());
    lines.push(String::new());
    lines.push("Legend: ✅ exists in Metal source  ❌ NOT found  — no reference".into());
    lines.push(String::new());
    lines.push("| Op | MT Kernel | Dtype | MLX Reference | Status |".into());
    lines.push("|---|---|---|---|---|".into());

    let mut prev_op = "";
    for spec in specs {
        let dtypes: Vec<&str> = if spec.dtypes.is_empty() {
            vec![""]
        } else {
            spec.dtypes.iter().copied().collect()
        };

        for &dtype in &dtypes {
            let ref_name = spec.ref_spec.resolve(dtype);

            let (ref_display, status) = match &spec.ref_spec {
                RefSpec::None(reason) => (
                    format!("*{reason}*"),
                    "—".to_string(),
                ),
                _ => {
                    let name = ref_name.as_deref().unwrap_or("?");
                    let found = kernel_map
                        .get(spec.metal_file)
                        .map(|ks| ks.contains(name))
                        .unwrap_or(false);
                    if found {
                        if let Some(name) = &ref_name {
                            benched
                                .entry(spec.metal_file.to_string())
                                .or_default()
                                .insert(name.clone());
                        }
                        (format!("`{name}`"), "✅".to_string())
                    } else {
                        (format!("`{name}`"), "❌".to_string())
                    }
                }
            };

            let op_label = if spec.op != prev_op { spec.op } else { "" };
            prev_op = spec.op;

            lines.push(format!(
                "| {op_label} | `{}` | {dtype} | {ref_display} | {status} |",
                spec.mt_kernel
            ));
        }
    }

    // Metal file coverage section
    lines.push(String::new());
    lines.push("## Metal File Coverage".into());
    lines.push(String::new());
    lines.push(
        "How many of each Metal file's instantiated kernels are used as references.".into(),
    );
    lines.push(String::new());
    lines.push(
        "| Metal File | Total kernels | Benchmarked | % | Unbenchmarked examples |".into(),
    );
    lines.push("|---|---|---|---|---|".into());

    let mut grand_total = 0usize;
    let mut grand_benched = 0usize;

    let mut sorted_files: Vec<_> = kernel_map.keys().collect();
    sorted_files.sort();

    for rel in sorted_files {
        let all_k = &kernel_map[rel];
        if all_k.is_empty() {
            continue;
        }
        let b = benched.get(rel.as_str()).cloned().unwrap_or_default();
        let unbench: Vec<&String> = {
            let mut v: Vec<&String> = all_k.difference(&b).collect();
            v.sort();
            v
        };
        let pct = 100 * b.len() / all_k.len();
        grand_total += all_k.len();
        grand_benched += b.len();

        let examples = {
            let mut parts: Vec<String> = unbench.iter().take(3).map(|k| format!("`{k}`")).collect();
            if unbench.len() > 3 {
                parts.push(format!("… (+{} more)", unbench.len() - 3));
            }
            if parts.is_empty() { "—".to_string() } else { parts.join(", ") }
        };

        let icon = if pct == 100 { "✅" } else if pct > 0 { "⚠️" } else { "❌" };
        lines.push(format!(
            "| `{rel}` | {} | {} | {icon} {pct}% | {examples} |",
            all_k.len(),
            b.len()
        ));
    }

    lines.push(String::new());
    let total_pct = if grand_total > 0 { 100 * grand_benched / grand_total } else { 0 };
    lines.push(format!(
        "**Total**: {grand_benched}/{grand_total} instantiated kernels benchmarked ({total_pct}%)"
    ));

    println!("{}", lines.join("\n"));
}

fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}
