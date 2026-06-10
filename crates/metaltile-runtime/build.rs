//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Link config for the optional GPU backends (CUDA, HIP, Vulkan).
//!
//! Per-feature; each enabled feature emits its own `cargo:rustc-link-*`
//! directives. macOS without any feature builds the Metal path unchanged.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(have_cutlass)");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        cuda();
    }
}

fn cuda() {
    let cuda_root = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // `libnvrtc` lives in the toolkit lib dir; `libcuda` (driver) link stub
    // lives under `lib64/stubs`. Both 64-bit dir names are tried.
    for sub in ["lib64", "lib64/stubs", "lib", "lib/stubs", "lib/x64"] {
        println!("cargo:rustc-link-search=native={cuda_root}/{sub}");
    }
    for p in ["/usr/lib/aarch64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"] {
        println!("cargo:rustc-link-search=native={p}");
    }

    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=cuda");
    // cuBLAS tensor-core GEMM escape hatch (Path A): ships in the CUDA toolkit.
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Optional: AOT-compile the CUTLASS grouped-MoE GEMM when CUTLASS_DIR points
    // at a CUTLASS checkout. nvcc → static lib → link; emits cfg(have_cutlass)
    // so the FFI/runtime path compiles only when present. Skipped (no-op) on any
    // build without CUTLASS_DIR, so Mac/Metal and CUTLASS-less CUDA boxes are
    // unaffected.
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=NEMOTRON_CUTLASS_ARCH");
    if let Ok(cutlass_dir) = std::env::var("CUTLASS_DIR") {
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
        let arch = std::env::var("NEMOTRON_CUTLASS_ARCH").unwrap_or_else(|_| "sm_121a".to_string());
        let nvcc = format!("{cuda_root}/bin/nvcc");
        let src = format!("{}/cuda/cutlass_moe.cu", env!("CARGO_MANIFEST_DIR"));
        println!("cargo:rerun-if-changed={src}");
        let obj = format!("{out_dir}/cutlass_moe.o");
        let status = std::process::Command::new(&nvcc)
            .args([
                "-O3",
                "-std=c++17",
                &format!("-arch={arch}"),
                "--expt-relaxed-constexpr",
                "-Xcompiler",
                "-fPIC",
            ])
            .args([
                "-I",
                &format!("{cutlass_dir}/include"),
                "-I",
                &format!("{cutlass_dir}/tools/util/include"),
            ])
            .args(["-c", &src, "-o", &obj])
            .status()
            .expect("nvcc invocation for cutlass_moe.cu failed to start");
        assert!(status.success(), "nvcc failed to compile cutlass_moe.cu");
        let lib = format!("{out_dir}/libcutlass_moe.a");
        let ar = std::process::Command::new("ar")
            .args(["crs", &lib, &obj])
            .status()
            .expect("ar failed to start");
        assert!(ar.success(), "ar failed to archive cutlass_moe.o");
        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static=cutlass_moe");
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-cfg=have_cutlass");
    }
}
