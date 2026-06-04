//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Regression guard: `dequant_gemv_int4` indirect Swift wrappers.
//!
//! FFAI's GPU-router dispatches `dequant_gemv_int4` indirectly (grid shape
//! from an `MTLBuffer` rather than a host `MTLSize`). This test lives in
//! `metaltile-std` — not the CLI — because the kernel corpus lives here and
//! `all_benches()` only returns entries when std is linked.

use std::collections::BTreeMap;

use metaltile::{
    codegen::emit::{dtype_suffix, render_swift_wrappers},
    core::{DType, ir::Kernel},
    harness::bench::KernelBench,
};
use metaltile_std::all_benches;

fn collect_kernels_with_indirect_flags() -> Vec<Kernel> {
    let mut by_name: BTreeMap<String, (&'static dyn KernelBench, Vec<DType>)> = BTreeMap::new();
    for entry in all_benches() {
        let b = entry.bench();
        let Some(&first_dt) = b.dtypes().first() else { continue };
        let name = b.setup(first_dt).kernel().name.to_string();
        let e = by_name.entry(name).or_insert((b, Vec::new()));
        for &dt in b.dtypes() {
            if !e.1.contains(&dt) {
                e.1.push(dt);
            }
        }
    }
    let mut kernels = Vec::new();
    for (name, (b, dtypes)) in &by_name {
        for &dt in dtypes {
            let suffix = dtype_suffix(dt);
            let mono = if dtypes.len() == 1 && name.ends_with(&format!("_{suffix}")) {
                name.clone()
            } else {
                format!("{name}_{suffix}")
            };
            let mut k = b.setup(dt).kernel().clone();
            k.name = mono.clone();
            if matches!(mono.as_str(), "dequant_gemv_int4_f16" | "dequant_gemv_int4_bf16") {
                k.wants_indirect_variant = true;
            }
            kernels.push(k);
        }
    }
    kernels
}

#[test]
fn dequant_gemv_int4_is_registered() {
    let kernels = collect_kernels_with_indirect_flags();
    assert!(
        kernels.iter().any(|k| k.name == "dequant_gemv_int4_f16"),
        "dequant_gemv_int4_f16 missing from kernel set"
    );
    assert!(
        kernels.iter().any(|k| k.name == "dequant_gemv_int4_bf16"),
        "dequant_gemv_int4_bf16 missing from kernel set"
    );
}

#[test]
fn dequant_gemv_int4_swift_wrappers_are_indirect() {
    let kernels = collect_kernels_with_indirect_flags();
    let swift = render_swift_wrappers(&kernels);
    assert!(
        swift.contains("func dequant_gemv_int4_f16_indirect("),
        "indirect Swift wrapper for dequant_gemv_int4_f16 dropped"
    );
    assert!(
        swift.contains("func dequant_gemv_int4_bf16_indirect("),
        "indirect Swift wrapper for dequant_gemv_int4_bf16 dropped"
    );
    assert!(
        swift.contains("dispatchThreadgroups(indirectBuffer:"),
        "indirect wrappers must use indirect buffer dispatch"
    );
}
