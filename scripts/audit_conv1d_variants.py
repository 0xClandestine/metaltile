#!/usr/bin/env python3
"""
audit_conv1d_variants.py
------------------------
Parses the legacy per-file conv1d kernels and the new consolidated
conv1d.rs variants block, then emits a coverage table showing which
old kernel maps to which consolidated variant (or is new / uncovered).

Run from the repo root:
    python3 scripts/audit_conv1d_variants.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "crates/metaltile-std/src/convolution"

# ---------------------------------------------------------------------------
# 1.  Old (pre-consolidation) kernel registry
# ---------------------------------------------------------------------------
# Manually enumerated from source files because the naming conventions
# vary per file.  Each entry is:
#   (old_fn_name, source_file, expected_consolidated_variant_or_note)
# ---------------------------------------------------------------------------

DENSE_OLD = [
    # fn name                           file                                 (T, D, DW)  consolidated suffix
    ("audio_conv1d",                    "audio_conv1d.rs",                   (0, 0, 0)),
    ("conv1d_dilated",                  "conv1d_dilated_transpose.rs",       (0, 1, 0)),
    ("conv1d_transpose",                "conv1d_dilated_transpose.rs",       (1, 0, 0)),
    ("ffai_conv1d_transpose_depthwise", "conv1d_transpose_depthwise.rs",     (1, 0, 1)),
]

# Format-name → (FMT index, BITS, WT type, ST type)
FMT_TABLE = {
    "mxfp4":      ( 0, 4,  "u32", "u8"),
    "nvfp4":      ( 1, 4,  "u32", "u8"),
    "mxint2":     ( 2, 2,  "u32", "u8"),   # NEW – no old kernel
    "mxint3":     ( 3, 3,  "u32", "u8"),   # NEW
    "mxint4":     ( 4, 4,  "u32", "u8"),   # NEW
    "mxint5":     ( 5, 5,  "u32", "u8"),   # NEW
    "mxint6":     ( 6, 6,  "u32", "u8"),   # NEW
    "fp4":        ( 7, 4,  "u32", "f32"),
    "int2":       ( 8, 2,  "u32", "f32"),  # NEW
    "int3":       ( 9, 3,  "u32", "f32"),  # NEW
    "int4":       (10, 4,  "u32", "f32"),  # NEW
    "int5":       (11, 5,  "u32", "f32"),  # NEW
    "int6":       (12, 6,  "u32", "f32"),  # NEW
    "mxfp8_e4m3": (13, 8,  "u8",  "u8"),
    "mxfp8_e5m2": (14, 8,  "u8",  "u8"),
    "mxint8":     (15, 8,  "u8",  "u8"),
    "fp8_e5m2":   (16, 8,  "u8",  "f32"),
    "nvfp8":      (17, 8,  "u8",  "f32"),
    "int8":       (18, 8,  "u8",  "f32"),
}

# Old block-scaled kernel names follow the pattern:
#   mt_{fmt}_{audio|fishspeech}_conv1d[_f16]
# The `_f16` suffix means the activation type T was hardcoded to f16 in the
# old file; the consolidated kernel is generic<T> so the same variant covers
# both T=f32 and T=f16.

def make_block_scaled_old():
    rows = []
    for path_label, dilated in [("audio_conv1d_block_scaled.rs", 0),
                                 ("fishspeech_conv1d_block_scaled.rs", 1)]:
        path = CONV_DIR / path_label
        if not path.exists():
            print(f"  [WARN] file not found: {path}", file=sys.stderr)
            continue
        src = path.read_text()
        for fn_name in re.findall(r"pub fn (mt_\w+)<T>", src):
            tag = "audio" if dilated == 0 else "fishspeech"
            # strip leading mt_ and trailing _{tag}_conv1d to get raw fmt label
            # handles both mt_{fmt}_{tag}_conv1d and mt_{fmt}_f16_{tag}_conv1d
            m = re.match(rf"mt_(.+?)_(f16_)?{tag}_conv1d$", fn_name)
            if m:
                fmt_label   = m.group(1)
                explicit_f16 = m.group(2) is not None
            else:
                fmt_label    = "???"
                explicit_f16 = False
            fmt_entry = FMT_TABLE.get(fmt_label)
            rows.append({
                "fn":          fn_name,
                "file":        path_label,
                "dilated":     dilated,
                "fmt_label":   fmt_label,
                "fmt_idx":     fmt_entry[0] if fmt_entry else None,
                "explicit_f16": explicit_f16,
            })
    return rows


# ---------------------------------------------------------------------------
# 2.  Parse consolidated conv1d.rs to extract the two variants blocks
# ---------------------------------------------------------------------------

def parse_consolidated():
    path = CONV_DIR / "consolidated/conv1d.rs"
    if not path.exists():
        print(f"  [WARN] consolidated file not found: {path}", file=sys.stderr)
        return [], []
    src = path.read_text()

    # --- Dense variants: look for TRANSPOSE / DILATED / DEPTHWISE arrays ---
    dense = []
    def parse_u32_dense(s):
        return [int(x.replace("u32","").strip()) for x in s.split(",") if x.strip()]
    t_vals  = parse_u32_dense(re.search(r"TRANSPOSE\s*=\s*\[([^\]]+)\]", src).group(1))
    d_vals  = parse_u32_dense(re.search(r"DILATED\s*=\s*\[([^\]]+)\]",   src).group(1))
    dw_vals = parse_u32_dense(re.search(r"DEPTHWISE\s*=\s*\[([^\]]+)\]", src).group(1))
    dense_suffix_pat = re.search(r'suffix\s*=\s*"([^"]+)"', src[:src.find("mt_conv1d_dense")+500])
    dense_suffix_tpl = dense_suffix_pat.group(1) if dense_suffix_pat else "fmt{TRANSPOSE}_{DILATED}_{DEPTHWISE}"
    fn_dense = "mt_conv1d_dense"
    for t, d, dw in zip(t_vals, d_vals, dw_vals):
        suffix = dense_suffix_tpl.replace("{TRANSPOSE}", str(t)).replace("{DILATED}", str(d)).replace("{DEPTHWISE}", str(dw))
        dense.append({
            "fn":   f"{fn_dense}_{suffix}",
            "TRANSPOSE": t, "DILATED": d, "DEPTHWISE": dw,
        })

    # --- Block-scaled variants: look in the section that contains pub fn mt_conv1d_block_scaled ---
    block = []
    # Find the pub fn declaration (not comments) and search backwards for the variants macro
    fn_pos = src.find("pub fn mt_conv1d_block_scaled")
    if fn_pos == -1:
        print("  [WARN] could not find pub fn mt_conv1d_block_scaled", file=sys.stderr)
    else:
        # The variants(...) macro starts before the pub fn; grab the section
        # from the second #[kernel(variants to the pub fn
        kernel_positions = [m.start() for m in re.finditer(r"#\[kernel\(variants", src)]
        # Pick the last #[kernel(variants before fn_pos
        bs_kernel_pos = max((p for p in kernel_positions if p < fn_pos), default=None)
        rest = src[bs_kernel_pos:fn_pos + 200] if bs_kernel_pos else src[fn_pos - 2000:fn_pos + 200]
        dilated_m = re.search(r"DILATED\s*=\s*\[([^\]]+)\]", rest)
        fmt_m     = re.search(r"\bFMT\s*=\s*\[([^\]]+)\]",   rest)
        suffix_m  = re.search(r'suffix\s*=\s*"([^"]+)"',      rest)
        if dilated_m and fmt_m:
            def parse_u32_list(s):
                return [int(x.replace("u32","").strip()) for x in s.split(",") if x.strip()]
            dil_list = parse_u32_list(dilated_m.group(1))
            fmt_list = parse_u32_list(fmt_m.group(1))
            bs_suffix_tpl = suffix_m.group(1) if suffix_m else "{DILATED}_{FMT}"
            fn_block = "mt_conv1d_block_scaled"
            for dil, fmt in zip(dil_list, fmt_list):
                suffix = bs_suffix_tpl.replace("{DILATED}", str(dil)).replace("{FMT}", str(fmt))
                block.append({
                    "fn":      f"{fn_block}_{suffix}",
                    "DILATED": dil,
                    "FMT":     fmt,
                })
    return dense, block


# ---------------------------------------------------------------------------
# 3.  Build and print coverage tables
# ---------------------------------------------------------------------------

def fmt_label_for(fmt_idx):
    for label, (idx, *_) in FMT_TABLE.items():
        if idx == fmt_idx:
            return label
    return "???"


def print_dense_table(dense_old, consolidated_dense):
    consolidated_by_key = {(r["TRANSPOSE"], r["DILATED"], r["DEPTHWISE"]): r["fn"]
                           for r in consolidated_dense}
    print("## Dense 1D conv variants\n")
    print(f"| {'Old kernel':<45} | {'Source file':<35} | {'Consolidated variant':<40} | Status |")
    print(f"|{'-'*46}|{'-'*36}|{'-'*41}|--------|")
    for fn, src_file, (t, d, dw) in dense_old:
        consolidated = consolidated_by_key.get((t, d, dw), "NOT FOUND")
        status = "✓" if consolidated != "NOT FOUND" else "MISSING"
        print(f"| {fn:<45} | {src_file:<35} | {consolidated:<40} | {status} |")
    print()


def print_block_scaled_table(block_old, consolidated_block):
    # Build lookup: (dilated, fmt) → consolidated fn name
    cons_by_key = {(r["DILATED"], r["FMT"]): r["fn"] for r in consolidated_block}

    print("## Block-scaled 1D conv variants\n")
    print(f"| {'Old kernel':<45} | {'Source file':<40} | {'D'} | {'FMT':<3} | {'Format':<12} | {'Consolidated variant':<38} | Status |")
    print(f"|{'-'*46}|{'-'*41}|---|-----|{'-'*14}|{'-'*39}|--------|")

    covered_keys = set()
    for r in block_old:
        dil  = r["dilated"]
        fmt  = r["fmt_idx"]
        flbl = r["fmt_label"]
        fn   = r["fn"]
        file = r["file"]
        f16_note = " (T=f16, covered by generic T)" if r["explicit_f16"] else ""
        if fmt is None:
            cons = "???"
            status = "UNKNOWN"
        else:
            cons = cons_by_key.get((dil, fmt), "NOT FOUND")
            status = "✓" if cons != "NOT FOUND" else "MISSING"
            if cons != "NOT FOUND":
                covered_keys.add((dil, fmt))
        print(f"| {fn:<45} | {file:<40} | {dil} | {fmt if fmt is not None else '?':>3} | {flbl:<12} | {cons:<38} | {status}{f16_note} |")
    print()

    # Show NEW variants (in consolidated but no old kernel)
    new_variants = [(r["DILATED"], r["FMT"], r["fn"])
                    for r in consolidated_block
                    if (r["DILATED"], r["FMT"]) not in covered_keys]
    if new_variants:
        print("### New formats (consolidated only — no legacy equivalent)\n")
        print(f"| {'Consolidated variant':<42} | {'D'} | {'FMT':<3} | {'Format':<12} | WT   | ST  | BITS |")
        print(f"|{'-'*43}|---|-----|{'-'*14}|------|-----|------|")
        for dil, fmt, cons_fn in new_variants:
            flbl = fmt_label_for(fmt)
            entry = FMT_TABLE.get(flbl)
            if entry:
                _, bits, wt, st = entry
            else:
                bits, wt, st = "?", "?", "?"
            print(f"| {cons_fn:<42} | {dil} | {fmt:>3} | {flbl:<12} | {wt:<4} | {st:<3} | {bits:>4} |")
        print()


def check_not_consolidated():
    """Kernels intentionally NOT in the consolidated file."""
    causal = CONV_DIR / "conv1d_causal_step_silu_cast_many.rs"
    if causal.exists():
        src = causal.read_text()
        fns = re.findall(r"pub fn (\w+)<T>", src)
        print("## Kernels intentionally NOT consolidated\n")
        print(f"| {'Kernel':<50} | {'File':<45} | Note |")
        print(f"|{'-'*51}|{'-'*46}|------|")
        for fn in fns:
            print(f"| {fn:<50} | conv1d_causal_step_silu_cast_many.rs          | separate SSM kernel |")
        print()


def main():
    print("# conv1d variant coverage audit\n")
    print(f"Repo: `{ROOT}`\n")

    consolidated_dense, consolidated_block = parse_consolidated()
    block_old = make_block_scaled_old()

    if not consolidated_dense:
        print("ERROR: could not parse dense variants from consolidated/conv1d.rs")
        sys.exit(1)
    if not consolidated_block:
        print("ERROR: could not parse block-scaled variants from consolidated/conv1d.rs")
        sys.exit(1)

    print(f"Consolidated dense variants parsed:        {len(consolidated_dense)}")
    print(f"Consolidated block-scaled variants parsed: {len(consolidated_block)}")
    print(f"Legacy block-scaled kernels found:         {len(block_old)}\n")
    print("---\n")

    print_dense_table(DENSE_OLD, consolidated_dense)
    print_block_scaled_table(block_old, consolidated_block)
    check_not_consolidated()

    # Summary counts
    missing_dense = sum(
        1 for _, _, (t,d,dw) in DENSE_OLD
        if not any(r["TRANSPOSE"]==t and r["DILATED"]==d and r["DEPTHWISE"]==dw
                   for r in consolidated_dense)
    )
    missing_block = sum(
        1 for r in block_old
        if r["fmt_idx"] is not None and
           not any(c["DILATED"]==r["dilated"] and c["FMT"]==r["fmt_idx"]
                   for c in consolidated_block)
    )
    new_fmts = len(consolidated_block) - len({
        (r["dilated"], r["fmt_idx"]) for r in block_old if r["fmt_idx"] is not None
    })

    print("---\n")
    print("## Summary\n")
    print(f"- Dense: {len(DENSE_OLD) - missing_dense}/{len(DENSE_OLD)} old kernels covered  "
          f"{'✓' if missing_dense == 0 else f'({missing_dense} MISSING)'}")
    print(f"- Block-scaled: {len(block_old) - missing_block}/{len(block_old)} old kernels covered  "
          f"{'✓' if missing_block == 0 else f'({missing_block} MISSING)'}")
    print(f"- New formats added (no legacy equivalent): {new_fmts}")
    if missing_dense == 0 and missing_block == 0:
        print("\n**All legacy kernels are covered by the consolidated variants.**")
    else:
        print("\n**WARNING: some legacy kernels have no consolidated equivalent.**")
        sys.exit(1)


if __name__ == "__main__":
    main()
