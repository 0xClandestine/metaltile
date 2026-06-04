# Kernel Specialization Spec

## Motivation

MetalTile kernels frequently need N structurally-identical variants that differ only
in compile-time constants. Today this is handled with `macro_rules!`, which:

- Cannot be used inside `#[kernel]` bodies (silently dropped by the DSL parser)
- Requires a separate Rust-level macro with its own syntax, hygiene, and error spans
- Forces kernel authors to maintain boilerplate invocation lists alongside the template
- Produces confusing rustdoc output (macro_rules! items, not fn items)

**Current state of the problem:**

| File | Pattern | Lines before | Lines after |
|---|---|---|---|
| `ffai/sdpa_decode.rs` | 5 arms × ~240 lines | 1,210 | ~150 |
| `ffai/moe.rs` (m8/m16/m32) | 3 manual copies | ~1,620 | ~170 |
| `ffai/rms_norm_qgemv.rs` | 3+ size variants | ~535 | ~60 |
| **Total** | | **~3,365** | **~380** |

All of these use `macro_rules!` solely to substitute integer literals into otherwise
identical kernel bodies.

The DSL's `UnrollPass` already handles the downstream effect: `for _i in range(0u32, ELEMS, 1u32)`
with a constant `ELEMS` unrolls to consecutive constant-indexed `stack_alloc` accesses,
producing the same MSL as hand-written named sequences. The missing piece is a way to
supply that constant at the Rust source level without `macro_rules!`.

---

## Design decision: merged into `#[kernel]`

Specialization determines *what kernels exist* — it is definitional, not a
transformation layer. It belongs inside `#[kernel]` alongside the rest of the kernel
definition. `#[bench]` and `#[test_kernel]` are separate attributes because they serve
a different concern (registration and testing); specialization does not.

Merging keeps the entire kernel definition in one place, avoids a second attribute to
explain, and leaves all existing `#[kernel]` call sites unaffected — no args means the
existing code path runs unchanged.

---

## Syntax

### Single parameter

```rust
#[kernel(specialize(M = [8, 16, 32], suffix = "m{M}"))]
pub fn mt_moe_gather_qmm_int4<T>(...) { ... }
```

Produces three kernel modules:
- `mt_moe_gather_qmm_int4_m8`
- `mt_moe_gather_qmm_int4_m16`
- `mt_moe_gather_qmm_int4_m32`

### Multiple parameters — zipped, not cartesian

```rust
#[kernel(specialize(
    ELEMS       = [2,  3,  4,  8,  16],
    PHASE_COUNT = [1,  1,  1,  2,  4 ],
    TG_SLOTS    = [2,  3,  4,  4,  4 ],
    suffix      = "d{ELEMS * 32}",
))]
pub fn ffai_sdpa_decode<T>(...) { ... }
```

Produces: `ffai_sdpa_decode_d64`, `_d96`, `_d128`, `_d256`, `_d512`.

All parameter lists must have equal length N. Variant `i` substitutes
`ELEMS=list[i], PHASE_COUNT=list[i], TG_SLOTS=list[i]` simultaneously.
Mismatched lengths are a compile error with a span on the attribute.

### Suffix template

`suffix = "..."` is appended to the source function name to form the variant name.
The string may contain `{expr}` segments where `expr` references any specialization
parameter and uses the operators `+`, `-`, `*`, `/`, `(`, `)`.

```rust
suffix = "m{M}"            // → base_m8, base_m16, base_m32
suffix = "d{ELEMS * 32}"   // → base_d64, base_d96, base_d128 ...
suffix = "{A}x{B}"         // → base_2x1, base_4x2 ...
```

The assembled name (`base_name + suffix_value`) must be a valid Rust identifier;
a compile error is emitted otherwise.

`suffix` is optional. When omitted, variant names are auto-derived by appending
`_{lowercase_param}{value}` for each parameter in declaration order:

```rust
// suffix omitted — auto-derives _m8, _m16, _m32
#[kernel(specialize(M = [8, 16, 32]))]
pub fn mt_moe_gather_qmm_int4<T>(...) { ... }
// → mt_moe_gather_qmm_int4_m8, _m16, _m32
```

Auto-derivation is unambiguous for single-param cases. For multi-param, an explicit
`suffix` is recommended to avoid names like `_elems8_phase_count2`.

### Substitution scope

In each variant's cloned function body, every bare `Ident` matching a specialization
parameter name is replaced with the corresponding integer literal. This covers:

- Size arguments: `stack_alloc("acc", M, "f32")`
- Loop bounds: `range(0u32, M, 1u32)`
- Arithmetic: `m_chunk * M`, `m_base + M`
- Any other expression-position occurrence of the parameter name

Substitution does **not** apply inside string literals. `stack_alloc("M_sized", ...)` is
unchanged; only the bare ident `M` in expression position is replaced.

The source function name itself is **not** emitted — only the N specialized variants are
registered in the kernel inventory.

---

## What it handles vs. what it doesn't

Specialization works when all variants share the same function signature and same body
structure. The only differences are integer constants.

It does **not** collapse variants with:
- Different function parameters (e.g. `has_sink: u32` present in some sdpa_decode
  variants but not others)
- Different body control flow (different `if` branches, different loop structure)

For `sdpa_decode.rs`, the five `macro_rules!` arms have structurally different
signatures and bodies. They map to separate `#[kernel]` functions. The d256 and d512
arms share a phased structure and collapse into one specialization:

```rust
// Three structurally distinct arms — separate #[kernel] bodies (~40-60 lines each)

#[kernel]
pub fn ffai_sdpa_decode_d64<T>(
    #[constexpr] has_sink: u32,
    #[constexpr] sink_logit: f32,
    ...
) { /* 2-slot sink body */ }

#[kernel]
pub fn ffai_sdpa_decode_d96<T>(...) { /* 3-slot simple body */ }

#[kernel]
pub fn ffai_sdpa_decode<T>(
    #[constexpr] sink_end: u32,
    #[constexpr] window_start: u32,
    ...
) { /* 4-slot sink+window body */ }

// d256 and d512 share phased structure — collapsed with specialize
#[kernel(specialize(ELEMS = [8, 16], PHASE_COUNT = [2, 4], suffix = "d{ELEMS * 32}"))]
pub fn ffai_sdpa_decode<T>(...) { /* phased body */ }
```

For `moe.rs` the bodies are structurally identical across m8/m16/m32 — full collapse:

```rust
// Before: m8 (265 L) + m16 (462 L) + m32 (892 L) = 1,619 lines
// After:

#[kernel(specialize(M = [8, 16, 32], suffix = "m{M}"))]
pub fn mt_moe_gather_qmm_int4<T>(
    x: Tensor<T>,
    weight_packed: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    expert_offsets: Tensor<u32>,
    mut out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] m_out: u32,
    #[constexpr] n_experts: u32,
    #[constexpr] group_size: u32,
) {
    let m_chunk = tgid_x;
    let row = tgid_y;
    let lane = tid;
    let m_base = m_chunk * M;
    let total_packs = k_in / 8u32;
    let groups_per_row = k_in / group_size;
    let weight_expert_base = expert * m_out * total_packs;
    let scale_expert_base = expert * m_out * groups_per_row;
    let x_row_base = row * k_in;
    stack_alloc("acc", M, "f32");
    for _m in range(0u32, M, 1u32) {
        stack_store("acc", _m, 0.0f32);
    }
    for pack_idx in range(lane, total_packs, 32u32) {
        let k_first = pack_idx * 8u32;
        let g = k_first / group_size;
        stack_alloc("xs", 8, "f32");
        for _k in range(0u32, 8u32, 1u32) {
            stack_store("xs", _k, load(x[x_row_base + k_first + _k]).cast::<f32>());
        }
        for _m in range(0u32, M, 1u32) {
            let wrb = weight_expert_base + (m_base + _m) * total_packs;
            let srb = scale_expert_base + (m_base + _m) * groups_per_row;
            let p = load(weight_packed[wrb + pack_idx]);
            let s = load(scales[srb + g]).cast::<f32>();
            let b = load(biases[srb + g]).cast::<f32>();
            let mut dot = 0.0f32;
            for _k in range(0u32, 8u32, 1u32) {
                dot = dot + ((p >> (_k * 4u32)) & 15u32).cast::<f32>() * s * stack_load("xs", _k)
                    + b * stack_load("xs", _k);
            }
            stack_store("acc", _m, stack_load("acc", _m) + dot);
        }
    }
    simd_sum_stack("acc", "acc_r", M);
    if lane == 0u32 {
        store_stack_n(out, row * m_out + m_base, "acc_r", M);
    }
}
```

Note: the nested `for _m` / `for _k` loops require the nested-loop unroll fix
(see companion spec) to unroll completely. Without it, `UnrollPass` skips the outer
loop because the body contains an inner loop.

---

## Error handling

All errors are reported at the `#[kernel(...)]` attribute span.

| Condition | Error |
|---|---|
| List lengths differ | `specialize: param lists must have equal length: ELEMS=5, PHASE_COUNT=4` |
| Empty list | `specialize: param list must not be empty` |
| Non-integer literal in list | `specialize: list values must be integer literals, got \`foo\`` |
| Unsupported suffix expr operator | `specialize: unsupported operator \`%\` in suffix expression` |
| Suffix references unknown param | `specialize: suffix references unknown param \`FOO\`` |
| Assembled name not a valid ident | `specialize: assembled name "foo bar" is not a valid identifier` |

---

## Implementation plan

### Step 1 — Extend `KernelAttr` in `crates/metaltile-macros/src/kernel/mod.rs`

`KernelAttr` currently errors on any arguments. Extend it to accept an optional
`specialize(...)` block:

```rust
pub(crate) struct KernelAttr {
    pub specialize: Option<SpecializeParams>,
}

pub(crate) struct SpecializeParams {
    pub params: Vec<(String, Vec<i64>)>,  // [("M", [8, 16, 32])]
    pub suffix: Option<String>,           // Some("m{M}") or None
    pub variant_count: usize,
}
```

Parsing in `KernelAttr::parse`:
- If `input.is_empty()` → existing behavior, `specialize: None`
- Otherwise expect `specialize(...)`, parse inner content as comma-separated
  `IDENT = [ lit, lit, ... ]` entries plus optional `suffix = "..."`.
- Validate equal lengths; error on mismatch.

### Step 2 — Extend `KernelMacroBuilder::expand` in `mod.rs`

```rust
pub(crate) fn expand(attr: KernelAttr, input_fn: ItemFn) -> TokenStream2 {
    match attr.specialize {
        None => KernelMacroBuilder::new(input_fn).expand_one(),
        Some(spec) => {
            let mut out = TokenStream2::new();
            for i in 0..spec.variant_count {
                let param_map: HashMap<String, i64> = spec.params.iter()
                    .map(|(name, vals)| (name.clone(), vals[i]))
                    .collect();
                let variant_fn = substitute_fn(input_fn.clone(), &param_map, &spec.suffix);
                out.extend(KernelMacroBuilder::new(variant_fn).expand_one());
            }
            out
        }
    }
}
```

`expand_one` is the current `expand` method renamed. For non-specialized kernels,
call path is identical to today.

### Step 3 — New file: `crates/metaltile-macros/src/kernel/specialize.rs`

Two functions:

**`substitute_fn(fn: ItemFn, params: &HashMap<String, i64>, suffix: &Option<String>) -> ItemFn`**

1. Walk the `ItemFn`'s token stream via `proc_macro2` and replace every `Ident`
   matching a param name with an integer literal. This is a recursive
   `TokenStream → TokenStream` rewrite; no `syn` AST required.
2. Convert back to `ItemFn` via `syn::parse2`.
3. Rewrite the function name: evaluate the suffix template (if present) or
   auto-derive the suffix, then assemble `format!("{}_{}", base_name, suffix_val)`
   as a new `syn::Ident`.

**`eval_suffix(template: &str, params: &HashMap<String, i64>) -> syn::Result<String>`**

Scan for `{...}` segments. For each, parse the inner text as a `syn::Expr` via
`syn::parse_str` and evaluate with the param map:

```rust
fn eval_expr(expr: &syn::Expr, params: &HashMap<String, i64>) -> syn::Result<i64> {
    match expr {
        syn::Expr::Lit(l)    => /* parse integer */,
        syn::Expr::Path(p)   => params.get(&name).copied().ok_or(/* unknown param error */),
        syn::Expr::Binary(b) => /* +, -, *, / */,
        syn::Expr::Paren(p)  => eval_expr(&p.expr, params),
        _                    => Err(/* unsupported */),
    }
}
```

Non-`{...}` segments are literal string fragments, concatenated as-is.

### Step 4 — Refactor `ffai/moe.rs`

Replace m8/m16/m32 kernels (~1,619 lines) with one `#[kernel(specialize(...))]` body
(~170 lines). Requires nested-loop unroll fix first, or can land with the inner `_k`
loop still hand-unrolled as an interim step.

### Step 5 — Refactor `ffai/sdpa_decode.rs`

Replace the 728-line `macro_rules! sdpa_decode_kernel` + invocations with five
`#[kernel]` functions (three structural variants + one `#[kernel(specialize(...))]`
for d256/d512). Net: ~1,060 lines removed.

### Step 6 — Refactor remaining sites

`grep -r 'macro_rules!' crates/metaltile-std/` to find remaining sites.
Primary target: `ffai/rms_norm_qgemv.rs` (~475 lines removable).

### Step 7 — Tests in `specialize.rs`

- Single param, correct variant count and names
- Multi-param zipped (not cartesian)
- Suffix arithmetic (`d{ELEMS * 32}`)
- Auto-derived suffix when `suffix` omitted
- Error: mismatched list lengths
- Error: empty list
- Substitution does not touch string literal contents

---

## Files changed

| File | Change |
|---|---|
| `crates/metaltile-macros/src/kernel/mod.rs` | Extend `KernelAttr`; add specialization dispatch in `expand` |
| `crates/metaltile-macros/src/kernel/specialize.rs` | New — `substitute_fn`, `eval_suffix`, `eval_expr` (~120 lines) |
| `crates/metaltile-std/src/ffai/moe.rs` | Replace m8/m16/m32 with one specialised body (~1,450 lines removed) |
| `crates/metaltile-std/src/ffai/sdpa_decode.rs` | Replace macro_rules! arms (~1,060 lines removed) |
| `crates/metaltile-std/src/ffai/rms_norm_qgemv.rs` | Replace size variants (~475 lines removed) |

**Total: ~120 lines added, ~2,985 lines removed.**

No changes to `metaltile-core`, `metaltile-codegen`, `metaltile-interp`,
`metaltile-runtime`, or any IR pass. Purely additive change to the macro layer
that composes with the existing `#[kernel]` expansion pipeline unchanged.
