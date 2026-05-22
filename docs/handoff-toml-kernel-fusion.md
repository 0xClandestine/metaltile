# TOML Kernel‑Body Fusion — Handoff & Design Doc

**Author**: AI coding agent (2026-05-22)
**Status**: design complete, implementation starting
**Branch**: `feat/metaltile-ml`

## 1. What We're Building

### Current State

The TOML model (`models/llama_decode.toml`) uses `fuse = "q_chain"` /
`fuse = "ffn_act"` tags on contiguous kernel groups. After commit `635daa6`
("single dispatch_chain per token"), **every kernel in the forward pass is
already dispatched in one MTLCommandBuffer**. The `fuse_group` annotation on
each `DispatchNode` is a **runtime no-op** — all nodes go into the same buffer
regardless of grouping.

Separately, commit `23b2d70` ("cross-kernel calling via KernelCallArg +
KernelInlinePass") added true kernel‑body inlining: one `#[kernel]` can call
another inline via `Op::KernelCall`, and `KernelInlinePass` (the first pass
in `standard_pipeline()`) splices the callee's IR directly. This is used by
hand‑authored fused kernels (`ffai_rms_norm_residual`, `ffai_rms_norm_rope`,
`ffai_rms_norm_qgemv`) to share the `mt_rms_inv_scalar` reduction helper
without copy‑pasting.

**Gap**: the TOML compiler does NOT connect these two mechanisms. Even with
`fuse = "ffn_act"`, the compilers still emits separate `DispatchNode`s and
separate `cached_kernels` — one per TOML `[[layer.kernel]]` entry.
Intermediates like `_gate`, `_gated`, `_up` still go through global‑memory
slot buffers with full `read → write → barrier → read` round‑trips.

### Target State

When TOML `fuse` tags group adjacent kernels, the compiler **synthesizes a
single compound `Kernel`** whose body performs the fused computation. All
intra‑group intermediate tensors stay in registers (scalar result of a gemv
reduction) rather than round‑tripping through global memory. The compound
kernel is dispatched once (one Metal function launch, one PSO) instead of
N separate launches.

Concretely, this TOML fragment:

```toml
[[layer.kernel]]
op = "gemv"          # gate projection → _gate[ffn_dim]
fuse = "ffn_act"
inputs  = { mat = "$layers.$idx.mlp.gate_proj", vec = "_ffn_normed" }
outputs = { out = "_gate" }
dispatch  = { rows = "$ffn_dim", tpg = "256", out_elems = "$ffn_dim" }

[[layer.kernel]]
op = "unary/silu"    # silu(_gate) → _gated[ffn_dim]
fuse = "ffn_act"
inputs  = { a = "_gate" }
outputs = { out = "_gated" }

[[layer.kernel]]
op = "gemv"          # up projection → _up[ffn_dim] (parallel with gate)
fuse = "ffn_act"
inputs  = { mat = "$layers.$idx.mlp.up_proj", vec = "_ffn_normed" }
outputs = { out = "_up" }

[[layer.kernel]]
op = "binary/mul"    # _gated * _up → _combined[ffn_dim]
fuse = "ffn_act"
inputs  = { a = "_gated", b = "_up" }
outputs = { out = "_combined" }
```

…will produce a single kernel with this body (pseudo‑IR):

```
row = program_id(0)
rs = row * k
re = rs + k

// gate gemv
acc_gate = strided_reduce_dot(mat_gate, vec, rs, rs, re)
gate = reduce_sum(acc_gate)

// silu  (applied per-threadgroup to the scalar result)
gated = Activation(Silu, gate)

// up gemv   (same vec, different mat)
acc_up = strided_reduce_dot(mat_up, vec, rs, rs, re)
up = reduce_sum(acc_up)

// elementwise multiply
combined = BinOp(Mul, gated, up)

store(out[row], combined)
```

Grid: `(ffn_dim, 1, 1)` with TPG = 256 (the gemv layout). Thread 0 of
each threadgroup does the silu + mul + store; the other 255 threads are
idle after the second `reduce_sum`.

Eliminated intermediates:
- `_gate` (ffn_dim × 2 bytes = ~28 KB for Llama 8B)
- `_gated` (same)
- `_up` (same)
- 3 kernel launches (silu, up_gemv, mul)

## 2. Architecture

### 2.1 New module: `crates/metaltile-model/src/fuse_group.rs`

This module is called by the compiler after TOML/Graph fusion groups are
assigned (step 2.5 in `compiler::compile`). It transforms contiguous
`DispatchNode` sequences sharing a `fuse_group` into single fused nodes.

### 2.2 Fusion strategies

We classify each fuse group into one of four strategies:

| Strategy | Pattern | Grid | Intermediate elimination |
|---|---|---|---|
| `GemvElementwise` | gemv → {unary\|binary}* | gemv's grid | `_gate`, `_gated`, `_up`, etc. → registers |
| `GemvScatter` | gemv → kv_cache_update | gemv's grid | intermediate → register, then scatter |
| `ElementwiseChain` | {unary\|binary}+ | unified elementwise | all → registers |
| `Incompatible` | anything else (gemv→rope, etc.) | N/A | fallback to dispatch batching |

### 2.3 Data structures

```rust
/// Result of analyzing a fuse group.
struct FuseGroupAnalysis {
    strategy: FuseStrategy,
    /// Subset of DispatchNode indices that can be fused.
    fusible_range: Range<usize>,
    /// Dependency DAG on the fusible nodes.
    dag: FuseDag,
}

/// A directed acyclic graph of sub-kernel operations within a fuse group.
/// Nodes are operations (loads, gemv reductions, elementwise ops).
struct FuseDag {
    /// Topologically sorted operation descriptors.
    ops: Vec<FuseOp>,
}

enum FuseOp {
    /// Cooperative matrix-vector dot product + reduction.
    /// Inputs: mat tensor, vec tensor
    /// Output: scalar ValueId
    Gemv { mat: String, vec: String, k: u32 },
    /// Elementwise unary activation (silu, gelu, etc.).
    /// Input: scalar ValueId
    /// Output: scalar ValueId
    Activation { kind: ActKind },
    /// Elementwise binary op (add, mul).
    /// Input: two scalar ValueIds
    /// Output: scalar ValueId
    Binary { kind: BinOpKind },
    /// Store the final result.
    Store { dst: String },
}

enum FuseStrategy {
    GemvElementwise,
    GemvScatter,
    ElementwiseChain,
    Incompatible,
}
```

### 2.4 Synthesis pipeline

```
                    TOML fuse groups assigned
                            │
                            ▼
         ┌──────────────────────────────────┐
         │  FuseGroupAnalysis::analyze()     │
         │  - Build dependency DAG           │
         │  - Classify strategy              │
         │  - Validate fusibility            │
         └──────────────┬───────────────────┘
                        │
              ┌─────────▼──────────┐
              │  Incompatible?      │──yes──▶ keep original nodes
              └─────────┬──────────┘
                        │ no
                        ▼
         ┌──────────────────────────────────┐
         │  FuseGroupSynthesizer::synthesize │
         │  - Build fused Kernel IR          │
         │  - Map params → fused params      │
         │  - Generate fused DispatchNode    │
         └──────────────┬───────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Replace N nodes with 1 fused     │
         │  node in the ExecutionPlan        │
         │  - Update slots (eliminate intra-  │
         │    group intermediates)            │
         │  - Update cached_kernels           │
         └──────────────────────────────────┘
```

### 2.5 Key design decisions

**Q: Why not use `KernelCall` / `KernelInlinePass` for the fusion?**
A: The sub‑kernels in a TOML fuse group have **incompatible grid topologies**.
For example, `gemv` uses `(n_elems, 1, 1)` grid with TPG=256 (all threads
cooperatively reduce to a scalar per threadgroup), while `silu` uses
`(n_elems, 1, 1)` grid with TPG=1 (one thread per element). Simply wrapping
both in `KernelCall` would produce a fused body where both sub‑kernels
reference `program_id(0)` and `tid` but interpret them differently. The
`KernelInlinePass` remaps callee `ProgramId` ops to the caller's, but it
doesn't transform the callee's thread‑level semantics.

Instead, the compiler **directly synthesizes the fused body** using the
gemv kernel's threadgroup layout, and transforms the elementwise ops to
operate on the scalar `reduce_sum` result within each threadgroup. This is
the same pattern hand‑fused kernels like `ffai_rms_norm_residual` use.

**Q: What about rope, kv_cache_update, sdpa?**
A: `rope` has grid `(n_heads, half_dim, 1)` with TPG=1 — fundamentally
incompatible with gemv's `(n_heads*head_dim, 1, 1)` with TPG=256. Different
threadgroup count, different element mapping. These remain dispatch‑batched
until we have a "grid remapping" transformation (future work).

`kv_cache_update` COULD be fused with gemv (the gemv result is a scalar,
`kv_cache_update` writes it to the cache at `position`). This is the
`GemvScatter` strategy — straightforward to implement.

`sdpa_vector` uses a grid of `(n_heads, 1, 1)` with TPG=1024. Different
grid semantics from gemv. Falls back to `Incompatible`.

**Q: How are intermediate buffers eliminated?**
A: The liveness analysis step (`assign_slots`) currently treats all
intermediates the same. After fusion synthesis, intra‑group intermediates
are NOT present in the fused node's `input_bindings` / `output_bindings`
(the fused node reads only the group's external inputs and writes only the
group's final outputs). The slot assignment runs after our synthesis and
will not allocate slots for the eliminated intermediates.

## 3. Implementation Plan

### Phase 1: Core synthesis engine (`fuse_group.rs`)

1. **`FuseGroupAnalysis`** — builds the dependency DAG from TOML
   input/output bindings, classifies the strategy.

2. **IR builder helpers** — functions that construct `Op::ProgramId`,
   `Op::StridedReduceDot`, `Op::ReduceSum`, `Op::Activation`,
   `Op::BinOp`, `Op::Store` in the fused `Kernel` body, allocating
   `ValueId`s sequentially.

3. **`GemvElementwiseSynthesizer`** — synthesizes the gemv→elementwise
   fused body. Handles:
   - Single gemv → unary (gate→silu)
   - Parallel gemv + gemv → binary (gate→silu, up→mul)
   - Single gemv → binary with weight tensor (o_proj→residual_add)

4. **`ElementwiseChainSynthesizer`** — synthesizes pure elementwise
   chains.

5. **`GemvScatterSynthesizer`** — synthesizes gemv→kv_cache_update.

### Phase 2: Compiler integration

1. Insert a new step **before** slot assignment (after step 2.5 in
   `compiler::compile`) that:
   - Groups `DispatchNode`s by `fuse_group`
   - For each group, runs `FuseGroupAnalysis`
   - For fusible groups, calls `FuseGroupSynthesizer`
   - Replaces the N nodes with 1 fused node in `nodes`, `cached_kernels`,
     `intermediate_outputs`, `intermediate_inputs`

2. Update the intermediate tracking to exclude intra‑group intermediates.

### Phase 3: Testing

1. **Unit tests** — test each `FuseStrategy` synthesizer with hand‑crafted
   node sequences.

2. **Integration test** — compile `llama_decode.toml` with `FusionMode::TomlDriven`,
   verify:
   - Fuse groups like `ffn_act` produce single fused nodes
   - Intermediate slot count is reduced
   - `cached_kernels.len() < original node count`

3. **GPU correctness test** — run a known model through `Session::step`
   with fusion enabled, compare output token sequence with unfused baseline.

### Phase 4: Benchmarking & tuning

1. Measure tokens/second before/after with `--fuse` (graph‑driven) and
   default (TOML‑driven).
2. Profile register pressure — doubling gemv accumulators in ffn_act (gate
   + up running interleaved) may spill if TPG is too high. May need a
   `max_tpg_for_fusion` heuristic.

## 4. Files Changed

| File | Change |
|---|---|
| `crates/metaltile-model/src/fuse_group.rs` | **New** — synthesis engine |
| `crates/metaltile-model/src/compiler.rs` | Insert synthesis step after fusion grouping |
| `crates/metaltile-model/src/lib.rs` | `pub mod fuse_group` |
| `crates/metaltile-model/src/plan.rs` | Minor: maybe add `is_fused: bool` to `DispatchNode` |
| `crates/metaltile-core/src/ir.rs` | No changes needed (existing ops suffice) |

## 5. Open Questions

1. **Register pressure**: ffn_act fuses two gemvs. Each gemv accumulator is
   1 f32 register (after `reduce_sum`). The two gemvs are sequential, so
   the fused kernel needs 2 f32 registers for the results + 1 for the gated
   intermediate = 3 f32 registers. No issue. But the `strided_reduce_dot`
   uses threadgroup memory for the cooperative reduction — the fused kernel
   reuses the same threadgroup memory for both reductions (sequentially).
   This means the fused kernel's threadgroup memory footprint equals the
   larger of the two individual gemvs, not their sum. ✓

2. **Mixed dtype**: TOML models use a single activation dtype (f16, bf16,
   or f32). All kernels in a fuse group share this dtype. No type casting
   between sub‑kernels. ✓

3. **`FusionMode::GraphDriven`**: The graph‑driven fusion pass
   (`fuse_dispatch_nodes`) also creates groups. We should run synthesis
   for graph‑driven groups too — the same synthesis engine applies.

4. **`KernelCall` future use**: Once the synthesis engine handles the
   reduction‑to‑elementwise pattern, `KernelCall` could be used for more
   exotic patterns where all sub‑kernels share a compatible grid. This is
   out of scope for this implementation but the module structure leaves
   room for it.

## 6. Non‑Goals (for this implementation)

- Grid‑remapping fusion (e.g., gemv→rope) — fundamentally different grid
  topologies
- Mixed‑mode fusion (reduction + another reduction)
- Thread‑specialization beyond the "thread 0 does post‑processing" pattern
- Automatic fusion of arbitrary kernel pairs