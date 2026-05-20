# CLAUDE.md — MetalTile

Protips and hazards for working on the MetalTile DSL + codegen. Process
conventions (branching, commits, PR checklist, CI) live in
[`CONTRIBUTING.md`](CONTRIBUTING.md) — this file is the technical "don't get
bitten" guide. Read it before writing or reviewing a kernel.

## ⚠️ A wrong dispatch can freeze the machine

Metal compute dispatches are **non-preemptive**: once a threadgroup starts, the
GPU runs it to completion. An infinite loop in a kernel never yields — the
WindowServer compositor starves of GPU time, the screen locks at the last
frame, and a hard power-cycle is the only recovery.

- **The trap:** reduction-mode kernels compute the simdgroup count as
  `n_simd = lsize / 32` (integer division). A loop strided by `n_simd` —
  `for _t in range(sg, n_kv, n_simd)` — becomes an **infinite GPU loop** when
  `n_simd == 0`, i.e. when the kernel is dispatched with **fewer than 32
  threads per threadgroup**.
- A 4-thread dispatch of a 1024-thread kernel once froze a dev machine for a
  day. The kernel was correct; the *dispatch geometry* was not.
- **Rules for any kernel using `simd_*` / `threadgroup_*`:**
  - TPG **must be a multiple of 32** and **≥ 32** (one full simdgroup).
  - The dispatch geometry is part of the kernel's contract — derive it from
    the kernel's invariants, never from an unrelated "elements" count.
  - GPU correctness tests and `BenchSpec`s set TPG from the kernel side, so
    they're safe. The danger is any *consumer* that turns a caller-supplied
    dimension into a dispatch shape — guard those.

## Dispatch modes — pick the right one

`Context::dispatch_with_grid(kernel, buffers, constexprs, grid_xyz, tg_xyz)`
calls `dispatchThreadgroups`. **`grid_xyz` is in threadgroups, not threads** —
total threads = `grid.{x*y*z} * tg.{x*y*z}`.

- **Grid3D** — one thread per output element, no cross-thread cooperation.
  `program_id::<i>()` lowers to the **thread** index.
  ```rust
  #[kernel] pub fn mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
      let i = program_id::<0>();
      store(out[i], load(a[i]) * load(b[i]));
  }
  // dispatch: grid=[1,1,1] tg=[N,1,1]  (or grid=[ceil(N/TPG),1,1] tg=[TPG,1,1])
  ```
- **Reduction** — uses `simd_*` / `threadgroup_*`. `program_id::<i>()` lowers
  to the **threadgroup** index; threads within a TG cooperate.
  ```rust
  #[kernel] pub fn rms_norm<T>(x: Tensor<T>, /* … */) {
      let row = program_id::<0>();   // = tgid_x, one TG per row
      // … reduce_sum across the TG …
  }
  // dispatch: grid=[rows,1,1] tg=[TPG,1,1]
  ```
- **Wrong:** `grid=[N,1,1] tg=[N,1,1]` for a Grid3D kernel — that is `N²`
  threads, most of them garbage indices. The product must equal exactly `N`.

## Document the dispatch contract: `## DISPATCH INVARIANTS`

A reduction kernel's threadgroup geometry is part of its API, but the kernel
cannot enforce it at runtime. Make the contract explicit in the kernel's `.rs`
doc comment so anyone dispatching it has a source of truth to verify against:

```rust
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: 1024 threads** (32 simdgroups × 32 lanes).
//! - **Grid: 1 threadgroup per q_head** (1D grid, tgid_x = q_head).
//! - **head_dim == 128.** Each lane owns 4 consecutive elements; loads are
//!   unconditional — other head dims read out of bounds / pin the GPU.
```

This is a young convention — most reduction kernels don't have a block yet.
Add one whenever you write or touch a reduction kernel; it costs four lines
and it is the only place the geometry contract is written down.

## Writing kernels

- **Improve the compiler, don't hand-write MSL.** If the DSL can't express a
  pattern, extend the codegen (body parser → IR → MSL emit). Don't bypass it.
- **One generic `<T>` kernel** beats five precision-specific copies — `f32` /
  `f16` / `bf16` all flow through the same `#[kernel] fn`.
- **For non-generic combinatorial variants** (bit widths baked in as `literal`
  constants, fixed group sizes), wrap the **entire `#[kernel] fn` declaration**
  in an outer `macro_rules!` — **never** put a `macro_rules!` call *inside* a
  `#[kernel]` body. The proc-macro does not expand inner declarative macros; it
  drops them and emits an **empty kernel body** that compiles fine and ships
  all-zeros output. The proc-macro now rejects the inner-body shape with a
  compile error — heed it.

  **Right — wrap the whole declaration; the compiler expands it before the
  `#[kernel]` proc-macro runs, so the body parser sees concrete tokens:**
  ```rust
  macro_rules! dequant_gather_kernel {
      ($name:ident, $bits:literal, $subop:literal) => {
          #[kernel]
          pub fn $name<T>(/* params */) {
              let bit_off = d * $bits;   // $bits already substituted
              // …
          }
          inventory::submit! { /* BenchSpec */ };
      };
  }
  dequant_gather_kernel!(dequant_gather_int4, 4u32, "int4");
  ```
  **Wrong — inner macro inside the body → empty MSL (now a compile error):**
  ```rust
  macro_rules! body { ($bits:literal) => { /* … */ }; }
  #[kernel] pub fn dequant_gather_int4<T>(/* … */) { body!(4); }
  ```
  Canonical reference: [`crates/metaltile-std/src/ffai/dequant_gather.rs`](crates/metaltile-std/src/ffai/dequant_gather.rs).
  For hand-unrolled tree reductions, replace `*_step!` macros with a DSL `for`
  loop over the halving strides — identical MSL, survives the proc-macro.

## Empty-body MSL — a silent all-zeros hazard

A kernel can emit MSL with a valid function/loop *header* but no *body*. The
symptom is identical regardless of cause: `xcrun metal` accepts the file, the
kernel dispatches without error, and the output buffer comes back **all
zeros**. Two known root causes:

1. **Macro composition** — inner `macro_rules!` in a `#[kernel]` body (above).
   Now caught at compile time by the proc-macro guard.
2. **Pass ordering** — a codegen pass eliminates a loop body but leaves the
   loop header, or a `Const` a later pass needs is still rolled inside a
   `BinOp`. Result: `for (…) { }` with an empty body.

Invariants for codegen passes (to avoid cause 2):

- A pass that rewrites blocks must walk **both `kernel.body` and every entry
  in `kernel.blocks`** — `kernel.body` is the entry block, not part of the map.
- A pass that removes a loop body must also remove the loop header.
- A pass consuming a `Const` must run after the pass that produces it.

**Detection** — emit all kernels, then scan for empty bodies:
```sh
cargo run --release -p metaltile-cli -- build --emit all -o /tmp/mt-smoke
awk '
  /for \(.*\) \{$/               { f=1; fn=FILENAME; l=FNR; next }
  f && /^[[:space:]]*\}$/        { print fn":"l": empty for-loop body"; f=0; next }
  f                              { f=0 }
  /^kernel void [A-Za-z_0-9]+\(/ { k=1; fn=FILENAME; l=FNR; next }
  k && /^\{$/                    { next }
  k && /^\}$/                    { print fn":"l": empty kernel body"; k=0; next }
  k                              { k=0 }
' /tmp/mt-smoke/Resources/kernels/*.metal
```
Empty output = clean. Any hit = ship-stopper. `xcrun metal` and MSL-snapshot
checks both pass an empty body — only a GPU correctness test catches it.

## Testing — every kernel ships a GPU correctness test

Three layers, no overlap:

| Layer | Catches | Where |
|---|---|---|
| **DSL / codegen unit tests** | Pass correctness, body-parser arms, emit paths; `trybuild` compile-fail fixtures | `cargo test`, `crates/metaltile-codegen` |
| **MSL snapshots** (`insta`) | Codegen output drift — reviewable text diffs in PRs | `crates/metaltile-codegen/tests/msl_snapshots.rs` |
| **GPU correctness** | Numeric disagreement vs a naive CPU oracle, on real Metal | `crates/metaltile-std/tests/<kernel>_gpu_correctness.rs` |

- **Every non-trivial kernel ships a paired `<kernel>_gpu_correctness.rs`** in
  the same commit. It is the *only* layer that catches empty-body MSL and
  numeric bugs — snapshots pin whatever the codegen emits (including wrong
  output), and `xcrun metal` only checks syntax.
- A new DSL primitive or emit path also lands an `insta` snapshot fixture.
- Skeleton: build small inputs → naive CPU reference in f32 → `dispatch_with_grid`
  → compare with `max_abs_diff < 1e-4`. Shared helpers in
  `crates/metaltile-std/tests/common/mod.rs`.

## Perf benching gotchas

- **"Too flat to be physical" means the harness is lying.** A latency that
  doesn't scale with input size is usually measurement overhead drowning the
  kernel cost, not an exceptional kernel. Verify the curve has the shape
  physics predicts before publishing a number.
- **Use `Context::upload_resident()` for inputs constant across iterations** —
  otherwise the bench re-uploads them every iteration and measures PCIe-ish
  traffic, not the kernel.
- **Dummy-dispatch once to warm the GPU clock** — cold DVFS gives the first
  measured shape a ~2× bandwidth deficit.
- Refresh the GB/s numbers cited in a commit with the `#[ignore]`'d perf bench:
  `cargo test --release -p metaltile-std --test <kernel>_gpu_correctness -- --ignored --nocapture`.

## CLI quick reference

`make build` · `make test` · `make bench` · `make fmt-check` · `make clippy`.
For per-kernel work, the `tile` CLI (`cargo run -p metaltile-cli --`):

- `inspect <kernel>` — IR and/or MSL for one kernel.
- `build --emit all -o <dir>` — emit every kernel's MSL (codegen smoke).
- `device` — GPU family, Metal version, native-bf16 flag.
- `snap` / `diff` — save and compare bench regression baselines.
