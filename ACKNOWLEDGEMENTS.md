# Acknowledgements

MetalTile is built on the work of many people. This document recognises the
open-source projects and individuals whose contributions made it possible.

---

## Contributors

### 0xClandestine

Founder and primary architect of MetalTile. Designed and built the
project from the ground up: the `#[kernel]` proc-macro DSL and body
parser, the IR (`Op` variants, shape algebra, DType system), the MSL
codegen pipeline and its optimisation passes, graph-driven fusion
(Patterns 1, 3, and 6), cross-kernel calling via `KernelCallArg` and
`KernelInlinePass`, the derive-based `Op` abstraction, and the full
`tile` CLI (`bench`, `build`, `inspect`, `snap`, `diff`). Also owns
the CI infrastructure, error handling strategy, tracing, and project
architecture.

### Tom Turney (TheTom)

MetalTile's most prolific kernel author, responsible for the majority
of the FFAI kernel library. Introduced the `metaltile-emit` crate and
Swift wrapper infrastructure, MoE orchestration kernels (`mt_moe_permute`,
`mt_moe_router_topk`, `mt_moe_unpermute`, `mt_moe_expert_indexed`),
gated delta network kernels (decode, chunked-prefill, chunked-WY, prep
variants), simdgroup-matrix quantised GEMM (`mt_qmm_mma`, `mt_qmm_bm2`,
`mt_qmm_bm4`), ICB `_record` kernel variants, `mt_scalar_fma_chain8`,
`mt_swiglu`, logits-processor kernels (temperature, repetition penalty,
top-K), and an extensive GPU correctness test suite covering KV cache,
SDPA, RoPE, SSM, and sampling.

### Eric Kryski (ekryski)

Led the GPU correctness testing infrastructure and kernel completeness
audit. Ported eleven FFAI kernels from `ekryski/mlx@alpha`, introduced
the AURA codec kernel set, and drove the CoopTile DSL migration that
replaced all hand-written `InlineMsl` blocks with composable primitives.
Delivered major performance work across attention head dimensions, MMA
convolution, FFT non-power-of-2, int4/int8 quantised paths, and
`sdpa_bidirectional` kernels for VLM vision towers. Also fixed
foundational codegen correctness bugs (SSA preservation, loop-body
cloning, FusedElementwise detection) and owns the CI bench-diff
pipeline.

### Ambisphaeric

Delivered high-impact performance work on the SDPA decode path:
sliding-window attention with sink-token specialisation (4× throughput
at N=16K, 8× at N=32K) and batched-Q SDPA decode variants (K=2/4/8/16).
Also contributed GEMV threadgroup-per-group tuning, the bench dirty-tree
guard with automatic baseline diff against the target branch, and GPU
correctness tests for gather, dequant_gemv, and arg_reduce.

---

## Inspiration and Prior Art

<!-- TODO: acknowledge projects, papers, or people that directly inspired
     MetalTile's design — e.g. Triton, MLX, IREE, or specific kernel
     techniques. Fill this in as a team. -->

---

## Special Thanks

<!-- TODO: anyone the team wants to call out individually — advisors,
     early testers, reviewers, or community members who shaped the
     project in a meaningful way. -->

---

## Open-Source Software

<details>
<summary>Third-party dependencies</summary>

### objc2 / objc2-metal / objc2-foundation

MetalTile's GPU runtime uses **objc2** and its companion crates
(`objc2-metal`, `objc2-foundation`) for safe, idiomatic Rust bindings to
Apple's Objective-C runtime, Metal GPU API, and Foundation frameworks.

- Repository: <https://github.com/madsmtm/objc2>
- License: MIT

> These crates provide the Metal device, command queue, buffer, and pipeline
> state objects that back every kernel dispatch in MetalTile.

---

### syn / quote / proc-macro2

The `#[kernel]`, `#[constexpr]`, `#[scalar]`, `#[strided]`, `shape!`, and
`tile!` proc macros are built on the **syn** parser, **quote** token-stream
builder, and **proc-macro2** bridge maintained by David Tolnay.

- syn: <https://github.com/dtolnay/syn> — MIT / Apache-2.0
- quote: <https://github.com/dtolnay/quote> — MIT / Apache-2.0
- proc-macro2: <https://github.com/dtolnay/proc-macro2> — MIT / Apache-2.0

---

### clap

The `tile` CLI is built with **clap**, the command-line argument parser for
Rust maintained by Ed Page and the clap contributors.

- Repository: <https://github.com/clap-rs/clap>
- License: MIT / Apache-2.0

---

### half

16-bit floating-point (`f16`, `bf16`) host-side types are provided by the
**half** crate, maintained by Kathryn Long.

- Repository: <https://github.com/starkat99/half-rs>
- License: MIT / Apache-2.0

---

### bytemuck

Safe byte-reinterpretation for buffer uploads and downloads is provided by
**bytemuck**, maintained by Lokathor.

- Repository: <https://github.com/Lokathor/bytemuck>
- License: MIT / Apache-2.0 / Zlib

---

### inventory

The compile-time kernel registry that powers `tile build` and `#[bench_kernel]`
is built on **inventory**, maintained by David Tolnay.

- Repository: <https://github.com/dtolnay/inventory>
- License: MIT / Apache-2.0

---

### serde / serde_json

Benchmark snapshots, baseline files, and IR manifests are serialised with
**serde** and **serde_json**, maintained by David Tolnay and the Serde
contributors.

- serde: <https://github.com/serde-rs/serde> — MIT / Apache-2.0
- serde_json: <https://github.com/serde-rs/json> — MIT / Apache-2.0

---

### tracing / tracing-subscriber

Structured diagnostics are emitted via the **tracing** ecosystem, maintained
by Tokio contributors.

- Repository: <https://github.com/tokio-rs/tracing>
- License: MIT

---

### thiserror

Error type derivation across MetalTile crates uses **thiserror**, maintained
by David Tolnay.

- Repository: <https://github.com/dtolnay/thiserror>
- License: MIT / Apache-2.0

---

### anstyle / anstream

Terminal colour output in the `tile` CLI uses **anstyle** and **anstream**,
maintained by the clap contributors.

- Repository: <https://github.com/rust-cli/anstyle>
- License: MIT / Apache-2.0

---

### smallvec

Inline-allocated small vectors used in hot IR paths are provided by
**smallvec**, originally developed by the Servo project.

- Repository: <https://github.com/servo/rust-smallvec>
- License: MIT / Apache-2.0

---

### rustc-hash

Fast, non-cryptographic hashing for IR maps and pass data structures uses
**rustc-hash**, extracted from the Rust compiler.

- Repository: <https://github.com/rust-lang/rustc-hash>
- License: MIT / Apache-2.0

</details>
