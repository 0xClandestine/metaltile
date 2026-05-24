# Individual Contributors

If you wish to be acknowledged for your contributions, please list your name
with a short description of your contribution(s) below. For example:

- Jane Smith: Added the `foo` and `bar` ops.

MetalTile was developed with contributions from the following individuals:

- 0xClandestine: Founded the project and designed the core architecture — `#[kernel]` proc-macro DSL and body parser, IR design, MSL codegen pipeline and optimisation passes, graph-driven fusion, cross-kernel calling, derive-based `Op` abstraction, `tile` CLI, CI infrastructure, error handling, and tracing.
- Tom Turney (TheTom): MetalTile's most prolific kernel author — FFAI kernel library, Swift wrapper infrastructure, MoE orchestration kernels, gated delta network kernels, simdgroup-matrix quantised GEMM, ICB `_record` variants, `mt_scalar_fma_chain8`, `mt_swiglu`, logits-processor kernels, and an extensive GPU correctness test suite covering KV cache, SDPA, RoPE, SSM, and sampling.
- Eric Kryski (ekryski): GPU correctness testing infrastructure and kernel completeness audit, CoopTile DSL migration (removing all hand-written `InlineMsl`), AURA codec kernels, eleven FFAI kernel ports, attention/conv/quant performance work, `sdpa_bidirectional` kernels for VLM vision towers, foundational codegen correctness fixes, and CI bench-diff pipeline.
- Ambisphaeric: SDPA decode sliding-window attention with sink-token specialisation (4× at N=16K, 8× at N=32K), batched-Q SDPA decode variants (K=2/4/8/16), bench dirty-tree guard with automatic baseline diff, GEMV threadgroup tuning, and GPU correctness tests for gather, dequant_gemv, and arg_reduce.

<a href="https://github.com/0xClandestine/metaltile/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=0xClandestine/metaltile&anon=0&columns=20&max=100&r=true" />
</a>

# Third-Party Software

MetalTile leverages several third-party libraries. Their repositories and
licenses are listed below.

- **objc2 / objc2-metal / objc2-foundation** — Safe Rust bindings to Apple's Objective-C runtime, Metal GPU API, and Foundation frameworks. Used for all Metal device, command queue, buffer, and pipeline state objects. [MIT](https://github.com/madsmtm/objc2)
- **syn / quote / proc-macro2** — Proc-macro parsing and token-stream generation, used by all MetalTile macros. [MIT / Apache-2.0](https://github.com/dtolnay/syn)
- **clap** — Command-line argument parsing for the `tile` binary. [MIT / Apache-2.0](https://github.com/clap-rs/clap)
- **half** — Host-side `f16` and `bf16` types used in bench and buffer utilities. [MIT / Apache-2.0](https://github.com/starkat99/half-rs)
- **bytemuck** — Safe byte-reinterpretation for buffer uploads and downloads. [MIT / Apache-2.0 / Zlib](https://github.com/Lokathor/bytemuck)
- **inventory** — Compile-time kernel registry powering `tile build` and `#[bench_kernel]`. [MIT / Apache-2.0](https://github.com/dtolnay/inventory)
- **serde / serde_json** — Serialisation for bench snapshots, baseline files, and IR manifests. [MIT / Apache-2.0](https://github.com/serde-rs/serde)
- **tracing / tracing-subscriber** — Structured diagnostics and event logging. [MIT](https://github.com/tokio-rs/tracing)
- **thiserror** — Error type derivation across MetalTile crates. [MIT / Apache-2.0](https://github.com/dtolnay/thiserror)
- **anstyle / anstream** — ANSI terminal colour output in the `tile` CLI. [MIT / Apache-2.0](https://github.com/rust-cli/anstyle)
- **smallvec** — Inline-allocated small vectors used in hot IR paths. [MIT / Apache-2.0](https://github.com/servo/rust-smallvec)
- **rustc-hash** — Fast non-cryptographic hashing for IR maps and pass data structures. [MIT / Apache-2.0](https://github.com/rust-lang/rustc-hash)
