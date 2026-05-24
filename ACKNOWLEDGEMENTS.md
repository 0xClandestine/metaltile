# Individual Contributors

If you wish to be acknowledged for your contributions, please list your name
with a short description of your contribution(s) below. For example:

- Jane Smith: Added the `foo` and `bar` ops.

MetalTile was developed with contributions from the following individuals:

- clandestine.eth (@0xClandestine) — Founded the project and designed the full stack: #[kernel] proc-macro DSL, body parser, IR, MSL codegen + optimization passes, graph-driven fusion, and cross-kernel calling.
- Tom Turney (@TheTom): one of MetalTile's most prolific kernel authors, with Flash-Attention-2 SDPA, simdgroup-matrix quantised GEMM/GEMV, MoE + gated-delta-network kernels, sampling + logits processors, fused activations, ICB `_record` codegen, FFAI kernel-pack infrastructure, and GPU correctness tests.
- Eric Kryski (@ekryski): Eric drove the bulk of kernel implementations — AURA codec family, SDPA variants, vision/audio front-ends, strided indexing, logits processors, conv2d, RMSNorm, fused activations, and quantized paths.
- Ambisphaeric (@Ambisphaeric): CI hygiene, bench dirty-tree guard, GPU correctness tests, a few kernel tuning wins, project bot, and PM duties.

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
