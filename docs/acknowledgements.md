# Acknowledgements

MetalTile is built on the work of many people. This document recognises the
open-source projects and individuals whose contributions made it possible.

---

## Open-Source Software

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

---

## Contributors

<!-- TODO: list individual contributors here -->

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
