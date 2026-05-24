<div align="center">
  <h1>MetalTile</h1>

  [![Apple Silicon][platform-badge]][platform-url]
  [![Rust][rust-badge]][rust-url]
  [![License][license-badge]][license-url]

  [platform-badge]: https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple&style=flat-square
  [platform-url]: https://developer.apple.com/metal/
  [rust-badge]: https://img.shields.io/badge/language-Rust-orange?logo=rust&style=flat-square
  [rust-url]: https://www.rust-lang.org/
  [license-badge]: https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square
  [license-url]: LICENSE

  **[Docs](docs/)** | **[Baselines](baselines/)** | **[Contributing](CONTRIBUTING.md)**

</div>

---

A Rust-embedded DSL for writing Apple Metal GPU kernels. Write tile-level algorithms in Rust, get optimized Metal Shading Language out — verified against, and frequently faster than, hand-tuned MLX.

## Installation

```sh
cargo install --path crates/metaltile-cli
```

## Getting Started

**1. Write a kernel.** Annotate a generic Rust function with `#[kernel]` and `#[bench_kernel]` — MetalTile generates `f32`, `f16`, and `bfloat16` Metal variants from a single definition and registers it against its MLX reference:

```rust
#[bench_kernel(
    op    = "unary",
    subop = "exp",
    class = Unary,
    input = Signed,
    tol   = 1e-4,
    mlx   = "v_Exp{tn}{tn}",
    metal_file = "unary.metal",
)]
#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}
```

**2. Install the CLI and run.**

```sh
cargo install --path crates/metaltile-cli
tile bench --filter exp
```

Read the [docs](docs/) to learn more.

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## License

<sup>
Licensed under the <a href="LICENSE">Apache License, Version 2.0</a>.
</sup>
