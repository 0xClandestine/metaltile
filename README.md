<div align="center">
  <h1>MetalTile</h1>

  [![CI][ci-badge]][ci-url]
  [![Apple Silicon][platform-badge]][platform-url]
  [![Rust][rust-badge]][rust-url]
  [![License][license-badge]][license-url]

  [ci-badge]: https://img.shields.io/github/actions/workflow/status/0xClandestine/metaltile/ci.yml?branch=main&style=flat-square
  [ci-url]: https://github.com/0xClandestine/metaltile/actions
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

- [**`#[kernel]`**](crates/metaltile-macros) — write a kernel once in Rust with generics, get `f32`, `f16`, and `bfloat16` Metal variants out automatically.
- [**`tile bench`**](crates/metaltile-cli) — benchmark every kernel against its hand-tuned MLX counterpart on real hardware.
- [**`tile build`**](crates/metaltile-cli) — compile all kernels to MSL; `--emit` writes `.metal` / `.metallib` / Swift wrappers.
- [**`tile inspect`**](crates/metaltile-cli) — print a kernel's IR and generated MSL for debugging.

> ⚠️ Early development — APIs are not yet stable. The core DSL, codegen, and runtime work today; the autotuner and type-level shape algebra are planned.

<table>
<tr>
<th>Rust DSL — what you write</th>
<th>Metal Shading Language — what you get</th>
</tr>
<tr>
<td>

```rust
#[kernel]
pub fn mt_exp<T>(
    a: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}
```

</td>
<td>

```cpp
kernel void mt_exp(
    const device float *a [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint v_idx = tid;
    auto v1 = a[v_idx];
    auto v2 = exp(v1);
    out[v_idx] = v2;
}
```

</td>
</tr>
</table>

MLX's Metal kernels are hand-written MSL — one kernel per type, thread indexing by hand, reduction boilerplate repeated everywhere. MetalTile is a `#[kernel]` proc-macro that lets you write the algorithm once in Rust using generics, and automatically get `f32`, `f16`, and `bfloat16` kernel variants out.

The compiler wires up thread indexing and expands tile primitives like `reduce_sum` and `dot` into the appropriate simdgroup and threadgroup machinery. The output is optimized MSL — the same thing you'd write by hand, minus the toil. Kernels are benchmarked against their MLX equivalents on real hardware; a number of them are faster.

## Installation

```sh
cargo install --path crates/metaltile-cli
```

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## License

<sup>
Licensed under the <a href="LICENSE">Apache License, Version 2.0</a>.
</sup>
