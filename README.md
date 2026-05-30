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
curl -fsSL https://github.com/0xClandestine/metaltile/releases/latest/download/install.sh | sh
```

Run `tile update` at any time to upgrade to the latest release.

For contributors building from source, see [Getting Started](docs/getting-started.md).

## Getting Started

**1. Write a kernel.** Annotate a generic Rust function with `#[kernel(bench(...))]` — MetalTile generates `f32`, `f16`, and `bfloat16` Metal variants from a single definition and registers it against its MLX reference:

<table>
<tr>
<th>Rust DSL — what you write</th>
<th>Metal Shading Language — what you get</th>
</tr>
<tr>
<td>

```rust
#[kernel(
    bench(
        op    = "unary",
        subop = "exp",
        class = Unary,
        input = Signed,
        tol   = 1e-4,
        mlx   = "v_Exp{tn}{tn}",
        metal_file = "unary.metal",
    )
)]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
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

**2. Install the CLI and run.**

```sh
cargo install --path crates/metaltile-cli
tile bench --filter exp
```

```
tile bench · Apple M4 Max
  logsumexp (logsumexp)
  Shape                                  │  Ref(GB/s) │  MT(GB/s) │   MT% │  ok
  ────────────────────────────────────────────────────────────────────────────────
  B=1024 N=4096 f32                      │      737.6 │     844.3 │  114% │   ✓
  B=1024 N=4096 f16                      │      369.5 │     504.7 │  137% │   ✓
  B=1024 N=4096 bf16                     │      366.8 │     508.5 │  139% │   ✓
```

Read the [docs](docs/) to learn more.

## Architecture

```mermaid
flowchart LR
    subgraph User["Your crate"]
        K["#[kernel]<br/>fn mt_exp&lt;T&gt;(..)"]
        B["#[bench]<br/>fn exp_bench(dt)"]
        T["#[test_kernel]<br/>fn exp_test(dt)"]
    end

    K --> IR["MetalTile IR<br/>(Op variants)"]
    IR --> Passes["codegen passes<br/>vectorize · unroll · DCE · ..."]
    Passes --> MSL[".metal source"]
    MSL --> Metallib["kernels.metallib<br/>(xcrun metal -c)"]

    subgraph CLI["tile CLI"]
        Build["tile build"]
        Bench["tile bench"]
        Test["tile test"]
        Inspect["tile inspect"]
    end

    B -.registers.-> Bench
    T -.registers.-> Test
    K -.discovered by.-> Build
    K -.discovered by.-> Inspect
    Build --> MSL
    Bench --> Runner["__tile_runner<br/>(subprocess)"]
    Test --> Runner
    Runner --> Metallib
```

`#[kernel]` lowers your DSL function to IR; the codegen passes optimise it; MSL emit produces a `.metal` source that `xcrun metal` compiles to a `metallib`. `#[bench]` / `#[test_kernel]` are optional annotations on the same function that register a setup callback the runner uses to dispatch the kernel and measure it (or diff against a CPU oracle).

## CLI reference

| Command | What it does |
|---|---|
| `tile build` | Compile every `#[kernel]` in the workspace to MSL and (optionally) a `metallib`. |
| `tile bench` | Run every `#[bench]`, report MetalTile GB/s vs the MLX reference + correctness. |
| `tile test` | Run every `#[test_kernel]` against its CPU oracle within tolerance. |
| `tile inspect` | Dump IR / per-pass IR / MSL for one kernel. |
| `tile device` | Print GPU device info and supported feature flags. |
| `tile snap` | Save bench results as a regression baseline. |
| `tile diff` | Compare bench results to a saved baseline. |
| `tile update` | Install the latest release (or build from a PR / commit). |

See [`docs/cli.md`](docs/cli.md) for the full flag surface.

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## Acknowledgements

MetalTile's benchmark suite and kernel library stand on the shoulders of the MLX ecosystem. A large
portion of the `metaltile-std` kernels are ports or faithful re-implementations of kernels from the following projects:

- [**ml-explore/mlx**](https://github.com/ml-explore/mlx) — primary source for reference kernels.
- [**ekryski/mlx**](https://github.com/ekryski/mlx) (`alpha`) — FFAI extensions: gated-delta, SSM replay, AURA codec.
- [**ml-explore/mlx-lm**](https://github.com/ml-explore/mlx-lm) — reference for GatedDeltaNet step semantics.

We are grateful to the MLX team at Apple and the broader MLX community, this wouldn't have been possible without you.

See [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) for the full list of individual contributors and third-party software.

## License

<sup>
Licensed under the <a href="LICENSE">Apache License, Version 2.0</a>.
</sup>
