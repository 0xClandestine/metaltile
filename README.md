<p align="center">
  <h1 align="center">MetalTile</h1>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

<p align="center">A Rust-embedded DSL for writing Apple Metal GPU kernels. Write tile-level algorithms in Rust, get optimized Metal Shading Language out — verified against, and frequently faster than, hand-tuned MLX.</p>

> ⚠️ Early development — APIs are not yet stable. The core DSL, codegen, and runtime work today; the autotuner and type-level shape algebra are planned.

<pre>
crates
├── <a href="crates/metaltile/README.md">metaltile</a>
├── <a href="crates/metaltile-cli/README.md">metaltile-cli</a>
├── <a href="crates/metaltile-codegen/README.md">metaltile-codegen</a>
├── <a href="crates/metaltile-core/README.md">metaltile-core</a>
├── <a href="crates/metaltile-macros/README.md">metaltile-macros</a>
├── <a href="crates/metaltile-runtime/README.md">metaltile-runtime</a>
└── <a href="crates/metaltile-std/README.md">metaltile-std</a>
</pre>

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

  MLX's Metal kernels are hand-written MSL — one kernel per type, thread
  indexing by hand, reduction boilerplate repeated everywhere. MetalTile is a
  #[kernel] proc-macro that lets you write the algorithm once in Rust using
  generics, and automatically get f32, f16, and bfloat16 kernel variants out.

  The compiler wires up thread indexing and expands tile primitives like
  reduce_sum and dot into the appropriate simdgroup and threadgroup machinery.
   The output is optimized MSL — the same thing you'd write by hand, minus the
   toil. Kernels are benchmarked against their MLX equivalents on real
  hardware; a number of them are faster.

## Benchmarks

`tile bench` dispatches every MetalTile kernel and its MLX Metal reference on identical buffers, then reports throughput and a numerical-equivalence check. Run the whole suite, or narrow with `--filter`:

```sh
tile bench                   # full suite
tile bench --filter softmax  # one op
```

A sample of what to expect: `mt_rms_norm_small` runs at **354% of MLX's hand-tuned `rms` kernel** on an Apple M4 Max (`B=1024 N=64`, f32). Full cross-hardware results live in [`baselines/`](baselines/) — committed snapshots per chip, refreshed as new hardware is benched. CI diffs every PR against the matching baseline.

See [`docs/cli.md`](docs/cli.md) for `-v` / `-vv` profiling, JSON output, and the `snap` / `diff` regression workflow.

## Contributing

Contributions — including AI-assisted ones — are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
