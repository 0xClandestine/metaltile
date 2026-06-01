# MetalTile Documentation

Table of contents for the MetalTile docs. The top-level [`README`](../README.md) is the curated landing page; this index lists every page so you can jump straight to a topic. New contributors (and their agents) should read [Getting started](getting-started.md) → [Developing](developing.md) → [Testing](testing.md) before opening a PR.

## Getting started

- [Getting started](getting-started.md) — toolchain, clone, first build, first kernel.

## Local development

- [Developing](developing.md) — repo layout, the `make` dev loop, branching, commits, debugging, and the **kernel-authoring hazards** (⚠️) that cause silent or catastrophic failure. Required reading before writing a kernel.
- [Testing](testing.md) — the test layers, what runs in CI vs locally, how to write a test, coverage targets, and the gaps in the test infrastructure that let bugs through silently.
- [Publishing](publishing.md) — the `dev` → `main` release flow.

## Reference

- [CLI](cli.md) — the `tile` binary: `bench`, `build`, `emit`, `inspect`, `device`, `snap`, `diff`.
- [Kernel audit](KERNEL_AUDIT.md) — per-op coverage table: which MLX / FFAI kernels are ported, partial, or still missing, with the gaps and open PRs called out.

## Design & planning

- [Bench metrics spec](BENCH_METRICS_SPEC.md) — planned `tile bench` measurement additions (latency, GFLOP/s, roofline/utilization, bottleneck) so kernels can actually be optimized and precisions compared; includes the precision-support roadmap (nvfp4/mxfp4/mxfp8) and M5 Neural Accelerator context.

## See also

- Top-level [`README`](../README.md) — project landing page.
- [`CONTRIBUTING`](../CONTRIBUTING.md) — issue / PR process, agentic-contribution disclosure, code of conduct.
