# MetalTile Documentation

Table of contents for the MetalTile docs. The top-level [`README`](../README.md) is the curated landing page; this index lists every page so you can jump straight to a topic. New contributors (and their agents) should read [Getting started](getting-started.md) → [Developing](developing.md) → [Testing](testing.md) before opening a PR.

## Getting started

- [Getting started](getting-started.md) — toolchain, clone, first build, first kernel.

## Local development

- [Developing](developing.md) — repo layout, the `make` dev loop, branching, commits, debugging, and the **kernel-authoring hazards** (⚠️) that cause silent or catastrophic failure. Required reading before writing a kernel.
- [Testing](testing.md) — the test layers, what runs in CI vs locally, how to write a test, coverage targets, and the gaps in the test infrastructure that let bugs through silently.
- [Publishing](publishing.md) — the `dev` → `main` release flow.

## Reference

- [Architecture](../specs/ARCHITECTURE.md) — how a `#[kernel]` becomes a compiled shader, and how the bench runner, test runner, and kernel profiling work end-to-end (current in-process vs planned-subprocess execution model).
- [CLI](cli.md) — the `tile` binary: `bench`, `build`, `emit`, `inspect`, `device`, `snap`, `diff`.
- [Kernel audit](../specs/KERNEL_AUDIT.md) — per-op coverage table: which MLX / FFAI kernels are ported, partial, or still missing, with the gaps and open PRs called out.

## Design & planning

Long-form specs and design docs live in [`specs/`](../specs/).

- [Bench metrics spec](../specs/BENCH_METRICS_SPEC.md) — planned `tile bench` measurement additions (latency, GFLOP/s, roofline/utilization, bottleneck) so kernels can actually be optimized and precisions compared; includes the precision-support roadmap (nvfp4/mxfp4/mxfp8) and M5 Neural Accelerator context.
- [Kernel organization spec](../specs/KERNEL_ORGANIZATION.md) — **proposal** for `metaltile-std`'s kernel source layout: file granularity, the canonical per-kernel file shape, and the family-by-family migration plan (coordinates with the CLI subprocess-rewrite spec, which governs the crate/CLI side).
- [Toolchain design](../specs/TOOLCHAIN_DESIGN.md) — the `#[kernel]` / `#[bench]` / `#[test_kernel]` macro surface and how the IR lowers to MSL.
- [Proposed optimizations](../specs/PROPOSED_OPTIMIZATIONS.md) — hot-path patterns that need codegen-layer support, with rationale and implementation sketches.

## Backend ports

Codegen-backend specs for taking the DSL beyond Metal. Each is a delta on the CUDA spec, which defines the shared backend seam.

- [CUDA / NVIDIA](../specs/CUDA_BACKEND_SPEC.md) — the backend-seam design (DSL → C++/PTX) the other ports build on.
- [AMD / ROCm](../specs/AMD_BACKEND_SPEC.md) — HIP/ROCm delta on the CUDA seam.
- [Vulkan / SPIR-V](../specs/VULKAN_BACKEND_SPEC.md) — portable compute via SPIR-V.
- [Apple ANE](../specs/ANE_BACKEND_SPEC.md) — the Neural Engine: why it's not directly programmable and what a port would entail.

## See also

- Top-level [`README`](../README.md) — project landing page.
- [`CONTRIBUTING`](../CONTRIBUTING.md) — issue / PR process, agentic-contribution disclosure, code of conduct.
