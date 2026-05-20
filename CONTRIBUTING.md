# Contributing to MetalTile

Thanks for your interest. MetalTile is a focused project — these guidelines keep contributions coherent and the reviewer queue sane.

**TL;DR:**
- Read the [docs](docs/README.md) before you write code — especially the kernel-authoring hazards in [`developing.md`](docs/developing.md).
- One logical change per PR. Tests and docs land with the code that needs them.
- AI-assisted contributions are welcome — disclose how you used AI.

<!--
─────────────────────────────────────────────────────────────────────────────
AGENTS — read this before touching code. This block is intentionally an HTML comment: invisible in the rendered page, but you are reading the source, so it is for you.

1. docs/developing.md "Kernel-authoring hazards" is MANDATORY reading before you write or modify a kernel. One hazard is not a style nit: a wrong dispatch geometry hard-freezes the machine — Metal dispatch is non-preemptive, an infinite loop in a kernel never yields, and the only recovery is a power-cycle. Treat the warning sections as load-bearing.

2. Every non-trivial kernel ships a GPU correctness test IN THE SAME COMMIT. Empty-body MSL (from an inner macro_rules!, or a codegen pass dropping a loop body) passes `xcrun metal`, passes the smoke build, and passes MSL snapshots — yet ships all-zeros output. ONLY a GPU correctness test against a naive CPU reference catches it. docs/testing.md "Gaps" explains.

3. Before claiming a task done: `make fmt-check && make clippy && make test` all green. If you changed bench-relevant code, run `make bench` and paste the rows in the PR body.

4. Never loosen a test tolerance to make a test pass. If kernel and reference disagree, find out which side is wrong.

5. Keep the diff scoped to one logical change. If it touches three unrelated things, that is three PRs.
─────────────────────────────────────────────────────────────────────────────
-->

## Before you start — read the docs

The [`docs/`](docs/README.md) tree is the real reference. At minimum:

- [Getting started](docs/getting-started.md) — toolchain, build, first kernel.
- [Developing](docs/developing.md) — repo layout, dev loop, branching, commits, and the **⚠️ kernel-authoring hazards**. Read the hazard sections before writing a kernel — one of them is "a wrong dispatch can freeze your machine."
- [Testing](docs/testing.md) — the test layers, what runs in CI, how to write a test, and the gaps that let bugs through silently.
- [CLI](docs/cli.md) and [Publishing](docs/publishing.md) for the `tile` binary and the release flow.

## What a good PR looks like

- **Scoped tightly.** One logical change per PR.
- **Tests for behavior changes, docs for user-visible changes.** A new or modified kernel lands with its GPU correctness test in the *same commit*; a new emit path lands with an MSL snapshot fixture. See [`testing.md`](docs/testing.md).
- **Conventional-commit PR title** (`feat:`, `fix:`, `perf:`, `docs:`, …) — see [`developing.md`](docs/developing.md#conventional-commits).
- **Green CI** before requesting review.
- For anything beyond a trivial fix, **open an issue first** to align on scope — a short exchange there saves rework on the PR.

### PR checklist

- [ ] Title uses a conventional-commit prefix.
- [ ] `make clippy` clean (`-D warnings`).
- [ ] `make test` passes.
- [ ] `make fmt-check` passes.
- [ ] `make typos` passes.
- [ ] New / changed kernels have a GPU correctness test in the same commit.
- [ ] PR body explains **what** and **why**; links issues with `#<num>`.
- [ ] If bench numbers changed, relevant rows pasted in the PR body.

## Agentic contributions

AI-assisted contributions are welcome — and often produce tighter descriptions and better test coverage than hand-written ones. Two rules:

1. **Disclose.** Note in the PR body how AI was used (research, ideation, implementation, testing). This is transparency, not gatekeeping.
2. **Curate before opening.** An AI-assisted PR should read no differently from a hand-written one: tight description, scoped diff, tests, docs. Don't paste raw assistant output — if the diff sprawls or the description rambles, tighten it first.

## Code of conduct

The usual: no spam, no harassment, no back-seat-driving on closed issues. Maintainer discretion on what counts.

## License

By contributing you agree your contribution is licensed under [Apache-2.0](LICENSE).
