# Publishing

How a release moves from `dev` to a tagged build on `main`. Maintainer-facing.

## Branch flow

`dev` is the integration branch; `main` holds stable, tagged releases only. Feature work merges into `dev` (see [Developing → branching](developing.md#branching-model)); a release is a single `dev` → `main` promotion.

## Cutting a release

1. **Confirm `dev` is green.** All [CI checks](testing.md#ci-vs-local) passing, and the changelog-worthy PRs for this release are merged.
2. **Open the release PR.** `dev` → `main`, titled `release: v0.x.0`. Merge it with a **merge commit** (not squash) so the history is preserved.
3. **Tag `main`** immediately after the merge:
   ```bash
   git checkout main
   git pull
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. **GitHub release and binary are created automatically.** Pushing the tag triggers `.github/workflows/release.yml`, which builds the `tile` binary on `macos-latest`, packages it as `tile-aarch64-apple-darwin.tar.gz`, and calls `gh release create --generate-notes`. `.github/release.yml` categorises merged PRs into release-notes sections by label — which is why [the conventional-commit prefix is enforced](developing.md#conventional-commits).

5. **Publish to crates.io** — see the section below.

## Publishing to crates.io

Six crates are published; `metaltile-cli` (`publish = false`) is binary-only and stays off the registry.

### One-time setup (first release only)

The workspace path dependencies in the root `Cargo.toml` need a `version` field alongside `path` so crates.io can resolve them. Without it, `cargo publish` rejects the package. Update `[workspace.dependencies]`:

```toml
metaltile         = { path = "crates/metaltile",          version = "0.1.0" }
metaltile-core    = { path = "crates/metaltile-core",     version = "0.1.0" }
metaltile-macros  = { path = "crates/metaltile-macros",   version = "0.1.0" }
metaltile-codegen = { path = "crates/metaltile-codegen",  version = "0.1.0" }
metaltile-runtime = { path = "crates/metaltile-runtime",  version = "0.1.0" }
metaltile-std     = { path = "crates/metaltile-std",      version = "0.1.0" }
```

This change lands in the `release: v0.x.0` PR so the `version` fields are already on `main` when you tag.

You also need a `CARGO_REGISTRY_TOKEN` secret in the GitHub repository settings (Settings → Secrets → Actions) set to a crates.io API token with publish scope.

### Dry-run verification

Before publishing, verify each crate packages cleanly:

```bash
cargo publish --dry-run -p metaltile-macros
cargo publish --dry-run -p metaltile-core
cargo publish --dry-run -p metaltile-codegen
cargo publish --dry-run -p metaltile-runtime
cargo publish --dry-run -p metaltile
cargo publish --dry-run -p metaltile-std
```

### Publish order

Crates must be published in dependency order. crates.io index propagation takes ~30 seconds per crate — wait between each step or the next publish will fail because the dependency isn't indexed yet.

```bash
cargo publish -p metaltile-macros
sleep 30
cargo publish -p metaltile-core
sleep 30
cargo publish -p metaltile-codegen
sleep 30
cargo publish -p metaltile-runtime
sleep 30
cargo publish -p metaltile
sleep 30
cargo publish -p metaltile-std
```

### Bumping the version for subsequent releases

Update `version` in `[workspace.package]` and the six `version` fields in `[workspace.dependencies]` together in the release PR. All crate `Cargo.toml` files inherit from the workspace, so there is nothing else to change.

## What we don't do (yet)

- **No MSRV policy.** The workspace tracks Rust nightly for `edition = 2024` and unstable `rustfmt` features. An MSRV will be declared once the project stabilises on a stable compiler.
- **No backport branches.** All fixes land on `dev` and ride the next release. A critical hotfix can be handled by cutting a `v0.x` branch retroactively if it ever becomes necessary.
- **No automated crates.io publish.** Publishing is a manual step until the release cadence is stable enough to trust a fully automated pipeline.
