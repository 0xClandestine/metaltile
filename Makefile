# MetalTile — Makefile
#
# Common dev-loop targets. See TOOLCHAIN_PLAN.md for the phased
# build-out and scripts/ for the longer-form scripts.

.DEFAULT_GOAL := help

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT := $(shell pwd)

# ─── Help ─────────────────────────────────────────────────────────────
.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────
.PHONY: setup
setup: ## one-time dev environment setup (toolchains, deps, first build)
	./scripts/setup-dev.sh

.PHONY: hooks
hooks: ## install git hooks (pre-commit, commit-msg, pre-push)
	./.github/scripts/install-hooks.sh

.PHONY: hooks-uninstall
hooks-uninstall: ## remove git hook installation
	git config --unset core.hooksPath || true
	@echo "✓ Uninstalled hooks (core.hooksPath cleared)"

# ─── Build ────────────────────────────────────────────────────────────
.PHONY: build
build: ## cargo build (debug)
	cargo build --workspace

.PHONY: build-release
build-release: ## cargo build (release)
	cargo build --workspace --release

# ─── Test ─────────────────────────────────────────────────────────────
.PHONY: test
test: ## cargo test --workspace
	cargo test --workspace

.PHONY: coverage
coverage: ## test coverage report (requires cargo-llvm-cov)
	./scripts/coverage.sh

# ─── Lint / format ────────────────────────────────────────────────────
.PHONY: clippy
clippy: ## run clippy on all targets with -D warnings
	cargo clippy --all-targets --all-features -- -D warnings

.PHONY: fmt
fmt: ## run rustfmt on all crates
	cargo fmt --all

.PHONY: fmt-check
fmt-check: ## check formatting without modifying files
	cargo fmt --all -- --check

.PHONY: typos
typos: ## run typos checker (same config CI uses)
	typos --config .github/configs/typos-cli.toml

# ─── Benchmark ────────────────────────────────────────────────────────
.PHONY: bench
bench: build-release ## run benchmark suite vs MLX (requires macOS + Metal)
	cargo run --release -p metaltile-cli -- bench

# ─── Clean ────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## remove target/ and build artifacts
	cargo clean
