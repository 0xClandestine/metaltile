#!/usr/bin/env sh
# install.sh — download and install the latest tile release binary.
#
# Usage:
#   curl -fsSL https://github.com/0xClandestine/metaltile/releases/latest/download/install.sh | sh
#
# By default installs to /usr/local/bin if writable, else ~/.local/bin.
# Override with TILE_INSTALL_DIR:
#   TILE_INSTALL_DIR=~/.cargo/bin curl -fsSL ... | sh
set -eu

REPO="0xClandestine/metaltile"
BINARY="tile"
ASSET="tile-aarch64-apple-darwin.tar.gz"

# ── Detect install directory ──────────────────────────────────────────────────

if [ -n "${TILE_INSTALL_DIR:-}" ]; then
    INSTALL_DIR="$TILE_INSTALL_DIR"
elif [ -w /usr/local/bin ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

# ── Resolve latest release tag ────────────────────────────────────────────────

echo "Fetching latest release..."
TAG=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    -H "Accept: application/vnd.github+json" \
    | grep '"tag_name"' \
    | head -1 \
    | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')

if [ -z "$TAG" ]; then
    echo "error: could not determine latest release tag." >&2
    exit 1
fi

echo "Installing tile ${TAG}..."

# ── Download and extract ──────────────────────────────────────────────────────

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
curl -fsSL --progress-bar "$URL" -o "$TMP/$ASSET"
tar xzf "$TMP/$ASSET" -C "$TMP"

# ── Install ───────────────────────────────────────────────────────────────────

chmod +x "$TMP/$BINARY"

if mv "$TMP/$BINARY" "$INSTALL_DIR/$BINARY" 2>/dev/null; then
    :
elif command -v sudo >/dev/null 2>&1; then
    echo "Requesting sudo to write to $INSTALL_DIR..."
    sudo mv "$TMP/$BINARY" "$INSTALL_DIR/$BINARY"
else
    echo "error: cannot write to $INSTALL_DIR — set TILE_INSTALL_DIR to a writable path." >&2
    exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo "Installed tile ${TAG} to ${INSTALL_DIR}/${BINARY}"

# Warn if the install dir isn't on PATH.
case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) ;;
    *)
        echo ""
        echo "note: ${INSTALL_DIR} is not in your PATH."
        echo "      Add the following to your shell profile:"
        echo "        export PATH=\"${INSTALL_DIR}:\$PATH\""
        ;;
esac
