#!/usr/bin/env bash
# rsync this worktree to the GX10 (sm_121 Blackwell) and run a command there.
# Usage: scripts/gx10-sync.sh [cargo args...]   e.g. gx10-sync.sh test -p metaltile-core
set -euo pipefail
HOST=gx10
REMOTE=metaltile-cuda  # relative to remote $HOME
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/"
ssh "$HOST" "mkdir -p $REMOTE"
rsync -az --delete \
  --exclude target/ --exclude .git/ --exclude '*.app' --exclude .cache/ \
  "$LOCAL" "$HOST:$REMOTE/"
ssh "$HOST" "bash -lc '\
  export CUDA_PATH=/usr/local/cuda; \
  export PATH=/usr/local/cuda/bin:\$PATH; \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH; \
  cd $REMOTE && cargo ${*:-build}'"
