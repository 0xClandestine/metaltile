#!/usr/bin/env bash
# check_bench_names.sh
# Verify that every #[bench(name = "...")] in the kernel crates uses one of the
# CI-sharded group prefixes (ffai/ or mlx/).  Kernels that don't will be silently
# skipped when CI runs `tile bench --match-group ffai` and `--match-group mlx`.
#
# Usage: ./check_bench_names.sh [repo-root]

set -uo pipefail   # note: no -e; we handle grep exit codes manually

ROOT="${1:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
CRATES_DIR="$ROOT/crates"

KNOWN_GROUPS=(ffai mlx)
GROUP_PATTERN="^($(IFS='|'; echo "${KNOWN_GROUPS[*]}"))"

bad=0
declare -a bad_lines=()

while IFS= read -r match; do
    file="${match%%:*}"
    rest="${match#*:}"
    lineno="${rest%%:*}"
    content="${rest#*:}"

    # Extract the string literal after name = "  (BSD sed: no \s)
    # Suppress grep exit 1 (no match) with || true.
    name=$(printf '%s' "$content" \
        | grep -oE 'name[[:space:]]*=[[:space:]]*"[^"]+"' \
        | sed 's/name[[:space:]]*=[[:space:]]*"//; s/"//' \
        || true)

    # Skip lines with no string literal (e.g. macro vars: name = $name)
    [[ -z "$name" ]] && continue

    if ! printf '%s' "$name" | grep -qE "$GROUP_PATTERN"; then
        bad_lines+=("  $file:$lineno  →  \"$name\"")
        ((bad++)) || true
    fi
done < <(grep -rn '#\[bench(' "$CRATES_DIR" \
            --include='*.rs' \
            --exclude-dir='metaltile-macros')

if [[ ${#bad_lines[@]} -eq 0 ]]; then
    echo "✓ All #[bench] names use a known CI group prefix (${KNOWN_GROUPS[*]})."
    exit 0
fi

echo "✗ ${bad} #[bench] name(s) will be SKIPPED by CI sharding."
echo "  CI runs: tile bench --match-group ffai  and  tile bench --match-group mlx"
echo "  Names that don't start with one of: ${KNOWN_GROUPS[*]}"
echo
for l in "${bad_lines[@]}"; do
    echo "$l"
done
echo
echo "Fix: rename the kernel to start with one of the known group prefixes,"
echo "     or add its group to the KNOWN_GROUPS list in this script and the"
echo "     CI matrix in .github/workflows/tile.yml."
exit 1
