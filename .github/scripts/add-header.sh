#!/usr/bin/env bash
set -euo pipefail

HEADER_LINE1="//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric"
HEADER_LINE2="//! SPDX-License-Identifier: Apache-2.0"

# Count how many files already have the header
already_had=0
updated=0

while IFS= read -r -d '' file; do
    # Check if the header already exists (first two lines)
    first=$(head -1 "$file" 2>/dev/null || true)
    second=$(sed -n '2p' "$file" 2>/dev/null || true)
    if [[ "$first" == "$HEADER_LINE1" && "$second" == "$HEADER_LINE2" ]]; then
        echo "  [SKIP] $file (already has header)"
        already_had=$((already_had + 1))
        continue
    fi

    # Prepend header — use a temp file for safety
    {
        echo "$HEADER_LINE1"
        echo "$HEADER_LINE2"
        cat "$file"
    } > "${file}.tmp"
    mv "${file}.tmp" "$file"
    echo "  [ADD]  $file"
    updated=$((updated + 1))
done < <(find . -name '*.rs' -type f -not -path './target/*' -print0)

echo ""
echo "Done. $updated file(s) updated, $already_had already had the header."