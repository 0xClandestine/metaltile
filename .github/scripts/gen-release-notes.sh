#!/usr/bin/env bash
# gen-release-notes.sh <tag> [<since-ref>]
#
# Generates release notes from conventional commits between <since-ref> (or
# the initial commit) and <tag>, then updates the GitHub release for <tag>.
#
# Usage:
#   .github/scripts/gen-release-notes.sh v0.1.0
#   .github/scripts/gen-release-notes.sh v0.2.0 v0.1.0
set -euo pipefail

TAG="${1:?usage: gen-release-notes.sh <tag> [<since-ref>]}"
SINCE="${2:-$(git rev-list --max-parents=0 HEAD)}"
REPO="${REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"

# ── Collect commits ───────────────────────────────────────────────────────────

declare -a feats fixes perfs refactors tests docs ci chore other

while IFS= read -r line; do
    sha="${line%% *}"
    msg="${line#* }"

    if   [[ "$msg" =~ ^feat ]];     then feats+=("$msg")
    elif [[ "$msg" =~ ^fix ]];      then fixes+=("$msg")
    elif [[ "$msg" =~ ^perf ]];     then perfs+=("$msg")
    elif [[ "$msg" =~ ^refactor ]]; then refactors+=("$msg")
    elif [[ "$msg" =~ ^test ]];     then tests+=("$msg")
    elif [[ "$msg" =~ ^docs? ]];    then docs+=("$msg")
    elif [[ "$msg" =~ ^ci ]];       then ci+=("$msg")
    elif [[ "$msg" =~ ^(chore|style|data|build) ]]; then chore+=("$msg")
    elif [[ "$msg" =~ ^release ]];  then : # skip release meta-commits
    else other+=("$msg")
    fi
done < <(git log "${SINCE}..${TAG}" --oneline --no-merges)

# ── Render section ────────────────────────────────────────────────────────────

render_section() {
    local title="$1"; shift
    local -a items=("$@")
    [[ ${#items[@]} -eq 0 ]] && return
    echo "### $title"
    echo
    for item in "${items[@]}"; do
        echo "- $item"
    done
    echo
}

# ── Build notes ───────────────────────────────────────────────────────────────

NOTES=$(
    render_section "✨ Features"       "${feats[@]+"${feats[@]}"}"
    render_section "🚀 Performance"    "${perfs[@]+"${perfs[@]}"}"
    render_section "🐛 Bug Fixes"      "${fixes[@]+"${fixes[@]}"}"
    render_section "♻️  Refactors"      "${refactors[@]+"${refactors[@]}"}"
    render_section "🧪 Tests"          "${tests[@]+"${tests[@]}"}"
    render_section "📚 Documentation"  "${docs[@]+"${docs[@]}"}"
    render_section "🔧 Other Changes"  "${ci[@]+"${ci[@]}"}" "${chore[@]+"${chore[@]}"}" "${other[@]+"${other[@]}"}"

    echo "## Contributors"
    echo
    git log "${SINCE}..${TAG}" --no-merges --format="%aN" \
        | sort -u \
        | while read -r name; do
            login=$(gh api graphql \
                -f query="{search(query:\"$name in:name type:user\",type:USER,first:1){nodes{...on User{login}}}}" \
                --jq '.data.search.nodes[0].login // empty' 2>/dev/null || true)
            if [[ -n "$login" ]]; then
                echo "- @$login"
            else
                echo "- $name"
            fi
          done
    echo
    echo "**Full Changelog**: https://github.com/${REPO}/commits/${TAG}"
)

echo "$NOTES"
echo
echo "--- Updating release ${TAG} on ${REPO} ---"
gh release edit "$TAG" --repo "$REPO" --notes "$NOTES"
echo "Done."
