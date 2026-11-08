#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."
OUT=".cache/ruff-preview"
mkdir -p "$OUT"

uv run ruff format --diff >"$OUT/1-format.diff" 2>/dev/null || true
uv run ruff check --diff >"$OUT/2-lint-safe-fixes.diff" 2>/dev/null || true
uv run ruff check --diff --unsafe-fixes >"$OUT/3-lint-safe-and-unsafe-fixes.diff" 2>/dev/null || true

summarize() {
    local file="$1" label="$2"
    printf '%-34s files: %3d   +%-5d -%-5d  %s\n' "$label" \
        "$(grep -c '^--- ' "$file" || true)" \
        "$(grep -c '^+[^+]' "$file" || true)" \
        "$(grep -c '^-[^-]' "$file" || true)" \
        "$file"
}

echo "Changes ruff WOULD make (nothing was modified):"
summarize "$OUT/1-format.diff" "ruff format"
summarize "$OUT/2-lint-safe-fixes.diff" "ruff check --fix"
summarize "$OUT/3-lint-safe-and-unsafe-fixes.diff" "ruff check --fix --unsafe-fixes"

echo
echo "Auto-fixable rules ([*] safe, [-] only with --unsafe-fixes):"
uv run ruff check --statistics 2>/dev/null | grep -E '^ *[0-9]+[[:space:]].*\[[*-]\]'

echo
echo "Remaining findings that need manual fixes:"
uv run ruff check --statistics 2>/dev/null | grep -E '^ *[0-9]+[[:space:]]' | grep -v '\[[*-]\]' | head -10

echo
echo "Review a diff with: code $OUT/1-format.diff"
