#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

python -m pip list --format=freeze \
    | cut -d'=' -f1 \
    | sed 's/-/_/g' \
    | tr '[:upper:]' '[:lower:]' \
    | sort -u >"$WORKDIR/installed_packages.txt"

pipreqs . --force --encoding=utf-8 \
    --ignore kvbiii-ml_venv,build,tests,examples \
    --savepath "$WORKDIR/requirements_from_imports.txt"

grep -o '^[^=<>]*' "$WORKDIR/requirements_from_imports.txt" \
    | sed 's/-/_/g' \
    | tr '[:upper:]' '[:lower:]' \
    | sort -u >"$WORKDIR/imported_packages.txt"

comm -23 "$WORKDIR/imported_packages.txt" "$WORKDIR/installed_packages.txt" \
    >"$WORKDIR/missing_packages.txt"

if [ -s "$WORKDIR/missing_packages.txt" ]; then
    echo "The following imports are not backed by installed dependencies:"
    cat "$WORKDIR/missing_packages.txt"
    exit 1
fi
