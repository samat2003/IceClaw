#!/usr/bin/env bash
# Run MOONATLAS's own tests + published-scope validator in its SEPARATE environment and save
# the pinned revision and raw output under data/processed/p1/ (git-ignored local run metadata).
# Prerequisite: git clone + venv as described in docs/UPSTREAM_MOONATLAS.md. No network, no API.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/.upstream/moonatlas-science"
OUT="$ROOT/data/processed/p1"
mkdir -p "$OUT"
[ -d "$UP/.venv" ] || { echo "MISSING: $UP/.venv (see docs/UPSTREAM_MOONATLAS.md)" >&2; exit 2; }

{
  echo "recorded_at_utc: $(date -u +%FT%TZ)"
  echo "upstream_url: $(git -C "$UP" remote get-url origin)"
  echo "upstream_sha: $(git -C "$UP" rev-parse HEAD)"
  echo "upstream_dirty: $(git -C "$UP" status --porcelain | wc -l) changed paths"
  echo "python: $("$UP/.venv/bin/python" --version 2>&1)"
  echo "--- upstream pytest"
  (cd "$UP" && .venv/bin/python -m pytest -q 2>&1 | tail -3); echo "pytest_exit: ${PIPESTATUS[0]}"
  echo "--- moonatlas-science validate --scope published"
  (cd "$UP" && .venv/bin/moonatlas-science validate --root examples/science-smoke --scope published 2>&1)
  echo "validate_exit: $?"
} | tee "$OUT/moonatlas_validation.txt"
