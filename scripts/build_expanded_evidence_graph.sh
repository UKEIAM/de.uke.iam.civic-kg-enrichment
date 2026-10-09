#!/usr/bin/env bash
# Builds data/processed/evidence_backed_expanded_graph_<date>.pkl — the
# evidence_graph expanded back out with native OptimusKG context beyond
# just the cancer-scoped population.
#
# This is a separate downstream branch off evidence_graph_*.pkl, not off
# cancer_scoped_graph_*.pkl — build_expanded_evidence_graph.py only needs
# evidence_graph_*.pkl (step 10 of build_cancer_scoped_graph.sh) plus the
# raw OptimusKG pickle, so it deliberately does NOT re-run the cancer-scope
# narrowing (build_cancer_scoped_variants.py/build_cancer_scoped_graph.py).
#
# Requires evidence_graph_*.pkl to already exist in data/processed/ — run
# scripts/build_cancer_scoped_graph.sh (or just
# `python3 src/graph/build_evidence_graph.py`) first if it doesn't.
#
# Usage:
#   ./scripts/build_expanded_evidence_graph.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! compgen -G "data/processed/evidence_graph_*.pkl" > /dev/null; then
    echo "error: no data/processed/evidence_graph_*.pkl found." >&2
    echo "Run scripts/build_cancer_scoped_graph.sh (or just" >&2
    echo "'python3 src/graph/build_evidence_graph.py') first." >&2
    exit 1
fi

echo "=== Building evidence_backed_expanded_graph ==="
python3 src/graph/build_expanded_evidence_graph.py

echo
echo "=== build_expanded_evidence_graph.sh done ==="
