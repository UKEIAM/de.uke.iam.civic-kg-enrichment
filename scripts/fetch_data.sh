#!/usr/bin/env bash
# Fetches all raw, externally-sourced data this pipeline depends on:
# CIViC evidence items (GraphQL), ClinVar's variant_summary.txt.gz (NCBI
# FTP), and OptimusKG's largest connected component (optimuskg library).
#
# The three fetchers are independent of each other and of everything else
# in this pipeline — safe to run in any order, including in parallel.
# They're run sequentially here only for clear, linear log output.
#
# Usage:
#   ./scripts/fetch_data.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== [1/3] Fetching CIViC evidence items ==="
python3 src/fetch_data/fetch_civic_data.py

echo
echo "=== [2/3] Fetching ClinVar variant_summary ==="
python3 src/fetch_data/fetch_clinvar_data.py

echo
echo "=== [3/3] Fetching OptimusKG (largest connected component) ==="
python3 src/fetch_data/save_optimuskg_pickle.py

echo
echo "=== fetch_data.sh done ==="
