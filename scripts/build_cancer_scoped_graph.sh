#!/usr/bin/env bash
# Builds data/processed/cancer_scoped_graph_<date>.pkl from scratch, running
# every stage in dependency order (each script's own "Inputs:"/"Outputs:"
# docstring documents the exact handoff; this just sequences them). Assumes
# scripts/fetch_data.sh has already been run at least once, so
# data/raw/{civic,clinvar,optimuskg}/ has something for the first few
# stages to read via find_latest_file().
#
# Every underlying script is a no-args, no-flags CLI that writes a fresh
# date-stamped output and picks up its own inputs via find_latest_file() —
# this script adds nothing beyond ordering and progress banners.
#
# Usage:
#   ./scripts/build_cancer_scoped_graph.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== [1/12] OptimusKG nodes -> parquet ==="
python3 src/process_data/save_optimuskg_nodes.py

echo
echo "=== [2/12] OptimusKG edges -> parquet ==="
python3 src/process_data/save_optimuskg_edges.py

echo
echo "=== [3/12] CIViC <-> OptimusKG entity resolution (disease/gene/drug) ==="
python3 src/alignment/civic_optimuskg_alignment.py

echo
echo "=== [4/12] CIViC <-> ClinVar variant alignment + gap candidates ==="
python3 src/alignment/civic_clinvar_gap_analysis.py

echo
echo "=== [5/12] ClinVar phenotype <-> OptimusKG disease alignment ==="
python3 src/alignment/clinvar_optimuskg_disease_alignment.py

echo
echo "=== [6/12] Graph nodes (gene/disease/drug/variant/evidence_item) ==="
python3 src/graph/save_graph_nodes.py

echo
echo "=== [7/12] Graph edges: hub (gene/disease/drug/variant <-> evidence) ==="
python3 src/graph/save_graph_edges_hub.py

echo
echo "=== [8/12] Graph edges: namespaced (gene-drug/drug-drug/drug-disease) ==="
python3 src/graph/save_graph_edges_namespaced.py

echo
echo "=== [9/12] Graph edges: projected (variant-drug/gene-disease) ==="
python3 src/graph/save_graph_edges_projected.py

echo
echo "=== [10/12] Assembling evidence_graph ==="
python3 src/graph/build_evidence_graph.py

echo
echo "=== [11/12] Cancer-scoped variant nodes/edges ==="
python3 src/graph/build_cancer_scoped_variants.py

echo
echo "=== [12/12] Assembling cancer_scoped_graph ==="
python3 src/graph/build_cancer_scoped_graph.py

echo
echo "=== build_cancer_scoped_graph.sh done ==="
