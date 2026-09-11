#!/usr/bin/env python3
"""
Assembles node and edge tables into an actual networkx graph object, prints a
build-statistics summary and runs the cross-consistency validations.

A single networkx.DiGraph, not a MultiGraph. Directed throughout,
matching OptimusKG's own MultiDiGraph representation, where
"undirected" is a boolean edge ATTRIBUTE rather than an actual second graph topology.

Graph scope:
this assembles exactly 5 node types, 11 edge
types, not OptimusKG's other native node types (Biological process,
Phenotype, Anatomy, Molecular function, Cellular component, Pathway,
Exposure) or its other native edges (GEN-GEN, DRG-PHE, DIS-DIS, etc.).

Cross-consistency validations (spec's "Further considerations"):
    1. VARIANT-EVIDENCE vs VARIANT-GENE/GENE-EVIDENCE:
       already run once in save_graph_edges_hub.py; re-run here for one complete
       report in a single place.
    2. GENE-DISEASE vs DISEASE-EVIDENCE:
       the edge's evidence_items and the evidence item's own
       disease_node_id both trace back to the exact same source column. This
       validates derive_projected_edges' own implementation.
    3. DRUG-DISEASE vs DISEASE-EVIDENCE:
       same reasoning as 2.

Inputs:
    - data/processed/graph_nodes_{gene,disease,drug,variant,evidence_item}_*.parquet
    - data/processed/graph_edges_{gene_evidence,disease_evidence,drug_evidence,
      variant_gene,variant_evidence,variant_disease,gene_disease,variant_drug,
      gene_drug,drug_drug,drug_disease}_*.parquet
Outputs:
    - data/processed/evidence_graph_<date>.pkl (the assembled networkx.DiGraph)
    - data/processed/evidence_graph_<date>_metadata.json (build provenance:
      OptimusKG snapshot, node/edge counts by type, timestamp)

Usage:
    python src/graph/build_evidence_graph.py
"""

# %%
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

PROCESSED_DIR = Path("data/processed")
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

NODE_TYPES = ["gene", "disease", "drug", "variant", "evidence_item"]
EDGE_TYPES = [
    "gene_evidence",
    "disease_evidence",
    "drug_evidence",
    "variant_gene",
    "variant_evidence",
    "variant_disease",
    "gene_disease",
    "variant_drug",
    "gene_drug",
    "drug_drug",
    "drug_disease",
]

# The lists above stay as-is (they match graph_nodes_{x}_*.parquet /
# graph_edges_{x}_*.parquet filenames) — only the node_type/edge_type
# ATTRIBUTE VALUES stored on the graph itself follow OptimusKG's own
# 3-letter node / 3-letter-dash-3-letter edge convention. Direction
# (from-first) matches each table's own already-established from/
# to columns, which for gene_drug ("DRG-GEN") and drug_disease
# ("DRG-DIS") already matches OptimusKG's native storage direction (Phase 4
# deliberately chose those directions for that reason). The one exception:
# gene_disease becomes "GEN-DIS" (gene-first, matching its actual
# from=gene column), not OptimusKG's native "DIS-GEN" (disease-first) —
# Phase 3 built that table gene-first to match the spec doc's prose
# ordering, and relabeling it "DIS-GEN" without also flipping its columns
# would make the label lie about what from/to actually mean.
NODE_TYPE_LABELS = {
    "gene": "GEN",
    "disease": "DIS",
    "drug": "DRG",
    "variant": "VAR",
    "evidence_item": "EVI",
}
EDGE_TYPE_LABELS = {
    "gene_evidence": "GEN-EVI",
    "disease_evidence": "DIS-EVI",
    "drug_evidence": "DRG-EVI",
    "variant_gene": "VAR-GEN",
    "variant_evidence": "VAR-EVI",
    "variant_disease": "VAR-DIS",
    "gene_disease": "GEN-DIS",  # see note above — not native DIS-GEN order
    "variant_drug": "VAR-DRG",
    "gene_drug": "DRG-GEN",  # matches native DRG-GEN direction
    "drug_drug": "DRG-DRG",  # matches native DRG-DRG
    "drug_disease": "DRG-DIS",  # matches native DRG-DIS direction
}

# %%
# ---------------------------------------------------------------------------
# 0. Load every node/edge table
# ---------------------------------------------------------------------------

node_records = {}
for node_type in NODE_TYPES:
    path = find_latest_file(PROCESSED_DIR, f"graph_nodes_{node_type}_*.parquet")
    records = pl.read_parquet(path).to_dicts()
    node_records[node_type] = records
    print(f"[load] {len(records)} {node_type} nodes from {path}")

edge_records = {}
for edge_type in EDGE_TYPES:
    path = find_latest_file(PROCESSED_DIR, f"graph_edges_{edge_type}_*.parquet")
    records = pl.read_parquet(path).to_dicts()
    edge_records[edge_type] = records
    print(f"[load] {len(records)} {edge_type} edges from {path}")

# %%
# ---------------------------------------------------------------------------
# 1. Assemble the graph.
# ---------------------------------------------------------------------------

G = nx.DiGraph()

for node_type, records in node_records.items():
    for row in records:
        node_id = row.pop("id")
        G.add_node(node_id, node_type=NODE_TYPE_LABELS[node_type], **row)

for edge_type, records in edge_records.items():
    for row in records:
        # "from"/"to" are read but NOT popped: they stay in row so **row
        # also stores them as real edge attributes, not just positionally
        # via G.add_edge(u, v, ...) — lets an edge's endpoints be read
        # straight off its attribute dict without reaching into the (u, v)
        # tuple networkx wraps it in.
        from_id, to_id = row["from"], row["to"]
        G.add_edge(from_id, to_id, edge_type=EDGE_TYPE_LABELS[edge_type], **row)

print(f"\n[assemble] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# %%
# ---------------------------------------------------------------------------
# 2. Build-statistics summary: node counts by type, edge counts by type
#    and origin, resolution rates.
# ---------------------------------------------------------------------------

print("\n[stats] nodes by type:")
for node_type, records in node_records.items():
    print(f"  {NODE_TYPE_LABELS[node_type]}: {len(records)}")

print("\n[stats] edges by type:")
for edge_type, records in edge_records.items():
    print(f"  {EDGE_TYPE_LABELS[edge_type]}: {len(records)}")

print("\n[stats] edges by origin (merged-shape edge types only):")
for edge_type in ["variant_disease", "gene_disease"]:
    origin_counts = {}
    for row in edge_records[edge_type]:
        key = tuple(sorted(row["origin"]))
        origin_counts[key] = origin_counts.get(key, 0) + 1
    print(f"  {EDGE_TYPE_LABELS[edge_type]}: {origin_counts}")

print("\n[stats] edges with CIViC backing (namespaced edge types only):")
for edge_type in ["gene_drug", "drug_drug", "drug_disease"]:
    n_with_civic = sum(
        1 for row in edge_records[edge_type] if row["n_evidence_items"] > 0
    )
    print(
        f"  {EDGE_TYPE_LABELS[edge_type]}: {n_with_civic} / "
        f"{len(edge_records[edge_type])} have civic_evidence"
    )

# %%
# ---------------------------------------------------------------------------
# 3. Cross-consistency validations.
# ---------------------------------------------------------------------------

# evidence_item's own "id" (CIVICEVI_<n>) is what edge tables reference for
# these lookups
evidence_item_path = find_latest_file(
    PROCESSED_DIR, "graph_nodes_evidence_item_*.parquet"
)
evidence_item_raw = pl.read_parquet(evidence_item_path).to_dicts()
evidence_id_to_genes = {
    row["id"]: set(row["gene_node_ids"] or []) for row in evidence_item_raw
}
evidence_id_to_disease = {
    row["id"]: row["disease_node_id"] for row in evidence_item_raw
}

variant_gene_path = find_latest_file(
    PROCESSED_DIR, "graph_edges_variant_gene_*.parquet"
)
variant_to_gene = {}
for row in pl.read_parquet(variant_gene_path).to_dicts():
    variant_to_gene.setdefault(row["from"], set()).add(row["to"])


def validate_variant_evidence() -> None:
    """Check 1: does the variant's VARIANT-GENE "to" overlap that same
    evidence item's own GENE-EVIDENCE "to"(s)?"""
    variant_evidence_path = find_latest_file(
        PROCESSED_DIR, "graph_edges_variant_evidence_*.parquet"
    )
    mismatches = 0
    no_gene = 0
    rows = pl.read_parquet(variant_evidence_path).to_dicts()
    for row in rows:
        variant_id, evidence_id = row["from"], row["to"]
        variant_genes = variant_to_gene.get(variant_id)
        if not variant_genes:
            no_gene += 1
            continue
        evidence_genes = evidence_id_to_genes.get(evidence_id, set())
        if not (variant_genes & evidence_genes):
            mismatches += 1
    print(
        f"[validation 1] VAR-EVI vs VAR-GEN/GEN-EVI: "
        f"{mismatches} / {len(rows)} edges have no gene overlap "
        f"({no_gene} because the variant didn't resolve to any gene at all)"
    )


def validate_projected_vs_disease_evidence(edge_type: str, evidence_field: str) -> None:
    """Does a projected edge's own evidence records point back
    to evidence items whose own disease_node_id actually matches the
    edge's to disease? This validates derive_projected_edges' implementation."""
    path = find_latest_file(PROCESSED_DIR, f"graph_edges_{edge_type}_*.parquet")
    rows = pl.read_parquet(path).to_dicts()
    n_checked = 0
    mismatches = 0
    for row in rows:
        for record in row[evidence_field]:
            n_checked += 1
            actual_disease = evidence_id_to_disease.get(record["id"])
            if actual_disease != row["to"]:
                mismatches += 1
    print(
        f"[validation] {EDGE_TYPE_LABELS[edge_type]} vs DIS-EVI: "
        f"{mismatches} / {n_checked} evidence-item references point to a "
        f"different disease than the edge's own to"
    )


validate_variant_evidence()
validate_projected_vs_disease_evidence("gene_disease", "evidence_items")
validate_projected_vs_disease_evidence("drug_disease", "civic_evidence")

# %%
# ---------------------------------------------------------------------------
# 4. Density-mismatch diagnostic
# ---------------------------------------------------------------------------

n_structural_only = sum(
    1
    for edge_type in ["gene_disease"]
    for row in edge_records[edge_type]
    if row["origin"] == ["optimuskg"]
) + sum(
    1
    for edge_type in ["drug_drug", "gene_drug", "drug_disease"]
    for row in edge_records[edge_type]
    if row["n_evidence_items"] == 0
)
n_evidence_backed = G.number_of_edges() - n_structural_only
print(
    f"\n {n_structural_only} edges are OptimusKG-structural-only "
    f"(no CIViC evidence), {n_evidence_backed} carry CIViC evidence of some "
    f"kind. A centrality/community-detection run on the full graph as-is "
    f"will be dominated by the structural majority; this script builds the full graph only"
)

# %%
# ---------------------------------------------------------------------------
# 5. Persist: pickle (preserves native list/dict attrs) + metadata
# ---------------------------------------------------------------------------

graph_path = PROCESSED_DIR / f"evidence_graph_{date_str}.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G, f)
print(f"\n[save] wrote {graph_path}")

optimuskg_snapshot = find_latest_file(
    "data/raw/optimuskg/", "optimuskg_lcc_graph_*.pkl"
).name
metadata = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "optimuskg_snapshot": optimuskg_snapshot,
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "nodes_by_type": {NODE_TYPE_LABELS[k]: len(v) for k, v in node_records.items()},
    "edges_by_type": {EDGE_TYPE_LABELS[k]: len(v) for k, v in edge_records.items()},
}
metadata_path = PROCESSED_DIR / f"evidence_graph_{date_str}_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[save] wrote {metadata_path}")

# %%
