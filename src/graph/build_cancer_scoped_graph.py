#!/usr/bin/env python3
"""
Builds a graph scoped by cancer-disease relevance rather than by
evidence-backing.

Two pickle-load stages:

Stage 1 (evidence_graph_*.pkl): source of GEN/DIS/DRG/EVI node attributes
and of every overlay edge type except VARIANT-GENE

Stage 2 (raw OptimusKG pickle): cancer-scope genes/drugs via native
DIS-GEN/DRG-DIS from the cancer disease seed, then the native non-overlay
edges restricted to that scope, except:
    - ANA-GEN: excluded entirely
    - DIS-GEN, DRG-DRG, DRG-GEN, DRG-DIS: skipped in this raw-pickle pass
      only. No edge is lost; the corresponding overlay tables
      (gene_disease, drug_drug, gene_drug, drug_disease) are full OUTER
      merges, so each already carries every native pair of its type
      (CIViC-enriched and bare OptimusKG-only alike).

Inputs:
    - data/graphs/evidence_graph_*.pkl (from build_evidence_graph.py)
    - data/processed/cancer_relevant_disease_ids_*.csv (from civic_optimuskg_alignment.py)
    - data/processed/cancer_scoped_variant_{nodes,gene_edges,disease_edges}_*.parquet
      (from build_cancer_scoped_variants.py)
    - data/raw/optimuskg/optimuskg_lcc_graph_*.pkl (the raw OptimusKG pickle)
Outputs:
    - data/graphs/cancer_scoped_graph_<date>.pkl
    - data/graphs/cancer_scoped_graph_<date>_metadata.json

Usage:
    python src/graph/build_cancer_scoped_graph.py
"""

# %%
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

PROCESSED_DIR = Path("data/processed")
GRAPHS_DIR = Path("data/graphs")  # final graph pickles, read/written here;
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)  # intermediate CSVs/parquets stay in PROCESSED_DIR
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

DENSE_EXCLUDED_NATIVE = {"ANA-GEN"}
RAW_LABELS_SKIPPED = {"DIS-GEN", "DRG-DRG", "DRG-GEN", "DRG-DIS"}
GDD_NATIVE_LABELS = {"GEN", "DIS", "DRG"}
# Overlay edge types kept as-is from evidence_graph_*.pkl. VAR-GEN is
# replaced by build_cancer_scoped_variants.py's widened table;
# VAR-DIS is kept here too (existing rows) AND unioned with that script's
# new rows in stage 3.
KEPT_OVERLAY_EDGE_TYPES = {
    "GEN-EVI",
    "DIS-EVI",
    "DRG-EVI",
    "VAR-EVI",
    "VAR-DIS",
    "GEN-DIS",
    "VAR-DRG",
    "DRG-GEN",
    "DRG-DRG",
    "DRG-DIS",
}

# %%
# ---------------------------------------------------------------------------
# Stage 1: load evidence_graph_*.pkl for GEN/DIS/DRG/EVI node attributes and
# the kept overlay edges; load the cancer disease seed and
# build_cancer_scoped_variants.py's widened variant tables.
# ---------------------------------------------------------------------------

evidence_graph_path = find_latest_file(GRAPHS_DIR, "evidence_graph_*.pkl")
print(f"[load] {evidence_graph_path}")
t0 = time.time()
with open(evidence_graph_path, "rb") as f:
    OUR_G = pickle.load(f)
print(
    f"[load] done in {time.time() - t0:.1f}s — {OUR_G.number_of_nodes()} nodes, "
    f"{OUR_G.number_of_edges()} edges"
)

cancer_disease_ids = set(
    pd.read_csv(
        find_latest_file(
            PROCESSED_DIR, "cancer_relevant_disease_ids_[0-9][0-9][0-9][0-9]-*.csv"
        )
    )["disease_node_id"].astype(str)
)
print(f"[setup] {len(cancer_disease_ids)} cancer-relevant disease node ids")

gdd_evi_node_attrs = {
    n: dict(attrs)
    for n, attrs in OUR_G.nodes(data=True)
    if attrs["node_type"] in ("GEN", "DIS", "DRG", "EVI")
}
all_evi_ids = {n for n, a in gdd_evi_node_attrs.items() if a["node_type"] == "EVI"}
print(
    f"[stage 1] snapshotted {len(gdd_evi_node_attrs)} GEN/DIS/DRG/EVI node attr "
    f"dicts ({len(all_evi_ids)} of them EVI)"
)

our_overlay_edges = [
    (u, v, dict(data))
    for u, v, data in OUR_G.edges(data=True)
    if data["edge_type"] in KEPT_OVERLAY_EDGE_TYPES
]
overlay_edge_type_breakdown = Counter(d["edge_type"] for _, _, d in our_overlay_edges)
print(
    f"[stage 1] {len(our_overlay_edges)} overlay edges kept: {dict(overlay_edge_type_breakdown)}"
)

del OUR_G  # free the 2.5GB+ graph before loading the widened variant tables

b0a_variant_nodes = pl.read_parquet(
    find_latest_file(PROCESSED_DIR, "cancer_scoped_variant_nodes_*.parquet")
).to_dicts()
b0a_variant_gene_edges = pl.read_parquet(
    find_latest_file(PROCESSED_DIR, "cancer_scoped_variant_gene_edges_*.parquet")
).to_dicts()
b0a_variant_disease_edges = pl.read_parquet(
    find_latest_file(PROCESSED_DIR, "cancer_scoped_variant_disease_edges_*.parquet")
).to_dicts()
print(
    f"[stage 1] loaded {len(b0a_variant_nodes)} cancer-scoped variant nodes, "
    f"{len(b0a_variant_gene_edges)} VAR-GENE edges, "
    f"{len(b0a_variant_disease_edges)} new VAR-DIS edges from build_cancer_scoped_variants.py"
)

# %%
# ---------------------------------------------------------------------------
# Stage 2: load the raw OptimusKG pickle. Cancer-scope genes/drugs via
# native DIS-GEN/DRG-DIS from the disease seed, then every other native
# edge type restricted to that scope (see module docstring for exclusions).
# ---------------------------------------------------------------------------

optimuskg_path = find_latest_file("data/raw/optimuskg/", "optimuskg_lcc_graph_*.pkl")
print(f"\n[load] {optimuskg_path} (large file, ~2 min to unpickle)")
t0 = time.time()
with open(optimuskg_path, "rb") as f:
    OKG = pickle.load(f)
print(
    f"[load] done in {time.time() - t0:.1f}s — {OKG.number_of_nodes()} nodes, "
    f"{OKG.number_of_edges()} edges total"
)

node_label = {n: attrs.get("label") for n, attrs in OKG.nodes(data=True)}
print(f"[setup] node-label lookup built for {len(node_label)} nodes")

t0 = time.time()
cancer_gene_ids = set()
cancer_drug_ids = set()
for u, v, attrs in OKG.edges(data=True):
    label = attrs.get("label")
    if label == "DIS-GEN" and u in cancer_disease_ids:
        cancer_gene_ids.add(v)
    elif label == "DRG-DIS" and v in cancer_disease_ids:
        cancer_drug_ids.add(u)
cancer_gdd_ids = cancer_disease_ids | cancer_gene_ids | cancer_drug_ids
print(
    f"[scan pass 1] done in {time.time() - t0:.1f}s — {len(cancer_gene_ids)} "
    f"cancer-scope genes, {len(cancer_drug_ids)} cancer-scope drugs "
    f"({len(cancer_gdd_ids)} total GEN/DIS/DRG)"
)

scoped_variant_ids = {
    row["from"] for row in b0a_variant_gene_edges if row["to"] in cancer_gene_ids
}
print(
    f"[setup] {len(scoped_variant_ids)} / {len(b0a_variant_nodes)} cancer-scoped "
    f"variants have their resolved gene in cancer_gene_ids "
    f"({len(b0a_variant_nodes) - len(scoped_variant_ids)} dropped)"
)

scoped_node_set = cancer_gdd_ids | scoped_variant_ids | all_evi_ids
print(f"[setup] {len(scoped_node_set)} total scoped nodes before native passthrough")

t0 = time.time()
native_edges = []
reached_passthrough_nodes = set()
native_edge_labels_seen = Counter()
n_excluded_ana_gen = 0
for u, v, attrs in OKG.edges(data=True):
    label = attrs.get("label")
    if label in RAW_LABELS_SKIPPED:
        continue
    if label in DENSE_EXCLUDED_NATIVE:
        n_excluded_ana_gen += 1
        continue

    u_is_gdd = node_label.get(u) in GDD_NATIVE_LABELS
    v_is_gdd = node_label.get(v) in GDD_NATIVE_LABELS
    if u_is_gdd and u not in cancer_gdd_ids:
        continue
    if v_is_gdd and v not in cancer_gdd_ids:
        continue
    if u not in cancer_gdd_ids and v not in cancer_gdd_ids:
        continue

    attrs = dict(attrs)
    if label == "GEN-GEN":
        # Relabeled undirected, kept in native (u, v) direction — see
        # save_graph_edges_namespaced.py's DRUG-DRUG precedent and the
        # earlier build_expanded_evidence_graph.py correction: GEN-GEN is a
        # single symmetric relation (INTERACTS_WITH) with zero genuine
        # reciprocal pairs, so there's nothing to canonicalize, just a
        # native undirected=False to correct.
        attrs["undirected"] = True
    native_edges.append((u, v, attrs))
    native_edge_labels_seen[label] += 1
    if not u_is_gdd:
        reached_passthrough_nodes.add(u)
    if not v_is_gdd:
        reached_passthrough_nodes.add(v)

print(f"[scan pass 2] done in {time.time() - t0:.1f}s")
print(f"[native] {n_excluded_ana_gen} ANA-GEN edges excluded")
print(
    f"[native] {len(native_edges)} native edges kept: {dict(native_edge_labels_seen)}"
)
print(f"[native] {len(reached_passthrough_nodes)} passthrough nodes reached")

passthrough_node_attrs = {n: dict(OKG.nodes[n]) for n in reached_passthrough_nodes}
passthrough_label_counts = Counter(
    a.get("label") for a in passthrough_node_attrs.values()
)
print(f"[native] passthrough nodes by label: {dict(passthrough_label_counts)}")

del OKG  # free the 6.8GB graph before assembling the final graph

# %%
# ---------------------------------------------------------------------------
# Stage 3: assemble and persist.
# ---------------------------------------------------------------------------

G = nx.DiGraph()

# sorted(), not a bare set union: node insertion order here fixes the
# iteration order find_figure_candidates.py's evi_nodes dict inherits (dict
# order = insertion order), which otherwise varies run-to-run with Python's
# randomized string-hash seed and silently reshuffles which EVIs land in
# each figure's saved top-N.
for n in sorted(cancer_gdd_ids | all_evi_ids):
    G.add_node(n, **gdd_evi_node_attrs[n])

for row in b0a_variant_nodes:
    node_id = row["id"]
    if node_id not in scoped_variant_ids:
        continue
    G.add_node(node_id, node_type="VAR", **{k: v for k, v in row.items() if k != "id"})

for n, attrs in passthrough_node_attrs.items():
    G.add_node(
        n,
        node_type=attrs.get("label"),
        **{k: v for k, v in attrs.items() if k != "label"},
    )

print(f"\n[assemble] {G.number_of_nodes()} nodes added")

# Overlay edges from evidence_graph_*.pkl (all kept types, VAR-GEN excluded).
# Both endpoints already in G guarantees
# add_edge() can't silently auto-create a node with an empty attr dict.
for u, v, data in our_overlay_edges:
    if u in G and v in G:
        G.add_edge(u, v, **data)

# The widened VAR-GENE table (replaces evidence_graph's narrower one).
n_var_gene_added = 0
for row in b0a_variant_gene_edges:
    u, v = row["from"], row["to"]
    if u in G and v in G:
        edge_data = {k: val for k, val in row.items() if k not in ("from", "to")}
        G.add_edge(u, v, edge_type="VAR-GEN", **{"from": u, "to": v, **edge_data})
        n_var_gene_added += 1

# The new VAR-DISEASE rows for cancer_scoped_variants.py's new-only
# variants, unioned with (not replacing) evidence_graph's existing ones
# already carried through via our_overlay_edges above.
n_var_dis_added = 0
for row in b0a_variant_disease_edges:
    u, v = row["from"], row["to"]
    if u in G and v in G:
        edge_data = {k: val for k, val in row.items() if k not in ("from", "to")}
        G.add_edge(u, v, edge_type="VAR-DIS", **{"from": u, "to": v, **edge_data})
        n_var_dis_added += 1

print(
    f"[assemble] {n_var_gene_added} VAR-GENE, {n_var_dis_added} new VAR-DIS edges added"
)

# Native edges from stage 2.
n_native_added = 0
for u, v, attrs in native_edges:
    if u in G and v in G:
        edge_data = {**attrs, "from": u, "to": v}
        G.add_edge(u, v, edge_type=attrs.get("label"), **edge_data)
        n_native_added += 1
print(f"[assemble] {n_native_added} native edges added")

print(f"\n[assemble] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges total")

node_type_breakdown = Counter(attrs.get("node_type") for _, attrs in G.nodes(data=True))
print(f"[stats] nodes by type: {dict(node_type_breakdown)}")

full_edge_type_breakdown = Counter(
    data.get("edge_type") for _, _, data in G.edges(data=True)
)
print(f"[stats] edges by type: {dict(full_edge_type_breakdown)}")

evidence_backed_gdd = sum(
    1
    for n, a in G.nodes(data=True)
    if a.get("node_type") in ("GEN", "DIS", "DRG") and a.get("in_civic_evidence_item")
)
not_evidence_backed_gdd = sum(
    1
    for n, a in G.nodes(data=True)
    if a.get("node_type") in ("GEN", "DIS", "DRG")
    and not a.get("in_civic_evidence_item")
)
print(
    f"[stats] GEN/DIS/DRG: {evidence_backed_gdd} evidence-backed, "
    f"{not_evidence_backed_gdd} not — the comparison population this graph exists for"
)

var_ids_in_g = {n for n, a in G.nodes(data=True) if a.get("node_type") == "VAR"}
var_evi_partners = {
    u for u, v, d in G.edges(data=True) if d.get("edge_type") == "VAR-EVI"
}
n_var_with_evi = len(var_ids_in_g & var_evi_partners)
print(
    f"[stats] {n_var_with_evi} / {len(var_ids_in_g)} scoped variants have >=1 "
    f"VAR-EVI edge (the rest are gap candidates by construction)"
)

graph_path = GRAPHS_DIR / f"cancer_scoped_graph_{date_str}.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G, f)
print(f"\n[save] wrote {graph_path}")

metadata = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "optimuskg_snapshot": Path(optimuskg_path).name,
    "evidence_graph_snapshot": Path(evidence_graph_path).name,
    "cancer_disease_csv": find_latest_file(
        PROCESSED_DIR, "cancer_relevant_disease_ids_[0-9][0-9][0-9][0-9]-*.csv"
    ).name,
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "nodes_by_type": {str(k): v for k, v in node_type_breakdown.items()},
    "edges_by_type": {str(k): v for k, v in full_edge_type_breakdown.items()},
    "cancer_disease_ids": len(cancer_disease_ids),
    "cancer_gene_ids": len(cancer_gene_ids),
    "cancer_drug_ids": len(cancer_drug_ids),
    "evidence_backed_gdd_nodes": evidence_backed_gdd,
    "not_evidence_backed_gdd_nodes": not_evidence_backed_gdd,
    "scoped_variants_total": len(var_ids_in_g),
    "scoped_variants_with_var_evi": n_var_with_evi,
    "scoped_variants_without_var_evi": len(var_ids_in_g) - n_var_with_evi,
    "dense_excluded_native_edge_types": sorted(DENSE_EXCLUDED_NATIVE),
    "raw_labels_skipped_in_stage_2": sorted(RAW_LABELS_SKIPPED),
    "n_ana_gen_excluded": n_excluded_ana_gen,
}
metadata_path = GRAPHS_DIR / f"cancer_scoped_graph_{date_str}_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[save] wrote {metadata_path}")

# %%
