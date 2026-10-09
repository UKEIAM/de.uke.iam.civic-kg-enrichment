#!/usr/bin/env python3
"""
Builds the evidence-backed graph + OptimusKG's full native graph, extending
build_evidence_graph.py three ways: (1) folds in OptimusKG's other 7
native node types (Biological process, Phenotype, Anatomy, Molecular
function, Cellular component, Pathway, Exposure) and native edges
(GEN-GEN, DRG-PHE, DIS-DIS, etc.),
(2) includes every Variant node, not just the ones with direct CIViC
evidence, so the graph supports variant-level gap analysis,
and (3) includes every Disease node that has a
variant_disease edge to any variant, even if that disease itself has no
CIViC evidence attached. This provides extra context for a gap-candidate variant beyond just its
gene (e.g. the ClinVar-phenotype-crosswalk disease it's actually
associated with).

Two pickle-load stages:

Stage 1 (evidence_graph_*.pkl): "evidence-backed" for
gene/disease/drug nodes is read directly off the in_civic_evidence_item
node attribute (computed in save_graph_nodes.py). The base node
set is: evidence-backed gene/disease/drug nodes, PLUS every disease that's
the target of any variant_disease edge (regardless of that disease's own
evidence-backing), PLUS ALL variant nodes and ALL
evidence_item nodes.

Stage 2 (raw OptimusKG pickle): grows the node set beyond the base set.
Pulls in everything touching an evidence-backed gene/disease/drug node, except:
    - DIS-GEN, DRG-DRG: excluded because they're too dense.
    - DRG-GEN, DRG-DIS: skipped in this raw-pickle pass ONLY, not excluded
      from the graph. These edges are built in save_graph_edges_namespaced.py,
      so they carry BOTH the OptimusKG relation AND civic_evidence.
      Re-pulling the bare pickle version here and calling
      G.add_edge(u, v, ...) again would REPLACE the richer edge's
      attributes rather than merge them.
    - GEN-GEN: relabeled undirected, not excluded. A directionality check
      of the full raw graph found GEN-GEN carries a single relation
      (INTERACTS_WITH, symmetric by nature) and is natively stored
      undirected=False — but with ZERO genuinely reciprocal pairs.
      So GEN-GEN edges are kept in their native (u, v) direction,
      just marked undirected=True since the relation itself is symmetric.

Inputs:
    - data/graphs/evidence_graph_*.pkl (from build_evidence_graph.py)
    - data/raw/optimuskg/optimuskg_lcc_graph_*.pkl (the raw OptimusKG pickle)
Outputs:
    - data/graphs/evidence_backed_expanded_graph_<date>.pkl
    - data/graphs/evidence_backed_expanded_graph_<date>_metadata.json

Usage:
    python src/graph/build_expanded_evidence_graph.py
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

GRAPHS_DIR = Path("data/graphs")  # final graph pickles only
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Density-excluded (measured, not assumed — see module docstring).
DENSE_EXCLUDED = {"GEN-DIS", "DRG-DRG"}
# Already covered more richly by this pipeline's own tables — skipped only
# in the stage-2 raw-pickle pass, not excluded from the graph.
RAW_LABELS_SKIPPED = {"DIS-GEN", "DRG-DRG", "DRG-GEN", "DRG-DIS"}

# %%
# ---------------------------------------------------------------------------
# Stage 1: load evidence_graph and build the base node set. This is the same as
# build_evidence_graph.py's evidence-backed node set, plus every disease
# reached via a variant_disease edge, plus all variant nodes and all
# evidence_item nodes. The base node set is the starting point for stage 2's
# native OptimusKG exploration.
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

evidence_backed_gdd_ids = {
    n
    for n, attrs in OUR_G.nodes(data=True)
    if attrs["node_type"] in ("GEN", "DIS", "DRG")
    and attrs.get("in_civic_evidence_item")
}

# Disease nodes reached via ANY variant_disease edge, so a gap-candidate
# variant still carries its full disease context (defined by ClinVar)
disease_ids_via_variant = {
    v for _, v, data in OUR_G.edges(data=True) if data["edge_type"] == "VAR-DIS"
}
new_diseases_via_variant = disease_ids_via_variant - evidence_backed_gdd_ids
print(
    f"[base] {len(disease_ids_via_variant)} distinct disease nodes reached via "
    f"variant_disease edges, {len(new_diseases_via_variant)} of them not "
    f"already evidence-backed (newly included for variant gap-candidate context)"
)

# Anchor for stage 2's native-OptimusKG hop too
native_hop_anchor_ids = evidence_backed_gdd_ids | new_diseases_via_variant

all_variant_ids = {
    n for n, attrs in OUR_G.nodes(data=True) if attrs["node_type"] == "VAR"
}
all_evidence_item_ids = {
    n for n, attrs in OUR_G.nodes(data=True) if attrs["node_type"] == "EVI"
}
base_node_set = native_hop_anchor_ids | all_variant_ids | all_evidence_item_ids

print(
    f"\n[base] {len(evidence_backed_gdd_ids)} evidence-backed gene/disease/drug "
    f"nodes + {len(new_diseases_via_variant)} additional diseases via variant_disease "
    f"+ {len(all_variant_ids)} variant nodes (all, unconditional) + "
    f"{len(all_evidence_item_ids)} evidence_item nodes (all) = "
    f"{len(base_node_set)} base nodes"
)

our_kept_edges = [
    (u, v, dict(data))
    for u, v, data in OUR_G.edges(data=True)
    if data["edge_type"] not in DENSE_EXCLUDED
    and u in base_node_set
    and v in base_node_set
]
print(
    f"[stage 1] {len(our_kept_edges)} of this pipeline's own edges kept "
    f"({DENSE_EXCLUDED} excluded entirely; kept only where both endpoints "
    f"are already in the base node set)"
)

edge_type_breakdown = Counter(data["edge_type"] for _, _, data in our_kept_edges)
print(f"[stage 1] by edge_type: {dict(edge_type_breakdown)}")

# Snapshot the node attribute dicts we need for stage 3 before freeing OUR_G.
base_node_attrs = {n: dict(OUR_G.nodes[n]) for n in base_node_set}

del OUR_G  # free the 2.5GB+ graph before loading the 6.8GB raw pickle

# %%
# ---------------------------------------------------------------------------
# Stage 2: load the raw OptimusKG pickle. Collect
# everything touching a node in native_hop_anchor_ids (evidence-backed
# gene/disease/drug, plus diseases newly reached via variant_disease),
# except the excluded edges (see docstring).
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

# Real inventory, not assumed — also finally surfaces GEN-GEN's true size.
node_labels = Counter(attrs.get("label") for _, attrs in OKG.nodes(data=True))
print(f"\n[discover] native node labels: {dict(node_labels)}")

t0 = time.time()
native_edges = []
reached_native_nodes = set()
n_gen_gen = 0
for u, v, attrs in OKG.edges(data=True):
    label = attrs.get("label")
    if label in RAW_LABELS_SKIPPED:
        continue
    if u in native_hop_anchor_ids or v in native_hop_anchor_ids:
        attrs = dict(attrs)
        if label == "GEN-GEN":
            # Relabeled undirected, kept in native (u, v) direction — see
            # docstring: a single symmetric relation (INTERACTS_WITH) with
            # zero genuine reciprocal pairs, so there's nothing to
            # canonicalize, just a native undirected=False to correct.
            attrs["undirected"] = True
            n_gen_gen += 1
        native_edges.append((u, v, attrs))
        reached_native_nodes.add(u)
        reached_native_nodes.add(v)
print(f"[scan] done in {time.time() - t0:.1f}s")
if n_gen_gen:
    print(f"[native] {n_gen_gen} GEN-GEN edges relabeled undirected=True")

native_edge_labels_seen = Counter(attrs.get("label") for _, _, attrs in native_edges)
print(
    f"\n[discover] native edge labels reached (skipped labels not counted): "
    f"{dict(native_edge_labels_seen)}"
)
print(f"[native] {len(native_edges)} native OptimusKG edges reached")
print(f"[native] {len(reached_native_nodes)} distinct nodes touched by those edges")

# Every node touched by a kept native edge that ISN'T already in
# base_node_set needs its own attributes pulled here. A GEN-GEN/ANA-GEN/PHE-GEN/etc. edge's non-
# evidence-backed endpoint (e.g. a gene interacting with an evidence-backed
# gene, but not itself evidence-backed) is just as real and just as
# missing from base_node_set; without this, G.add_edge() in stage 3 would
# silently auto-create it with an empty attribute dict.
newly_reached_node_ids = reached_native_nodes - base_node_set
other_native_node_ids = {
    n
    for n in newly_reached_node_ids
    if OKG.nodes[n].get("label") not in ("GEN", "DIS", "DRG")
}
same_type_not_backed_ids = newly_reached_node_ids - other_native_node_ids
print(
    f"[native] {len(newly_reached_node_ids)} reached nodes need attributes pulled "
    f"(not already in base_node_set): {len(other_native_node_ids)} are 'other' "
    f"native types (Biological process, Phenotype, Anatomy, Molecular function, "
    f"Cellular component, Pathway, Exposure), {len(same_type_not_backed_ids)} are "
    f"gene/disease/drug nodes reached via a native edge but not themselves "
    f"evidence-backed (e.g. a GEN-GEN interaction partner)"
)

other_native_nodes = {n: dict(OKG.nodes[n]) for n in newly_reached_node_ids}
other_node_label_counts = Counter(
    attrs.get("label") for attrs in other_native_nodes.values()
)
print(
    f"[native] newly-reached node breakdown by native label: {dict(other_node_label_counts)}"
)

del OKG  # free the 6.8GB graph before assembling the final graph

# %%
# ---------------------------------------------------------------------------
# Stage 3: assemble and persist.
# ---------------------------------------------------------------------------

G = nx.DiGraph()

for node_id in base_node_set:
    G.add_node(node_id, **base_node_attrs[node_id])

# Nodes newly pulled here from the raw pickle are already labeled GEN/DIS/
# DRG/ANA/EXP/PHE/BPO/CCO/MFN/PWY — OptimusKG's own raw label already IS
# the target 3-letter node_type format (build_evidence_graph.py's
# NODE_TYPE_LABELS maps the friendly "gene"/"disease"/"drug" names in
# base_node_set to these exact same 3-letter values)
for node_id, attrs in other_native_nodes.items():
    node_type = attrs.get("label")
    G.add_node(
        node_id, node_type=node_type, **{k: v for k, v in attrs.items() if k != "label"}
    )

for u, v, data in our_kept_edges:
    G.add_edge(u, v, **data)

# Same reasoning: OptimusKG's own raw edge label, so no "optimuskg_native_" prefix.
# Unlike our_kept_edges (whose attrs already carry "from"/"to" from
# build_evidence_graph.py), these come straight off the raw OptimusKG
# pickle and never had them — add them explicitly so every edge in the
# final graph carries "from"/"to" as real attributes.
for u, v, attrs in native_edges:
    edge_data = {**attrs, "from": u, "to": v}
    G.add_edge(u, v, edge_type=attrs.get("label"), **edge_data)

print(f"\n[assemble] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

node_type_breakdown = Counter(attrs.get("node_type") for _, attrs in G.nodes(data=True))
print(f"[stats] nodes by type: {dict(node_type_breakdown)}")

full_edge_type_breakdown = Counter(
    data.get("edge_type") for _, _, data in G.edges(data=True)
)
print(f"[stats] edges by type: {dict(full_edge_type_breakdown)}")

graph_path = GRAPHS_DIR / f"evidence_backed_expanded_graph_{date_str}.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G, f)
print(f"\n[save] wrote {graph_path}")

metadata = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "optimuskg_snapshot": Path(optimuskg_path).name,
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "nodes_by_type": {str(k): v for k, v in node_type_breakdown.items()},
    "edges_by_type": {str(k): v for k, v in full_edge_type_breakdown.items()},
    "evidence_backed_gene_disease_drug_nodes": len(evidence_backed_gdd_ids),
    "additional_diseases_via_variant_disease": len(new_diseases_via_variant),
    "all_variant_nodes": len(all_variant_ids),
    "all_evidence_item_nodes": len(all_evidence_item_ids),
    "native_edges_added": len(native_edges),
    "native_edge_labels": {str(k): v for k, v in native_edge_labels_seen.items()},
    "gen_gen_relabeled_undirected": n_gen_gen,
    "newly_reached_node_labels": {
        str(k): v for k, v in other_node_label_counts.items()
    },
    "dense_excluded_edge_types": sorted(DENSE_EXCLUDED),
    "raw_labels_skipped_in_stage_2": sorted(RAW_LABELS_SKIPPED),
}
metadata_path = (
    GRAPHS_DIR / f"evidence_backed_expanded_graph_{date_str}_metadata.json"
)
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[save] wrote {metadata_path}")

# %%
