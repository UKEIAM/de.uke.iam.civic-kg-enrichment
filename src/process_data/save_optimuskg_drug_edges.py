#!/usr/bin/env python3
"""Script to save OptimusKG's drug-drug (DRG-DRG) edges from the pickle graph
to a parquet file — same source as save_optimuskg_nodes.py, but edges rather
than nodes, since the optimuskg library itself doesn't expose edge access and
the pickle (networkx MultiDiGraph) is the only way to get at them.

Loading the pickle is the expensive part (~6.8GB, ~100s to unpickle) — worth
running this once and reading the parquet output afterward rather than
re-loading the graph for every drug-drug edge question.
"""

import os
import pickle
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

GRAPH_PATH = find_latest_file("data/raw/optimuskg/", "optimuskg_lcc_graph_*.pkl")
OUT_DIR = Path("data/raw/optimuskg/edges")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the cached full graph
print(f"[load] {GRAPH_PATH} (large file, unpickling takes ~2 min)")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)
print(f"[load] done — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges total")

drg_ids = {node_id for node_id, attrs in G.nodes(data=True) if attrs.get("label") == "DRG"}
print(f"[setup] {len(drg_ids)} drug (DRG) nodes")

# Drug-drug edges: both endpoints are DRG nodes. Edge attrs vary by relation
# (SYNERGISTIC_INTERACTION carries interaction_description, PARENT doesn't) —
# from_dicts with infer_schema_length=None handles the ragged schema the same
# way save_optimuskg_nodes.py does for struct-valued node columns.
rows = []
for source, target, attrs in G.edges(data=True):
    if source in drg_ids and target in drg_ids:
        row = {"source": source, "target": target}
        row.update(attrs)
        rows.append(row)

df = pl.from_dicts(rows, infer_schema_length=None)
out_path = OUT_DIR / "drug_drug_edges.parquet"
df.write_parquet(out_path)
print(f"drug-drug edges: wrote {out_path} shape={df.shape}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())
