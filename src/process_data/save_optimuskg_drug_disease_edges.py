#!/usr/bin/env python3
"""Script to save OptimusKG's drug-disease (DRG-DIS) edges from the pickle
graph to a parquet file — same source/reasoning as the other
save_optimuskg_*_edges.py scripts, needed here for src/graph/'s namespaced
DRUG-DISEASE edge. Confirmed in an earlier check (see
civic_optimuskg_alignment.py-era conversation, "Phase 0") to carry three
distinct, semantically opposed relations — INDICATION/CONTRAINDICATION/
OFF_LABEL_USE — unlike DIS-GEN's single generic ASSOCIATED_WITH, which is
exactly why DRUG-DISEASE needs the namespaced treatment rather than
GENE-DISEASE's merged origin/evidence_items shape.

Loading the pickle is the expensive part (~6.8GB, ~2 min to unpickle) —
worth running this once and reading the parquet output afterward.
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

print(f"[load] {GRAPH_PATH} (large file, unpickling takes ~2 min)")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)
print(f"[load] done — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges total")

rows = []
for source, target, attrs in G.edges(data=True):
    if attrs.get("label") == "DRG-DIS":
        row = {"source": source, "target": target}
        row.update(attrs)
        rows.append(row)

df = pl.from_dicts(rows, infer_schema_length=None)
out_path = OUT_DIR / "drug_disease_edges.parquet"
df.write_parquet(out_path)
print(f"drug-disease edges: wrote {out_path} shape={df.shape}")
print(f"[verify] columns: {df.columns}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())

dupe_pairs = (
    df.group_by(["source", "target"])
    .agg(pl.len().alias("n_edges"), pl.col("relation").n_unique().alias("n_relations"))
    .filter(pl.col("n_edges") > 1)
)
print(f"\n[verify] {dupe_pairs.height} (source, target) pairs have >1 edge")
multi_relation = dupe_pairs.filter(pl.col("n_relations") > 1)
print(f"[verify] {multi_relation.height} of those have genuinely DIFFERENT relations")
