#!/usr/bin/env python3
"""Script to save OptimusKG's drug-gene (DRG-GEN) mechanism-of-action
edges from the pickle graph to a parquet file — same source/reasoning as
save_optimuskg_drug_edges.py and save_optimuskg_disease_gene_edges.py,
needed here for src/graph/'s namespaced GENE-DRUG edge.

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
labels_seen = set()
for source, target, attrs in G.edges(data=True):
    label = attrs.get("label")
    if label in ("DRG-GEN", "GEN-DRG"):
        labels_seen.add(label)
        row = {"source": source, "target": target}
        row.update(attrs)
        rows.append(row)

print(f"[verify] edge labels matched: {labels_seen}")

df = pl.from_dicts(rows, infer_schema_length=None)
out_path = OUT_DIR / "gene_drug_edges.parquet"
df.write_parquet(out_path)
print(f"gene-drug edges: wrote {out_path} shape={df.shape}")
print(f"[verify] columns: {df.columns}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())

print("\n[sample] first 5 edges:")
for row in df.head(5).iter_rows(named=True):
    print(f"  {row}")

# Duplicate (source, target) check — if OptimusKG's own dedup/conflict
# resolution isn't reflected in relation_assertions/relation_conflict
# fields (neither drug_drug_edges.parquet nor disease_gene_edges.parquet
# has them), this is the only way to tell whether a (drug, gene) pair
# legitimately carries more than one relation from different sources.
dupe_pairs = (
    df.group_by(["source", "target"])
    .agg(pl.len().alias("n_edges"), pl.col("relation").n_unique().alias("n_relations"))
    .filter(pl.col("n_edges") > 1)
)
print(f"\n[verify] {dupe_pairs.height} (source, target) pairs have >1 edge")
multi_relation = dupe_pairs.filter(pl.col("n_relations") > 1)
print(f"[verify] {multi_relation.height} of those have genuinely DIFFERENT relations")
