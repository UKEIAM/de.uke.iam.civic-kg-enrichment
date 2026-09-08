#!/usr/bin/env python3
"""Script to save OptimusKG's disease-gene (DIS-GEN) edges from the pickle
graph to a parquet file. Needed here for src/graph/'s GENE-DISEASE
projected edge: its origin field (optimuskg-only / civic-only / both) needs
to know which (gene, disease) pairs OptimusKG's own DIS-GEN edge already
connects, confirmed (Phase 0 check, see civic_optimuskg_alignment.py-era
conversation) to carry a single generic relation, ASSOCIATED_WITH, across
all 9.7M edges — safe to treat as one flat set of pairs, no relation-type
disambiguation needed the way DRG-DIS would require.

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
    if attrs.get("label") == "DIS-GEN":
        row = {"source": source, "target": target}
        row.update(attrs)
        rows.append(row)

df = pl.from_dicts(rows, infer_schema_length=None)
out_path = OUT_DIR / "disease_gene_edges.parquet"
df.write_parquet(out_path)
print(f"disease-gene edges: wrote {out_path} shape={df.shape}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())
