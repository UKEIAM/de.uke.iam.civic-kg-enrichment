#!/usr/bin/env python3
"""Script to save the nodes of the OptimusKG graph (pickle file) to parquet files."""

import os
import pickle
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_file import find_latest_file

GRAPH_PATH = find_latest_file("data/raw/optimuskg/", "optimuskg_lcc_graph_*.pkl")
OUT_DIR = Path("data/raw/optimuskg/nodes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the cached full graph
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

label_to_name = {"GEN": "gene", "DIS": "disease", "DRG": "drug"}

for label, name in label_to_name.items():
    rows = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("label") == label:
            row = {"id": node_id}
            row.update(attrs)
            rows.append(row)

    df = pl.from_dicts(rows, infer_schema_length=None)
    out_path = OUT_DIR / f"{name}.parquet"
    df.write_parquet(out_path)
    print(f"{name}: wrote {out_path} shape={df.shape}")
