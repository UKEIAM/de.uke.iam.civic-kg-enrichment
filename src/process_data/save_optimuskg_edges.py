#!/usr/bin/env python3
"""Consolidated script to save OptimusKG's drug-drug (DRG-DRG), disease-gene
(DIS-GEN), gene-drug (DRG-GEN/GEN-DRG) and drug-disease (DRG-DIS) edges
from the pickle graph to parquet files, in a single pickle load.

Replaces the four separate save_optimuskg_{drug,disease_gene,gene_drug,
drug_disease}_edges.py scripts, which each independently paid the
~6.8GB/~2min unpickle cost — one script, one load, saves ~6 min on every
full rerun of this pipeline's edge-extraction stage. The optimuskg library
itself doesn't expose edge access, so the pickle (networkx MultiDiGraph)
is the only way to get at any of these.

Per-edge-type notes:
    - DRUG-DRUG: both endpoints are DRG nodes — matched structurally (by
      node label), not by an edge "label" attribute. Edge attrs vary by
      relation (SYNERGISTIC_INTERACTION carries interaction_description,
      PARENT doesn't).
    - DISEASE-GENE: edge label "DIS-GEN". Carries a single generic
      relation, ASSOCIATED_WITH, across all 9.7M edges — safe to treat as
      one flat set of pairs.
    - GENE-DRUG: edge label "DRG-GEN" (or "GEN-DRG", both accepted) — a
      drug's molecular mechanism of action.
    - DRUG-DISEASE: edge label "DRG-DIS". Carries three distinct,
      semantically opposed relations — INDICATION/CONTRAINDICATION/
      OFF_LABEL_USE — unlike DIS-GEN's single generic ASSOCIATED_WITH,
      which is why DRUG-DISEASE needs the namespaced treatment rather than
      GENE-DISEASE's merged origin/evidence_items shape (see
      save_graph_edges_namespaced.py).
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

drg_ids = {node_id for node_id, attrs in G.nodes(data=True) if attrs.get("label") == "DRG"}
print(f"[setup] {len(drg_ids)} drug (DRG) nodes")

# One pass over all 21.8M edges, sorted into the four buckets below.
# DRUG-DRUG is matched structurally (both endpoints in drg_ids) and is
# independent of the other three, which are matched by edge label — an
# edge can't land in more than one bucket, but each check is kept
# independent (not one big elif).
drug_drug_rows = []
disease_gene_rows = []
gene_drug_rows = []
gene_drug_labels_seen = set()
drug_disease_rows = []

for source, target, attrs in G.edges(data=True):
    label = attrs.get("label")
    row = {"from": source, "to": target}

    if source in drg_ids and target in drg_ids:
        drug_drug_rows.append({**row, **attrs})

    if label == "DIS-GEN":
        disease_gene_rows.append({**row, **attrs})
    elif label in ("DRG-GEN", "GEN-DRG"):
        gene_drug_labels_seen.add(label)
        gene_drug_rows.append({**row, **attrs})
    elif label == "DRG-DIS":
        drug_disease_rows.append({**row, **attrs})

print(
    f"[scan] matched {len(drug_drug_rows)} DRG-DRG, {len(disease_gene_rows)} "
    f"DIS-GEN, {len(gene_drug_rows)} DRG-GEN/GEN-DRG, {len(drug_disease_rows)} "
    f"DRG-DIS edges"
)

del G  # free the 6.8GB graph before the parquet-writing/reporting below

# %%
# ---------------------------------------------------------------------------
# 1. DRUG-DRUG
# ---------------------------------------------------------------------------

df = pl.from_dicts(drug_drug_rows, infer_schema_length=None)
out_path = OUT_DIR / "drug_drug_edges.parquet"
df.write_parquet(out_path)
print(f"\ndrug-drug edges: wrote {out_path} shape={df.shape}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())

# %%
# ---------------------------------------------------------------------------
# 2. DISEASE-GENE
# ---------------------------------------------------------------------------

df = pl.from_dicts(disease_gene_rows, infer_schema_length=None)
out_path = OUT_DIR / "disease_gene_edges.parquet"
df.write_parquet(out_path)
print(f"\ndisease-gene edges: wrote {out_path} shape={df.shape}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())

# %%
# ---------------------------------------------------------------------------
# 3. GENE-DRUG
# ---------------------------------------------------------------------------

print(f"\n[verify] edge labels matched: {gene_drug_labels_seen}")

df = pl.from_dicts(gene_drug_rows, infer_schema_length=None)
out_path = OUT_DIR / "gene_drug_edges.parquet"
df.write_parquet(out_path)
print(f"gene-drug edges: wrote {out_path} shape={df.shape}")
print(f"[verify] columns: {df.columns}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())

print("\n[sample] first 5 edges:")
for row in df.head(5).iter_rows(named=True):
    print(f"  {row}")

# Duplicate (from, to) check — if OptimusKG's own dedup/conflict
# resolution isn't reflected in relation_assertions/relation_conflict
# fields (neither drug_drug_edges.parquet nor disease_gene_edges.parquet
# has them), this is the only way to tell whether a (drug, gene) pair
# legitimately carries more than one relation from different sources.
dupe_pairs = (
    df.group_by(["from", "to"])
    .agg(pl.len().alias("n_edges"), pl.col("relation").n_unique().alias("n_relations"))
    .filter(pl.col("n_edges") > 1)
)
print(f"\n[verify] {dupe_pairs.height} (from, to) pairs have >1 edge")
multi_relation = dupe_pairs.filter(pl.col("n_relations") > 1)
print(f"[verify] {multi_relation.height} of those have genuinely DIFFERENT relations")

# %%
# ---------------------------------------------------------------------------
# 4. DRUG-DISEASE
# ---------------------------------------------------------------------------

df = pl.from_dicts(drug_disease_rows, infer_schema_length=None)
out_path = OUT_DIR / "drug_disease_edges.parquet"
df.write_parquet(out_path)
print(f"\ndrug-disease edges: wrote {out_path} shape={df.shape}")
print(f"[verify] columns: {df.columns}")

print("\n[breakdown] by relation:")
print(df["relation"].value_counts())

dupe_pairs = (
    df.group_by(["from", "to"])
    .agg(pl.len().alias("n_edges"), pl.col("relation").n_unique().alias("n_relations"))
    .filter(pl.col("n_edges") > 1)
)
print(f"\n[verify] {dupe_pairs.height} (from, to) pairs have >1 edge")
multi_relation = dupe_pairs.filter(pl.col("n_relations") > 1)
print(f"[verify] {multi_relation.height} of those have genuinely DIFFERENT relations")
