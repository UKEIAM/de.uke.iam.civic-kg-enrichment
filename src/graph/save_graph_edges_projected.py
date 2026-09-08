#!/usr/bin/env python3
"""
Builds the Projected edges: pairs of entities that co-occur inside a shared CIViC
evidence item.

Both edges here are built with derive_projected_edges.

VARIANT-DRUG has no native OptimusKG counterpart, so its origin is always
["civic"].

GENE-DISEASE enriches OptimusKG's existing DIS-GEN edge
(DIS-GEN carries a single generic ASSOCIATED_WITH relation, unlike DRG-DIS's
INDICATION/CONTRAINDICATION/OFF_LABEL_USE split, there's no semantic
conflict risk in combining CIViC's claim with it):
    - ["optimuskg"] only: the gap already planned; structurally
      suggested, no curated evidence.
    - ["civic"] only: the inverse; a gene-disease pair with real, curated evidence that OptimusKG's own
      structure doesn't represent at all.
    - ["civic", "optimuskg"]: confirmation, both sources agree.

Inputs:
    - data/processed/graph_nodes_evidence_item_*.parquet (from save_graph_nodes.py)
    - data/raw/optimuskg/edges/disease_gene_edges.parquet
      (from save_optimuskg_disease_gene_edges.py)
Outputs:
    - data/processed/graph_edges_variant_drug_<date>.parquet
    - data/processed/graph_edges_gene_disease_<date>.parquet

Usage:
    python src/graph/save_graph_edges_projected.py
"""

# %%
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

PROCESSED_DIR = Path("data/processed")
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# %%
# ---------------------------------------------------------------------------
# 0. Load evidence_item node table + shared projection helpers
#    (identical to save_graph_edges_hub.py's).
# ---------------------------------------------------------------------------

evidence_path = find_latest_file(PROCESSED_DIR, "graph_nodes_evidence_item_*.parquet")
evidence_item_nodes = pl.read_parquet(evidence_path).to_pandas()
print(
    f"[setup] loaded {len(evidence_item_nodes)} evidence_item nodes from {evidence_path}"
)


def derive_projected_edges(
    evidence_df: pd.DataFrame, entity_col_a: str, entity_col_b: str
) -> pd.DataFrame:
    """Group CIViC evidence items by the pair of entities they share, emit
    one edge per pair carrying the grouped evidence-item records.
    entity_col_a/b may be list-valued or scalar — explode() is a no-op on
    an already-scalar column, so the same double-explode handles list x
    list, list x scalar, and scalar x scalar pairings uniformly."""
    pairs = evidence_df[[entity_col_a, entity_col_b, "id", "evidence_level"]].copy()
    pairs = pairs.explode(entity_col_a).explode(entity_col_b)
    pairs = pairs.dropna(subset=[entity_col_a, entity_col_b])

    def _agg(group: pd.DataFrame) -> pd.Series:
        records = [
            {"id": eid, "evidence_level": lvl}
            for eid, lvl in zip(group["id"], group["evidence_level"])
        ]
        return pd.Series({"evidence_items": records, "n_evidence_items": len(records)})

    grouped = (
        pairs.groupby([entity_col_a, entity_col_b], group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={entity_col_a: "source", entity_col_b: "target"})
    )
    return grouped


def merge_with_other_origin(
    civic_edges: pd.DataFrame,
    civic_origin_name: str,
    other_pairs: pd.DataFrame,
    other_origin_name: str,
) -> pd.DataFrame:
    """Outer-merges a CIViC-projected edge set (source, target,
    evidence_items, n_evidence_items) with another set of (source, target)
    pairs from a second, independent origin, producing one row per pair
    with an origin list: [civic_origin_name] only, [other_origin_name]
    only, or both when they agree."""
    merged = civic_edges.merge(
        other_pairs[["source", "target"]].drop_duplicates().assign(_other=True),
        on=["source", "target"],
        how="outer",
    )
    merged["evidence_items"] = merged["evidence_items"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged["n_evidence_items"] = merged["n_evidence_items"].fillna(0).astype(int)
    merged["_other"] = merged["_other"].astype("boolean").fillna(False).astype(bool)
    merged["_civic"] = merged["n_evidence_items"] > 0

    def _origin(row):
        origins = []
        if row["_civic"]:
            origins.append(civic_origin_name)
        if row["_other"]:
            origins.append(other_origin_name)
        return sorted(origins)

    merged["origin"] = merged.apply(_origin, axis=1)
    return merged[["source", "target", "origin", "evidence_items", "n_evidence_items"]]


# %%
# ---------------------------------------------------------------------------
# 1. VARIANT-DRUG: projected, no native OptimusKG counterpart;
#    origin is always ["civic"].
# ---------------------------------------------------------------------------

variant_drug_edges = derive_projected_edges(
    evidence_item_nodes, "variant_node_ids", "drug_node_ids"
)
variant_drug_edges["origin"] = variant_drug_edges.apply(lambda r: ["civic"], axis=1)
variant_drug_edges["undirected"] = True

print(f'[VARIANT-DRUG] {len(variant_drug_edges)} edges (origin always ["civic"])')

pl.from_pandas(variant_drug_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_variant_drug_{date_str}.parquet"
)
print("[projected] wrote VARIANT-DRUG edge table")

# %%
# ---------------------------------------------------------------------------
# 2. GENE-DISEASE: projected + merged with OptimusKG's own DIS-GEN
# ---------------------------------------------------------------------------

gene_disease_civic = derive_projected_edges(
    evidence_item_nodes, "gene_node_ids", "disease_node_id"
)

dis_gen_path = Path("data/raw/optimuskg/edges/disease_gene_edges.parquet")
dis_gen = pl.read_parquet(dis_gen_path).to_pandas()
print(f"[setup] loaded {len(dis_gen)} OptimusKG DIS-GEN edges from {dis_gen_path}")

# DIS-GEN edges are stored disease -> gene (e.g. "DOID_0050890 -> ENSG00000000971");
# flip to (gene, disease) to match this edge table's own source=gene, target=disease
# convention (GENE-DISEASE name order).
optimuskg_gene_disease_pairs = pd.DataFrame(
    {"source": dis_gen["target"], "target": dis_gen["source"]}
).drop_duplicates()
print(
    f"[GENE-DISEASE] {len(optimuskg_gene_disease_pairs)} unique (gene, disease) "
    f"pairs from OptimusKG's own DIS-GEN edge"
)

gene_disease_edges = merge_with_other_origin(
    gene_disease_civic, "civic", optimuskg_gene_disease_pairs, "optimuskg"
)
gene_disease_edges["undirected"] = True

n_civic_only = (gene_disease_edges["origin"].apply(lambda o: o == ["civic"])).sum()
n_optimuskg_only = (
    gene_disease_edges["origin"].apply(lambda o: o == ["optimuskg"])
).sum()
n_both = (gene_disease_edges["origin"].apply(lambda o: len(o) == 2)).sum()
print(
    f"[GENE-DISEASE] {len(gene_disease_edges)} total edges: "
    f"{n_optimuskg_only} optimuskg-only (structural gap, no curated evidence), "
    f"{n_civic_only} civic-only (curated evidence OptimusKG's own structure "
    f"doesn't represent at all), {n_both} confirmed (both agree)"
)

pl.from_pandas(gene_disease_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_gene_disease_{date_str}.parquet"
)
print("[projected] wrote GENE-DISEASE edge table")

# %%
