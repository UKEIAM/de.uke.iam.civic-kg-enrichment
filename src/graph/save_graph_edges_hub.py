#!/usr/bin/env python3
"""
Builds the direct "hub" edges connecting Gene/Disease/Drug/Variant to Evidence item, plus
Variant-Gene and Variant-Disease. All six are Direct edges per the spec
(as opposed to Phase 3's Projected edges, built via one shared
derive_projected_edges-style function on top of these).

Five of the six are mechanical explodes of list columns
(save_graph_nodes.py) already computed:
    - GENE-EVIDENCE / DRUG-EVIDENCE: explode Evidence item's gene_node_ids /
      drug_node_ids.
    - DISEASE-EVIDENCE: Evidence item's disease_node_id is already scalar.
    - VARIANT-EVIDENCE: explode Evidence item's variant_node_ids (itself
      already filtered in Phase 1 to VariationIDs that resolved to a real
      Variant node).
    - VARIANT-DISEASE: two independent origins, merged where they agree —
      clinvar_phenotype (Variant's resolved_phenotype_disease_ids, the
      MONDO crosswalk from clinvar_optimuskg_disease_alignment.py) and
      civic_evidence (projected: variant + disease sharing an evidence
      item). A pair present in both gets origin=["civic_evidence",
      "clinvar_phenotype"] rather than two separate edges.

VARIANT-GENE: Variant nodes only carry ClinVar's own gene_id_entrez, not an OptimusKG gene_node_id.
Reuses the same Entrez -> OptimusKG crosswalk civic_optimuskg_alignment.py's gene
resolution already established (hgnc_hub and direct_ensembl tiers).

Section 5 is a cross-consistency validation the spec calls out directly:
for every VARIANT-EVIDENCE edge, confirm the variant's VARIANT-GENE
target(s) overlap that same evidence item's own GENE-EVIDENCE target(s). A
mismatch doesn't necessarily mean a bug (ClinVar and CIViC can legitimately
attribute a variant to different genes at a locus, or the variant may not
have resolved to a gene at all).

Namespacing/provenance: derived tables referencing node ids, written to data/processed/.
Every edge defaults to undirected.

Inputs:
    - data/processed/graph_nodes_{gene,disease,drug,variant,evidence_item}_*.parquet
      (from save_graph_nodes.py)
    - HGNC complete set (fetched live from HGNC, for the Entrez -> OptimusKG
      gene crosswalk — VARIANT-GENE only)
Outputs:
    - data/processed/graph_edges_gene_evidence_<date>.parquet
    - data/processed/graph_edges_disease_evidence_<date>.parquet
    - data/processed/graph_edges_drug_evidence_<date>.parquet
    - data/processed/graph_edges_variant_gene_<date>.parquet
    - data/processed/graph_edges_variant_evidence_<date>.parquet
    - data/processed/graph_edges_variant_disease_<date>.parquet

Usage:
    python src/graph/save_graph_edges_hub.py
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

NODES_DIR = Path("data/raw/optimuskg/nodes")
PROCESSED_DIR = Path("data/processed")
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_node_table(name: str) -> pd.DataFrame:
    path = find_latest_file(PROCESSED_DIR, f"graph_nodes_{name}_*.parquet")
    df = pl.read_parquet(path).to_pandas()
    print(f"[setup] loaded {len(df)} {name} nodes from {path}")
    return df


# %%
# ---------------------------------------------------------------------------
# 0. Load node tables
# ---------------------------------------------------------------------------

evidence_item_nodes = _load_node_table("evidence_item")
variant_nodes = _load_node_table("variant")

# %%
# ---------------------------------------------------------------------------
# 1. GENE-EVIDENCE / DISEASE-EVIDENCE / DRUG-EVIDENCE: mechanical explodes
#    of Evidence item's own gene_node_ids/drug_node_ids (list) and
#    disease_node_id (scalar).
# ---------------------------------------------------------------------------

## Convention:
## "source" is always the non-evidence-item entity (gene/disease/drug/variant),
## "target" is the Evidence item ("source" is the more specific/leaf node).


def build_scalar_hub_edges(
    df: pd.DataFrame, id_col: str, edge_name: str
) -> pd.DataFrame:
    edges = (
        df[[id_col, "id"]]
        .dropna(subset=[id_col])
        .rename(columns={id_col: "source", "id": "target"})
    )
    edges["undirected"] = True
    print(f"[{edge_name}] {len(edges)} edges")
    return edges.reset_index(drop=True)


def build_list_hub_edges(df: pd.DataFrame, id_col: str, edge_name: str) -> pd.DataFrame:
    edges = df[[id_col, "id"]].explode(id_col).dropna(subset=[id_col])
    edges = edges.rename(columns={id_col: "source", "id": "target"})
    edges["undirected"] = True
    print(f"[{edge_name}] {len(edges)} edges")
    return edges.reset_index(drop=True)


gene_evidence_edges = build_list_hub_edges(
    evidence_item_nodes, "gene_node_ids", "GENE-EVIDENCE"
)
disease_evidence_edges = build_scalar_hub_edges(
    evidence_item_nodes, "disease_node_id", "DISEASE-EVIDENCE"
)
drug_evidence_edges = build_list_hub_edges(
    evidence_item_nodes, "drug_node_ids", "DRUG-EVIDENCE"
)

pl.from_pandas(gene_evidence_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_gene_evidence_{date_str}.parquet"
)
pl.from_pandas(disease_evidence_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_disease_evidence_{date_str}.parquet"
)
pl.from_pandas(drug_evidence_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_drug_evidence_{date_str}.parquet"
)
print("[hub] wrote GENE/DISEASE/DRUG-EVIDENCE edge tables")

# %%
# ---------------------------------------------------------------------------
# 2. VARIANT-GENE: resolve each Variant's ClinVar gene_id_entrez to an
#    OptimusKG gene_node_id via the same hgnc_hub/direct_ensembl crosswalk
#    civic_optimuskg_alignment.py's gene resolution already established.
# ---------------------------------------------------------------------------

genes = pl.read_parquet(NODES_DIR / "gene.parquet")
genes_id_set = set(genes["id"].to_list())

gene_xrefs_flat = (
    genes.select([pl.col("id").alias("gene_node_id"), "xrefs"])
    .explode("xrefs")
    .drop_nulls()
    .unnest("xrefs")
)
hgnc_to_node = dict(
    zip(
        gene_xrefs_flat.filter(gene_xrefs_flat["source"] == "HGNC")["id"],
        gene_xrefs_flat.filter(gene_xrefs_flat["source"] == "HGNC")["gene_node_id"],
    )
)

hgnc_complete = pd.read_csv(
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    sep="\t",
    dtype=str,
)
hgnc_pairs = hgnc_complete.dropna(subset=["entrez_id", "hgnc_id"])
entrez_to_hgnc_bare = dict(
    zip(hgnc_pairs["entrez_id"], hgnc_pairs["hgnc_id"].str.replace("HGNC:", ""))
)
hgnc_ensembl_pairs = hgnc_complete.dropna(subset=["entrez_id", "ensembl_gene_id"])
entrez_to_ensembl = dict(
    zip(hgnc_ensembl_pairs["entrez_id"], hgnc_ensembl_pairs["ensembl_gene_id"])
)


def resolve_gene_by_entrez(entrez_id) -> str | None:
    if pd.isna(entrez_id):
        return None
    entrez_id = str(entrez_id)

    hgnc_bare = entrez_to_hgnc_bare.get(entrez_id)
    if hgnc_bare:
        node_id = hgnc_to_node.get(hgnc_bare)
        if node_id:
            return node_id

    ensembl_id = entrez_to_ensembl.get(entrez_id)
    if ensembl_id and ensembl_id in genes_id_set:
        return ensembl_id

    return None


variant_nodes["_resolved_gene_node_id"] = variant_nodes["gene_id_entrez"].apply(
    resolve_gene_by_entrez
)

variant_gene_edges = variant_nodes.dropna(subset=["_resolved_gene_node_id"])[
    ["id", "_resolved_gene_node_id"]
].rename(columns={"id": "source", "_resolved_gene_node_id": "target"})
variant_gene_edges["undirected"] = True
variant_gene_edges = variant_gene_edges.reset_index(drop=True)

n_variant_gene_resolved = len(variant_gene_edges)
print(
    f"[VARIANT-GENE] {n_variant_gene_resolved} / {len(variant_nodes)} variants "
    f"resolved to an OptimusKG gene node (via hgnc_hub/direct_ensembl — no "
    f"symbol-match fallback needed, ClinVar's GeneID is essentially always "
    f"populated)"
)

pl.from_pandas(variant_gene_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_variant_gene_{date_str}.parquet"
)
print("[hub] wrote VARIANT-GENE edge table")

# %%
# ---------------------------------------------------------------------------
# 3. VARIANT-EVIDENCE: explode Evidence item's variant_node_ids (already
#    filtered in Phase 1 to VariationIDs that resolved to a real Variant
#    node).
# ---------------------------------------------------------------------------

variant_evidence_edges = build_list_hub_edges(
    evidence_item_nodes, "variant_node_ids", "VARIANT-EVIDENCE"
)
pl.from_pandas(variant_evidence_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_variant_evidence_{date_str}.parquet"
)
print("[hub] wrote VARIANT-EVIDENCE edge table")

# %%
# ---------------------------------------------------------------------------
# 4. VARIANT-DISEASE: two independent origins, merged on (variant, disease)
#    where they agree.
#    - clinvar_phenotype: Variant's own resolved_phenotype_disease_ids
#      (the MONDO crosswalk from clinvar_optimuskg_disease_alignment.py).
#    - civic_evidence: projected; a variant and a disease that co-occur on
#      the same CIViC evidence item. Built via derive_projected_edges.
# ---------------------------------------------------------------------------


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


variant_disease_civic = derive_projected_edges(
    evidence_item_nodes, "variant_node_ids", "disease_node_id"
)

phenotype_pairs = variant_nodes[["id", "resolved_phenotype_disease_ids"]].explode(
    "resolved_phenotype_disease_ids"
)
phenotype_pairs = phenotype_pairs.dropna(subset=["resolved_phenotype_disease_ids"])
phenotype_pairs = phenotype_pairs.rename(
    columns={"id": "source", "resolved_phenotype_disease_ids": "target"}
)

variant_disease_edges = merge_with_other_origin(
    variant_disease_civic, "civic_evidence", phenotype_pairs, "clinvar_phenotype"
)
variant_disease_edges["undirected"] = True

n_both_origins = variant_disease_edges["origin"].apply(lambda o: len(o) > 1).sum()
print(
    f"[VARIANT-DISEASE] {len(variant_disease_edges)} edges "
    f"({(variant_disease_edges['origin'].apply(lambda o: o == ['clinvar_phenotype'])).sum()} "
    f"clinvar_phenotype-only, "
    f"{(variant_disease_edges['origin'].apply(lambda o: o == ['civic_evidence'])).sum()} "
    f"civic_evidence-only, {n_both_origins} agree on both)"
)

pl.from_pandas(variant_disease_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_variant_disease_{date_str}.parquet"
)
print("[hub] wrote VARIANT-DISEASE edge table")

# %%
# ---------------------------------------------------------------------------
# 5. Cross-consistency validation: for every VARIANT-EVIDENCE edge, does the
#    variant's VARIANT-GENE target overlap that same evidence item's own
#    GENE-EVIDENCE target(s)? A mismatch isn't necessarily a bug (ClinVar
#    and CIViC can legitimately attribute a variant to different genes, or
#    the variant may not have resolved to a gene at all.
# ---------------------------------------------------------------------------

variant_to_gene = variant_gene_edges.groupby("source")["target"].apply(set).to_dict()
# gene_node_ids comes back from pl.read_parquet(...).to_pandas() as a numpy
# array, not a Python list. set(ids) works fine on either, but an
# isinstance(ids, list) guard would silently treat every row as empty.
evidence_to_genes = evidence_item_nodes.set_index("id")["gene_node_ids"].apply(
    lambda ids: set(ids) if ids is not None else set()
)

mismatches = 0
no_variant_gene = 0
for _, row in variant_evidence_edges.iterrows():
    variant_id, evidence_id = row["source"], row["target"]
    variant_genes = variant_to_gene.get(variant_id)
    if not variant_genes:
        no_variant_gene += 1
        continue
    evidence_genes = evidence_to_genes.get(evidence_id, set())
    if not (variant_genes & evidence_genes):
        mismatches += 1

print(
    f"[validation] {mismatches} / {len(variant_evidence_edges)} VARIANT-EVIDENCE "
    f"edges have NO gene overlap between the variant's VARIANT-GENE target(s) "
    f"and that evidence item's own GENE-EVIDENCE target(s) "
    f"({no_variant_gene} of those because the variant didn't resolve to any "
    f"gene at all — see VARIANT-GENE resolution rate above)"
)

# %%
