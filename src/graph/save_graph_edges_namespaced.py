#!/usr/bin/env python3
"""
Builds the Namespaced edges: GENE-DRUG, DRUG-DRUG, DRUG-DISEASE. All three sit on an
edge OptimusKG already has, where OptimusKG's own relation means something
categorically different from CIViC's claim: DRG-GEN's relation is a drug's
molecular mechanism of action (INHIBITOR/AGONIST/TARGET/ENZYME/TRANSPORTER/...),
DRG-DRG's is a context-free pharmacology fact (SYNERGISTIC_INTERACTION/PARENT), and
DRG-DIS's is a specific clinical claim with three semantically opposed
values (INDICATION/CONTRAINDICATION/OFF_LABEL_USE).
So CIViC's evidence goes in its own civic_evidence field on the same edge,
never merged into OptimusKG's relation vocabulary
or blended into a shared origin list.

Directionality: GENE-DRUG and DRUG-DISEASE are kept directed (source=drug,
target=gene/disease, matching OptimusKG's own DRG-GEN/DRG-DIS storage
direction) since both relations are directional (a drug acts on
a gene's product, a drug treats/is-contraindicated-for a disease, not the
reverse). DRUG-DRUG stays undirected since
SYNERGISTIC_INTERACTION/PARENT are symmetric facts about a pair even though
OptimusKG's own raw storage isn't marked that way; pairs are canonicalized
(alphabetically sorted) on both the CIViC and OptimusKG side before merging
so A-B and B-A aren't double-counted as separate edges.

Two secondary analyses run here as diagnostics:
    - GENE-DRUG: for PREDICTIVE evidence items specifically, does the
      biomarker gene match a documented DRG-GEN mechanism relation? A hit
      means the biomarker is the drug's own documented target (or enzyme,
      transporter, etc.); a miss means the predictive relationship runs
      through something DRG-GEN doesn't cover (downstream pathway, a
      resistance mutation). Stratified by clinical_significance.
    - DRUG-DRUG: (a) flag any pair where OptimusKG says PARENT (same
      substance, different chemical form) AND CIViC lists a combination/
      substitute claim (bug); (b) for CIViC's COMBINATION pairs specifically (not
      SUBSTITUTES), how often does OptimusKG
      independently also have a SYNERGISTIC_INTERACTION relation for that
      pair (real cross-validation signal).

Inputs:
    - data/processed/graph_nodes_evidence_item_*.parquet (from save_graph_nodes.py)
    - data/raw/optimuskg/edges/{gene_drug,drug_drug,drug_disease}_edges.parquet
      (from save_optimuskg_{gene_drug,drug_edges,drug_disease}_edges.py)
Outputs:
    - data/processed/graph_edges_gene_drug_<date>.parquet
    - data/processed/graph_edges_drug_drug_<date>.parquet
    - data/processed/graph_edges_drug_disease_<date>.parquet

Usage:
    python src/graph/save_graph_edges_namespaced.py
"""

# %%
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

PROCESSED_DIR = Path("data/processed")
EDGES_DIR = Path("data/raw/optimuskg/edges")
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# %%
# ---------------------------------------------------------------------------
# 0. Load evidence_item node table + shared helpers.
# ---------------------------------------------------------------------------

evidence_path = find_latest_file(PROCESSED_DIR, "graph_nodes_evidence_item_*.parquet")
evidence_item_nodes = pl.read_parquet(evidence_path).to_pandas()
print(
    f"[setup] loaded {len(evidence_item_nodes)} evidence_item nodes from {evidence_path}"
)


def derive_projected_edges(
    evidence_df: pd.DataFrame, entity_col_a: str, entity_col_b: str
) -> pd.DataFrame:
    """Same shared logic as save_graph_edges_hub.py/save_graph_edges_projected.py.
    Group CIViC evidence items by the pair of entities they share,
    emit one edge per pair. entity_col_a becomes "source", entity_col_b
    becomes "target"; pass them in the direction this file's edges need
    (drug first, to match OptimusKG's own DRG-GEN/DRG-DIS storage
    direction), not necessarily the edge's own name order."""
    pairs = evidence_df[[entity_col_a, entity_col_b, "id", "evidence_level"]].copy()
    pairs = pairs.explode(entity_col_a).explode(entity_col_b)
    pairs = pairs.dropna(subset=[entity_col_a, entity_col_b])

    def _agg(group: pd.DataFrame) -> pd.Series:
        records = [
            {"id": eid, "evidence_level": lvl}
            for eid, lvl in zip(group["id"], group["evidence_level"])
        ]
        return pd.Series({"civic_evidence": records, "n_evidence_items": len(records)})

    grouped = (
        pairs.groupby([entity_col_a, entity_col_b], group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={entity_col_a: "source", entity_col_b: "target"})
    )
    return grouped


def merge_namespaced(
    civic_edges: pd.DataFrame, optimuskg_edges: pd.DataFrame
) -> pd.DataFrame:
    """Outer-merge on (source, target): OptimusKG's own columns carried
    through as-is (null where OptimusKG has no edge for that pair), CIViC's
    civic_evidence/n_evidence_items kept in their own fields (empty/0 where
    CIViC doesn't reference that pair)."""
    merged = civic_edges.merge(optimuskg_edges, on=["source", "target"], how="outer")
    merged["civic_evidence"] = merged["civic_evidence"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged["n_evidence_items"] = merged["n_evidence_items"].fillna(0).astype(int)
    return merged


# %%
# ---------------------------------------------------------------------------
# 1. GENE-DRUG: directed (source=drug, target=gene, matching DRG-GEN's own
#    storage direction).
# ---------------------------------------------------------------------------

gene_drug_civic = derive_projected_edges(
    evidence_item_nodes, "drug_node_ids", "gene_node_ids"
)

drg_gen_path = EDGES_DIR / "gene_drug_edges.parquet"
drg_gen = pl.read_parquet(drg_gen_path).to_pandas()
drg_gen = drg_gen[
    ["source", "target", "relation", "mechanisms_of_action", "source_urls", "sources"]
]
print(f"[setup] loaded {len(drg_gen)} OptimusKG DRG-GEN edges from {drg_gen_path}")

gene_drug_edges = merge_namespaced(gene_drug_civic, drg_gen)
gene_drug_edges["undirected"] = False

n_relation = gene_drug_edges["relation"].notna().sum()
n_civic = (gene_drug_edges["n_evidence_items"] > 0).sum()
n_both = (
    (gene_drug_edges["relation"].notna()) & (gene_drug_edges["n_evidence_items"] > 0)
).sum()
print(
    f"[GENE-DRUG] {len(gene_drug_edges)} total edges: {n_relation} have an "
    f"OptimusKG DRG-GEN relation, {n_civic} have CIViC evidence, {n_both} have both"
)

# Secondary analysis: does the biomarker gene match a documented mechanism
# relation, for PREDICTIVE evidence specifically? A hit means the biomarker
# is the drug's own documented target/enzyme/transporter/etc; a miss means
# the predictive relationship runs through something DRG-GEN doesn't cover.
drg_gen_pairs = set(zip(drg_gen["source"], drg_gen["target"]))

predictive_rows = (
    evidence_item_nodes[evidence_item_nodes["evidence_type"] == "PREDICTIVE"]
    .explode("drug_node_ids")
    .explode("gene_node_ids")
)
predictive_rows = predictive_rows.dropna(subset=["drug_node_ids", "gene_node_ids"])
predictive_rows["_has_mechanism"] = predictive_rows.apply(
    lambda r: (r["drug_node_ids"], r["gene_node_ids"]) in drg_gen_pairs, axis=1
)
mechanism_breakdown = (
    predictive_rows.groupby("clinical_significance_civic")["_has_mechanism"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "has_documented_mechanism", "count": "total"})
)
print(
    "\n[GENE-DRUG mechanism-vs-biomarker] PREDICTIVE (drug, gene) pairs by clinical_significance_civic:"
)
print(mechanism_breakdown.to_string())

pl.from_pandas(gene_drug_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_gene_drug_{date_str}.parquet"
)
print("\n[namespaced] wrote GENE-DRUG edge table")

# %%
# ---------------------------------------------------------------------------
# 2. DRUG-DRUG: undirected. An N-drug combination
#    evidence item projects to C(N,2) pairwise edges, not one N-ary edge;
#    combination_size (the number of distinct drugs on that evidence item's
#    own row) marks which pairs came from a 3+-drug regimen.
# ---------------------------------------------------------------------------


def derive_pairwise_projected_edges(
    evidence_df: pd.DataFrame, list_col: str
) -> pd.DataFrame:
    """Same grouping idea as derive_projected_edges, but for pairs drawn
    from WITHIN a single list column on the same evidence item (e.g. every
    drug on one combination-therapy row), not across two different
    columns. Pairs are canonicalized (sorted) so A-B and B-A collapse to
    one edge."""
    records = []
    for row in evidence_df.itertuples(index=False):
        ids = getattr(row, list_col)
        unique_ids = (
            sorted({i for i in ids if i is not None}) if ids is not None else []
        )
        if len(unique_ids) < 2:
            continue
        for a, b in combinations(unique_ids, 2):
            records.append(
                {
                    "source": a,
                    "target": b,
                    "evidence_item_id": row.id,
                    "evidence_level": row.evidence_level,
                    "interaction_type": row.therapy_interaction_type,
                    "combination_size": len(unique_ids),
                }
            )
    if not records:
        return pd.DataFrame(
            columns=["source", "target", "civic_evidence", "n_evidence_items"]
        )

    pairs = pd.DataFrame(records)

    def _agg(group: pd.DataFrame) -> pd.Series:
        civic_evidence = [
            {
                "id": r.evidence_item_id,
                "interaction_type": r.interaction_type,
                "evidence_level": r.evidence_level,
                "combination_size": r.combination_size,
            }
            for r in group.itertuples(index=False)
        ]
        return pd.Series(
            {
                "civic_evidence": civic_evidence,
                "n_evidence_items": group["evidence_item_id"].nunique(),
            }
        )

    return (
        pairs.groupby(["source", "target"], group_keys=False)
        .apply(_agg, include_groups=False)
        .reset_index()
    )


drug_drug_civic = derive_pairwise_projected_edges(evidence_item_nodes, "drug_node_ids")
print(f"[DRUG-DRUG] {len(drug_drug_civic)} CIViC-projected drug-drug pairs")

drg_drg_path = EDGES_DIR / "drug_drug_edges.parquet"
drg_drg = pl.read_parquet(drg_drg_path).to_pandas()
print(f"[setup] loaded {len(drg_drg)} OptimusKG DRG-DRG edges from {drg_drg_path}")

# Canonicalize (source, target) the same way the CIViC side already is;
# OptimusKG's raw storage isn't guaranteed to store only one direction per
# pair (DrugBank interaction data is often entered from both drugs' own
# pages). Collapsing duplicates: keep the first non-null
# interaction_description per canonical pair, report how many collided.
drg_drg["source_c"] = drg_drg.apply(lambda r: min(r["source"], r["target"]), axis=1)
drg_drg["target_c"] = drg_drg.apply(lambda r: max(r["source"], r["target"]), axis=1)
n_before = len(drg_drg)
drg_drg_canonical = (
    drg_drg.groupby(["source_c", "target_c"])
    .agg(
        relation=("relation", "first"),
        interaction_description=("interaction_description", "first"),
        sources=("sources", "first"),
        n_raw_edges=("relation", "size"),
    )
    .reset_index()
    .rename(columns={"source_c": "source", "target_c": "target"})
)
n_collided = (drg_drg_canonical["n_raw_edges"] > 1).sum()
print(
    f"[DRUG-DRUG] {n_before} raw OptimusKG edges canonicalized to "
    f"{len(drg_drg_canonical)} unique pairs ({n_collided} pairs had both "
    f"directions stored separately — kept the first non-null values)"
)
drg_drg_canonical = drg_drg_canonical.drop(columns=["n_raw_edges"])

drug_drug_edges = merge_namespaced(drug_drug_civic, drg_drg_canonical)
drug_drug_edges["undirected"] = True

print(
    f"[DRUG-DRUG] {len(drug_drug_edges)} total edges: "
    f"{drug_drug_edges['relation'].notna().sum()} have an OptimusKG relation, "
    f"{(drug_drug_edges['n_evidence_items'] > 0).sum()} have CIViC evidence"
)

# Secondary analysis (a): PARENT + combination/substitute claim on the same
# pair is almost certainly a resolution bug (same substance, two node ids),
# not a genuine combination-therapy finding.
parent_with_civic = drug_drug_edges[
    (drug_drug_edges["relation"] == "PARENT")
    & (drug_drug_edges["n_evidence_items"] > 0)
]
print(
    f"\n[DRUG-DRUG resolution-bug check] {len(parent_with_civic)} pairs have "
    f"OptimusKG relation=PARENT AND a CIViC combination/substitute claim — "
    f"likely two node ids for one real drug, not a genuine finding"
)


# Secondary analysis (b): for CIViC's COMBINATION pairs specifically (not
# SUBSTITUTES — interchangeable alternatives, a different kind of claim),
# how often does OptimusKG independently also have SYNERGISTIC_INTERACTION?
def _has_combination(civic_evidence: list) -> bool:
    return any(e["interaction_type"] == "COMBINATION" for e in civic_evidence)


combination_pairs = drug_drug_edges[
    drug_drug_edges["civic_evidence"].apply(_has_combination)
]
combination_with_synergy = combination_pairs[
    combination_pairs["relation"] == "SYNERGISTIC_INTERACTION"
]
print(
    f"[DRUG-DRUG cross-validation] {len(combination_with_synergy)} / "
    f"{len(combination_pairs)} CIViC COMBINATION pairs also have an "
    f"independent OptimusKG SYNERGISTIC_INTERACTION relation"
)

pl.from_pandas(drug_drug_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_drug_drug_{date_str}.parquet"
)
print("\n[namespaced] wrote DRUG-DRUG edge table")

# %%
# ---------------------------------------------------------------------------
# 3. DRUG-DISEASE: directed (source=drug, target=disease, matching
#    DRG-DIS's own storage direction).
# ---------------------------------------------------------------------------

drug_disease_civic = derive_projected_edges(
    evidence_item_nodes, "drug_node_ids", "disease_node_id"
)

drg_dis_path = EDGES_DIR / "drug_disease_edges.parquet"
drg_dis = pl.read_parquet(drg_dis_path).to_pandas()
drg_dis = drg_dis[
    [
        "source",
        "target",
        "relation",
        "highest_clinical_trial_phase",
        "reference_ids",
        "sources",
    ]
]
print(f"\n[setup] loaded {len(drg_dis)} OptimusKG DRG-DIS edges from {drg_dis_path}")

drug_disease_edges = merge_namespaced(drug_disease_civic, drg_dis)
drug_disease_edges["undirected"] = False

n_relation = drug_disease_edges["relation"].notna().sum()
n_civic = (drug_disease_edges["n_evidence_items"] > 0).sum()
n_contraindication_with_civic = (
    (drug_disease_edges["relation"] == "CONTRAINDICATION")
    & (drug_disease_edges["n_evidence_items"] > 0)
).sum()
print(
    f"[DRUG-DISEASE] {len(drug_disease_edges)} total edges: {n_relation} have "
    f"an OptimusKG DRG-DIS relation, {n_civic} have CIViC evidence "
    f"({n_contraindication_with_civic} of those sit on a CONTRAINDICATION "
    f"relation — worth a look, since CIViC evidence there is a candidate a "
    f"predictive/therapeutic claim, categorically different from "
    f"'this drug should not be used for this disease')"
)

pl.from_pandas(drug_disease_edges).write_parquet(
    PROCESSED_DIR / f"graph_edges_drug_disease_{date_str}.parquet"
)
print("[namespaced] wrote DRUG-DISEASE edge table")

# %%
