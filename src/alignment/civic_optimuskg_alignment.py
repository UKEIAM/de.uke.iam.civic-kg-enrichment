#!/usr/bin/env python3
"""
Resolves CIViC evidence items' disease, gene/molecular profile and drug
fields against OptimusKG nodes, scoped to a cancer-relevant subset (disease = cancer).

Practical observations: OptimusKG's own precomputed hierarchy
convenience fields (`descendants`, `ancestors`, `parents`, `children`) are
not reliable enough to trust directly. The cancer
scope below is instead computed from the external MONDO ontology's own
`is_a` hierarchy directly, unioned with an OncoTree-derived signal and with
CIViC's items.

CIViC CSV note: disease_doid is stored as a bare number (no "DOID:" prefix) —
the prefix is constructed explicitly below before any lookup.

Inputs:
    - CIViC evidence items CSV (output from fetch_civic_data.py)
    - OptimusKG disease, gene, drug nodes parquet files (output from save_optimuskg_nodes.py)
    - MONDO ontology (fetched live from OBO)
    - HGNC complete set (fetched live from HGNC)
Outputs:
    - data/processed/civic_resolved_YYYY-MM-DD.csv: CIViC evidence items with resolved disease, gene, and drug node ids, plus resolution metadata.
    - data/processed/civic_predictive_unresolved_drugs_YYYY-MM-DD.csv: PREDICTIVE evidence items with at least one unresolved drug slot, categorized by why.

Companion script:
    - civic_clinvar_gap_analysis.py picks up from this script's civic_resolved_*.csv
      output to cross-reference against ClinVar and find curation gaps —
      not part of this script's job (OptimusKG resolution), kept separate.

Usage:
    python src/alignment/civic_optimuskg_alignment.py
"""

# %%
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import obonet
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

# Paths and constants
NODES_DIR = Path("data/raw/optimuskg/nodes")
CIVIC_RAW_PATH = find_latest_file(Path("data/raw/civic"), "civic_raw_*.csv")
OUTPUT_DIR = Path("data/processed")
os.makedirs(OUTPUT_DIR.parent, exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# 1. Load OptimusKG disease nodes from parquet file (output from save_optimuskg_nodes.py)
# ---------------------------------------------------------------------------

CANCER_ROOT = "MONDO_0004992"  # "cancer" (malignant neoplasm)
CANCER_ROOT_COLON = "MONDO:0004992"  # same term, MONDO's own colon format
CANCER_THERAPEUTIC_AREA = (
    "MONDO_0045024"  # "cancer therapeutic area" (OncoTree-derived)
)

diseases = pl.read_parquet(Path.joinpath(NODES_DIR, "disease.parquet"))
diseases_id_set = set(diseases["id"].to_list())  # unique disease node ids in OptimusKG

# %%
# ---------------------------------------------------------------------------
# 2. Load external MONDO ontology (authoritative source; do not rely on
#    OptimusKG's own precomputed ancestors/descendants fields)
# ---------------------------------------------------------------------------

print("[setup] loading MONDO ontology (one-time, may take a moment)...")
mondo = obonet.read_obo("https://purl.obolibrary.org/obo/mondo.obo")

# Clean is_a-only subgraph, used both for scope and for the resolution fallback
is_a_graph = nx.DiGraph()
for u, v, key in mondo.edges(keys=True):
    if key == "is_a":
        is_a_graph.add_edge(u, v)

# %%
# ---------------------------------------------------------------------------
# 3. Cancer-relevant disease SCOPE (cancer relevance filter + network-analysis
#    boundary). This is not a hard pruning boundary for enrichment/gap-detection.
#    Computed as a three-way union:
#      a) authoritative MONDO is_a descendants of "cancer"
#      b) therapeutic_areas containing the OncoTree-derived cancer code
#      c) any disease node CIViC evidence items actually resolves to
# ---------------------------------------------------------------------------

cancer_mondo_descendants = nx.ancestors(is_a_graph, CANCER_ROOT_COLON) | {
    CANCER_ROOT_COLON
}
cancer_mondo_underscore = {d.replace(":", "_") for d in cancer_mondo_descendants}
cancer_scope_mondo = cancer_mondo_underscore & diseases_id_set

cancer_scope_therapeutic_area = set(
    diseases.filter(
        diseases["therapeutic_areas"].list.contains(CANCER_THERAPEUTIC_AREA)
    )["id"].to_list()
)

cancer_disease_ids = cancer_scope_mondo | cancer_scope_therapeutic_area
print(
    f"[scope] {len(cancer_disease_ids)} cancer-relevant disease nodes before CIViC union "
    f"({len(cancer_scope_mondo)} via MONDO is_a, {len(cancer_scope_therapeutic_area)} via therapeutic_areas)"
)

# %%
# ---------------------------------------------------------------------------
# 4. Disease resolution
#    Priority order:
#      1. CIViC's own disease_mondo_id field, direct match against OptimusKG node ids
#      2. same disease_mondo_id, hop-fallback up MONDO's is_a hierarchy
#         (max 3 hops)
#      3. disease_doid (bare number in the civic_raw CSV -> "DOID:<n>"), two-tier:
#         a) OptimusKG disease nodes' own xrefs fields
#         b) external MONDO ontology's own DOID xrefs
# ---------------------------------------------------------------------------

# Tier 3a lookup: OptimusKG disease nodes' own xrefs -> node id
xref_table = diseases.select(["id", "xrefs"]).explode("xrefs").drop_nulls()
dupes = (
    xref_table.group_by("xrefs")
    .agg(pl.col("id").n_unique().alias("n_nodes"))
    .filter(pl.col("n_nodes") > 1)
)
if dupes.height > 0:
    print(f"[warning] {dupes.height} xref values map to multiple disease nodes")
xref_to_node = dict(zip(xref_table["xrefs"], xref_table["id"]))

# Tier 3b lookup: external MONDO ontology's own DOID xrefs -> MONDO id
doid_to_mondo = {}
for node_id, data in mondo.nodes(data=True):
    for xref in data.get("xref", []):
        if xref.startswith("DOID:"):
            doid_to_mondo[xref] = node_id


def find_resolvable_ancestor(
    mondo_colon_id: str, max_hops: int = 5
) -> tuple[str | None, int]:
    """Walk up MONDO's is_a hierarchy from mondo_colon_id until hitting a
    term that exists as an OptimusKG node. Returns (node_id_or_None, hops)."""
    current_ids = {mondo_colon_id}
    visited = set()
    for hop in range(1, max_hops + 1):
        next_ids = set()
        for cid in current_ids:
            if cid in visited:
                continue
            visited.add(cid)
            for _, parent in is_a_graph.out_edges(cid):
                next_ids.add(parent)
        for parent in next_ids:
            parent_underscore = parent.replace(":", "_")
            if parent_underscore in diseases_id_set:
                return parent_underscore, hop
        if not next_ids:
            return None, hop
        current_ids = next_ids
    return None, max_hops


def resolve_doid(doid: str) -> dict:
    node_id = xref_to_node.get(doid)
    if node_id:
        return {"node_id": node_id, "resolved_via": "direct_xref", "resolution_hops": 0}

    mondo_id = doid_to_mondo.get(doid)
    if mondo_id:
        optimuskg_id = mondo_id.replace(":", "_")
        if optimuskg_id in diseases_id_set:
            return {
                "node_id": optimuskg_id,
                "resolved_via": "mondo_hub",
                "resolution_hops": 0,
            }
        # A valid MONDO id was found via the external crosswalk, but it isn't
        # itself an OptimusKG node — try the same hop-fallback used for
        # CIViC's own disease_mondo_id, rather than giving up here.
        ancestor_id, hops = find_resolvable_ancestor(mondo_id)
        if ancestor_id:
            return {
                "node_id": ancestor_id,
                "resolved_via": "mondo_hub_parent_fallback",
                "resolution_hops": hops,
            }

    return {"node_id": None, "resolved_via": "unresolved", "resolution_hops": None}


def resolve_civic_disease_row(row: pd.Series) -> dict:
    mondo_id_colon = row["disease_mondo_id"]
    doid_raw = row["disease_doid"]

    if pd.notna(mondo_id_colon):
        node_id = str(mondo_id_colon).replace(":", "_")
        if node_id in diseases_id_set:
            return {
                "disease_node_id": node_id,
                "disease_resolved_via": "civic_mondo_id",
                "resolution_hops": 0,
            }

        ancestor_id, hops = find_resolvable_ancestor(str(mondo_id_colon))
        if ancestor_id:
            return {
                "disease_node_id": ancestor_id,
                "disease_resolved_via": "mondo_parent_fallback",
                "resolution_hops": hops,
            }

    if pd.notna(doid_raw):
        result = resolve_doid(f"DOID:{int(doid_raw)}")
        if result["node_id"]:
            return {
                "disease_node_id": result["node_id"],
                "disease_resolved_via": result["resolved_via"],
                "resolution_hops": result["resolution_hops"],
            }
    if pd.isna(mondo_id_colon) and pd.isna(doid_raw):
        return {
            "disease_node_id": None,
            "disease_resolved_via": "no_disease_identifier",
            "resolution_hops": None,
        }
    if pd.notna(mondo_id_colon):
        return {
            "disease_node_id": None,
            "disease_resolved_via": "civic_mondo_id_not_in_optimuskg",
            "resolution_hops": None,
        }
    return {
        "disease_node_id": None,
        "disease_resolved_via": "unresolved",
        "resolution_hops": None,
    }


# %%
# ---------------------------------------------------------------------------
# 5. Gene resolution
# - OptimusKG gene node primary ids are a mix of Ensembl (61,108 nodes) and NCBIGene (198
#    nodes). Two independent, confirmed working paths:
#      1. hgnc_hub: CIViC Entrez id -> HGNC id (via HGNC's own complete set)
#         -> OptimusKG gene node (via that node's own xrefs, source="HGNC")
#      2. direct_ensembl: CIViC Entrez id -> Ensembl id (via the same HGNC
#         complete set) -> matched directly against OptimusKG gene node ids
#      3. symbol_match: CIViC gene symbol -> OptimusKG gene node symbol, with
#         symbol_synonyms and obsolete_symbols as fallbacks
# ---------------------------------------------------------------------------

genes = pl.read_parquet(Path.joinpath(NODES_DIR, "gene.parquet"))
genes_id_set = set(genes["id"].to_list())

# OptimusKG's own HGNC -> gene node lookup, built from gene xrefs (struct
# fields confirmed as {id, source}
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

# HGNC's own authoritative complete set — external hub, used to bridge CIViC's Entrez ids to OptimusKG gene nodes
hgnc_complete = pd.read_csv(
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    sep="\t",
    dtype=str,
)
# HGNC's own hgnc_id carries a "HGNC:" prefix; OptimusKG's xrefs store the
# bare number — strip it explicitly rather than assuming a match as-is.
hgnc_pairs = hgnc_complete.dropna(subset=["entrez_id", "hgnc_id"])
entrez_to_hgnc_bare = dict(
    zip(hgnc_pairs["entrez_id"], hgnc_pairs["hgnc_id"].str.replace("HGNC:", ""))
)

hgnc_ensembl_pairs = hgnc_complete.dropna(subset=["entrez_id", "ensembl_gene_id"])
entrez_to_ensembl = dict(
    zip(hgnc_ensembl_pairs["entrez_id"], hgnc_ensembl_pairs["ensembl_gene_id"])
)

# Tier 3: direct match against OptimusKG's own symbol column, with
# symbol_synonyms/obsolete_symbols as further fallbacks.
symbol_to_node = dict(zip(genes["symbol"], genes["id"]))

synonym_to_node = {}
for gene_id, synonyms in zip(genes["id"], genes["symbol_synonyms"].to_list()):
    for syn in synonyms or []:
        synonym_to_node.setdefault(syn["label"], gene_id)
for gene_id, obsolete in zip(genes["id"], genes["obsolete_symbols"].to_list()):
    for sym in obsolete or []:
        synonym_to_node.setdefault(sym["label"], gene_id)


def resolve_civic_gene(entrez_id: str, symbol: str | None = None) -> dict:
    hgnc_bare = entrez_to_hgnc_bare.get(str(entrez_id))
    if hgnc_bare:
        node_id = hgnc_to_node.get(hgnc_bare)
        if node_id:
            return {"gene_node_id": node_id, "gene_resolved_via": "hgnc_hub"}

    ensembl_id = entrez_to_ensembl.get(str(entrez_id))
    if ensembl_id and ensembl_id in genes_id_set:
        return {"gene_node_id": ensembl_id, "gene_resolved_via": "direct_ensembl"}

    if symbol:
        node_id = symbol_to_node.get(symbol) or synonym_to_node.get(symbol)
        if node_id:
            return {"gene_node_id": node_id, "gene_resolved_via": "symbol_match"}

    return {"gene_node_id": None, "gene_resolved_via": "unresolved"}


def resolve_civic_gene_row(row: pd.Series) -> dict:
    entrez_ids_raw = row.get("variant_feature_entrez_ids")
    if pd.isna(entrez_ids_raw):
        return {
            "gene_node_ids": [],
            "gene_resolved_via_list": [],
            "n_genes_total": 0,
            "n_genes_resolved": 0,
            "gene_all_resolved": False,
            "gene_any_resolved": False,
        }

    raw_entrez_list = [e.strip() for e in str(entrez_ids_raw).split(",") if e.strip()]

    names_raw = row.get("variant_feature_names")
    raw_symbols = (
        [s.strip() for s in str(names_raw).split(",")] if pd.notna(names_raw) else []
    )
    raw_symbols += [None] * (len(raw_entrez_list) - len(raw_symbols))

    seen = {}
    for entrez_id, symbol in zip(raw_entrez_list, raw_symbols):
        if entrez_id not in seen:
            seen[entrez_id] = symbol
    unique_entrez_ids = list(seen.keys())

    node_ids, resolved_via_list = [], []
    for entrez_id in unique_entrez_ids:
        result = resolve_civic_gene(entrez_id, symbol=seen[entrez_id])
        node_ids.append(result["gene_node_id"])
        resolved_via_list.append(result["gene_resolved_via"])

    n_resolved = sum(1 for n in node_ids if n is not None)
    return {
        "gene_node_ids": node_ids,  # one entry per DISTINCT gene, None where unresolved
        "gene_resolved_via_list": resolved_via_list,
        "n_genes_total": len(
            unique_entrez_ids
        ),  # distinct genes, not variant-record count
        "n_genes_resolved": n_resolved,
        "gene_all_resolved": n_resolved == len(unique_entrez_ids)
        and len(unique_entrez_ids) > 0,
        "gene_any_resolved": n_resolved > 0,
    }


# %%
# ---------------------------------------------------------------------------
# 6. Drug resolution
#    - No xrefs field on drug nodes (unlike disease/gene).
#    - Ids = DRUGBANK (1,227 nodes, OptimsKG format = "DRUGBANK:DB00001"; CIViC = "DB00530")
#    and CHEMBL (14,557 nodes, OptimsKG format = "CHEMBL1000"; CIViC = "CHEMBL1000"), no
#    transformation needed). InChIKey exists on both sides but is null for
#    biologics/antibody therapies — will not be used.
#    therapy_drugbank_id/therapy_chembl_id can each hold multiple
#    comma-separated values per row (real combination therapies), so both fields are
#    parsed position-aligned per row, not just the first value.
# ---------------------------------------------------------------------------
drugs = pl.read_parquet(Path.joinpath(NODES_DIR, "drug.parquet"))
drugs_id_set = set(drugs["id"].to_list())


# Name-based fallback tier: some CIViC rows carry a populated drugbank_id or
# chembl_id that still doesn't match any OptimusKG node
# NOTE: many unresolved rows that are NOT
# individual compounds at all — drug classes ("PD1 Inhibitor"), molecular
# targets ("Anti-CD33"), non-pharmacological procedures ("Cytoreductive
# Surgery", "Ionizing Radiotherapy"), and named multi-drug regimens ("FAC
# Regimen"). No crosswalk can resolve these; will be reported as a
# separate, not counted as a resolution failure alongside genuine gaps.
def _normalize_name(s: str) -> str:
    if s:
        return s.strip().lower()


# Case/whitespace-normalized lookups
name_to_node = {_normalize_name(n): i for n, i in zip(drugs["name"], drugs["id"])}
synonym_to_drug_node = {}
for drug_id, synonyms in zip(drugs["id"], drugs["synonyms"].to_list()):
    for syn in synonyms or []:
        synonym_to_drug_node.setdefault(_normalize_name(syn), drug_id)
for drug_id, trade_names in zip(drugs["id"], drugs["trade_names"].to_list()):
    for tn in trade_names or []:
        synonym_to_drug_node.setdefault(_normalize_name(tn), drug_id)


def resolve_civic_drug(
    drugbank_id: str | None, chembl_id: str | None, name: str | None = None
) -> dict:
    if drugbank_id:
        candidate = f"DRUGBANK:{drugbank_id}"
        if candidate in drugs_id_set:
            return {"drug_node_id": candidate, "drug_resolved_via": "drugbank_id"}
    if chembl_id and chembl_id in drugs_id_set:
        return {"drug_node_id": chembl_id, "drug_resolved_via": "chembl_id"}
    if name:
        normalized = _normalize_name(name)
        node_id = name_to_node.get(normalized) or synonym_to_drug_node.get(normalized)
        if node_id:
            return {"drug_node_id": node_id, "drug_resolved_via": "name_match"}
        # Targeted fallback: some CIViC therapy names embed a class-descriptor
        # prefix ahead of the actual compound code as one unsplit string
        tokens = name.split()
        if tokens:
            last_token = _normalize_name(tokens[-1])
            node_id = name_to_node.get(last_token) or synonym_to_drug_node.get(
                last_token
            )
            if node_id:
                return {
                    "drug_node_id": node_id,
                    "drug_resolved_via": "name_match_last_token",
                }
    return {"drug_node_id": None, "drug_resolved_via": "unresolved"}


def resolve_civic_drug_row(row: pd.Series) -> dict:
    drugbank_raw = row.get("therapy_drugbank_id")
    chembl_raw = row.get("therapy_chembl_id")
    name_raw = row.get("therapy_name")

    drugbank_list = (
        [d.strip() for d in str(drugbank_raw).split(",") if d.strip()]
        if pd.notna(drugbank_raw)
        else []
    )
    chembl_list = (
        [c.strip() for c in str(chembl_raw).split(",") if c.strip()]
        if pd.notna(chembl_raw)
        else []
    )
    name_list = (
        [n.strip() for n in str(name_raw).split(",") if n.strip()]
        if pd.notna(name_raw)
        else []
    )

    n = max(len(drugbank_list), len(chembl_list), len(name_list))
    if n == 0:
        return {
            "drug_node_ids": [],
            "drug_resolved_via_list": [],
            "n_drugs_total": 0,
            "n_drugs_resolved": 0,
            "drug_all_resolved": False,
            "drug_any_resolved": False,
        }

    # Confirmed that CIViC's therapy_drugbank_id and therapy_chembl_id fields are position-aligned
    drugbank_list += [None] * (n - len(drugbank_list))
    chembl_list += [None] * (n - len(chembl_list))
    name_list += [None] * (n - len(name_list))

    node_ids, resolved_via_list = [], []
    seen_triples = set()
    for db_id, chembl_id, name in zip(drugbank_list, chembl_list, name_list):
        triple_key = (db_id, chembl_id, name)
        if triple_key in seen_triples:
            continue  # same dedup safety used for repeated gene entries
        seen_triples.add(triple_key)
        result = resolve_civic_drug(db_id, chembl_id, name)
        node_ids.append(result["drug_node_id"])
        resolved_via_list.append(result["drug_resolved_via"])

    n_resolved = sum(1 for n_id in node_ids if n_id is not None)
    return {
        "drug_node_ids": node_ids,
        "drug_resolved_via_list": resolved_via_list,
        "n_drugs_total": len(node_ids),
        "n_drugs_resolved": n_resolved,
        "drug_all_resolved": n_resolved == len(node_ids) and len(node_ids) > 0,
        "drug_any_resolved": n_resolved > 0,
    }


# %%
# ---------------------------------------------------------------------------
# 7. Apply across the full CIViC dataframe
# ---------------------------------------------------------------------------

civic_df = pd.read_csv(CIVIC_RAW_PATH)

disease_results = civic_df.apply(
    resolve_civic_disease_row, axis=1, result_type="expand"
)
gene_results = civic_df.apply(resolve_civic_gene_row, axis=1, result_type="expand")
drug_results = civic_df.apply(resolve_civic_drug_row, axis=1, result_type="expand")

civic_df = pd.concat([civic_df, disease_results, gene_results, drug_results], axis=1)

# Finalize scope: union in every disease CIViC actually resolved to, even if not in the MONDO/therapeutic_area scope
resolved_civic_diseases = set(civic_df["disease_node_id"].dropna().unique())
cancer_disease_ids_final = cancer_disease_ids | resolved_civic_diseases
print(
    f"[scope] {len(cancer_disease_ids_final)} cancer-relevant disease nodes after CIViC union "
    f"(+{len(resolved_civic_diseases - cancer_disease_ids)} from CIViC not already in scope)"
)

civic_df["disease_in_cancer_scope"] = civic_df["disease_node_id"].apply(
    lambda x: x in cancer_disease_ids_final if pd.notna(x) else None
)

# Drug resolution bar depends on therapy_interaction_type: COMBINATION (and
# anything else/unlabeled) = drugs used together, so ALL must resolve to be meaningful.
# SUBSTITUTES = drugs are interchangeable (evidence is still usable if just one resolves).
# SEQUENTIAL= grouped with COMBINATION rather than getting its own case, since a
# defined drug sequence is still a claim about a specific, complete regimen.
civic_df["drug_resolution_ok"] = civic_df.apply(
    lambda r: (
        True
        if r.get("evidence_type") != "PREDICTIVE"
        else (
            r["drug_any_resolved"]
            if r.get("therapy_interaction_type") == "SUBSTITUTES"
            else r["drug_all_resolved"]
        )
    ),
    axis=1,
)

# fully_resolved =  disease + every gene in the molecular profile +
# the drug resolution bar above (automatically satisfied for
# non-PREDICTIVE evidence types)
civic_df["fully_resolved"] = (
    civic_df["disease_node_id"].notna()
    & civic_df["gene_all_resolved"]
    & civic_df["drug_resolution_ok"]
)

# %%
# ---------------------------------------------------------------------------
# 8. Coverage report
# ---------------------------------------------------------------------------
n_total = len(civic_df)
print(f"\n[coverage] {n_total} total CIViC evidence items")
print(
    f"  disease resolved: {civic_df['disease_node_id'].notna().sum()} "
    f"({civic_df['disease_node_id'].notna().mean():.1%})"
)
print("  disease resolved_via breakdown:")
print(civic_df["disease_resolved_via"].value_counts().to_string())
print(
    "  resolution_hops distribution (mondo_parent_fallback and mondo_hub_parent_fallback cases):"
)
print(
    civic_df.loc[
        civic_df["disease_resolved_via"].isin(
            ["mondo_parent_fallback", "mondo_hub_parent_fallback"]
        ),
        "resolution_hops",
    ]
    .value_counts()
    .sort_index()
    .to_string()
)
print(
    f"  gene: all genes resolved: {civic_df['gene_all_resolved'].sum()} "
    f"({civic_df['gene_all_resolved'].mean():.1%})"
)
print(
    f"  gene: at least one gene resolved: {civic_df['gene_any_resolved'].sum()} "
    f"({civic_df['gene_any_resolved'].mean():.1%})"
)
print(
    f"  multi-gene profiles (n_genes_total > 1): {(civic_df['n_genes_total'] > 1).sum()}"
)
# Drug are scoped to PREDICTIVE evidence items only — the other five
# evidence_type values don't concern a therapy at all, so including them in
# a drug-resolution rate would misleadingly deflate it
predictive_df = civic_df[civic_df["evidence_type"] == "PREDICTIVE"]
n_predictive = len(predictive_df)
print(
    f"  PREDICTIVE evidence items: {n_predictive} (drug resolution only applies to these)"
)
print(
    f"  drug (PREDICTIVE only): all drugs resolved: {predictive_df['drug_all_resolved'].sum()} "
    f"({predictive_df['drug_all_resolved'].mean():.1%})"
)
print(
    f"  drug (PREDICTIVE only): at least one drug resolved: {predictive_df['drug_any_resolved'].sum()} "
    f"({predictive_df['drug_any_resolved'].mean():.1%})"
)
print(f"  multi-drug rows (n_drugs_total > 1): {(civic_df['n_drugs_total'] > 1).sum()}")
print(
    f"  fully resolved (all three, drug bar auto-satisfied for non-PREDICTIVE): "
    f"{civic_df['fully_resolved'].sum()} ({civic_df['fully_resolved'].mean():.1%})"
)

# %%
# ---------------------------------------------------------------------------
# 9. Categorize residual unresolved drug slots for reporting.
#    Two categories, kept distinct: STRUCTURALLY_NOT_SINGLE_COMPOUND (a class,
#    molecular target, procedure, or named regimen — no crosswalk could ever
#    resolve these) vs OUT_OF_OPTIMUSKG_SCOPE (genuine coverage gap, not a
#    matching failure). Anything that doesn't cleanly fit either of those is flagged
#    NEEDS_MANUAL_REVIEW rather than forced into one of the two.
# ---------------------------------------------------------------------------

import re

NON_COMPOUND_KEYWORDS = [
    "inhibitor",
    "inhibition",
    "antibody",
    "therapy",
    "regimen",
    "surgery",
    "radiotherapy",
    "chemotherapy",
    "compound",
    "oligonucleotide",
]
# Rough compound-code pattern: letters mixed with digits, optional hyphen
# (e.g. "TAE684", "PP242", "GDC-0623", "KU-0060648", "JQ1")
CODE_LIKE_PATTERN = re.compile(r"[A-Za-z]{1,6}-?\d{1,7}")


def _loose_search(term: str, drugs_df) -> list:
    term_lower = term.lower()
    hits = []
    for row in drugs_df.iter_rows(named=True):
        candidates = (
            [row["name"]] + (row["synonyms"] or []) + (row["trade_names"] or [])
        )
        if any(term_lower in (c or "").lower() for c in candidates):
            hits.append((row["id"], row["name"]))
    return hits


def classify_unresolved_slot(slot_name: str) -> str:
    has_code = CODE_LIKE_PATTERN.search(slot_name)
    has_class_keyword = any(kw in slot_name.lower() for kw in NON_COMPOUND_KEYWORDS)

    if has_code:
        hits = _loose_search(has_code.group(0), drugs)
        if not hits:
            return "out_of_optimuskg_scope"
        return "needs_manual_review"  # exists under some form but still didn't match via the normal tiers
    if has_class_keyword:
        return "structurally_not_single_compound"
    return "needs_manual_review"


def classify_unresolved_row(row) -> list:
    name_raw = row.get("therapy_name")
    if pd.isna(name_raw):
        return []
    names = [n.strip() for n in str(name_raw).split(",") if n.strip()]
    resolved_via = row.get("drug_resolved_via_list", [])
    return [
        classify_unresolved_slot(name) if via == "unresolved" else "resolved"
        for name, via in zip(names, resolved_via)
    ]


predictive_df = predictive_df.copy()
predictive_df["drug_unresolved_categories"] = predictive_df.apply(
    classify_unresolved_row, axis=1
)

from collections import Counter

category_counts = Counter(
    cat
    for cats in predictive_df["drug_unresolved_categories"]
    for cat in cats
    if cat != "resolved"
)

print("\n[unresolved drug slot categories]")
print(category_counts)

# Subset of predictive_df containing only rows with at least one unresolved drug slot for later inspection
unresolved_drugs_predictive_df = predictive_df[
    predictive_df["drug_unresolved_categories"].apply(
        lambda cats: any(
            cat != "resolved" for cat in cats
        )  # corresponds to ['out_of_optimuskg_scope', 'structurally_not_single_compound', 'needs_manual_review'])
    )
]

# Removing description column to avoid cluttering the output CSV with long text fields
unresolved_drugs_predictive_df = unresolved_drugs_predictive_df.drop(
    ["description"], axis=1
)

# %%
# Save the resolved dataset to a timestamped CSV file
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
output_path = OUTPUT_DIR / f"civic_resolved_{date_str}.csv"
civic_df.to_csv(output_path, index=False)
print(f"\nSaved resolved dataset to {output_path}")

# save cancer-relevant disease ids to a separate CSV for reference
cancer_disease_ids_output_path = (
    OUTPUT_DIR / f"cancer_relevant_disease_ids_{date_str}.csv"
)
pd.DataFrame({"disease_node_id": list(cancer_disease_ids_final)}).to_csv(
    cancer_disease_ids_output_path, index=False
)
print(f"Saved cancer-relevant disease ids to {cancer_disease_ids_output_path}")

# Optional:
# # Save predictive subset to a separate CSV file for easier downstream analysis
# predictive_output_path = OUTPUT_DIR / f"civic_resolved_predictive_subset_{date_str}.csv"
# predictive_df.to_csv(predictive_output_path, index=False)
# print(f"Saved predictive subset to {predictive_output_path}")

# Save list of drugs that were not resolved to a separate CSV for reporting
unresolved_drugs_output_path = (
    OUTPUT_DIR / f"civic_predictive_unresolved_drugs_{date_str}.csv"
)
unresolved_drugs_predictive_df.to_csv(unresolved_drugs_output_path, index=False)
print(f"✅Saved unresolved drugs to {unresolved_drugs_output_path}")
