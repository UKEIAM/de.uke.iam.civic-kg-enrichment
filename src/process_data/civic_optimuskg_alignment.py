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

Usage:
    python src/process_data/civic_optimuskg_alignment.py
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
CLINVAR_RAW_PATH = find_latest_file(
    Path("data/raw/clinvar"), "clinvar_variant_summary_*.tsv"
)
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
#    nodes) — no reliable direct-Entrez match. Two independent, confirmed
#    working paths, tried in order and combined since they fail on different rows:
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

# %%
# ---------------------------------------------------------------------------
# 10a. Multi-tier ClinVar VariationID resolution for CIViC evidence items,
#    and resulting coverage by evidence_type.
#    variant_clinvar_id (singular) alone under-counts CIViC's real ClinVar
#    coverage. All three tiers (variant_clinvar_id, variant_clinvar_ids, variant_dbsnp_rsids)
#    are combined into one set of VariationIDs per evidence item, used for ALL matching in
#    sections 10-12, reducing false "gap" positives caused by missing crosswalk metadata rather
#    than missing curation. HGVS-string matching against ClinVar's own Name column
#    remains out of scope.
# ---------------------------------------------------------------------------
# ClinVar summary file (output from src/fetch_data/fetch_clinvar_data.py)
clinvar_summary = pd.read_csv(CLINVAR_RAW_PATH, sep="\t", dtype=str)

# Deduplicate variants by VariationID, keeping the first row for each variant (different genome versions)
clinvar_summary = clinvar_summary.drop_duplicates(subset="VariationID", keep="first")
print(
    f"[verify] clinvar_summary rows after de-duplicating by VariationID: {len(clinvar_summary)}"
)

# dbSNP rsID -> ClinVar VariationID, bare-number keyed
rsid_to_variationid = dict(
    zip(
        clinvar_summary.loc[clinvar_summary["RS# (dbSNP)"] != "-1", "RS# (dbSNP)"],
        clinvar_summary.loc[clinvar_summary["RS# (dbSNP)"] != "-1", "VariationID"],
    )
)


def get_civic_clinvar_variation_ids(row: pd.Series) -> set:
    """Collect every ClinVar VariationID associated with a CIViC evidence
    item, combining three independent sources — none of which is complete
    on its own."""
    ids = set()

    def _add_if_numeric(candidate: str):
        # Validating numeric-only entries here, since CIViC's variant_clinvar_id and
        # variant_clinvar_ids columns are free-text and can contain non-numeric junk
        # (e.g. "N/A", "unknown", "multiple", "see description", etc.) that would
        # otherwise throw off the coverage counts.
        candidate = candidate.strip()
        if candidate.isdigit():
            ids.add(candidate)

    # Tier 0a: singular column
    # Split on comma to handle CIViC's multi-ClinVar entries, which are not consistently
    # captured in the plural column (variant_clinvar_ids)
    singular_raw = row.get("variant_clinvar_id")
    if pd.notna(singular_raw):
        for v in str(singular_raw).split(","):
            _add_if_numeric(v)

    # Tier 0b: plural column
    plural_raw = row.get("variant_clinvar_ids")
    if pd.notna(plural_raw):
        for v in str(plural_raw).split(","):
            _add_if_numeric(v)

    # Tier 1: dbSNP rsID -> ClinVar VariationID, via ClinVar's own file.
    # Strip the "rs" prefix CIViC uses that ClinVar's bare-number column
    # doesn't carry.
    dbsnp_raw = row.get("variant_dbsnp_rsids")
    if pd.notna(dbsnp_raw):
        for rsid in str(dbsnp_raw).split(","):
            bare_rsid = rsid.strip().lower().replace("rs", "")
            variation_id = rsid_to_variationid.get(bare_rsid)
            if variation_id:
                ids.add(str(variation_id))

    return ids


civic_df["clinvar_variation_ids_all"] = civic_df.apply(
    get_civic_clinvar_variation_ids, axis=1
)

# Coverage by evidence_type, using the combined multi-tier set
# Where coverage is still low even after combining all three
# tiers, a "gap candidate" found by ID-matching cannot be distinguished from
# a variant CIViC has ALREADY curated under an evidence item whose ClinVar
# reference simply wasn't captured by any of these three tiers. Gap counts
# for low-coverage evidence types should be read as upper bounds on the true
# curation gap, not confirmed gaps.
clinvar_id_coverage = (
    civic_df.groupby("evidence_type")["clinvar_variation_ids_all"]
    .apply(lambda s: s.apply(lambda ids: len(ids) > 0).mean())
    .sort_values(ascending=False)
)
print("[clinvar coverage] combined (multi-tier) ClinVar match rate by evidence_type:")
print(clinvar_id_coverage.to_string())
print(
    "NOTE: even after combining variant_clinvar_id, variant_clinvar_ids, and "
    "dbSNP-based matching, low coverage for ONCOGENIC/PREDICTIVE/DIAGNOSTIC/"
    "PROGNOSTIC means gap candidates likely still overestimate "
    "the true curation gap."
)


# %%
# ---------------------------------------------------------------------------
# 10. Predisposing-variant gap detection via ClinVar
#    CIViC = Predisposing; Clinvar = Pathogenic/likely pathogenic
#    Scope: genes already resolved from at least one CIViC evidence item
#    ("aligned genes"). A gap candidate is a ClinVar pathogenic/likely-pathogenic
#    variant on an aligned gene with NO matching CIViC PREDISPOSING evidence item.
#    This is the only evidence type that directly corresponds to ClinVar's
#    pathogenicity classifications.
#    Matching is by ClinVar VariationID via
#    CIViC's own variant_clinvar_id column where populated; PREDISPOSING
#    items lacking variant_clinvar_id can't be matched this way and are
#    reported separately as a residual limitation, not resolved via
#    HGVS/coordinate matching.
# ---------------------------------------------------------------------------

# 1. Establish "aligned genes"
# Aligned gene symbols: genes with at least one resolved CIViC evidence item
aligned_gene_node_ids = {
    gid for ids in civic_df["gene_node_ids"] for gid in (ids or []) if gid is not None
}
aligned_gene_symbols = set(
    genes.filter(genes["id"].is_in(aligned_gene_node_ids))["symbol"].to_list()
)
print(f"[clinvar gap] {len(aligned_gene_symbols)} aligned gene symbols")

# Aligned gene Entrez ids
ensembl_to_entrez = dict(
    zip(hgnc_ensembl_pairs["ensembl_gene_id"], hgnc_ensembl_pairs["entrez_id"])
)
aligned_entrez_ids = set()
for node_id in aligned_gene_node_ids:
    if node_id in ensembl_to_entrez:
        aligned_entrez_ids.add(ensembl_to_entrez[node_id])
    elif node_id.startswith("NCBIGene:"):
        aligned_entrez_ids.add(node_id.split(":", 1)[1])
print(
    f"[clinvar gap] {len(aligned_entrez_ids)} aligned gene Entrez ids "
    f"(of {len(aligned_gene_node_ids)} aligned OptimusKG gene nodes — some "
    f"may be unmapped if neither the Ensembl nor NCBIGene path applied)"
)


# 2. Define what counts as "pathogenic"
def is_pathogenic(clinical_significance) -> bool:
    if pd.isna(clinical_significance):
        return False
    # Case-sensitive: ClinVar capitalizes "Pathogenic"/"Likely pathogenic" as
    # asserted classifications. Lowercase "pathogenicity" appears inside
    # "Conflicting interpretations of pathogenicity", which should NOT count
    # as a confirmed pathogenic call — case-sensitivity is what excludes it
    # here.
    return "Pathogenic" in clinical_significance


# 3. Pull ClinVar's own pathogenic variants for the aligned genes
clinvar_pathogenic_aligned = clinvar_summary[
    clinvar_summary["GeneID"].isin(aligned_entrez_ids)
    & clinvar_summary["ClinicalSignificance"].apply(is_pathogenic)
].copy()
clinvar_pathogenic_aligned["VariationID"] = clinvar_pathogenic_aligned[
    "VariationID"
].astype(str)
print(
    f"[clinvar gap] {len(clinvar_pathogenic_aligned)} pathogenic/likely-pathogenic "
    f"ClinVar variants on aligned genes"
)

# 4. Build the set of what CIViC has already covered
# CIViC's own PREDISPOSING ClinVar coverage, using the combined multi-tier
# set from section 10a (singular id + plural ids + dbSNP-based match)
predisposing_df = civic_df[civic_df["evidence_type"] == "PREDISPOSING"].copy()

# count how many Predisposing CIViC items have no matched id at all (residual limitation)
n_predisposing_no_clinvar_id = (
    predisposing_df["clinvar_variation_ids_all"].apply(len).eq(0).sum()
)
print(
    f"[clinvar gap] {n_predisposing_no_clinvar_id} / {len(predisposing_df)} "
    f"PREDISPOSING evidence items have no ClinVar match via any of the three "
    f"tiers"
)

# get every ClinVar id any Predisposing evidence item references
covered_clinvar_ids = {
    vid for ids in predisposing_df["clinvar_variation_ids_all"] for vid in ids
}

print(
    f"[clinvar gap] {len(covered_clinvar_ids)} unique ClinVar IDs are referenced by PREDISPOSING evidence items"
)

# 5. Compute actual gap candidates
# Gap candidates: pathogenic ClinVar variants on aligned genes, not covered
# by any CIViC PREDISPOSING evidence item
clinvar_gap_candidates = clinvar_pathogenic_aligned[
    ~clinvar_pathogenic_aligned["VariationID"].isin(covered_clinvar_ids)
]
print(
    f"[clinvar gap] {len(clinvar_gap_candidates)} gap candidates: pathogenic/"
    f"likely-pathogenic ClinVar variants on aligned genes with no matching "
    f"CIViC PREDISPOSING evidence item"
)
# % of ClinVar's pathogenic variants on aligned genes that CIViC's
# Predisposing evidence already references
n_predisposing_total = len(clinvar_pathogenic_aligned)
predisposing_coverage_pct = (
    (n_predisposing_total - len(clinvar_gap_candidates)) / n_predisposing_total * 100
    if n_predisposing_total > 0
    else None
)
print(
    f"[clinvar gap] PREDISPOSING coverage: {predisposing_coverage_pct:.1f}% of "
    f"pathogenic/likely-pathogenic ClinVar variants on aligned genes are "
    f"referenced by a CIViC PREDISPOSING evidence item"
)

# %%
# Save outputs
# Aligned with OptimusKG
predisposing_aligned_output_path = (
    OUTPUT_DIR / f"clinvar_predisposing_aligned_variants_{date_str}.csv"
)
clinvar_pathogenic_aligned.to_csv(predisposing_aligned_output_path, index=False)
print(
    f"Saved full aligned pathogenic variant set to {predisposing_aligned_output_path}"
)

# Gap candidates
clinvar_gap_output_path = (
    OUTPUT_DIR / f"clinvar_predisposing_gap_candidates_{date_str}.csv"
)
clinvar_gap_candidates.to_csv(clinvar_gap_output_path, index=False)
print(f"Saved ClinVar gap candidates to {clinvar_gap_output_path}")

# %%
# How many of these gap candidates are high-confidence?
# Stratify by ClinVar's own ReviewStatus (a proxy for its star-rating
# system: 0-star "no assertion criteria provided" up to 3-star "reviewed by
# expert panel") — distinguishes low-confidence, high-volume routine
# classifications (which plausibly explain much of the raw gap count via
# ClinVar's broader submission scope vs. CIViC's literature-curation bar)
# from well-vetted, high-confidence classifications that represent genuine,
# actionable curation opportunities rather than database-scope noise.
review_status_breakdown = clinvar_gap_candidates["ReviewStatus"].value_counts()
print("[clinvar gap] PREDISPOSING gap candidates by ClinVar review status:")
print(review_status_breakdown.to_string())

HIGH_CONFIDENCE_REVIEW_STATUSES = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}
high_confidence_gap_candidates = clinvar_gap_candidates[
    clinvar_gap_candidates["ReviewStatus"].isin(HIGH_CONFIDENCE_REVIEW_STATUSES)
]
print(
    f"[clinvar gap] {len(high_confidence_gap_candidates)} / {len(clinvar_gap_candidates)} "
    f"gap candidates are high-confidence (2-star-or-better ClinVar review status) — "
    f"these are the most clinically actionable curation candidates, as opposed to "
    f"single-submitter or no-criteria entries that more plausibly reflect ClinVar's "
    f"broader submission scope than a genuine high-value curation gap"
)

high_confidence_gap_output_path = (
    OUTPUT_DIR / f"clinvar_predisposing_gap_candidates_highconfidence_{date_str}.csv"
)
high_confidence_gap_candidates.to_csv(high_confidence_gap_output_path, index=False)
print(
    f"Saved high-confidence PREDISPOSING gap candidates to "
    f"{high_confidence_gap_output_path}"
)


# %%
# ---------------------------------------------------------------------------
# 11. Oncogenic-variant gap detection via ClinVar Oncogenicity classification
#    CIViC = Oncogenic; Clinvar = Oncogenic
#    Same structure as section 10, different classification column and CIViC
#    evidence type. IMPORTANT: unlike germline ClinicalSignificance, real
#    (non-"-") Oncogenicity classifications are globally sparse (~3,800 rows
#    across all of ClinVar) — but empirically confirmed to concentrate almost
#    entirely (~78%, 2,981 variants) on genes already aligned with CIViC, a
#    selection effect: the genes CIViC curators prioritize are the same genes
#    clinical labs prioritize for oncogenicity submission. This makes the
#    track worth building despite the low global base rate.
# ---------------------------------------------------------------------------

# 1. Define what counts as oncogenic
ONCOGENIC_POSITIVE_VALUES = {"Oncogenic", "Likely oncogenic"}

# 2. Pull ClinVar's own oncogenic-classified variants for the aligned genes
clinvar_oncogenic_aligned = clinvar_summary[
    clinvar_summary["GeneID"].isin(aligned_entrez_ids)
    & clinvar_summary["Oncogenicity"].isin(ONCOGENIC_POSITIVE_VALUES)
].copy()
clinvar_oncogenic_aligned["VariationID"] = clinvar_oncogenic_aligned[
    "VariationID"
].astype(str)
print(
    f"[clinvar gap] {len(clinvar_oncogenic_aligned)} oncogenic/likely-oncogenic "
    f"ClinVar variants on aligned genes"
)

# 3. Build the set of what CIViC has already covered
oncogenic_df = civic_df[civic_df["evidence_type"] == "ONCOGENIC"].copy()

# count how many Oncogenic CIViC items have no matched id at all (residual limitation)
n_oncogenic_no_clinvar_id = (
    oncogenic_df["clinvar_variation_ids_all"].apply(len).eq(0).sum()
)
print(
    f"[clinvar gap] {n_oncogenic_no_clinvar_id} / {len(oncogenic_df)} "
    f"ONCOGENIC CIViC evidence items have no ClinVar match via any of the three "
    f"tiers"
)

# get every ClinVar id any Oncogenic evidence item references
covered_oncogenic_clinvar_ids = {
    vid for ids in oncogenic_df["clinvar_variation_ids_all"] for vid in ids
}
print(
    f"[clinvar gap] {len(covered_oncogenic_clinvar_ids)} unique ClinVar IDs are referenced by ONCOGENIC evidence items"
)

# 4. Compute the gap and the coverage percentage
# A ClinVar oncogenic variant on an aligned gene is a gap candidate if its VariationID isn't in the covered set
oncogenic_gap_candidates = clinvar_oncogenic_aligned[
    ~clinvar_oncogenic_aligned["VariationID"].isin(covered_oncogenic_clinvar_ids)
]
print(
    f"[clinvar gap] {len(oncogenic_gap_candidates)} gap candidates: oncogenic/"
    f"likely-oncogenic ClinVar variants on aligned genes with no matching "
    f"CIViC ONCOGENIC evidence item"
)
n_oncogenic_total = len(clinvar_oncogenic_aligned)
oncogenic_coverage_pct = (
    (n_oncogenic_total - len(oncogenic_gap_candidates)) / n_oncogenic_total * 100
    if n_oncogenic_total > 0
    else None
)
print(
    f"[clinvar gap] ONCOGENIC coverage: {oncogenic_coverage_pct:.1f}% of "
    f"oncogenic/likely-oncogenic ClinVar variants on aligned genes are "
    f"referenced by a CIViC ONCOGENIC evidence item"
)
# No high-confidence stratification for oncogenic variants, due to the lower coverage (~40%) of ClinVar matches.
# Of all CIViC evidence items with evidence_type == "ONCOGENIC", only about 40% have some ClinVar id that the
# pipeline could actually extract. The other ~60% have none of those fields populated with anything the resolver
# could turn into a real numeric id (empty, or only placeholder text like "N/A"/"NONE FOUND".
# A variant behind one of those ~60% of evidence items could very well have a real, findable ClinVar record,
# but CIViC's curator just never entered a ClinVar id for it, or entered it in a form none of the three tiers can parse.

# %%
# Save outputs
# Aligned with OptimusKG
oncogenic_aligned_output_path = (
    OUTPUT_DIR / f"clinvar_oncogenic_aligned_variants_{date_str}.csv"
)
clinvar_oncogenic_aligned.to_csv(oncogenic_aligned_output_path, index=False)
print(f"Saved full aligned oncogenic variant set to {oncogenic_aligned_output_path}")

# Gap candidates
oncogenic_gap_output_path = (
    OUTPUT_DIR / f"clinvar_oncogenic_gap_candidates_{date_str}.csv"
)
oncogenic_gap_candidates.to_csv(oncogenic_gap_output_path, index=False)
print(f"Saved ClinVar oncogenic gap candidates to {oncogenic_gap_output_path}")

# %%
# ---------------------------------------------------------------------------
# 12. Clinical-impact-variant gap detection via ClinVar SomaticClinicalImpact
#    CIViC = PREDICTIVE, DIAGNOSTIC, and PROGNOSTIC; Clinvar = SomaticClinicalImpact
#    IMPORTANT SCOPING NOTE: ClinVar's own
#    documentation describes clinical-impact classifications as covering
#    therapeutic/diagnostic/prognostic sub-assertions — but the
#    variant_summary.txt file only carries the aggregate TIER (I-IV), not
#    which sub-assertion type it reflects.
#    This is a limitation of the data source, not the analysis.
#    Tier IV (Benign/Likely benign) is excluded.
# ---------------------------------------------------------------------------

# 1. Define the positive classification values
CLINICAL_IMPACT_POSITIVE_VALUES = {
    "Tier I - Strong",
    "Tier II - Potential",
    "Tier III - Unknown",
}

clinvar_clinical_impact_aligned = clinvar_summary[
    clinvar_summary["GeneID"].isin(aligned_entrez_ids)
    & clinvar_summary["SomaticClinicalImpact"].isin(CLINICAL_IMPACT_POSITIVE_VALUES)
].copy()
clinvar_clinical_impact_aligned["VariationID"] = clinvar_clinical_impact_aligned[
    "VariationID"
].astype(str)
print(
    f"[clinvar gap] {len(clinvar_clinical_impact_aligned)} clinical-impact-classified "
    f"(Tier I-III) ClinVar variants on aligned genes"
)

# 2. Pull ClinVar's own clinical-impact-classified variants for the aligned genes
# Clinvar's SomaticClinicalImpact tier doesn't distinguish which of the three assertion types
CLINICAL_IMPACT_EVIDENCE_TYPES = {"PREDICTIVE", "DIAGNOSTIC", "PROGNOSTIC"}
clinical_impact_df = civic_df[
    civic_df["evidence_type"].isin(CLINICAL_IMPACT_EVIDENCE_TYPES)
].copy()

# 3. Build the set of what CIViC has already covered, combined across three evidence types
# count how many Predictive/Diagnostic/Prognostic CIViC items have no matched id at all (residual limitation)
n_clinical_impact_no_clinvar_id = (
    clinical_impact_df["clinvar_variation_ids_all"].apply(len).eq(0).sum()
)
print(
    f"[clinvar gap] {n_clinical_impact_no_clinvar_id} / {len(clinical_impact_df)} "
    f"PREDICTIVE/DIAGNOSTIC/PROGNOSTIC evidence items have no ClinVar match via "
    f"any of the three tiers"
)

# get every ClinVar id any Predictive/Diagnostic/Prognostic evidence item references
covered_clinical_impact_clinvar_ids = {
    vid for ids in clinical_impact_df["clinvar_variation_ids_all"] for vid in ids
}
print(
    f"[clinvar gap] {len(covered_clinical_impact_clinvar_ids)} unique ClinVar IDs referenced by CIViC PREDICTIVE/DIAGNOSTIC/PROGNOSTIC evidence items"
)

# 4. Compute the gap and coverage percentage
clinical_impact_gap_candidates = clinvar_clinical_impact_aligned[
    ~clinvar_clinical_impact_aligned["VariationID"].isin(
        covered_clinical_impact_clinvar_ids
    )
]
print(
    f"[clinvar gap] {len(clinical_impact_gap_candidates)} gap candidates: "
    f"clinical-impact-classified ClinVar variants on aligned genes with no "
    f"matching CIViC PREDICTIVE/DIAGNOSTIC/PROGNOSTIC evidence item"
)
n_clinical_impact_total = len(clinvar_clinical_impact_aligned)
clinical_impact_coverage_pct = (
    (n_clinical_impact_total - len(clinical_impact_gap_candidates))
    / n_clinical_impact_total
    * 100
    if n_clinical_impact_total > 0
    else None
)
print(
    f"[clinvar gap] Clinical-impact coverage: {clinical_impact_coverage_pct:.1f}% "
    f"of Tier I-III ClinVar variants on aligned genes are referenced by a "
    f"CIViC PREDICTIVE/DIAGNOSTIC/PROGNOSTIC evidence item"
)
# %%
# Save outputs
# Aligned with OptimusKG
clinical_impact_aligned_output_path = (
    OUTPUT_DIR / f"clinvar_clinical_impact_aligned_variants_{date_str}.csv"
)
clinvar_clinical_impact_aligned.to_csv(clinical_impact_aligned_output_path, index=False)
print(
    f"Saved full aligned clinical-impact variant set to {clinical_impact_aligned_output_path}"
)

# Gap candidates
clinical_impact_gap_output_path = (
    OUTPUT_DIR / f"clinvar_clinical_impact_gap_candidates_{date_str}.csv"
)
clinical_impact_gap_candidates.to_csv(clinical_impact_gap_output_path, index=False)
print(
    f"Saved ClinVar clinical-impact gap candidates to {clinical_impact_gap_output_path}"
)

# %%
