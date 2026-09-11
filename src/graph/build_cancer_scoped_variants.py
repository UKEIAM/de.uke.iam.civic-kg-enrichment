#!/usr/bin/env python3
"""
Builds the widened variant population for the cancer-scoped graph
(build_cancer_scoped_graph.py). The existing graph_nodes_variant_*.parquet
is scoped to ClinVar variants on genes CIViC has curated at
least one evidence item for (aligned-genes). The cancer-scoped graph instead wants variants
from every gene reachable from a cancer-relevant disease. The total number of
ClinVar variants on those genes was 4,457,555, which
would make the graph 99% variant nodes. Narrowing to
Pathogenic/Likely-pathogenic with a high-confidence review status fixed this.

Scope of new variants added here: on a cancer-scope gene (via native
DIS-GEN from the cancer disease seed) AND Pathogenic/Likely-pathogenic
(excluding "Conflicting classifications") AND a review status in
HIGH_CONFIDENCE_REVIEW_STATUSES. Each variant's GRCh38 row is preferred for
its genomic-coordinate fields where one exists, but GRCh38 is not a
membership requirement — the classification fields this filter keys on are
identical between a variant's GRCh37 and GRCh38 rows, so requiring GRCh38
would only drop otherwise-qualifying variants that ClinVar hasn't lifted
over yet, for no analytical reason. Unioned with the existing aligned-gene
variant table (a variant that's both CIViC-aligned and meets the new filter
keeps its richer existing row).

Inputs:
    - data/processed/cancer_relevant_disease_ids_*.csv (from civic_optimuskg_alignment.py)
    - data/raw/optimuskg/edges/disease_gene_edges.parquet (from save_optimuskg_edges.py)
    - data/raw/optimuskg/nodes/{gene,disease}.parquet (from save_optimuskg_nodes.py)
    - HGNC complete set (fetched live, same Entrez->OptimusKG crosswalk as
      save_graph_edges_hub.py's VARIANT-GENE section)
    - data/raw/clinvar/clinvar_variant_summary_*.tsv (from fetch_clinvar_data.py)
    - data/processed/graph_nodes_variant_*.parquet (from save_graph_nodes.py)
Outputs:
    - data/processed/cancer_scoped_variant_nodes_<date>.parquet
    - data/processed/cancer_scoped_variant_gene_edges_<date>.parquet
    - data/processed/cancer_scoped_variant_disease_edges_<date>.parquet

Usage:
    python src/graph/build_cancer_scoped_variants.py
"""

# %%
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

NODES_DIR = Path("data/raw/optimuskg/nodes")
EDGES_DIR = Path("data/raw/optimuskg/edges")
PROCESSED_DIR = Path("data/processed")
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

HIGH_CONFIDENCE_REVIEW_STATUSES = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}

# %%
# ---------------------------------------------------------------------------
# 1. Cancer-scope genes: OptimusKG genes reached from the cancer disease
#    seed via native DIS-GEN. Same disease scope build_cancer_scoped_graph.py uses.
# ---------------------------------------------------------------------------

cancer_scope_path = find_latest_file(
    PROCESSED_DIR, "cancer_relevant_disease_ids_[0-9][0-9][0-9][0-9]-*.csv"
)
cancer_disease_ids = set(pd.read_csv(cancer_scope_path)["disease_node_id"].astype(str))
print(
    f"[setup] {len(cancer_disease_ids)} cancer-relevant disease node ids "
    f"loaded from {cancer_scope_path}"
)

disease_gene_edges = pl.read_parquet(EDGES_DIR / "disease_gene_edges.parquet")
cancer_gene_ids = set(
    disease_gene_edges.filter(pl.col("from").is_in(list(cancer_disease_ids)))[
        "to"
    ].to_list()
)
print(
    f"[setup] {len(cancer_gene_ids)} cancer-scope genes (DIS-GEN neighbours "
    f"of the {len(cancer_disease_ids)} cancer diseases)"
)

diseases_id_set = set(pl.read_parquet(NODES_DIR / "disease.parquet")["id"].to_list())

# %%
# ---------------------------------------------------------------------------
# 2. Entrez -> OptimusKG gene node crosswalk (same as save_graph_edges_hub.py's VARIANT-GENE section)
# to find which Entrez ids resolve into a cancer-scope gene node.
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


# %%
# ---------------------------------------------------------------------------
# 3. Load ClinVar, resolve every distinct GeneID once, then filter to the
#    new-variant scope.
# ---------------------------------------------------------------------------

clinvar_raw_path = find_latest_file(
    Path("data/raw/clinvar"), "clinvar_variant_summary_*.tsv"
)
clinvar_summary_raw = pd.read_csv(clinvar_raw_path, sep="\t", dtype=str)
print(
    f"[setup] {len(clinvar_summary_raw)} raw ClinVar variant_summary rows (all assemblies)"
)

distinct_gene_ids = clinvar_summary_raw["GeneID"].dropna().unique()
entrez_to_gene_node = {gid: resolve_gene_by_entrez(gid) for gid in distinct_gene_ids}
cancer_scope_entrez_ids = {
    gid for gid, node in entrez_to_gene_node.items() if node in cancer_gene_ids
}
print(
    f"[setup] {len(cancer_scope_entrez_ids)} / {len(distinct_gene_ids)} distinct "
    f"ClinVar GeneIDs resolve into a cancer-scope gene node"
)

# Prefer each variant's GRCh38 row over GRCh37/other when picking one row
# per VariationID (same fix, same reasoning, as
# civic_clinvar_gap_analysis.py/save_graph_nodes.py): filtering to GRCh38
# only would silently drop the ClinVar variants with no GRCh38 row
# at all, not just their coordinates, and the classification fields this
# filter actually keys on (GeneID, ClinicalSignificance, ReviewStatus) are
# identical between a variant's GRCh37 and GRCh38 rows regardless, so there
# is no reason to require GRCh38 as a hard membership condition here, only
# as a coordinate preference. Sorting so GRCh38 rows sort first, then
# deduplicating keep="first", gets one row per VariationID with GRCh38
# coordinates wherever available.
clinvar_summary_raw["_grch38_first"] = (
    clinvar_summary_raw["Assembly"] != "GRCh38"
).astype(int)
clinvar_deduped = clinvar_summary_raw.sort_values("_grch38_first").drop_duplicates(
    subset="VariationID", keep="first"
)
clinvar_deduped = clinvar_deduped.drop(columns=["_grch38_first"])
print(
    f"[setup] {len(clinvar_deduped)} distinct VariationIDs "
    f"({(clinvar_deduped['Assembly'] != 'GRCh38').sum()} without a GRCh38 row)"
)

sig = clinvar_deduped["ClinicalSignificance"].fillna("")
is_plp = sig.str.contains("Pathogenic", case=False) & ~sig.str.contains(
    "Conflicting", case=False
)
is_high_confidence = clinvar_deduped["ReviewStatus"].isin(
    HIGH_CONFIDENCE_REVIEW_STATUSES
)
is_cancer_scope_gene = clinvar_deduped["GeneID"].isin(cancer_scope_entrez_ids)

new_candidates = clinvar_deduped[
    is_cancer_scope_gene & is_plp & is_high_confidence
].copy()
print(
    f"[filter] {len(new_candidates)} ClinVar variants are on a cancer-scope gene, "
    f"Pathogenic/Likely-pathogenic, high-confidence review status "
    f"({(new_candidates['Assembly'] != 'GRCh38').sum()} without a GRCh38 row)"
)

# %%
# ---------------------------------------------------------------------------
# 4. Build new-candidate variant node rows in the same 28-column shape as
#    graph_nodes_variant_*.parquet, so the union in section 5 is a clean
#    concat.
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "GeneSymbol": "gene_symbol",
    "GeneID": "gene_id_entrez",
    "Name": "hgvs_name",
    "Type": "type",
    "HGNC_ID": "hgnc_id",
    "Chromosome": "chromosome",
    "Start": "start",
    "Stop": "stop",
    "ReferenceAlleleVCF": "reference_allele_vcf",
    "AlternateAlleleVCF": "alternate_allele_vcf",
    "NumberSubmitters": "number_submitters_clinvar",
    "Assembly": "assembly",
    "OtherIDs": "other_ids_clinvar",
    "ReviewStatusClinicalImpact": "review_status_clinical_impact_clinvar",
    "ReviewStatusOncogenicity": "review_status_oncogenicity_clinvar",
    "ClinicalSignificance": "clinical_significance_clinvar",
    "Oncogenicity": "oncogenicity",
    "SomaticClinicalImpact": "somatic_clinical_impact_tier",
    "OriginSimple": "variant_origin",
    "RS# (dbSNP)": "dbsnp_rsid",
    "ReviewStatus": "review_status_clinvar",
}

new_variant_nodes = new_candidates.rename(columns=COLUMN_MAP)[
    list(COLUMN_MAP.values()) + ["VariationID"]
].copy()
new_variant_nodes["id"] = "CLINVAR_" + new_variant_nodes["VariationID"].astype(str)
new_variant_nodes["resolved_phenotype_disease_ids"] = [
    [] for _ in range(len(new_variant_nodes))
]
new_variant_nodes["phenotype_is_cancer_relevant"] = False
new_variant_nodes["tracks"] = [
    ["cancer_scoped_plp_hiconf"] for _ in range(len(new_variant_nodes))
]
new_variant_nodes["in_civic_evidence_item"] = False
new_variant_nodes["n_evidence_items"] = 0
new_variant_nodes["highest_evidence_level"] = None

# %%
# ---------------------------------------------------------------------------
# 5. Union with the existing aligned-gene variant table, dedup on
#    VariationID keep="first" with existing rows first, so a variant that's
#    both CIViC-aligned and meets the new filter keeps its richer existing
#    row.
# ---------------------------------------------------------------------------

existing_variant_path = find_latest_file(PROCESSED_DIR, "graph_nodes_variant_*.parquet")
existing_variant_nodes = pl.read_parquet(existing_variant_path).to_pandas()
print(
    f"[setup] {len(existing_variant_nodes)} existing (aligned-gene) variant nodes "
    f"loaded from {existing_variant_path}"
)

# existing_variant_nodes carries proper numeric dtypes (its own upstream CSVs
# were read without dtype=str, so pandas inferred them); new_variant_nodes'
# fields are all raw strings straight off the dtype=str ClinVar read. Align
# both to pandas' nullable Int64 before concatenating; a plain concat would
# otherwise leave these columns as mixed-type "object", which
# pl.from_pandas() below can't convert to parquet.
NUMERIC_COLUMNS = [
    "gene_id_entrez",
    "start",
    "stop",
    "number_submitters_clinvar",
    "dbsnp_rsid",
]
for df in (existing_variant_nodes, new_variant_nodes):
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

combined = pd.concat(
    [existing_variant_nodes, new_variant_nodes], ignore_index=True, sort=False
)
combined = combined.drop_duplicates(subset="VariationID", keep="first")
n_new_only = len(combined) - len(existing_variant_nodes)
print(
    f"[union] {len(existing_variant_nodes)} existing + {len(new_variant_nodes)} "
    f"new-candidate -> {len(combined)} distinct variants "
    f"({n_new_only} genuinely new, {len(new_variant_nodes) - n_new_only} were "
    f"already in the existing table and kept their existing row)"
)

new_only_mask = ~combined["VariationID"].isin(existing_variant_nodes["VariationID"])

# %%
# ---------------------------------------------------------------------------
# 6. Phenotype crosswalk for the new-only variants (existing variants
#    already have variant_disease rows in
#    graph_edges_variant_disease_*.parquet); same regex + set-membership
#    logic as clinvar_optimuskg_disease_alignment.py.
# ---------------------------------------------------------------------------

phenotype_ids_by_variation_id = dict(
    zip(new_candidates["VariationID"], new_candidates["PhenotypeIDS"])
)


def resolve_and_flag_phenotypes(phenotype_ids_raw) -> tuple:
    """Handles the doubled "MONDO:MONDO:0013342"-style prefix quirk observed
    in raw ClinVar data. Identical logic to
    clinvar_optimuskg_disease_alignment.py's resolve_and_flag_phenotypes."""
    if pd.isna(phenotype_ids_raw):
        return [], False
    matches = re.findall(r"MONDO:(?:MONDO:)?(\d+)", str(phenotype_ids_raw))
    candidate_ids = [f"MONDO_{m}" for m in matches]
    resolved = [m for m in candidate_ids if m in diseases_id_set]
    is_cancer_relevant = any(m in cancer_disease_ids for m in resolved)
    return resolved, is_cancer_relevant


new_only_variation_ids = combined.loc[new_only_mask, "VariationID"]
phenotype_results = new_only_variation_ids.map(
    lambda vid: resolve_and_flag_phenotypes(phenotype_ids_by_variation_id.get(vid))
)
combined.loc[new_only_mask, "resolved_phenotype_disease_ids"] = phenotype_results.apply(
    lambda x: x[0]
)
combined.loc[new_only_mask, "phenotype_is_cancer_relevant"] = phenotype_results.apply(
    lambda x: x[1]
)

n_resolved = (
    combined.loc[new_only_mask, "resolved_phenotype_disease_ids"].apply(len).gt(0).sum()
)
print(
    f"[phenotype] {n_resolved} / {new_only_mask.sum()} new-only variants have a "
    f"phenotype resolving to a real OptimusKG disease node"
)

variant_nodes_path = PROCESSED_DIR / f"cancer_scoped_variant_nodes_{date_str}.parquet"
pl.from_pandas(combined).write_parquet(variant_nodes_path)
print(f"\n[save] wrote {variant_nodes_path} shape={combined.shape}")

# %%
# ---------------------------------------------------------------------------
# 7. VAR-GENE edges for the full union (relation="LOCATED_IN").
# ---------------------------------------------------------------------------

combined["_resolved_gene_node_id"] = combined["gene_id_entrez"].apply(
    resolve_gene_by_entrez
)
variant_gene_edges = combined.dropna(subset=["_resolved_gene_node_id"])[
    ["id", "_resolved_gene_node_id"]
].rename(columns={"id": "from", "_resolved_gene_node_id": "to"})
variant_gene_edges["undirected"] = True
variant_gene_edges["relation"] = "LOCATED_IN"
variant_gene_edges = variant_gene_edges.reset_index(drop=True)
print(
    f"[VAR-GENE] {len(variant_gene_edges)} / {len(combined)} variants resolved "
    f"to an OptimusKG gene node"
)

variant_gene_path = (
    PROCESSED_DIR / f"cancer_scoped_variant_gene_edges_{date_str}.parquet"
)
pl.from_pandas(variant_gene_edges).write_parquet(variant_gene_path)
print(f"[save] wrote {variant_gene_path} shape={variant_gene_edges.shape}")

# %%
# ---------------------------------------------------------------------------
# 8. VAR-DISEASE edges for the new-only variants
# ---------------------------------------------------------------------------

phenotype_pairs = combined.loc[new_only_mask, ["id", "resolved_phenotype_disease_ids"]]
phenotype_pairs = phenotype_pairs.explode("resolved_phenotype_disease_ids")
phenotype_pairs = phenotype_pairs.dropna(subset=["resolved_phenotype_disease_ids"])
phenotype_pairs = phenotype_pairs.rename(
    columns={"id": "from", "resolved_phenotype_disease_ids": "to"}
)
phenotype_pairs["origin"] = [["clinvar_phenotype"] for _ in range(len(phenotype_pairs))]
phenotype_pairs["undirected"] = True
phenotype_pairs = phenotype_pairs.reset_index(drop=True)
print(f"[VAR-DIS] {len(phenotype_pairs)} new clinvar_phenotype-origin edges")

variant_disease_path = (
    PROCESSED_DIR / f"cancer_scoped_variant_disease_edges_{date_str}.parquet"
)
pl.from_pandas(phenotype_pairs).write_parquet(variant_disease_path)
print(f"[save] wrote {variant_disease_path} shape={phenotype_pairs.shape}")

# %%
