#!/usr/bin/env python3
"""
Cross-references CIViC evidence items (already resolved to OptimusKG gene
nodes) against ClinVar's variant_summary to find "gap candidates" — ClinVar
variants on CIViC-aligned genes with a matching clinical/oncogenic
classification that no existing CIViC evidence item currently references.

Companion to civic_optimuskg_alignment.py, not a replacement: this script
reads that script's civic_resolved_*.csv output rather than re-deriving
gene resolution itself, and does not touch MONDO/disease resolution at all
(none of the sections below need it).

Three independent gap-detection tracks, structured identically:
    1. Multi-tier ClinVar VariationID resolution + coverage-by-evidence_type
       (shared groundwork for tracks 2-4 below)
    2. PREDISPOSING vs ClinVar ClinicalSignificance (pathogenic/likely pathogenic)
    3. ONCOGENIC vs ClinVar Oncogenicity (oncogenic/likely oncogenic)
    4. PREDICTIVE/DIAGNOSTIC/PROGNOSTIC vs ClinVar SomaticClinicalImpact (Tier I-III)

Section 5 is not a gap-detection track. It's every ClinVar variant any CIViC evidence item
references (any evidence_type). Feeds src/graph/save_graph_nodes.py's
Variant node build via clinvar_optimuskg_disease_alignment.py, so a
CIViC-referenced ClinVar variant still gets a node even where it doesn't
carry a pathogenic/oncogenic/Tier I-III classification (e.g. Benign, or
unclassified).

Note: gene_node_ids in civic_resolved_*.csv round-trips through
to_csv/read_csv as a Python-list repr string (e.g. "['ENSG00000146648']"),
not a real list — parsed back explicitly via ast.literal_eval below.

Inputs:
    - data/processed/civic_resolved_YYYY-MM-DD.csv (output from civic_optimuskg_alignment.py)
    - OptimusKG gene nodes parquet (output from save_optimuskg_nodes.py)
    - ClinVar variant_summary TSV (output from fetch_clinvar_data.py)
    - HGNC complete set (fetched live from HGNC, for the Ensembl -> Entrez crosswalk)
Outputs:
    - data/processed/clinvar_predisposing_aligned_variants_YYYY-MM-DD.csv
    - data/processed/clinvar_predisposing_gap_candidates_YYYY-MM-DD.csv
    - data/processed/clinvar_predisposing_gap_candidates_highconfidence_YYYY-MM-DD.csv
    - data/processed/clinvar_oncogenic_aligned_variants_YYYY-MM-DD.csv
    - data/processed/clinvar_oncogenic_gap_candidates_YYYY-MM-DD.csv
    - data/processed/clinvar_clinical_impact_aligned_variants_YYYY-MM-DD.csv
    - data/processed/clinvar_clinical_impact_gap_candidates_YYYY-MM-DD.csv
    - data/processed/clinvar_evidence_referenced_variants_YYYY-MM-DD.csv

Usage:
    python src/alignment/civic_clinvar_gap_analysis.py
"""

# %%
import ast
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

# Paths and constants
NODES_DIR = Path("data/raw/optimuskg/nodes")
CLINVAR_RAW_PATH = find_latest_file(
    Path("data/raw/clinvar"), "clinvar_variant_summary_*.tsv"
)
CIVIC_RESOLVED_PATH = find_latest_file(Path("data/processed"), "civic_resolved_*.csv")
OUTPUT_DIR = Path("data/processed")
os.makedirs(OUTPUT_DIR.parent, exist_ok=True)

date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# %%
# ---------------------------------------------------------------------------
# 0. Load civic_optimuskg_alignment.py's resolved output + supporting
#    reference data (OptimusKG gene nodes, HGNC's Ensembl<->Entrez crosswalk)
# ---------------------------------------------------------------------------

civic_df = pd.read_csv(CIVIC_RESOLVED_PATH)
# gene_node_ids survives the CSV round-trip as a Python-list repr string
# (e.g. "['ENSG00000146648']" or "[None]"), not an actual list — parse it
# back explicitly, the same way it was produced by civic_optimuskg_alignment.py.
civic_df["gene_node_ids"] = civic_df["gene_node_ids"].apply(ast.literal_eval)

genes = pl.read_parquet(Path.joinpath(NODES_DIR, "gene.parquet"))

# HGNC's own authoritative complete set — same external hub used by
# civic_optimuskg_alignment.py, re-fetched here since gap-detection needs
# the Ensembl -> Entrez direction to match ClinVar's Entrez-keyed GeneID
# column, independent of that script's own in-memory state.
hgnc_complete = pd.read_csv(
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    sep="\t",
    dtype=str,
)
hgnc_ensembl_pairs = hgnc_complete.dropna(subset=["entrez_id", "ensembl_gene_id"])

# %%
# ---------------------------------------------------------------------------
# 1. Multi-tier ClinVar VariationID resolution for CIViC evidence items,
#    and resulting coverage by evidence_type.
#    variant_clinvar_id (singular) alone under-counts CIViC's real ClinVar
#    coverage. All three tiers (variant_clinvar_id, variant_clinvar_ids, variant_dbsnp_rsids)
#    are combined into one set of VariationIDs per evidence item, used for ALL matching in
#    sections 2-4, reducing false "gap" positives caused by missing crosswalk metadata rather
#    than missing curation. HGVS-string matching against ClinVar's own Name column
#    remains out of scope.
# ---------------------------------------------------------------------------
# ClinVar summary file (output from src/fetch_data/fetch_clinvar_data.py)
clinvar_summary = pd.read_csv(CLINVAR_RAW_PATH, sep="\t", dtype=str)

# Prefer each variant's GRCh38 row over its GRCh37/other row when picking
# one row per VariationID, rather than deduplicating by VariationID first
# Sorting so GRCh38 rows sort first, then deduplicating keep="first":
# every variant kept, GRCh38 coordinates wherever ClinVar has
# them, the row's own Assembly column truthfully reporting GRCh37/other for
# the minority that don't.
clinvar_summary["_grch38_first"] = (clinvar_summary["Assembly"] != "GRCh38").astype(int)
clinvar_summary = clinvar_summary.sort_values("_grch38_first").drop_duplicates(
    subset="VariationID", keep="first"
)
clinvar_summary = clinvar_summary.drop(columns=["_grch38_first"])
print(
    f"[verify] clinvar_summary rows after de-duplicating by VariationID "
    f"(GRCh38 preferred): {len(clinvar_summary)}"
)
print(
    f"[verify] {(clinvar_summary['Assembly'] != 'GRCh38').sum()} kept rows are NOT "
    f"GRCh38 (no GRCh38 row existed for that variant)"
)

# dbSNP rsID -> ClinVar VariationID(s), bare-number keyed. One-to-many, not
# one-to-one: 222,427 rsIDs (8.4% of all rsIDs with a VariationID, checked
# directly) genuinely correspond to more than one distinct ClinVar record
# (different submitters/interpretations of the same variant, multi-allelic
# sites, …). Building this as a plain dict(zip(...)) would silently keep
# only whichever VariationID happened to appear last for a colliding rsID
rsid_to_variationids = (
    clinvar_summary.loc[clinvar_summary["RS# (dbSNP)"] != "-1"]
    .groupby("RS# (dbSNP)")["VariationID"]
    .apply(set)
    .to_dict()
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
            variation_ids = rsid_to_variationids.get(bare_rsid)
            if variation_ids:
                ids.update(str(vid) for vid in variation_ids)

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
# 2. Predisposing-variant gap detection via ClinVar
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
# set from section 1 (singular id + plural ids + dbSNP-based match)
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
# 3. Oncogenic-variant gap detection via ClinVar Oncogenicity classification
#    CIViC = Oncogenic; Clinvar = Oncogenic
#    Same structure as section 2, different classification column and CIViC
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
# 4. Clinical-impact-variant gap detection via ClinVar SomaticClinicalImpact
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
# ---------------------------------------------------------------------------
# 5. Evidence-referenced variants: every ClinVar variant any CIViC evidence
#    item references, regardless of classification or gene alignment. Not a
#    gap-detection track.
#    A CIViC-referenced ClinVar variant gets a node even where it doesn't
#    carry a pathogenic/oncogenic/Tier I-III classification.
# ---------------------------------------------------------------------------

all_referenced_variation_ids = set().union(*civic_df["clinvar_variation_ids_all"])
print(
    f"[evidence-referenced] {len(all_referenced_variation_ids)} unique ClinVar "
    f"VariationIDs referenced by CIViC evidence items (any evidence_type, "
    f"any classification)"
)

clinvar_evidence_referenced = clinvar_summary[
    clinvar_summary["VariationID"].isin(all_referenced_variation_ids)
].copy()
print(
    f"[evidence-referenced] {len(clinvar_evidence_referenced)} / "
    f"{len(all_referenced_variation_ids)} referenced VariationIDs found in "
    f"clinvar_summary (the rest reference a VariationID not present in this "
    f"ClinVar snapshot)"
)

evidence_referenced_output_path = (
    OUTPUT_DIR / f"clinvar_evidence_referenced_variants_{date_str}.csv"
)
clinvar_evidence_referenced.to_csv(evidence_referenced_output_path, index=False)
print(f"Saved evidence-referenced variant set to {evidence_referenced_output_path}")

# %%
