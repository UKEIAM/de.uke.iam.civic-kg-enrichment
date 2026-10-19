#!/usr/bin/env python3
"""
Builds the node tables the graph will be assembled from.

Two node types are genuinely new to OptimusKG::
    - Evidence item (CIVICEVI_<id>): one row per CIViC evidence item, from
      civic_resolved.csv.
    - Variant (CLINVAR_<VariationID>): one row per distinct ClinVar variant
      across all four tracks (predisposing/oncogenic/clinical_impact/
      evidence_referenced), deduplicated, with an in_civic_evidence_item flag
      computed against the SAME evidence items (via the same three
      ClinVar-id-matching tiers civic_clinvar_gap_analysis.py already
      established: variant_clinvar_id, variant_clinvar_ids, dbSNP rsid
      crosswalk). The fourth track, evidence_referenced, is every ClinVar
      variant any CIViC evidence item references (built by
      civic_clinvar_gap_analysis.py section 5) — it gets a node even
      without a pathogenic/oncogenic/Tier I-III classification (e.g.
      Benign, or unclassified), since it's referenced by a CIViC
      evidence item.

      Also carries civic_clinvar_gap_analysis.py's actual gap verdict,
      which that script computes but (until now) never wrote anywhere past
      its own CSVs/stdout: {predisposing,oncogenic,clinical_impact}_gap_candidate,
      each True iff the variant qualifies for that track (Pathogenic/
      Likely-pathogenic, Oncogenic/Likely-oncogenic, or Tier I-III
      respectively) AND no referencing CIViC evidence item has the matching
      evidence_type. This is deliberately NOT the same as
      `not in_civic_evidence_item` — that flag is global (true if ANY
      evidence item references the variant, regardless of type) and would
      under-count real gaps: a variant with a PREDICTIVE evidence item but
      no PREDISPOSING one is still a genuine predisposing gap.
      is_gap_candidate ORs the three together; high_confidence_gap_candidate
      additionally requires a 2-star-or-better ClinVar review status
      (predisposing track only, mirroring civic_clinvar_gap_analysis.py's
      own stratification — oncogenic/clinical_impact gaps aren't
      confidence-stratified there either).

Gene/Disease/Drug aren't new node types, but this script also produces
derived, enriched copies of them adding in_civic_evidence_item/n_evidence_items/
highest_evidence_level, aggregated from civic_resolved.csv. Gene/Disease/
Drug all use the same aggregation helper (aggregate_evidence_by_entity);
Variant needs an extra VariationID-matching step first since it isn't
already a column on civic_resolved.csv the way gene/disease/drug node ids
are. Any remaining gap means a referenced VariationID wasn't
found in clinvar_summary at all.

Evidence item's own variant_node_ids reuses that same VariationID match:
the ClinVar ids an evidence item references, filtered to ones that actually
got a Variant node. Also carries molecular_profile_is_multi_variant (from
civic_resolved.csv) so a consumer can tell apart a variant_node_ids entry
that resolved as the sole candidate from one that resolved as 1-of-N
sub-variants in an "OR" molecular profile — a distinction variant_node_ids
alone can't make, since the matching tiers union all sub-variants' ClinVar
ids into one flat set per evidence item.

Namespacing/provenance: outputs are a derived dataset that references
OptimusKG's ids, written to data/processed/ and this can be regenerated
against a new OptimusKG snapshot without hand-editing anything.

Inputs:
    - data/raw/optimuskg/nodes/{gene,disease,drug}.parquet (from save_optimuskg_nodes.py)
    - data/processed/civic_resolved_*.csv (from civic_optimuskg_alignment.py)
    - data/processed/clinvar_{predisposing,oncogenic,clinical_impact,evidence_referenced}_aligned_variants_phenotype_enriched_*.csv
      (from clinvar_optimuskg_disease_alignment.py)
    - data/raw/clinvar/clinvar_variant_summary_*.tsv (from fetch_clinvar_data.py;
      only needed for the dbSNP rsid -> VariationID crosswalk, tier 3 of the
      ClinVar-id matching)
Outputs:
    - data/processed/graph_nodes_gene_<date>.parquet
    - data/processed/graph_nodes_disease_<date>.parquet
    - data/processed/graph_nodes_drug_<date>.parquet
    - data/processed/graph_nodes_variant_<date>.parquet
    - data/processed/graph_nodes_evidence_item_<date>.parquet

Usage:
    python src/graph/save_graph_nodes.py
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

NODES_DIR = Path("data/raw/optimuskg/nodes")
PROCESSED_DIR = Path("data/processed")
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# CIViC's evidence_level convention: A = highest quality (validated
# association) down to E = lowest (inferential association).
EVIDENCE_LEVEL_ORDER = ["A", "B", "C", "D", "E"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(EVIDENCE_LEVEL_ORDER)}

CLINVAR_TRACKS = ["predisposing", "oncogenic", "clinical_impact", "evidence_referenced"]

# Same type-matching civic_clinvar_gap_analysis.py's three gap-detection
# tracks use: a variant qualifying for a track is only a genuine gap if no
# referencing CIViC evidence item has one of these evidence_type values.
TRACK_REQUIRED_EVIDENCE_TYPES = {
    "predisposing": {"PREDISPOSING"},
    "oncogenic": {"ONCOGENIC"},
    "clinical_impact": {"PREDICTIVE", "DIAGNOSTIC", "PROGNOSTIC"},
}
# Same set civic_clinvar_gap_analysis.py uses to stratify predisposing gap
# candidates into clinically-actionable vs. database-scope noise.
HIGH_CONFIDENCE_REVIEW_STATUSES = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}

# %%
# ---------------------------------------------------------------------------
# 0. Load civic_optimuskg_alignment.py's resolved output — the single
#    source of truth for every evidence-item -> entity aggregation below.
# ---------------------------------------------------------------------------

civic_resolved_path = find_latest_file(PROCESSED_DIR, "civic_resolved_*.csv")
civic_df = pd.read_csv(civic_resolved_path)
print(f"[setup] loaded {len(civic_df)} CIViC evidence items from {civic_resolved_path}")

# gene_node_ids/drug_node_ids round-trip through to_csv/read_csv as a
# Python-list repr string (e.g. "['ENSG00000146648']" or "[None]"), not an
# actual list — parsed back the same way clinvar_gap_analysis.py already does.
civic_df["gene_node_ids"] = civic_df["gene_node_ids"].apply(ast.literal_eval)
civic_df["drug_node_ids"] = civic_df["drug_node_ids"].apply(ast.literal_eval)


def _best_level(levels: pd.Series) -> str | None:
    ranked = [lvl for lvl in levels.dropna() if lvl in LEVEL_RANK]
    if not ranked:
        return None
    return min(ranked, key=lambda lvl: LEVEL_RANK[lvl])


def aggregate_evidence_by_entity(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Shared aggregation for Gene/Disease/Drug enrichment: explode a
    civic_df column that holds either a list of node ids (gene_node_ids,
    drug_node_ids) or a single scalar node id (disease_node_id) into one
    row per (entity node id, evidence item), then aggregate to
    in_civic_evidence_item/n_evidence_items/highest_evidence_level per entity."""
    exploded = df[[id_col, "id", "evidence_level"]].explode(id_col)
    exploded = exploded.dropna(subset=[id_col])

    agg = (
        exploded.groupby(id_col)
        .agg(
            n_evidence_items=("id", "nunique"),
            highest_evidence_level=("evidence_level", _best_level),
        )
        .reset_index()
        .rename(columns={id_col: "node_id"})
    )
    agg["in_civic_evidence_item"] = True
    return agg


# %%
# ---------------------------------------------------------------------------
# 1. Gene / Disease / Drug: derived, enriched copies of the existing
#    OptimusKG node tables (originals in data/raw/ untouched).
# ---------------------------------------------------------------------------

ENTITY_ENRICHMENT_SPECS = [
    ("gene", "gene_node_ids"),
    ("disease", "disease_node_id"),
    ("drug", "drug_node_ids"),
]

for entity_name, id_col in ENTITY_ENRICHMENT_SPECS:
    nodes = pl.read_parquet(NODES_DIR / f"{entity_name}.parquet").to_pandas()
    enrichment = aggregate_evidence_by_entity(civic_df, id_col)

    enriched = nodes.merge(enrichment, left_on="id", right_on="node_id", how="left")
    enriched = enriched.drop(columns=["node_id"])
    enriched["in_civic_evidence_item"] = (
        enriched["in_civic_evidence_item"].astype("boolean").fillna(False).astype(bool)
    )
    enriched["n_evidence_items"] = enriched["n_evidence_items"].fillna(0).astype(int)

    n_in_evidence = enriched["in_civic_evidence_item"].sum()
    print(
        f"[{entity_name}] {n_in_evidence} / {len(enriched)} nodes appear in at "
        f"least one CIViC evidence item"
    )

    out_path = PROCESSED_DIR / f"graph_nodes_{entity_name}_{date_str}.parquet"
    pl.from_pandas(enriched).write_parquet(out_path)
    print(f"[{entity_name}] wrote {out_path} shape={enriched.shape}")

# %%
# ---------------------------------------------------------------------------
# 2. ClinVar VariationID matching (feeds both Variant and Evidence item
#    below) + Variant node table: dedup across the three ClinVar
#    classification tracks, then resolve in_civic_evidence_item/
#    n_evidence_items/highest_evidence_level against civic_df via the same
#    three ClinVar-id-matching tiers civic_clinvar_gap_analysis.py already
#    established.
# ---------------------------------------------------------------------------

clinvar_raw_path = find_latest_file(
    Path("data/raw/clinvar"), "clinvar_variant_summary_*.tsv"
)
clinvar_summary = pd.read_csv(clinvar_raw_path, sep="\t", dtype=str)
# Prefer each variant's GRCh38 row over GRCh37/other, don't filter to
# GRCh38-only (see civic_clinvar_gap_analysis.py's identical fix for the
# full reasoning)
clinvar_summary["_grch38_first"] = (clinvar_summary["Assembly"] != "GRCh38").astype(int)
clinvar_summary = clinvar_summary.sort_values("_grch38_first").drop_duplicates(
    subset="VariationID", keep="first"
)
clinvar_summary = clinvar_summary.drop(columns=["_grch38_first"])

# One-to-many, not one-to-one — some rsIDs map to more than one
# VariationID; dict(zip(...)) would silently keep only one, dropping a
# real match (see civic_clinvar_gap_analysis.py's identical fix).
rsid_to_variationids = (
    clinvar_summary.loc[clinvar_summary["RS# (dbSNP)"] != "-1"]
    .groupby("RS# (dbSNP)")["VariationID"]
    .apply(set)
    .to_dict()
)


def get_civic_clinvar_variation_ids(row: pd.Series) -> set:
    """Same three-tier matching as civic_clinvar_gap_analysis.py's
    get_civic_clinvar_variation_ids."""
    ids = set()

    def _add_if_numeric(candidate: str):
        candidate = candidate.strip()
        if candidate.isdigit():
            ids.add(candidate)

    singular_raw = row.get("variant_clinvar_id")
    if pd.notna(singular_raw):
        for v in str(singular_raw).split(","):
            _add_if_numeric(v)

    plural_raw = row.get("variant_clinvar_ids")
    if pd.notna(plural_raw):
        for v in str(plural_raw).split(","):
            _add_if_numeric(v)

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

# Per-VariationID evidence aggregation, global:
# in_civic_evidence_item is True for a variant if ANY CIViC
# evidence item references it via any of the three tiers, regardless of
# which classification track(s) it happened to be scoped under below.
# evidence_type is carried alongside (not just id/level) so the
# track-specific gap flags below can check for a *matching-type* reference,
# not just any reference — see module docstring for why that distinction
# matters.
variation_id_to_evidence = {}
for _, row in civic_df.iterrows():
    for vid in row["clinvar_variation_ids_all"]:
        variation_id_to_evidence.setdefault(vid, []).append(
            (row["id"], row["evidence_level"], row["evidence_type"])
        )
variation_id_to_evidence_types = {
    vid: {et for _, _, et in items} for vid, items in variation_id_to_evidence.items()
}

# %%
track_frames = []
for track in CLINVAR_TRACKS:
    path = find_latest_file(
        PROCESSED_DIR, f"clinvar_{track}_aligned_variants_phenotype_enriched_*.csv"
    )
    df = pd.read_csv(path)
    df["VariationID"] = df["VariationID"].astype(str)
    df["track"] = track
    track_frames.append(df)
    print(f"[variant] loaded {len(df)} rows from {path} (track={track})")

all_tracks_df = pd.concat(track_frames, ignore_index=True)

# resolved_phenotype_disease_ids is itself a list-repr string on reload
# same round-trip quirk as gene_node_ids/drug_node_ids.
all_tracks_df["resolved_phenotype_disease_ids"] = all_tracks_df[
    "resolved_phenotype_disease_ids"
].apply(ast.literal_eval)


def _first_non_null(s: pd.Series):
    non_null = s.dropna()
    return non_null.iloc[0] if len(non_null) else None


variant_nodes = (
    all_tracks_df.groupby("VariationID")
    .agg(
        gene_symbol=("GeneSymbol", _first_non_null),
        gene_id_entrez=("GeneID", _first_non_null),
        hgvs_name=("Name", _first_non_null),
        type=("Type", _first_non_null),
        hgnc_id=("HGNC_ID", _first_non_null),
        chromosome=("Chromosome", _first_non_null),
        start=("Start", _first_non_null),
        stop=("Stop", _first_non_null),
        reference_allele_vcf=("ReferenceAlleleVCF", _first_non_null),
        alternate_allele_vcf=("AlternateAlleleVCF", _first_non_null),
        number_submitters_clinvar=("NumberSubmitters", _first_non_null),
        assembly=("Assembly", _first_non_null),
        other_ids_clinvar=("OtherIDs", _first_non_null),
        review_status_clinical_impact_clinvar=(
            "ReviewStatusClinicalImpact",
            _first_non_null,
        ),
        review_status_oncogenicity_clinvar=(
            "ReviewStatusOncogenicity",
            _first_non_null,
        ),
        clinical_significance_clinvar=("ClinicalSignificance", _first_non_null),
        oncogenicity=("Oncogenicity", _first_non_null),
        somatic_clinical_impact_tier=("SomaticClinicalImpact", _first_non_null),
        variant_origin=("OriginSimple", _first_non_null),
        dbsnp_rsid=("RS# (dbSNP)", _first_non_null),
        review_status_clinvar=("ReviewStatus", _first_non_null),
        resolved_phenotype_disease_ids=(
            "resolved_phenotype_disease_ids",
            lambda lists: sorted(set().union(*lists)) if len(lists) else [],
        ),
        phenotype_is_cancer_relevant=("phenotype_is_cancer_relevant", "any"),
        tracks=("track", lambda s: sorted(set(s))),
    )
    .reset_index()
)

variant_nodes["id"] = "CLINVAR_" + variant_nodes["VariationID"]
variant_nodes["in_civic_evidence_item"] = variant_nodes["VariationID"].isin(
    variation_id_to_evidence
)
variant_nodes["n_evidence_items"] = variant_nodes["VariationID"].apply(
    lambda vid: len(variation_id_to_evidence.get(vid, []))
)
variant_nodes["highest_evidence_level"] = variant_nodes["VariationID"].apply(
    lambda vid: _best_level(
        pd.Series([lvl for _, lvl, _ in variation_id_to_evidence.get(vid, [])])
    )
)

n_resolved = variant_nodes["in_civic_evidence_item"].sum()
print(
    f"[variant] {len(variant_nodes)} distinct ClinVar variants across "
    f"{len(CLINVAR_TRACKS)} tracks; {n_resolved} / {len(variant_nodes)} "
    f"in_civic_evidence_item"
)

# %%
# ---------------------------------------------------------------------------
# 2b. Type-matched gap verdict (civic_clinvar_gap_analysis.py's actual
#     analytical output, re-derived here so it lands on the node instead of
#     dead-ending in that script's standalone CSVs — see module docstring).
# ---------------------------------------------------------------------------


def _track_gap_flags(row: pd.Series) -> pd.Series:
    referenced_types = variation_id_to_evidence_types.get(row["VariationID"], set())
    row_tracks = set(row["tracks"])
    return pd.Series(
        {
            f"{track}_gap_candidate": track in row_tracks
            and not (referenced_types & required_types)
            for track, required_types in TRACK_REQUIRED_EVIDENCE_TYPES.items()
        }
    )


gap_flag_cols = list(TRACK_REQUIRED_EVIDENCE_TYPES)
variant_nodes[[f"{t}_gap_candidate" for t in gap_flag_cols]] = variant_nodes.apply(
    _track_gap_flags, axis=1
)
variant_nodes["is_gap_candidate"] = variant_nodes[
    [f"{t}_gap_candidate" for t in gap_flag_cols]
].any(axis=1)
variant_nodes["high_confidence_gap_candidate"] = variant_nodes[
    "predisposing_gap_candidate"
] & variant_nodes["review_status_clinvar"].isin(HIGH_CONFIDENCE_REVIEW_STATUSES)

n_gap = variant_nodes["is_gap_candidate"].sum()
n_high_conf_gap = variant_nodes["high_confidence_gap_candidate"].sum()
print(
    f"[variant] {n_gap} / {len(variant_nodes)} are a type-matched gap candidate "
    f"(qualifying classification, no matching-evidence-type CIViC evidence item); "
    f"{n_high_conf_gap} of those are high-confidence predisposing gaps "
    f"(review status in {sorted(HIGH_CONFIDENCE_REVIEW_STATUSES)})"
)

variant_out_path = PROCESSED_DIR / f"graph_nodes_variant_{date_str}.parquet"
pl.from_pandas(variant_nodes).write_parquet(variant_out_path)
print(f"[variant] wrote {variant_out_path} shape={variant_nodes.shape}")

# %%
# ---------------------------------------------------------------------------
# 3. Evidence item: one row per CIViC evidence item. variant_node_ids reuses
#    clinvar_variation_ids_all from section 2, filtered to ids that actually
#    resolved to a Variant node above (see module docstring for why some
#    referenced ClinVar ids won't).
# ---------------------------------------------------------------------------

variant_id_set = set(variant_nodes["VariationID"])

evidence_item_nodes = pd.DataFrame(
    {
        "id": "CIVICEVI_" + civic_df["id"].astype(str),
        "civic_evidence_id": civic_df["id"],
        "evidence_level": civic_df["evidence_level"],
        "evidence_type": civic_df["evidence_type"],
        "clinical_significance_civic": civic_df["significance"],
        "evidence_direction": civic_df["evidence_direction"],
        "evidence_rating": civic_df["evidence_rating"],
        "description": civic_df["description"],
        "therapy_interaction_type": civic_df["therapy_interaction_type"],
        "source_type": civic_df["source_type"],
        "source_citation": civic_df["source_citation"],
        "source_publication_id": civic_df["source_citation_id"],
        "disease_node_id": civic_df["disease_node_id"],
        "gene_node_ids": civic_df["gene_node_ids"],
        "drug_node_ids": civic_df["drug_node_ids"],
        "variant_node_ids": civic_df["clinvar_variation_ids_all"].apply(
            lambda ids: sorted(f"CLINVAR_{vid}" for vid in ids if vid in variant_id_set)
        ),
        # Without this, a resolved variant_node_ids entry is indistinguishable
        # between "the only candidate, and it resolved" and "1 of N OR'd
        # sub-variants resolved, the other N-1 didn't" — the molecular
        # profile's own variant candidates aren't otherwise carried onto the
        # EVI node.
        "molecular_profile_is_multi_variant": civic_df["molecular_profile_is_multi_variant"],
    }
)

n_with_clinvar_ref = (civic_df["clinvar_variation_ids_all"].apply(len) > 0).sum()
n_with_variant_node = evidence_item_nodes["variant_node_ids"].apply(len).gt(0).sum()
print(
    f"[evidence_item] {n_with_clinvar_ref} evidence items reference at least one "
    f"ClinVar id via the three matching tiers; {n_with_variant_node} of those "
    f"resolve to an actual Variant node (the rest reference ClinVar variants "
    f"outside Phase 1's pathogenic/oncogenic/Tier I-III-on-aligned-genes scope "
    f"— a real, expected gap, not a bug)"
)

evidence_item_out_path = PROCESSED_DIR / f"graph_nodes_evidence_item_{date_str}.parquet"
pl.from_pandas(evidence_item_nodes).write_parquet(evidence_item_out_path)
print(
    f"[evidence_item] wrote {evidence_item_out_path} shape={evidence_item_nodes.shape}"
)
