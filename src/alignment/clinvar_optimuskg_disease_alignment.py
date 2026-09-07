#!/usr/bin/env python3
"""
Aligns ClinVar variant phenotypes (PhenotypeIDS) to OptimusKG disease nodes,
for the ClinVar variant sets already aligned to CIViC-covered genes.

It only needs three things civic_optimuskg_alignment.py has
already produced or can cheaply reload:
  1. OptimusKG's own disease nodes (to know which node ids are real)
  2. The final cancer-relevant disease scope (cancer_relevant_disease_ids_*.csv)
  3. The ClinVar aligned-variant CSVs (clinvar_*_aligned_variants_*.csv),
     which already carry PhenotypeIDS, VariationID, GeneSymbol, etc.
     These are: genes for which CIViC has curated at least one evidence item,
     of any evidence type.

Inputs:
    - data/raw/optimuskg/nodes/disease.parquet (from save_optimuskg_nodes.py)
    - data/processed/cancer_relevant_disease_ids_*.csv (from civic_optimuskg_alignment.py)
    - data/processed/clinvar_predisposing_aligned_variants_*.csv
    - data/processed/clinvar_oncogenic_aligned_variants_*.csv
    - data/processed/clinvar_clinical_impact_aligned_variants_*.csv
Outputs:
    - data/processed/clinvar_<track>_aligned_variants_phenotype_enriched_<date>.csv
      for each track found, with two new columns:
        resolved_phenotype_disease_ids: OptimusKG disease node ids matched
        phenotype_is_cancer_relevant: whether any matched disease is in scope

Usage:
    python src/alignment/clinvar_optimuskg_disease_alignment.py
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
PROCESSED_DIR = Path("data/processed")

# Each entry: (input filename pattern, output name fragment)
CLINVAR_ALIGNED_TRACKS = [
    ("clinvar_predisposing_aligned_variants_*.csv", "predisposing"),
    ("clinvar_oncogenic_aligned_variants_*.csv", "oncogenic"),
    ("clinvar_clinical_impact_aligned_variants_*.csv", "clinical_impact"),
]

# %%
# ---------------------------------------------------------------------------
# 1. Load OptimusKG disease node ids and the cancer-relevant disease scope
# ---------------------------------------------------------------------------

diseases = pl.read_parquet(Path.joinpath(NODES_DIR, "disease.parquet"))
diseases_id_set = set(diseases["id"].to_list())
print(f"[setup] {len(diseases_id_set)} total OptimusKG disease node ids loaded")

cancer_scope_path = find_latest_file(PROCESSED_DIR, "cancer_relevant_disease_ids_*.csv")
cancer_disease_ids_final = set(
    pd.read_csv(cancer_scope_path)["disease_node_id"].astype(str)
)
print(
    f"[setup] {len(cancer_disease_ids_final)} cancer-relevant disease node ids "
    f"loaded from {cancer_scope_path}"
)

# %%
# ---------------------------------------------------------------------------
# 2. Phenotype resolution: extract MONDO ids from ClinVar's PhenotypeIDS,
#    keep only ones that resolve to a real OptimusKG disease node, and flag
#    whether any resolved disease is cancer-relevant. Not all resolved
#    phenotypes are cancer-relevant.
# ---------------------------------------------------------------------------


def resolve_and_flag_phenotypes(phenotype_ids_raw) -> tuple:
    """Handles the doubled "MONDO:MONDO:0013342"-style prefix quirk observed
    in raw ClinVar data."""
    if pd.isna(phenotype_ids_raw):
        return [], False
    matches = re.findall(r"MONDO:(?:MONDO:)?(\d+)", str(phenotype_ids_raw))
    candidate_ids = [f"MONDO_{m}" for m in matches]
    resolved = [m for m in candidate_ids if m in diseases_id_set]
    is_cancer_relevant = any(m in cancer_disease_ids_final for m in resolved)
    return resolved, is_cancer_relevant


# %%
# ---------------------------------------------------------------------------
# 3. Apply to each ClinVar aligned-variant track found, save enriched output
# ---------------------------------------------------------------------------

date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

for pattern, track_name in CLINVAR_ALIGNED_TRACKS:
    try:
        input_path = find_latest_file(PROCESSED_DIR, pattern)
    except FileNotFoundError:
        print(f"[skip] no input file found for pattern {pattern}")
        continue

    df = pd.read_csv(input_path)
    print(f"\n[{track_name}] loaded {len(df)} rows from {input_path}")

    phenotype_results = df["PhenotypeIDS"].apply(resolve_and_flag_phenotypes)
    df["resolved_phenotype_disease_ids"] = phenotype_results.apply(lambda x: x[0])
    df["phenotype_is_cancer_relevant"] = phenotype_results.apply(lambda x: x[1])

    n_resolved = df["resolved_phenotype_disease_ids"].apply(len).gt(0).sum()
    n_cancer_relevant = df["phenotype_is_cancer_relevant"].sum()
    print(
        f"[{track_name}] {n_resolved} / {len(df)} rows have a phenotype resolving "
        f"to a real OptimusKG disease node"
    )
    print(
        f"[{track_name}] {n_cancer_relevant} / {n_resolved if n_resolved else 1} "
        f"resolved rows are cancer-relevant "
        f"({n_cancer_relevant} / {len(df)} of all rows combined)"
    )

    output_path = (
        PROCESSED_DIR
        / f"clinvar_{track_name}_aligned_variants_phenotype_enriched_{date_str}.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"[{track_name}] saved enriched output to {output_path}")
