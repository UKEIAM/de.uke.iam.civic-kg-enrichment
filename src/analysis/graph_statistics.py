#!/usr/bin/env python3
"""
Overall statistics of the cancer-scoped graph for the paper's Results section:
basic graph counts, entity-resolution coverage, variant-resolution coverage
(reported separately from entity resolution — see section 4b), ClinVar
gap-detection coverage, origin=civic-only edge counts, catalog/variant
curation coverage, disease-hierarchy richness, and the hit/miss rates behind
Figures 1, 2, and 5.

Figures 1/2/5 here are NOT the same computation as find_figure_candidates.py:
that script picks good individual examples across ALL evidence types (for
Figs 2/5) and doesn't track a "neither" bucket (for Fig 1). This script restates
all three PREDICTIVE-only, scoped exactly as the paper's results text needs.

ClinVar gap-detection coverage (section 5) is restricted to the ~53k variants
in the original CIViC-aligned-gene tracks (predisposing/oncogenic/
clinical_impact). The broader ~74.6k-variant cancer_scoped_plp_hiconf
population build_cancer_scoped_variants.py adds on top was explicitly ruled
out of scope for gap detection in an earlier conversation; those variants have
is_gap_candidate=None (never evaluated) and are reported as a separate
"not_evaluated" count, never folded into covered or gap.

Inputs:
    - data/graphs/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
    - data/processed/civic_resolved_*.csv (from civic_optimuskg_alignment.py)
    - data/processed/civic_predictive_unresolved_drugs_*.csv (from civic_optimuskg_alignment.py)
Outputs:
    - printed report
    - data/processed/graph_statistics/graph_statistics_<date>.json

Usage:
    python src/analysis/graph_statistics.py
"""

# %%
import ast
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

OUT_DIR = Path("data/processed/graph_statistics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAP_DETECTION_TRACKS = {
    "predisposing": {"PREDISPOSING"},
    "oncogenic": {"ONCOGENIC"},
    "clinical_impact": {"PREDICTIVE", "DIAGNOSTIC", "PROGNOSTIC"},
}

# Per-edge-type CIViC-only origin tag — NOT one literal string. VAR-DIS edges
# use a different origin vocabulary (civic_evidence/clinvar_phenotype, from
# clinvar_optimuskg_disease_alignment.py) than GEN-DIS/VAR-DRG (civic/
# optimuskg, from civic_optimuskg_alignment.py).
CIVIC_ONLY_ORIGIN = {
    "GEN-DIS": ["civic"],
    "VAR-DIS": ["civic_evidence"],
    "VAR-DRG": ["civic"],
}

# %%
# ---------------------------------------------------------------------------
# 0. Load the cancer-scoped graph. Node/edge pass builds everything every
#    section below needs — same relation-lookup shapes as
#    find_figure_candidates.py, plus the per-track/origin/coverage counters.
# ---------------------------------------------------------------------------

graph_path = find_latest_file("data/graphs", "cancer_scoped_graph_*.pkl")
print(f"[load] {graph_path}")
with open(graph_path, "rb") as f:
    CSG = pickle.load(f)
print(f"[load] done — {CSG.number_of_nodes()} nodes, {CSG.number_of_edges()} edges")

report = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "graph_snapshot": Path(graph_path).name,
}

# %%
# Node pass: evi_nodes/var_nodes lookups, nodes_by_type, catalog coverage (sec 7)
evi_nodes: dict[str, dict] = {}
var_nodes: dict[str, dict] = {}
node_type_counts = Counter()
catalog_total = Counter()
catalog_with_evidence = Counter()
CATALOG_TYPES = {"GEN", "DIS", "DRG"}

for n, a in CSG.nodes(data=True):
    nt = a.get("node_type")
    node_type_counts[nt] += 1
    if nt == "EVI":
        evi_nodes[n] = a
    elif nt == "VAR":
        var_nodes[n] = a
    if nt in CATALOG_TYPES:
        catalog_total[nt] += 1
        if (a.get("n_evidence_items") or 0) > 0:
            catalog_with_evidence[nt] += 1

print(f"[setup] {len(evi_nodes)} EVI nodes, {len(var_nodes)} VAR nodes")

# %%
# Edge pass: relation lookups (same shapes as find_figure_candidates.py),
# edges_by_type, origin=civic-only counters, VAR-EVI endpoint set.
drg_gen_relation: dict[tuple[str, str], str] = {}
gen_gen_adj: dict[str, set[str]] = {}
drg_phe_by_drug: dict[str, list[tuple[str, str]]] = {}
drg_drg_interaction: dict[frozenset, dict] = {}
dis_parent_of: dict[str, list[str]] = {}
dis_parent_children: dict[str, list[str]] = {}
edge_type_counts = Counter()
civic_only_origin_counts = Counter()
edge_type_totals_for_origin = Counter()
var_evi_endpoints: set[str] = set()

for u, v, d in CSG.edges(data=True):
    et = d.get("edge_type")
    edge_type_counts[et] += 1

    if et == "DRG-GEN" and d.get("relation"):
        drg_gen_relation[(u, v)] = d["relation"]
    elif et == "GEN-GEN":
        gen_gen_adj.setdefault(u, set()).add(v)
        gen_gen_adj.setdefault(v, set()).add(u)
    elif et == "DRG-PHE":
        drg_phe_by_drug.setdefault(u, []).append((v, d.get("relation")))
    elif et == "DRG-DRG" and d.get("relation") == "SYNERGISTIC_INTERACTION":
        drg_drg_interaction[frozenset((u, v))] = d
    elif et == "DIS-DIS" and d.get("relation") == "PARENT":
        dis_parent_of.setdefault(v, []).append(u)  # child -> [parents]
        dis_parent_children.setdefault(u, []).append(v)  # parent -> [children]
    elif et == "VAR-EVI":
        var_evi_endpoints.add(u)
        var_evi_endpoints.add(v)

    if et in CIVIC_ONLY_ORIGIN:
        edge_type_totals_for_origin[et] += 1
        if d.get("origin") == CIVIC_ONLY_ORIGIN[et]:
            civic_only_origin_counts[et] += 1

print(
    f"[setup] {len(drg_gen_relation)} DRG-GEN relations, "
    f"{len(gen_gen_adj)} genes with GEN-GEN neighbors, "
    f"{len(drg_phe_by_drug)} drugs with DRG-PHE edges, "
    f"{len(drg_drg_interaction)} drug pairs with SYNERGISTIC_INTERACTION, "
    f"{len(dis_parent_of)} diseases with a PARENT"
)

report["basic"] = {
    "n_nodes": CSG.number_of_nodes(),
    "n_edges": CSG.number_of_edges(),
    "nodes_by_type": dict(node_type_counts),
    "edges_by_type": dict(edge_type_counts),
}
print(f"\n[basic] {report['basic']['n_nodes']} nodes, {report['basic']['n_edges']} edges")
print(f"[basic] nodes_by_type: {report['basic']['nodes_by_type']}")
print(f"[basic] edges_by_type: {report['basic']['edges_by_type']}")

# %%
# ---------------------------------------------------------------------------
# 1. Figure 1 — mechanism-of-action hit / miss / neither (PREDICTIVE,
#    single-gene/single-drug EVIs — same filter as
#    find_figure_candidates.py's Figure 1 search).
# ---------------------------------------------------------------------------

pop1 = []
for eid, a in evi_nodes.items():
    if a.get("evidence_type") != "PREDICTIVE":
        continue
    genes = [g for g in (a.get("gene_node_ids") or []) if g]
    drugs = [d for d in (a.get("drug_node_ids") or []) if d]
    if len(genes) == 1 and len(drugs) == 1:
        pop1.append((eid, genes[0], drugs[0]))

n_hit = n_miss = n_neither = 0
for eid, gene, drug in pop1:
    if (drug, gene) in drg_gen_relation:
        n_hit += 1
        continue
    if any((drug, other) in drg_gen_relation for other in gen_gen_adj.get(gene, ())):
        n_miss += 1
        continue
    n_neither += 1

assert n_hit + n_miss + n_neither == len(pop1)

report["fig1"] = {
    "population": len(pop1),
    "hit": {"n": n_hit, "pct": 100 * n_hit / len(pop1)},
    "miss": {"n": n_miss, "pct": 100 * n_miss / len(pop1)},
    "neither": {"n": n_neither, "pct": 100 * n_neither / len(pop1)},
}
print(
    f"\n[fig1] population={len(pop1)} (PREDICTIVE, single gene + single drug) — "
    f"hit={n_hit} ({report['fig1']['hit']['pct']:.1f}%), "
    f"miss={n_miss} ({report['fig1']['miss']['pct']:.1f}%), "
    f"neither={n_neither} ({report['fig1']['neither']['pct']:.1f}%)"
)

# %%
# ---------------------------------------------------------------------------
# 2. Figure 2 — contraindications, PREDICTIVE-scoped. Counts EVIs (not
#    EVI-drug pairs) with >=1 CONTRAINDICATION among their resolved drugs.
# ---------------------------------------------------------------------------

fig2_denominator = 0
fig2_hit = 0
for eid, a in evi_nodes.items():
    if a.get("evidence_type") != "PREDICTIVE":
        continue
    drugs = {d for d in (a.get("drug_node_ids") or []) if d}
    if not drugs:
        continue
    fig2_denominator += 1
    has_contra = any(
        r == "CONTRAINDICATION"
        for drug in drugs
        for _, r in drg_phe_by_drug.get(drug, [])
    )
    if has_contra:
        fig2_hit += 1

report["fig2"] = {
    "denominator": fig2_denominator,
    "n": fig2_hit,
    "pct": 100 * fig2_hit / fig2_denominator if fig2_denominator else None,
}
print(
    f"\n[fig2] denominator={fig2_denominator} (PREDICTIVE, >=1 resolved drug) — "
    f"{fig2_hit} ({report['fig2']['pct']:.1f}%) have >=1 documented contraindication"
)

# %%
# ---------------------------------------------------------------------------
# 5. Figure 5 — synergistic interaction, PREDICTIVE-scoped. Counts EVIs
#    with >=1 qualifying drug pair, among EVIs with >=2 resolved drugs.
# ---------------------------------------------------------------------------

fig5_denominator = 0
fig5_hit = 0
fig5_by_therapy_type = Counter()
for eid, a in evi_nodes.items():
    if a.get("evidence_type") != "PREDICTIVE":
        continue
    drugs = [d for d in (a.get("drug_node_ids") or []) if d]
    if len(drugs) < 2:
        continue
    fig5_denominator += 1
    has_interaction = False
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            if frozenset((drugs[i], drugs[j])) in drg_drg_interaction:
                has_interaction = True
                break
        if has_interaction:
            break
    if has_interaction:
        fig5_hit += 1
        fig5_by_therapy_type[a.get("therapy_interaction_type")] += 1

report["fig5"] = {
    "denominator": fig5_denominator,
    "n": fig5_hit,
    "pct": 100 * fig5_hit / fig5_denominator if fig5_denominator else None,
    "by_therapy_interaction_type": dict(fig5_by_therapy_type),
}
print(
    f"\n[fig5] denominator={fig5_denominator} (PREDICTIVE, >=2 resolved drugs) — "
    f"{fig5_hit} ({report['fig5']['pct']:.1f}%) have >=1 documented synergistic "
    f"interaction; by therapy_interaction_type: {dict(fig5_by_therapy_type)}"
)

# %%
# ---------------------------------------------------------------------------
# 4. Entity resolution coverage — from civic_resolved_*.csv, upstream of the
#    graph. drug_resolution_ok/fully_resolved recomputed verbatim from
#    civic_optimuskg_alignment.py rather than trusted from the stored CSV.
# ---------------------------------------------------------------------------

civic_resolved_path = find_latest_file("data/processed", "civic_resolved_*.csv")
print(f"\n[load] {civic_resolved_path}")
civic_df = pd.read_csv(civic_resolved_path)
# drug_resolved_via_list round-trips through to_csv/read_csv as a Python-list
# repr string (same quirk as gene_node_ids/drug_node_ids elsewhere in this
# pipeline) — parse it back before any PREDICTIVE-only subset is taken from
# civic_df, so that subset doesn't keep a stale unparsed copy of the column.
civic_df["drug_resolved_via_list"] = civic_df["drug_resolved_via_list"].apply(ast.literal_eval)

recomputed_drug_ok = civic_df.apply(
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
recomputed_fully_resolved = (
    civic_df["disease_node_id"].notna()
    & civic_df["gene_all_resolved"]
    & recomputed_drug_ok
)

# Free sanity check: the CSV already carries its own drug_resolution_ok/
# fully_resolved columns from civic_optimuskg_alignment.py — confirm this
# recomputation matches rather than silently drifting from the source formula.
for col, recomputed in (
    ("drug_resolution_ok", recomputed_drug_ok),
    ("fully_resolved", recomputed_fully_resolved),
):
    if col in civic_df.columns:
        n_mismatch = int((civic_df[col].astype(bool) != recomputed).sum())
        if n_mismatch:
            print(f"[warning] {n_mismatch} rows where recomputed {col} differs from stored CSV column")

civic_df["drug_resolution_ok"] = recomputed_drug_ok
civic_df["fully_resolved"] = recomputed_fully_resolved

n_total = len(civic_df)
predictive_df = civic_df[civic_df["evidence_type"] == "PREDICTIVE"]

# Per-slot drug resolution tiers (PREDICTIVE only — drug resolution doesn't
# apply to other evidence types). Each PREDICTIVE evidence item's
# drug_resolved_via_list has one entry per resolved-or-not drug slot.
drug_resolved_via_counts = Counter()
for lst in predictive_df["drug_resolved_via_list"]:
    drug_resolved_via_counts.update(lst)
n_drug_slots_predictive = sum(drug_resolved_via_counts.values())

# Residual unresolved-slot classification — why a specific unresolved drug
# slot couldn't be matched, computed only for PREDICTIVE evidence items with
# >=1 unresolved slot (civic_optimuskg_alignment.py section 9). "resolved"
# entries are placeholders for slots that DID resolve on an otherwise-
# unresolved row, and are excluded here — only genuine failure reasons count.
unresolved_drugs_path = find_latest_file(
    "data/processed", "civic_predictive_unresolved_drugs_*.csv"
)
unresolved_drugs_df = pd.read_csv(unresolved_drugs_path)
unresolved_drugs_df["drug_unresolved_categories"] = unresolved_drugs_df[
    "drug_unresolved_categories"
].apply(ast.literal_eval)
drug_unresolved_category_counts = Counter()
for lst in unresolved_drugs_df["drug_unresolved_categories"]:
    drug_unresolved_category_counts.update(cat for cat in lst if cat != "resolved")

# Evidence items with zero gene candidates at all (n_genes_total==0) are not
# resolution failures — civic_optimuskg_alignment.py never got a gene to try
# resolving in the first place. Overwhelmingly gene fusions (molecular_profile
# contains "::", e.g. "BCR::ABL1 Fusion"), whose fusion-partner genes aren't
# captured in CIViC's variant_feature_entrez_ids the way single-gene variants
# are; the rest are gene-independent biomarkers (TMB, MSI, methylation
# signatures, etc.) that genuinely have no associated gene.
no_gene_df = civic_df[civic_df["n_genes_total"] == 0]
no_gene_is_fusion = no_gene_df["molecular_profile"].astype(str).str.contains("::", na=False)
n_no_gene = len(no_gene_df)
n_no_gene_fusion = int(no_gene_is_fusion.sum())
n_no_gene_other = n_no_gene - n_no_gene_fusion

report["resolution_coverage"] = {
    "n_total_evidence_items": n_total,
    "disease": {
        "n": int(civic_df["disease_node_id"].notna().sum()),
        "pct": float(civic_df["disease_node_id"].notna().mean() * 100),
    },
    "disease_resolved_via": {
        via: {"n": int(n), "pct": float(n / n_total * 100)}
        for via, n in civic_df["disease_resolved_via"].value_counts().items()
    },
    "gene_all_resolved": {
        "n": int(civic_df["gene_all_resolved"].sum()),
        "pct": float(civic_df["gene_all_resolved"].mean() * 100),
    },
    "gene_any_resolved": {
        "n": int(civic_df["gene_any_resolved"].sum()),
        "pct": float(civic_df["gene_any_resolved"].mean() * 100),
    },
    "gene_missing_entirely": {
        "n": n_no_gene,
        "pct": float(n_no_gene / n_total * 100),
        "fusion_molecular_profile": {
            "n": n_no_gene_fusion,
            "pct": float(n_no_gene_fusion / n_total * 100),
        },
        "other_gene_independent_profile": {
            "n": n_no_gene_other,
            "pct": float(n_no_gene_other / n_total * 100),
        },
    },
    "n_predictive": len(predictive_df),
    "drug_all_resolved_predictive_only": {
        "n": int(predictive_df["drug_all_resolved"].sum()),
        "pct": float(predictive_df["drug_all_resolved"].mean() * 100),
    },
    "drug_any_resolved_predictive_only": {
        "n": int(predictive_df["drug_any_resolved"].sum()),
        "pct": float(predictive_df["drug_any_resolved"].mean() * 100),
    },
    "drug_resolved_via_predictive_only": {
        via: {"n": int(n), "pct": float(n / n_drug_slots_predictive * 100)}
        for via, n in drug_resolved_via_counts.items()
    },
    "drug_unresolved_categories": {
        cat: {
            "n": int(n),
            "pct": float(n / sum(drug_unresolved_category_counts.values()) * 100),
        }
        for cat, n in drug_unresolved_category_counts.items()
    },
    "fully_resolved": {
        "n": int(civic_df["fully_resolved"].sum()),
        "pct": float(civic_df["fully_resolved"].mean() * 100),
    },
}
rc = report["resolution_coverage"]
print(
    f"[resolution] {n_total} total evidence items — "
    f"disease: {rc['disease']['n']} ({rc['disease']['pct']:.1f}%), "
    f"gene (all): {rc['gene_all_resolved']['n']} ({rc['gene_all_resolved']['pct']:.1f}%), "
    f"gene (any): {rc['gene_any_resolved']['n']} ({rc['gene_any_resolved']['pct']:.1f}%)"
)
print("[resolution] disease_resolved_via breakdown:")
for via, stats in rc["disease_resolved_via"].items():
    print(f"  {via}: {stats['n']} ({stats['pct']:.1f}%)")
gm = rc["gene_missing_entirely"]
print(
    f"[resolution] gene missing entirely (no candidate to resolve): {gm['n']} "
    f"({gm['pct']:.1f}%) — fusion molecular profile: "
    f"{gm['fusion_molecular_profile']['n']} ({gm['fusion_molecular_profile']['pct']:.1f}%), "
    f"other gene-independent profile: {gm['other_gene_independent_profile']['n']} "
    f"({gm['other_gene_independent_profile']['pct']:.1f}%)"
)
print(
    f"[resolution] PREDICTIVE ({rc['n_predictive']}) drug (all): "
    f"{rc['drug_all_resolved_predictive_only']['n']} "
    f"({rc['drug_all_resolved_predictive_only']['pct']:.1f}%), drug (any): "
    f"{rc['drug_any_resolved_predictive_only']['n']} "
    f"({rc['drug_any_resolved_predictive_only']['pct']:.1f}%)"
)
print(f"[resolution] drug_resolved_via breakdown ({n_drug_slots_predictive} drug slots, PREDICTIVE):")
for via, stats in rc["drug_resolved_via_predictive_only"].items():
    print(f"  {via}: {stats['n']} ({stats['pct']:.1f}%)")
print(
    f"[resolution] unresolved drug slot categories "
    f"({sum(drug_unresolved_category_counts.values())} unresolved slots):"
)
for cat, stats in rc["drug_unresolved_categories"].items():
    print(f"  {cat}: {stats['n']} ({stats['pct']:.1f}%)")
print(
    f"[resolution] fully resolved: {rc['fully_resolved']['n']} "
    f"({rc['fully_resolved']['pct']:.1f}%)"
)

# %%
# ---------------------------------------------------------------------------
# 4b. Variant resolution — reported separately from fully_resolved (disease/
#     gene/drug) rather than folded in. Variant-node resolution is scoped far
#     more narrowly (save_graph_nodes.py's 3-tier ClinVar VariationID
#     matching, restricted to pathogenic/oncogenic/Tier I-III variants on
#     CIViC-aligned genes) than disease/gene/drug resolution — an evidence
#     item with zero resolved variant is often out of scope by design
#     (gene-level or fusion evidence with no single citable variant), not a
#     resolution failure, so conflating it with fully_resolved would
#     misrepresent resolution quality.
# ---------------------------------------------------------------------------

fully_resolved_by_civic_id = dict(zip(civic_df["id"].astype(str), civic_df["fully_resolved"]))

n_evi_total = len(evi_nodes)
n_with_variant = 0
n_fully_resolved_and_variant = 0
for eid, a in evi_nodes.items():
    civic_id = eid.removeprefix("CIVICEVI_")
    has_variant = len(a.get("variant_node_ids") or []) > 0
    if has_variant:
        n_with_variant += 1
        if fully_resolved_by_civic_id.get(civic_id):
            n_fully_resolved_and_variant += 1

report["variant_resolution"] = {
    "n_total_evidence_items": n_evi_total,
    "with_resolved_variant": {
        "n": n_with_variant,
        "pct": 100 * n_with_variant / n_evi_total if n_evi_total else None,
    },
    "fully_resolved_including_variant": {
        "n": n_fully_resolved_and_variant,
        "pct": 100 * n_fully_resolved_and_variant / n_evi_total if n_evi_total else None,
    },
    "note": (
        "Not folded into resolution_coverage.fully_resolved — variant "
        "resolution is scoped far more narrowly (Phase 1 pathogenic/oncogenic/"
        "Tier I-III variants on CIViC-aligned genes only) than disease/gene/"
        "drug resolution, so a missing variant is often an out-of-scope "
        "evidence item (gene-level/fusion), not a resolution failure."
    ),
}
vr = report["variant_resolution"]
print(
    f"\n[variant_resolution] {vr['with_resolved_variant']['n']} / {n_evi_total} "
    f"({vr['with_resolved_variant']['pct']:.2f}%) evidence items resolve to >=1 "
    "ClinVar variant node; combined with disease+gene+drug: "
    f"{vr['fully_resolved_including_variant']['n']} / {n_evi_total} "
    f"({vr['fully_resolved_including_variant']['pct']:.2f}%) fully resolved "
    "including variant"
)

# %%
# ---------------------------------------------------------------------------
# 5. ClinVar gap-detection coverage — restricted to the predisposing/
#    oncogenic/clinical_impact tracks (the CIViC-aligned-gene population).
#    cancer_scoped_plp_hiconf is explicitly out of scope — reported as
#    "not_evaluated", never folded into covered or gap.
# ---------------------------------------------------------------------------

per_track = {}
for track in GAP_DETECTION_TRACKS:
    y = 0
    x_covered = 0
    for n, a in var_nodes.items():
        if track not in (a.get("tracks") or []):
            continue
        y += 1
        if a.get(f"{track}_gap_candidate") is False:
            x_covered += 1
    per_track[track] = {
        "covered": x_covered,
        "total": y,
        "pct_covered": 100 * x_covered / y if y else None,
    }

gap_tracks = set(GAP_DETECTION_TRACKS)
y_any = x_any = gap_any = not_evaluated = 0
for n, a in var_nodes.items():
    tracks = set(a.get("tracks") or [])
    if tracks & gap_tracks:
        y_any += 1
        if a.get("is_gap_candidate") is False:
            x_any += 1
        elif a.get("is_gap_candidate") is True:
            gap_any += 1
    if a.get("is_gap_candidate") is None:
        not_evaluated += 1

report["clinvar_gap_coverage"] = {
    "per_track": per_track,
    "any_track": {
        "total": y_any,
        "covered": x_any,
        "gap": gap_any,
        "pct_covered": 100 * x_any / y_any if y_any else None,
    },
    "not_evaluated": {
        "n": not_evaluated,
        "note": (
            "VAR nodes with is_gap_candidate=None — the cancer_scoped_plp_hiconf "
            "population, explicitly out of scope for gap detection, not folded "
            "into covered or gap above"
        ),
    },
}
print("\n[clinvar_gap] per-track coverage (covered / total qualifying variants):")
for track, stats in per_track.items():
    print(f"  {track}: {stats['covered']} / {stats['total']} ({stats['pct_covered']:.2f}%)")
print(
    f"[clinvar_gap] any track: {x_any} / {y_any} covered "
    f"({report['clinvar_gap_coverage']['any_track']['pct_covered']:.2f}%), "
    f"{gap_any} gap, {not_evaluated} not evaluated (cancer_scoped_plp_hiconf)"
)

# %%
# ---------------------------------------------------------------------------
# 6. Origin=civic-only edges — GEN-DIS and VAR-DRG use origin==["civic"];
#    VAR-DIS uses a different vocabulary, origin==["civic_evidence"].
# ---------------------------------------------------------------------------

origin_report = {}
for et, tag in CIVIC_ONLY_ORIGIN.items():
    total = edge_type_totals_for_origin.get(et, 0)
    n = civic_only_origin_counts.get(et, 0)
    origin_report[et] = {
        "n": n,
        "total": total,
        "pct": 100 * n / total if total else None,
        "origin_tag_used": tag,
    }
origin_report["note"] = (
    "VAR-DIS does not use the civic/optimuskg origin vocabulary GEN-DIS and "
    "VAR-DRG use (it's set by clinvar_optimuskg_disease_alignment.py, a "
    "different script) — its CIViC-only equivalent is origin==['civic_evidence'], "
    "not origin==['civic']."
)
report["origin_civic_only_edges"] = origin_report
print("\n[origin_civic_only] edges with a CIViC-only origin tag:")
for et in CIVIC_ONLY_ORIGIN:
    r = origin_report[et]
    print(f"  {et}: {r['n']} / {r['total']} ({r['pct']:.4f}%) [tag={r['origin_tag_used']}]")

# %%
# ---------------------------------------------------------------------------
# 7. Evidence-backed catalog coverage per node type — of OptimusKG's own
#    GEN/DIS/DRG catalog, what fraction CIViC evidence actually touches.
# ---------------------------------------------------------------------------

catalog_coverage = {
    nt: {
        "n": catalog_with_evidence.get(nt, 0),
        "total": catalog_total.get(nt, 0),
        "pct": 100 * catalog_with_evidence.get(nt, 0) / catalog_total[nt]
        if catalog_total.get(nt)
        else None,
    }
    for nt in CATALOG_TYPES
}
report["catalog_coverage"] = catalog_coverage
print("\n[catalog_coverage] fraction of OptimusKG's own catalog CIViC touches:")
for nt, r in catalog_coverage.items():
    print(f"  {nt}: {r['n']} / {r['total']} ({r['pct']:.2f}%)")

# %%
# ---------------------------------------------------------------------------
# 8. Variant-level CIViC curation coverage — flat yes/no curation-touch
#    rate, cross-checked two independent ways.
# ---------------------------------------------------------------------------

n_var_total = len(var_nodes)
n_var_in_evidence_attr = sum(
    1 for a in var_nodes.values() if a.get("in_civic_evidence_item") is True
)
n_var_via_edges = sum(1 for n in var_nodes if n in var_evi_endpoints)
assert n_var_in_evidence_attr == n_var_via_edges, (
    f"in_civic_evidence_item flag ({n_var_in_evidence_attr}) disagrees with "
    f"VAR-EVI edge endpoints ({n_var_via_edges}) — investigate before trusting "
    "this number"
)

report["variant_coverage"] = {
    "n": n_var_in_evidence_attr,
    "total": n_var_total,
    "pct": 100 * n_var_in_evidence_attr / n_var_total if n_var_total else None,
    "cross_check_via_var_evi_edges": n_var_via_edges,
}
print(
    f"\n[variant_coverage] {n_var_in_evidence_attr} / {n_var_total} "
    f"({report['variant_coverage']['pct']:.2f}%) VAR nodes referenced by >=1 "
    f"CIViC evidence item (cross-checked via VAR-EVI edges: match)"
)

# %%
# ---------------------------------------------------------------------------
# 9. Disease hierarchy richness — among EVIs with a resolved disease, how
#    many have a disease with total DIS-DIS/PARENT degree (parents +
#    children) >= 2.
# ---------------------------------------------------------------------------

evi_with_disease = [
    (eid, a["disease_node_id"]) for eid, a in evi_nodes.items() if a.get("disease_node_id")
]
n_rich = 0
for eid, dis_id in evi_with_disease:
    degree = len(dis_parent_of.get(dis_id, [])) + len(dis_parent_children.get(dis_id, []))
    if degree >= 2:
        n_rich += 1

report["disease_hierarchy"] = {
    "denominator": len(evi_with_disease),
    "n": n_rich,
    "pct": 100 * n_rich / len(evi_with_disease) if evi_with_disease else None,
}
print(
    f"\n[disease_hierarchy] denominator={len(evi_with_disease)} (EVIs with a "
    f"resolved disease) — {n_rich} ({report['disease_hierarchy']['pct']:.2f}%) "
    f"have a disease with total PARENT-degree >= 2"
)

# %%
report["civic_resolved_snapshot"] = Path(civic_resolved_path).name
report["civic_predictive_unresolved_drugs_snapshot"] = Path(unresolved_drugs_path).name
report_path = OUT_DIR / f"graph_statistics_{datetime.now(timezone.utc):%Y-%m-%d}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[save] wrote {report_path}")

print("\n[done]")

# %%
