#!/usr/bin/env python3
"""
Step 1 of building the paper's figures — run this before
src/app/candidate_browser.py (the Streamlit app the paper's actual figures
were picked and exported from). Surveys candidate evidence items for each
figure's story and prints (and saves) the top N, so a specific example can
be picked from the app. Each search dedupes to one row per EVI.

Figure numbering matches the paper: 1 mechanism of action (miss presented
before hit), 2 drug safety/contraindications, 3 disease hierarchy, 4
adverse drug interactions (synergistic interaction), 5 gene-disease
structural gap, 6 ClinVar variant gap.

src/analysis/exploratory/graph_ego_figures.py is an earlier, superseded
approach to building these same 5 figures directly via matplotlib/
networkx rather than through the Streamlit app — kept for reference, not
part of the current figure-building path. Its FIG1_HIT_EVI/FIG1_MISS_EVI/
FIG2_EVI/FIG3_EVI/FIG5_EVI constants predate this script's
PINNED_EVI_IDS mechanism below.

Figure 6 (ClinVar curation gap) is centered on a VAR node rather than an
EVI, so it doesn't fit the per-EVI pattern the other 5 figures share.

Inputs:
    - data/graphs/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
Outputs:
    - printed report
    - data/processed/find_figure_candidates/find_figure_candidates_<date>.json

Usage:
    python src/analysis/find_figure_candidates.py
"""

# %%
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

TOP_N = 15

# EVI ids already picked (by hand, from a prior run's top-N) and hardcoded
# into the paper's figures. Candidate order below is otherwise just
# "first N found walking evi_nodes" with no stable secondary sort, so a
# re-run with a different underlying node ordering can drop a previously
# -chosen example out of the saved top-N even though it still genuinely
# qualifies — pin_first() guarantees these stay in the saved list (and
# thus stay selectable in the Streamlit app) regardless.
PINNED_EVI_IDS: dict[str, list[str]] = {
    "fig1_miss": ["CIVICEVI_2880"],
    "fig2": ["CIVICEVI_1988"],
    "fig3": ["CIVICEVI_2475"],
    "fig4_combination": ["CIVICEVI_677"],
    "fig5": ["CIVICEVI_9220"],
}


def pin_first(candidates: list[tuple], pinned_ids: list[str], top_n: int) -> list[tuple]:
    """Reorders candidates (each a tuple whose first element is the EVI id)
    so any id in pinned_ids survives the top_n truncation, preserved in
    pinned_ids' own order, ahead of the rest in their original order."""
    pinned = sorted(
        (c for c in candidates if c[0] in pinned_ids),
        key=lambda c: pinned_ids.index(c[0]),
    )
    rest = [c for c in candidates if c[0] not in pinned_ids]
    return (pinned + rest)[:top_n]


OUT_DIR = Path("data/processed/find_figure_candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def paper_node_label(node_id: str, attrs: dict) -> str:
    """Same as graph_viz.py's paper_node_label() — short symbol for
    genes/drugs, protein-change notation for variants, falling back to
    the coding-level change (e.g. "c.340+1G>A") for the many ClinVar
    variants with no protein-level annotation at all (non-coding,
    intronic, splice, UTR), rather than the full raw HGVS string."""
    if attrs.get("node_type") == "VAR":
        hgvs = attrs.get("hgvs_name")
        if hgvs:
            protein_match = re.search(r"\(p\.[^)]+\)", hgvs)
            if protein_match:
                return protein_match.group(0)[1:-1]
            coding_match = re.search(r"c\..+$", hgvs)
            if coding_match:
                return coding_match.group(0)
            return str(hgvs)
        return node_id
    for key in ("symbol", "gene_symbol", "hgvs_name", "name"):
        if attrs.get(key):
            return str(attrs[key])
    return node_id


# %%
# ---------------------------------------------------------------------------
# 0. Load the cancer-scoped graph, build the same relation lookups
#    graph_ego_figures.py uses (plus dis_parent_children, the reverse of
#    dis_parent_of — Figure 3's search needs to walk both directions).
# ---------------------------------------------------------------------------

graph_path = find_latest_file("data/graphs", "cancer_scoped_graph_*.pkl")
print(f"[load] {graph_path}")
with open(graph_path, "rb") as f:
    CSG = pickle.load(f)
print(f"[load] done — {CSG.number_of_nodes()} nodes, {CSG.number_of_edges()} edges")

evi_nodes = {n: a for n, a in CSG.nodes(data=True) if a.get("node_type") == "EVI"}

drg_gen_relation: dict[tuple[str, str], str] = {}
gen_gen_adj: dict[str, set[str]] = {}
drg_phe_by_drug: dict[str, list[tuple[str, str]]] = {}
dis_parent_of: dict[str, list[str]] = {}
dis_parent_children: dict[str, list[str]] = {}
gen_dis_origin: dict[str, list[tuple[str, list]]] = {}
drg_drg_interaction: dict[frozenset, dict] = {}
for u, v, d in CSG.edges(data=True):
    et = d.get("edge_type")
    if et == "DRG-GEN" and d.get("relation"):
        drg_gen_relation[(u, v)] = d["relation"]
    elif et == "GEN-GEN":
        gen_gen_adj.setdefault(u, set()).add(v)
        gen_gen_adj.setdefault(v, set()).add(u)
    elif et == "DRG-PHE":
        drg_phe_by_drug.setdefault(u, []).append((v, d.get("relation")))
    elif et == "DIS-DIS" and d.get("relation") == "PARENT":
        dis_parent_of.setdefault(v, []).append(u)  # child -> [parents]
        dis_parent_children.setdefault(u, []).append(v)  # parent -> [children]
    elif et == "GEN-DIS":
        gen_dis_origin.setdefault(u, []).append((v, d.get("origin")))
        gen_dis_origin.setdefault(v, []).append((u, d.get("origin")))
    elif et == "DRG-DRG" and d.get("relation") == "SYNERGISTIC_INTERACTION":
        drg_drg_interaction[frozenset((u, v))] = d
print(
    f"[setup] {len(evi_nodes)} EVI nodes, {len(drg_gen_relation)} DRG-GEN relations, "
    f"{len(gen_gen_adj)} genes with GEN-GEN neighbors, "
    f"{len(drg_phe_by_drug)} drugs with DRG-PHE edges, "
    f"{len(dis_parent_of)} diseases with a PARENT, "
    f"{len(gen_dis_origin)} genes/diseases with a GEN-DIS edge, "
    f"{len(drg_drg_interaction)} drug pairs with a native SYNERGISTIC_INTERACTION"
)

report = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "graph_snapshot": Path(graph_path).name,
    "top_n": TOP_N,
}


def label(node_id: str) -> str:
    return paper_node_label(node_id, CSG.nodes[node_id])


def all_variant_labels(eid: str) -> list[str]:
    """Every variant the EVI resolves to, labeled — filtered to ids that
    are actually nodes in this cancer-scoped graph (an EVI's
    variant_node_ids can reference an id outside this graph's scoped
    variant population, same as drug_node_ids can for drugs)."""
    a = evi_nodes[eid]
    return [label(vid) for vid in (a.get("variant_node_ids") or []) if vid and vid in CSG]


# %%
# ---------------------------------------------------------------------------
# Figure 1: mechanism of action. Restricted to EVIs resolving to exactly
# one gene and one drug — the pairing has to be unambiguous for "this
# EVI's own (gene, drug) pair" to mean anything. hit = a direct native
# DRG-GEN relation between them; miss = no direct relation, but a GEN-GEN
# bridge to a gene that does have one to the same drug.
# ---------------------------------------------------------------------------

predictive = [
    (eid, a)
    for eid, a in evi_nodes.items()
    if a.get("evidence_type") == "PREDICTIVE"
    and a.get("gene_node_ids")
    and a.get("drug_node_ids")
]

hit_candidates = []
miss_candidates = []
for eid, a in predictive:
    genes = [g for g in a["gene_node_ids"] if g]
    drugs = [d for d in a["drug_node_ids"] if d]
    if len(genes) != 1 or len(drugs) != 1:
        continue
    gene, drug = genes[0], drugs[0]
    if (drug, gene) in drg_gen_relation:
        hit_candidates.append((eid, gene, drug, drg_gen_relation[(drug, gene)]))
        continue
    for other_gene in gen_gen_adj.get(gene, ()):
        if (drug, other_gene) in drg_gen_relation:
            miss_candidates.append(
                (eid, gene, drug, other_gene, drg_gen_relation[(drug, other_gene)])
            )
            break

print(
    f"\n[Fig1 miss] {len(miss_candidates)} candidates (no direct relation, "
    f"GEN-GEN bridge exists)"
)
miss_top = pin_first(miss_candidates, PINNED_EVI_IDS["fig1_miss"], TOP_N)
for eid, gene, drug, bridge, rel in miss_top:
    print(f"  {eid}: {label(gene)} / {label(drug)} — bridge {label(bridge)} ({rel})")

print(
    f"\n[Fig1 hit] {len(hit_candidates)} candidates (single-gene/drug PREDICTIVE "
    f"EVIs, direct DRG-GEN relation)"
)
for eid, gene, drug, rel in hit_candidates[:TOP_N]:
    print(f"  {eid}: {label(gene)} / {label(drug)} — {rel}")

report["fig1_miss"] = {
    "total": len(miss_candidates),
    "top": [
        {
            "evi": eid, "gene": label(gene), "drug": label(drug),
            "bridge_gene": label(bridge), "bridge_relation": rel,
            "all_variants": all_variant_labels(eid),
        }
        for eid, gene, drug, bridge, rel in miss_top
    ],
}
report["fig1_hit"] = {
    "total": len(hit_candidates),
    "top": [
        {
            "evi": eid, "gene": label(gene), "drug": label(drug), "relation": rel,
            "all_variants": all_variant_labels(eid),
        }
        for eid, gene, drug, rel in hit_candidates[:TOP_N]
    ],
}

# %%
# ---------------------------------------------------------------------------
# Figure 2: EVI with a resolved drug that has >=1 DRG-PHE CONTRAINDICATION
# edge. One row per (EVI, drug) — an EVI with two distinct resolved drugs
# both having contraindications is two genuine candidates, not a
# duplicate. Also reports the [2, 6] fan-out band used to avoid a
# starburst figure.
# ---------------------------------------------------------------------------

fig2_candidates = []
for eid, a in evi_nodes.items():
    seen_drugs = set()
    for drug in (d for d in (a.get("drug_node_ids") or []) if d):
        if drug in seen_drugs:
            continue  # drug_node_ids can carry a literal duplicate id
        seen_drugs.add(drug)
        contra = [p for p, r in drg_phe_by_drug.get(drug, []) if r == "CONTRAINDICATION"]
        if contra:
            fig2_candidates.append((eid, drug, contra))

fig2_moderate = [c for c in fig2_candidates if 2 <= len(c[2]) <= 6]
print(
    f"\n[Fig2] {len(fig2_candidates)} EVI-drug pairs with >=1 CONTRAINDICATION "
    f"DRG-PHE edge ({len(fig2_moderate)} with fan-out in [2, 6])"
)
fig2_top = pin_first(fig2_moderate, PINNED_EVI_IDS["fig2"], TOP_N)
for eid, drug, contra in fig2_top:
    phenotypes = ", ".join(label(p) for p in contra)
    print(f"  {eid}: {label(drug)} — {len(contra)} contraindications: {phenotypes}")

report["fig2"] = {
    "total": len(fig2_candidates),
    "total_with_moderate_fanout": len(fig2_moderate),
    "top": [
        {
            "evi": eid, "drug": label(drug),
            "contraindications": [label(p) for p in contra],
        }
        for eid, drug, contra in fig2_top
    ],
}

# %%
# ---------------------------------------------------------------------------
# Figure 3: EVI with a resolved disease reachable via >=1 DIS-DIS PARENT
# edge in EITHER direction (disease has a parent, or has a child — either
# way it sits inside a documented hierarchy); multi-level = walking one
# more PARENT hop (a grandparent above, or a grandchild below) succeeds.
# ---------------------------------------------------------------------------

fig3_candidates = []
fig3_multilevel = []
for eid, a in evi_nodes.items():
    dis = a.get("disease_node_id")
    if not dis or (dis not in dis_parent_of and dis not in dis_parent_children):
        continue
    fig3_candidates.append((eid, dis))
    found = False
    for parent in dis_parent_of.get(dis, []):
        if parent in dis_parent_of:  # grandparent
            fig3_multilevel.append((eid, dis, "up", parent, dis_parent_of[parent][0]))
            found = True
            break
    if not found:
        for child in dis_parent_children.get(dis, []):
            if child in dis_parent_children:  # grandchild
                fig3_multilevel.append(
                    (eid, dis, "down", child, dis_parent_children[child][0])
                )
                break

print(
    f"\n[Fig3] {len(fig3_candidates)} candidates with >=1 DIS-DIS PARENT edge "
    f"({len(fig3_multilevel)} reach a 3rd level)"
)
fig3_top = pin_first(fig3_multilevel, PINNED_EVI_IDS["fig3"], TOP_N)
for eid, dis, direction, mid, far in fig3_top:
    chain = f"{label(dis)} -> {label(mid)} -> {label(far)}" if direction == "up" else \
        f"{label(far)} -> {label(mid)} -> {label(dis)}"
    print(f"  {eid}: {chain}")

report["fig3"] = {
    "total": len(fig3_candidates),
    "total_multilevel": len(fig3_multilevel),
    "top": [
        {
            "evi": eid,
            "chain": (
                [label(dis), label(mid), label(far)] if direction == "up"
                else [label(far), label(mid), label(dis)]
            ),
        }
        for eid, dis, direction, mid, far in fig3_top
    ],
}

# %%
# ---------------------------------------------------------------------------
# Figure 4: EVI with >=2 resolved drugs where a direct native
# SYNERGISTIC_INTERACTION edge exists between two of them. Despite the
# name, this relation is sourced entirely from DrugBank's drug-drug
# INTERACTION data (checked directly: 0/157,329 descriptions carry
# positive-synergy language; all are pharmacokinetic/toxicity cautions) —
# so this is a safety flag, not a validation that a combination works.
# Split by the EVI's own therapy_interaction_type: COMBINATION (drugs
# actually co-administered — flags a real interaction risk within CIViC's
# own combination claim) vs. SUBSTITUTES (interchangeable alternatives — a
# different, not-yet-built story about flagging a risk between drugs
# that aren't meant to be combined at all). One row per EVI, first
# qualifying drug pair highlighted — but all_drugs/all_variants carry
# every drug/variant the EVI itself resolves to (not just the interacting
# pair), so the figure can show the full regimen/molecular context, not
# just the two drugs that happen to have a flagged interaction.
# ---------------------------------------------------------------------------

fig4_combination = []
fig4_substitutes = []
for eid, a in evi_nodes.items():
    drugs = [d for d in (a.get("drug_node_ids") or []) if d]
    if len(drugs) < 2:
        continue
    therapy_type = a.get("therapy_interaction_type")
    pair = None
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            if frozenset((drugs[i], drugs[j])) in drg_drg_interaction:
                pair = (drugs[i], drugs[j])
                break
        if pair:
            break
    if not pair:
        continue
    if therapy_type == "COMBINATION":
        fig4_combination.append((eid, *pair))
    elif therapy_type == "SUBSTITUTES":
        fig4_substitutes.append((eid, *pair))

print(
    f"\n[Fig4] {len(fig4_combination)} COMBINATION EVIs with a corroborating "
    f"native interaction, {len(fig4_substitutes)} SUBSTITUTES EVIs likewise"
)
fig4_combination_top = pin_first(fig4_combination, PINNED_EVI_IDS["fig4_combination"], TOP_N)
for eid, d1, d2 in fig4_combination_top:
    print(f"  {eid}: {label(d1)} + {label(d2)}")

def fig4_row(eid: str, d1: str, d2: str) -> dict:
    # An EVI's drug_node_ids/variant_node_ids can reference an id that
    # isn't actually a node in this cancer-scoped graph (e.g. a drug
    # outside its scoped drug population) — skip those rather than crash.
    a = evi_nodes[eid]
    return {
        "evi": eid,
        "drug_1": label(d1),
        "drug_2": label(d2),
        "all_drugs": [label(d) for d in (a.get("drug_node_ids") or []) if d and d in CSG],
        "all_variants": all_variant_labels(eid),
    }


report["fig4_combination"] = {
    "total": len(fig4_combination),
    "top": [fig4_row(eid, d1, d2) for eid, d1, d2 in fig4_combination_top],
}
report["fig4_substitutes"] = {
    "total": len(fig4_substitutes),
    "top": [fig4_row(eid, d1, d2) for eid, d1, d2 in fig4_substitutes[:TOP_N]],
}

# %%
# ---------------------------------------------------------------------------
# Figure 5: EVI with a resolved gene+disease where the gene has native
# GEN-DIS edges (origin == ["optimuskg"] only) to OTHER diseases — i.e.
# structurally connected but never reviewed by CIViC. Reformulated to
# surface ALL such diseases per gene (previously just the first match) —
# CIViC-aligned genes turn out to have hundreds of these on average
# (verified directly against the graph: median 610, max 1866 across the
# 337 CIViC-aligned genes), far too many to show at once, so the top
# FIG5_TOP_DISEASES are selected to prefer diverse examples: a greedy
# pick that skips a candidate disease if it shares a DIS-DIS/PARENT
# "family" with one already picked — the same connected hierarchy
# cluster, reached by walking PARENT edges in either direction, so two
# diseases that are merely siblings under a common ancestor (e.g. two
# subtypes of the same broader syndrome) are excluded from co-occurring
# too, not just direct ancestor/descendant pairs. This is deliberately
# stricter than just excluding direct ancestor/descendant pairs: checked
# directly against 5 real examples, the stricter version sometimes yields
# well under 10 truly-diverse diseases (as few as 2 for one gene whose
# candidates cluster into one or two large hereditary-syndrome families)
# — the fallback below still pads out to FIG5_TOP_DISEASES with
# (possibly related) leftovers in that case, since showing fewer than 10
# is worse than showing 10 with a few related ones mixed in.
# This is the one search that's also committed, unmodified, in
# graph_ego_figures.py's build_figure4_gene_disease_gap() (which picks a
# single hardcoded example for the paper's static figure, under that
# script's own, unrenumbered FIG4 naming) — not updated here, since this
# reformulation is specifically for find_figure_candidates.py's own
# candidate listing and the Streamlit browser, not the paper figure itself.
# ---------------------------------------------------------------------------

FIG5_TOP_DISEASES = 10

_dis_family_cache: dict[str, set[str]] = {}


def _disease_family(dis_id: str) -> set[str]:
    """Every disease reachable from dis_id by walking DIS-DIS/PARENT
    edges in either direction (ancestors, descendants, AND siblings
    reached via a shared ancestor) — the full connected hierarchy
    cluster dis_id belongs to, not just its direct ancestor/descendant
    chain."""
    seen: set[str] = {dis_id}
    frontier = [dis_id]
    while frontier:
        nxt = []
        for d in frontier:
            for neighbor in dis_parent_of.get(d, []) + dis_parent_children.get(d, []):
                if neighbor not in seen:
                    seen.add(neighbor)
                    nxt.append(neighbor)
        frontier = nxt
    seen.discard(dis_id)
    return seen


def is_hierarchy_related(a: str, b: str) -> bool:
    """True iff a and b belong to the same DIS-DIS/PARENT hierarchy
    cluster — ancestor, descendant, or sibling-via-shared-ancestor."""
    if a not in _dis_family_cache:
        _dis_family_cache[a] = _disease_family(a)
    return b in _dis_family_cache[a]


def diverse_top_n(candidates: list[str], n: int) -> list[str]:
    selected: list[str] = []
    leftover: list[str] = []
    for c in candidates:
        if len(selected) < n and not any(is_hierarchy_related(c, s) for s in selected):
            selected.append(c)
        else:
            leftover.append(c)
    if len(selected) < n:
        selected.extend(leftover[: n - len(selected)])
    return selected


fig5_candidates = []
for eid, a in evi_nodes.items():
    if not (a.get("gene_node_ids") and a.get("disease_node_id")):
        continue
    own_dis = a["disease_node_id"]
    for gene in (g for g in a["gene_node_ids"] if g):
        others = sorted(
            dis
            for dis, origin in gen_dis_origin.get(gene, [])
            if dis != own_dis and origin == ["optimuskg"]
        )
        if others:
            fig5_candidates.append((eid, gene, own_dis, others))
            break

print(f"\n[Fig5] {len(fig5_candidates)} EVIs with >=1 qualifying gene-disease gap")
fig5_top = pin_first(fig5_candidates, PINNED_EVI_IDS["fig5"], TOP_N)
for eid, gene, own_dis, others in fig5_top:
    top_others = diverse_top_n(others, FIG5_TOP_DISEASES)
    print(
        f"  {eid}: {label(gene)} — civic: {label(own_dis)}, optimuskg-only "
        f"({len(others)} total, top {len(top_others)} shown): "
        f"{', '.join(label(d) for d in top_others)}"
    )

report["fig5"] = {
    "total": len(fig5_candidates),
    "top": [
        {
            "evi": eid, "gene": label(gene),
            "civic_disease": label(own_dis),
            "optimuskg_only_diseases": [
                label(d) for d in diverse_top_n(others, FIG5_TOP_DISEASES)
            ],
            "optimuskg_only_diseases_total": len(others),
        }
        for eid, gene, own_dis, others in fig5_top
    ],
}

# %%
# ---------------------------------------------------------------------------
# Figure 6: ClinVar curation gap. A VAR node qualifies for one of three
# classification tracks (predisposing/oncogenic/clinical_impact — set by
# save_graph_nodes.py from civic_clinvar_gap_analysis.py's output) but has
# no CIViC evidence item of the matching evidence_type connected via
# VAR-EVI. This re-derives that type-matched gap check directly from the
# graph rather than trusting VAR's existing in_civic_evidence_item flag,
# which is global (true if ANY evidence item references the variant,
# regardless of type) and would under-count real gaps — e.g. a variant
# with a PREDICTIVE evidence item but no PREDISPOSING one is still a
# genuine predisposing gap. This mismatch is exactly why this search is
# being run standalone before deciding whether/how to add the
# type-matched flags onto the VAR node itself in save_graph_nodes.py.
#
# review_status_clinvar (the plain ClinVar ReviewStatus column, meant to
# mirror civic_clinvar_gap_analysis.py's high-confidence stratification
# for the predisposing track) is always None on this graph snapshot — a
# real gap in the current pipeline, not a bug in this search — so
# number_submitters_clinvar is used as the confidence proxy instead.
#
# Candidates are restricted to genes CIViC already curates (n_evidence_items
# > 0) for narrative strength: "this gene is a CIViC priority area, but this
# specific variant's <track> status has never been curated."
# ---------------------------------------------------------------------------

TRACK_REQUIRED_EVIDENCE_TYPES = {
    "predisposing": {"PREDISPOSING"},
    "oncogenic": {"ONCOGENIC"},
    "clinical_impact": {"PREDICTIVE", "DIAGNOSTIC", "PROGNOSTIC"},
}
TRACK_CLASSIFICATION_FIELD = {
    "predisposing": "clinical_significance_clinvar",
    "oncogenic": "oncogenicity",
    "clinical_impact": "somatic_clinical_impact_tier",
}

var_gene_edge: dict[str, str] = {}
var_evi_types: dict[str, set[str]] = {}
for u, v, d in CSG.edges(data=True):
    et = d.get("edge_type")
    if et == "VAR-GEN":
        var_gene_edge[u] = v
    elif et == "VAR-EVI":
        var_id, evi_id = (u, v) if CSG.nodes[u].get("node_type") == "VAR" else (v, u)
        evi_type = CSG.nodes[evi_id].get("evidence_type")
        if evi_type:
            var_evi_types.setdefault(var_id, set()).add(evi_type)

var_nodes = {n: a for n, a in CSG.nodes(data=True) if a.get("node_type") == "VAR"}
print(
    f"\n[setup] {len(var_nodes)} VAR nodes, {len(var_gene_edge)} with a VAR-GEN edge, "
    f"{len(var_evi_types)} with >=1 VAR-EVI edge"
)

fig6_candidates = []
for vid, a in var_nodes.items():
    gene_id = var_gene_edge.get(vid)
    if not gene_id:
        continue
    gene_attrs = CSG.nodes[gene_id]
    n_gene_evidence = gene_attrs.get("n_evidence_items") or 0
    if n_gene_evidence == 0:
        continue
    for track in a.get("tracks") or []:
        required_types = TRACK_REQUIRED_EVIDENCE_TYPES.get(track)
        if not required_types:
            continue
        if var_evi_types.get(vid, set()) & required_types:
            continue  # covered by a matching-type evidence item — not a gap
        fig6_candidates.append(
            {
                "variant": vid,
                "gene": label(gene_id),
                "track": track,
                "classification": a.get(TRACK_CLASSIFICATION_FIELD[track]),
                "hgvs_name": a.get("hgvs_name"),
                "number_submitters_clinvar": a.get("number_submitters_clinvar") or 0,
                "gene_n_evidence_items": n_gene_evidence,
            }
        )

fig6_candidates.sort(
    key=lambda c: (c["gene_n_evidence_items"], c["number_submitters_clinvar"]),
    reverse=True,
)

# Deduped-by-gene view for browsing: sorting by gene_n_evidence_items alone
# makes the raw top-N a wall of the single most CIViC-studied gene (VHL, 659
# evidence items, dwarfing every other gene) — one entry per gene instead,
# keeping each gene's best-evidenced (most ClinVar submitters) gap variant,
# surfaces a diverse set of genes to pick an example from.
best_per_gene: dict[str, dict] = {}
for c in fig6_candidates:
    existing = best_per_gene.get(c["gene"])
    if existing is None or c["number_submitters_clinvar"] > existing["number_submitters_clinvar"]:
        best_per_gene[c["gene"]] = c
fig6_candidates_by_gene = sorted(
    best_per_gene.values(),
    key=lambda c: (c["gene_n_evidence_items"], c["number_submitters_clinvar"]),
    reverse=True,
)

print(
    f"\n[Fig6] {len(fig6_candidates)} ClinVar curation-gap candidates (type-matched, "
    f"on genes CIViC already curates), {len(fig6_candidates_by_gene)} distinct genes"
)
print(f"[Fig6] top {TOP_N} by raw count (gene-dominated):")
for c in fig6_candidates[:TOP_N]:
    print(
        f"  {c['variant']} ({c['gene']}): {c['track']} = {c['classification']} — "
        f"gene has {c['gene_n_evidence_items']} CIViC evidence items, "
        f"{c['number_submitters_clinvar']} ClinVar submitters"
    )

print(f"\n[Fig6] top {TOP_N} diversified (one candidate per gene):")
for c in fig6_candidates_by_gene[:TOP_N]:
    print(
        f"  {c['variant']} ({c['gene']}): {c['track']} = {c['classification']} — "
        f"gene has {c['gene_n_evidence_items']} CIViC evidence items, "
        f"{c['number_submitters_clinvar']} ClinVar submitters"
    )

report["fig6"] = {
    "total": len(fig6_candidates),
    "total_distinct_genes": len(fig6_candidates_by_gene),
    "top": fig6_candidates[:TOP_N],
    "top_by_gene": fig6_candidates_by_gene[:TOP_N],
}

report_path = OUT_DIR / f"find_figure_candidates_{datetime.now(timezone.utc):%Y-%m-%d}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[save] wrote {report_path}")

print("\n[done]")

# %%
