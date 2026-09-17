#!/usr/bin/env python3
"""
Step 1 of building the paper's 5 figures — run this before
graph_ego_figures.py. Surveys candidate evidence items for each figure's
story and prints (and saves) the top N, so a specific example can be
picked and hardcoded into graph_ego_figures.py's FIG1_HIT_EVI/
FIG1_MISS_EVI/FIG2_EVI/FIG3_EVI/FIG5_EVI (Figure 4 needs no manual pick —
its own search is deterministic and reproduced here too, for visibility
into alternatives). Not imported from graph_ego_figures.py (that script
builds all 5 figures on import) — the small shared bits are duplicated
here instead. Each search dedupes to one row per EVI.

Inputs:
    - data/processed/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
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

OUT_DIR = Path("data/processed/find_figure_candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def paper_node_label(node_id: str, attrs: dict) -> str:
    """Same as graph_ego_figures.py's paper_node_label() — short symbol
    for genes/drugs, protein-change notation for variants."""
    if attrs.get("node_type") == "VAR":
        hgvs = attrs.get("hgvs_name")
        if hgvs:
            match = re.search(r"\(p\.[^)]+\)", hgvs)
            return match.group(0)[1:-1] if match else str(hgvs)
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

graph_path = find_latest_file("data/processed", "cancer_scoped_graph_*.pkl")
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
    f"\n[Fig1 hit] {len(hit_candidates)} candidates (single-gene/drug PREDICTIVE "
    f"EVIs, direct DRG-GEN relation)"
)
for eid, gene, drug, rel in hit_candidates[:TOP_N]:
    print(f"  {eid}: {label(gene)} / {label(drug)} — {rel}")

print(
    f"\n[Fig1 miss] {len(miss_candidates)} candidates (no direct relation, "
    f"GEN-GEN bridge exists)"
)
for eid, gene, drug, bridge, rel in miss_candidates[:TOP_N]:
    print(f"  {eid}: {label(gene)} / {label(drug)} — bridge {label(bridge)} ({rel})")

report["fig1_hit"] = {
    "total": len(hit_candidates),
    "top": [
        {"evi": eid, "gene": label(gene), "drug": label(drug), "relation": rel}
        for eid, gene, drug, rel in hit_candidates[:TOP_N]
    ],
}
report["fig1_miss"] = {
    "total": len(miss_candidates),
    "top": [
        {
            "evi": eid, "gene": label(gene), "drug": label(drug),
            "bridge_gene": label(bridge), "bridge_relation": rel,
        }
        for eid, gene, drug, bridge, rel in miss_candidates[:TOP_N]
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
for eid, drug, contra in fig2_moderate[:TOP_N]:
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
        for eid, drug, contra in fig2_moderate[:TOP_N]
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
for eid, dis, direction, mid, far in fig3_multilevel[:TOP_N]:
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
        for eid, dis, direction, mid, far in fig3_multilevel[:TOP_N]
    ],
}

# %%
# ---------------------------------------------------------------------------
# Figure 4: EVI with a resolved gene+disease where the gene has another
# GEN-DIS edge (origin == ["optimuskg"] only) to a DIFFERENT disease. This
# is the one search that's also committed, unmodified, in
# graph_ego_figures.py's build_figure4_gene_disease_gap() — reproduced
# here (same one-row-per-EVI, first-qualifying-gene behavior) to show
# alternatives beyond its first match.
# ---------------------------------------------------------------------------

fig4_candidates = []
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
            fig4_candidates.append((eid, gene, own_dis, others[0]))
            break

print(f"\n[Fig4] {len(fig4_candidates)} EVIs with >=1 qualifying gene-disease gap")
for eid, gene, own_dis, gap_dis in fig4_candidates[:TOP_N]:
    print(f"  {eid}: {label(gene)} — civic: {label(own_dis)}, optimuskg-only: {label(gap_dis)}")

report["fig4"] = {
    "total": len(fig4_candidates),
    "top": [
        {
            "evi": eid, "gene": label(gene),
            "civic_disease": label(own_dis), "optimuskg_only_disease": label(gap_dis),
        }
        for eid, gene, own_dis, gap_dis in fig4_candidates[:TOP_N]
    ],
}

# %%
# ---------------------------------------------------------------------------
# Figure 5: EVI with >=2 resolved drugs where a direct native
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
# qualifying drug pair.
# ---------------------------------------------------------------------------

fig5_combination = []
fig5_substitutes = []
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
        fig5_combination.append((eid, *pair))
    elif therapy_type == "SUBSTITUTES":
        fig5_substitutes.append((eid, *pair))

print(
    f"\n[Fig5] {len(fig5_combination)} COMBINATION EVIs with a corroborating "
    f"native interaction, {len(fig5_substitutes)} SUBSTITUTES EVIs likewise"
)
for eid, d1, d2 in fig5_combination[:TOP_N]:
    print(f"  {eid}: {label(d1)} + {label(d2)}")

report["fig5_combination"] = {
    "total": len(fig5_combination),
    "top": [{"evi": eid, "drug_1": label(d1), "drug_2": label(d2)} for eid, d1, d2 in fig5_combination[:TOP_N]],
}
report["fig5_substitutes"] = {
    "total": len(fig5_substitutes),
    "top": [{"evi": eid, "drug_1": label(d1), "drug_2": label(d2)} for eid, d1, d2 in fig5_substitutes[:TOP_N]],
}

report_path = OUT_DIR / f"find_figure_candidates_{datetime.now(timezone.utc):%Y-%m-%d}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[save] wrote {report_path}")

print("\n[done]")

# %%
