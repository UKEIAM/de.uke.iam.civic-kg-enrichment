#!/usr/bin/env python3
"""
EXPLORATORY / SUPERSEDED: the paper's published figures were ultimately
picked and exported from src/app/candidate_browser.py (Streamlit), not
from this script. Kept for reference only.

5 purpose-built figures:
1. mechanism of action (hit/miss pair)
2. drug safety/usage
3. disease hierarchy
4. gene-disease structural gap
5. drug-drug interaction safety flag.

These source from cancer_scoped_graph_*.pkl

Output is static and vector (SVG + PDF), not the generic picker's
interactive HTML;
relation labels and the node-type legend are baked directly into the
image. Shared node-type color/label/tooltip/legend/export helpers live in
src/utils/graph_viz.py — this file keeps only what's specific to an
EVI-centered shell-layout figure (draw_ego_panel, other_resolved_nodes,
has_relation_at_hop) and the 5 build_figureN functions themselves.

Figures 1-3 and 5's example evidence items (FIG1_HIT_EVI/FIG1_MISS_EVI/
FIG2_EVI/FIG3_EVI/FIG5_EVI below) are hardcoded picks from the candidate
pools find_figure_candidates.py surveys — run that script first to see the
full pool and choose a different example. Figure 4 needs no manual pick:
its own search is deterministic and runs live below.

Inputs:
    - data/graphs/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
Outputs:
    - data/processed/ego_networks/fig{1,2,3,4,5}_*.{svg,pdf} (the print figures)
    - data/processed/ego_networks/fig{1,2,3,4,5}_*.html (interactive pyvis
      counterparts of the same subgraphs, for exploration)

Usage:
    python src/analysis/exploratory/graph_ego_figures.py
"""

# %%
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from graph_viz import (
    CONTEXT_COLOR,
    GAP_COLOR,
    HIGHLIGHT_COLOR,
    NODE_COLORS,
    NODE_MARKERS,
    NODE_TYPE_ORDER,
    context_legend_handle,
    edge_legend_handle,
    node_label,
    node_legend_handles,
    paper_node_label,
    render_interactive_panel,
    save_static_figure,
)
from load_files import find_latest_file

OUT_DIR = Path("data/processed/ego_networks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_LABEL = "other entity resolved by this evidence item (not highlighted)"


def other_resolved_nodes(evi_attrs: dict, exclude: set, include: set) -> list[str]:
    """Every other gene/drug/disease/variant an EVI resolves to — narrowed
    to `include` (any of {"gene", "drug", "disease", "variant"}, since
    each figure only wants the entity types it isn't already showing as
    part of the highlighted story) and excluding ids already shown —
    returned for display as muted context nodes so the EVI's full
    real-world resolution is visible, not just the one relation the
    figure exists to illustrate."""
    ids = []
    if "gene" in include:
        ids += list(evi_attrs.get("gene_node_ids") or [])
    if "drug" in include:
        ids += list(evi_attrs.get("drug_node_ids") or [])
    if "disease" in include and evi_attrs.get("disease_node_id"):
        ids.append(evi_attrs["disease_node_id"])
    if "variant" in include:
        ids += list(evi_attrs.get("variant_node_ids") or [])
    return [i for i in ids if i and i not in exclude]


def has_relation_at_hop(
    G: nx.DiGraph,
    center_id: str,
    mid_type: str,
    target_type: str,
    relation_field: str = "relation",
    relation_value=None,
    exclude_node: str | None = None,
):
    """Does center_id have a mid_type neighbor that itself connects to a
    target_type node via an edge whose relation_field == relation_value (or
    any value, if relation_value is None)? exclude_node lets a caller rule
    out a node it already knows about (Figure 4 excludes the EVI's own
    disease, so the search finds a genuinely different one).

    Returns the (mid, target) pair found, or None — unlike the boolean the
    figures spec's own pseudocode returns, the figures below need to know
    *which* mid/target nodes satisfied the check to actually draw them."""
    for mid in list(G.successors(center_id)) + list(G.predecessors(center_id)):
        if G.nodes[mid].get("node_type") != mid_type:
            continue
        for far in list(G.successors(mid)) + list(G.predecessors(mid)):
            if far == center_id or far == exclude_node:
                continue
            if G.nodes[far].get("node_type") != target_type:
                continue
            edge_data = G.get_edge_data(mid, far) or G.get_edge_data(far, mid)
            if (
                relation_value is None
                or edge_data.get(relation_field) == relation_value
            ):
                return mid, far
    return None


def draw_ego_panel(
    ax,
    sub: nx.DiGraph,
    nlist: list[list[str]],
    highlighted_edges: list[dict],
    synthetic_edges: list[dict] | None = None,
    center_id: str | None = None,
    panel_title: str | None = None,
    context_node_ids: set | None = None,
    force_center_view: bool = True,
) -> None:
    """Draw one figure/panel: concentric shell layout by hop distance,
    context edges muted, the figure's specific edge(s) highlighted and
    labeled with their relation. synthetic_edges draws a line between two
    node positions that isn't a real edge in sub (Figure 1 panel B's "no
    direct edge" connector) — nx.draw_networkx_edges only needs pos for
    the endpoints, not a real G edge, so this works the same way.
    context_node_ids are drawn in muted gray regardless of node_type —
    other entities the center EVI resolves to that aren't part of this
    figure's highlighted story (see other_resolved_nodes())."""
    context_node_ids = context_node_ids or set()
    pos = nx.shell_layout(sub, nlist=nlist)
    # shell_layout places a single-node ring 0 at a small nonzero radius,
    # not the origin, once there are 3+ rings (only coincidentally at
    # (0, 0) for a 2-ring layout) — force the actual center node there so
    # "evidence item in the center" holds regardless of ring count.
    if center_id is not None and center_id in pos:
        pos[center_id] = (0.0, 0.0)
    # Without this, matplotlib's default unequal-aspect autoscale stretches
    # a near-zero-variance axis (e.g. a 2-node ring, which shell_layout
    # places ~180° apart at y≈0) to fill the plot — circles render as
    # ellipses, and the fixed-size label offset below ends up scaled far
    # outside the visible canvas on the near-flat axis.
    ax.set_aspect("equal", adjustable="datalim")

    highlighted_pairs = {(h["u"], h["v"]) for h in highlighted_edges} | {
        (h["v"], h["u"]) for h in highlighted_edges
    }
    # Self-loops are real data (e.g. JAK2's own GEN-GEN "INTERACTS_WITH"
    # homodimer record from PRIMEKG) but add nothing to any of these 4
    # figures' stories — drawing them just looks like a rendering glitch.
    context_edges = [
        (u, v) for u, v in sub.edges() if u != v and (u, v) not in highlighted_pairs
    ]
    nx.draw_networkx_edges(
        sub,
        pos,
        edgelist=context_edges,
        ax=ax,
        edge_color=CONTEXT_COLOR,
        width=1.0,
        arrows=True,
        arrowsize=8,
        connectionstyle="arc3,rad=0.05",
    )

    for h in highlighted_edges:
        style = h.get("style", "solid")
        nx.draw_networkx_edges(
            sub,
            pos,
            edgelist=[(h["u"], h["v"])],
            ax=ax,
            edge_color=h.get("color", HIGHLIGHT_COLOR),
            width=h.get("width", 2.5),
            style=style,
            arrows=True,
            arrowsize=14,
        )
        if h.get("label"):
            nx.draw_networkx_edge_labels(
                sub,
                pos,
                edge_labels={(h["u"], h["v"]): h["label"]},
                ax=ax,
                font_size=8,
                font_color=h.get("color", HIGHLIGHT_COLOR),
            )

    for s in synthetic_edges or []:
        nx.draw_networkx_edges(
            sub,
            pos,
            edgelist=[(s["u"], s["v"])],
            ax=ax,
            edge_color=s.get("color", GAP_COLOR),
            width=s.get("width", 1.5),
            style=s.get("style", "dashed"),
            arrows=False,
        )
        if s.get("label"):
            (x0, y0), (x1, y1) = pos[s["u"]], pos[s["v"]]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            # Offset radially outward from the center node rather than
            # sitting exactly on the connector's midpoint — the midpoint
            # of a synthetic gene-drug connector can otherwise land right
            # on top of the center EVI (forced to (0, 0) above): an
            # *arbitrary* perpendicular offset has a 50/50 chance of
            # picking the side that points straight at it, so this picks
            # the one direction guaranteed to point away instead.
            ccx, ccy = pos.get(center_id, (0.0, 0.0))
            rx, ry = mx - ccx, my - ccy
            rlen = (rx**2 + ry**2) ** 0.5 or 1.0
            offset = 0.15
            ax.text(
                mx + rx / rlen * offset,
                my + ry / rlen * offset,
                s["label"],
                fontsize=7,
                style="italic",
                color=s.get("color", GAP_COLOR),
                ha="center",
                va="center",
                backgroundcolor="white",
            )

    for node_type, marker in NODE_MARKERS.items():
        nodes_of_type = [
            n for n in sub.nodes() if sub.nodes[n].get("node_type") == node_type
        ]
        if not nodes_of_type:
            continue
        sizes = [700 if n == center_id else 260 for n in nodes_of_type]
        colors = [
            CONTEXT_COLOR
            if n in context_node_ids
            else NODE_COLORS.get(node_type, "#828C97")
            for n in nodes_of_type
        ]
        nx.draw_networkx_nodes(
            sub,
            pos,
            nodelist=nodes_of_type,
            ax=ax,
            node_color=colors,
            node_shape=marker,
            node_size=sizes,
            edgecolors="black",
            linewidths=0.5,
        )

    # Labels offset above the marker (rather than centered on it) so text
    # doesn't sit directly on top of the node glyph; a white backing box
    # keeps them legible where a context edge crosses underneath. The
    # offset scales down with ring count — a fixed 0.09 is fine for a
    # sparse 2-3 ring layout (Figs 1/2/4) but is comparable to the
    # inter-ring spacing on Figure 3's 9-ring chain, which visually
    # drifted labels toward the neighboring ring's node.
    label_offset = min(0.09, 0.4 / max(len(nlist), 1))
    label_pos = {n: (x, y + label_offset) for n, (x, y) in pos.items()}
    labels = {n: paper_node_label(n, sub.nodes[n]) for n in sub.nodes()}
    nx.draw_networkx_labels(
        sub,
        label_pos,
        labels=labels,
        ax=ax,
        font_size=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
    )

    # shell_layout packs nodes into a fixed radius with no inherent margin,
    # so a long edge-of-plot label clips against the axes border without
    # this — margins expand the data limits, not the node layout itself.
    if force_center_view and center_id is not None and center_id in pos:
        # Keep the center node visually centered in the rendered panel,
        # not just at (0, 0) in data coordinates — a lopsided ring (e.g.
        # Figure 1 panel B's single-node bridge-gene ring sitting far to
        # one side) otherwise drags the rendered view off-center even
        # though the center node's own coordinate is correct. Only makes
        # sense for a roughly star-shaped subgraph, though — forcing a
        # square window symmetric around the center wastes most of the
        # canvas on a one-sided layout like Figure 3's arc chain, which
        # opts out via force_center_view=False and keeps the plain-margin
        # behavior below instead.
        cx, cy = pos[center_id]
        max_extent = max(max(abs(x - cx), abs(y - cy)) for x, y in pos.values())
        radius = max_extent * 1.5 + 0.15
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
    else:
        ax.margins(0.22)
        # Extra headroom above the highest node specifically — a long
        # chain's topmost label (Figure 3's arc peak) can otherwise
        # collide with the panel title sitting just above the axes.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))
    if panel_title:
        ax.set_title(panel_title, fontsize=9, wrap=True)
    ax.axis("off")


# %%
# ---------------------------------------------------------------------------
# 0. Load the cancer-scoped graph, build the relation lookups every figure
# below shares.
# ---------------------------------------------------------------------------

cancer_scoped_graph_path = find_latest_file(
    "data/graphs", "cancer_scoped_graph_*.pkl"
)
print(f"[load] {cancer_scoped_graph_path}")
with open(cancer_scoped_graph_path, "rb") as f:
    CSG = pickle.load(f)
print(f"[load] done — {CSG.number_of_nodes()} nodes, {CSG.number_of_edges()} edges")

print("\n[setup] building relation lookups from the cancer-scoped graph")
drg_gen_relation: dict[tuple[str, str], str] = {}
gen_gen_adj: dict[str, set[str]] = {}
drg_phe_by_drug: dict[str, list[tuple[str, str]]] = {}
dis_parent_of: dict[str, list[str]] = {}
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
        # from = parent, to = child — confirmed against the raw nodes'
        # own children/parents attrs (see the plan's verification table),
        # NOT the reverse "child is_a parent" reading MONDO's own RDF uses.
        dis_parent_of.setdefault(v, []).append(u)
    elif et == "GEN-DIS":
        gen_dis_origin.setdefault(u, []).append((v, d.get("origin")))
        gen_dis_origin.setdefault(v, []).append((u, d.get("origin")))
    elif et == "DRG-DRG" and d.get("relation") == "SYNERGISTIC_INTERACTION":
        # undirected=True (canonicalized pair) — keyed by frozenset so
        # lookup doesn't depend on which drug happened to sort first.
        drg_drg_interaction[frozenset((u, v))] = d
print(
    f"[setup] {len(drg_gen_relation)} DRG-GEN relations, "
    f"{len(gen_gen_adj)} genes with GEN-GEN neighbors, "
    f"{len(drg_phe_by_drug)} drugs with DRG-PHE edges, "
    f"{len(dis_parent_of)} diseases with a PARENT, "
    f"{len(gen_dis_origin)} genes/diseases with a GEN-DIS edge, "
    f"{len(drg_drg_interaction)} drug pairs with a native SYNERGISTIC_INTERACTION"
)

# %%
# ---------------------------------------------------------------------------
# Figure 1: mechanism of action, hit vs. miss.
# ---------------------------------------------------------------------------

FIG1_HIT_EVI = "CIVICEVI_1598"  # BRAF / vemurafenib, colorectal cancer
FIG1_MISS_EVI = "CIVICEVI_5914"  # BRCA1 / olaparib, pancreatic neoplasm


def build_figure1_moa() -> None:
    if FIG1_HIT_EVI not in CSG or FIG1_MISS_EVI not in CSG:
        raise RuntimeError(
            "Figure 1's example EVIs are missing from the current "
            "cancer_scoped_graph snapshot."
        )

    # gene_node_ids/drug_node_ids are lists (an EVI can resolve to more than
    # one of either) — search every (gene, drug) combination rather than
    # assuming index 0 is "the" pair; that assumption is what silently
    # picked the wrong drug the first time this ran (CIVICEVI_1598's first
    # listed drug wasn't vemurafenib).
    hit_attrs = CSG.nodes[FIG1_HIT_EVI]
    hit_pair = None
    for hit_gene in (g for g in hit_attrs["gene_node_ids"] if g):
        for hit_drug in (dgg for dgg in hit_attrs["drug_node_ids"] if dgg):
            if (hit_drug, hit_gene) in drg_gen_relation:
                hit_pair = (hit_gene, hit_drug, drg_gen_relation[(hit_drug, hit_gene)])
                break
        if hit_pair:
            break
    if hit_pair is None:
        raise RuntimeError(
            f"No direct DRG-GEN relation found for {FIG1_HIT_EVI} in the current snapshot."
        )
    hit_gene, hit_drug, hit_relation = hit_pair

    miss_attrs = CSG.nodes[FIG1_MISS_EVI]
    bridge = None
    for miss_gene in (g for g in miss_attrs["gene_node_ids"] if g):
        for miss_drug in (dgg for dgg in miss_attrs["drug_node_ids"] if dgg):
            if (miss_drug, miss_gene) in drg_gen_relation:
                continue  # this pair is a hit, not a miss
            for other_gene in gen_gen_adj.get(miss_gene, ()):
                if (miss_drug, other_gene) in drg_gen_relation:
                    bridge = (
                        miss_gene,
                        miss_drug,
                        other_gene,
                        drg_gen_relation[(miss_drug, other_gene)],
                    )
                    break
            if bridge:
                break
        if bridge:
            break
    if bridge is None:
        raise RuntimeError(
            f"No GEN-GEN bridge found for {FIG1_MISS_EVI} in the current snapshot."
        )
    miss_gene, miss_drug, bridge_gene, bridge_relation = bridge

    hit_highlighted = [{"u": hit_drug, "v": hit_gene, "label": hit_relation}]
    miss_highlighted = [
        {"u": miss_gene, "v": bridge_gene, "label": "interacts with"},
        {"u": miss_drug, "v": bridge_gene, "label": bridge_relation},
    ]
    miss_synthetic = [
        {
            "u": miss_gene,
            "v": miss_drug,
            "label": "no direct MoA edge",
            "color": GAP_COLOR,
        },
    ]
    edge_legend = [
        (HIGHLIGHT_COLOR, "solid", "documented DRG-GEN mechanism"),
        (GAP_COLOR, "dashed", "no direct mechanism edge"),
    ]

    # Every other gene/drug/variant these two EVIs resolve to, beyond the
    # one (gene, drug) pair each panel highlights — shown as muted context
    # so the panel isn't a misleadingly narrow view of what CIViC actually
    # curated for that evidence item.
    entity_types = {"gene", "drug", "disease", "variant"}
    hit_extra = other_resolved_nodes(
        hit_attrs, exclude={hit_gene, hit_drug}, include=entity_types
    )
    miss_extra = other_resolved_nodes(
        miss_attrs, exclude={miss_gene, miss_drug, bridge_gene}, include=entity_types
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 10))

    sub_a = CSG.subgraph([FIG1_HIT_EVI, hit_gene, hit_drug, *hit_extra])
    draw_ego_panel(
        ax_a,
        sub_a,
        [[FIG1_HIT_EVI], [hit_gene, hit_drug, *hit_extra]],
        highlighted_edges=hit_highlighted,
        center_id=FIG1_HIT_EVI,
        panel_title=f"A — hit: {node_label(hit_gene, CSG.nodes[hit_gene])} / "
        f"{node_label(hit_drug, CSG.nodes[hit_drug])}",
        context_node_ids=set(hit_extra),
    )

    sub_b = CSG.subgraph(
        [FIG1_MISS_EVI, miss_gene, miss_drug, bridge_gene, *miss_extra]
    )
    draw_ego_panel(
        ax_b,
        sub_b,
        [[FIG1_MISS_EVI], [miss_gene, miss_drug, *miss_extra], [bridge_gene]],
        highlighted_edges=miss_highlighted,
        synthetic_edges=miss_synthetic,
        center_id=FIG1_MISS_EVI,
        panel_title=f"B — miss: {node_label(miss_gene, CSG.nodes[miss_gene])} / "
        f"{node_label(miss_drug, CSG.nodes[miss_drug])}",
        context_node_ids=set(miss_extra),
    )

    types_present = sorted(
        {sub_a.nodes[n]["node_type"] for n in sub_a.nodes()}
        | {sub_b.nodes[n]["node_type"] for n in sub_b.nodes()},
        key=NODE_TYPE_ORDER.index,
    )
    legend = node_legend_handles(types_present)
    if hit_extra or miss_extra:
        legend.append(context_legend_handle(CONTEXT_LABEL))
    legend += [edge_legend_handle(*e) for e in edge_legend]
    fig.suptitle("Figure 1 — Mechanism of action: hit vs. miss", fontsize=12)
    save_static_figure(fig, legend, str(OUT_DIR / "fig1_mechanism_of_action"))

    render_interactive_panel(
        sub_a,
        hit_highlighted,
        OUT_DIR / "fig1_mechanism_of_action_hit.html",
        center_id=FIG1_HIT_EVI,
        title="Figure 1A — hit: mechanism of action",
        context_node_ids=set(hit_extra),
        context_label=CONTEXT_LABEL if hit_extra else None,
        legend_edges=edge_legend,
    )
    render_interactive_panel(
        sub_b,
        miss_highlighted,
        OUT_DIR / "fig1_mechanism_of_action_miss.html",
        synthetic_edges=miss_synthetic,
        center_id=FIG1_MISS_EVI,
        title="Figure 1B — miss: mechanism of action",
        context_node_ids=set(miss_extra),
        context_label=CONTEXT_LABEL if miss_extra else None,
        legend_edges=edge_legend,
    )


# %%
# ---------------------------------------------------------------------------
# Figure 2: drug safety/usage context (DRG-PHE, CONTRAINDICATION).
# ADVERSE_DRUG_REACTION doesn't exist in native DRG-PHE data at all — see
# the plan's verification table; CONTRAINDICATION is the real, populated
# relation used here instead.
# ---------------------------------------------------------------------------

FIG2_EVI = "CIVICEVI_1395"  # gefitinib, 2 contraindicated phenotypes
FIG2_MAX_CONTRAINDICATIONS = 5


def build_figure2_drug_safety() -> None:
    if FIG2_EVI not in CSG:
        raise RuntimeError(
            f"Figure 2's vetted example EVI {FIG2_EVI} is missing from the current snapshot."
        )
    # drug_node_ids is a list — search every resolved drug rather than
    # assuming index 0 is the one with contraindications (same fix as
    # Figure 1's gene/drug pairing).
    drug, contra = None, []
    for candidate_drug in (d for d in CSG.nodes[FIG2_EVI]["drug_node_ids"] if d):
        # No independent priority/severity signal in the data — capped
        # deterministically by node id, not a clinical-severity ordering.
        candidate_contra = sorted(
            p
            for p, r in drg_phe_by_drug.get(candidate_drug, [])
            if r == "CONTRAINDICATION"
        )
        if candidate_contra:
            drug, contra = candidate_drug, candidate_contra[:FIG2_MAX_CONTRAINDICATIONS]
            break
    if drug is None:
        raise RuntimeError(
            f"{FIG2_EVI} has no drug with a CONTRAINDICATION DRG-PHE edge in the current snapshot."
        )

    highlighted = [{"u": drug, "v": p, "label": "CONTRAINDICATION"} for p in contra]
    edge_legend = [(HIGHLIGHT_COLOR, "solid", "DRG-PHE: CONTRAINDICATION")]

    # The EVI's disease, gene(s), and variant(s) — not part of the
    # contraindication story, but shown muted so the figure doesn't look
    # like the drug is this EVI's only resolved entity.
    extra = other_resolved_nodes(
        CSG.nodes[FIG2_EVI], exclude={drug}, include={"disease", "gene", "variant"}
    )

    sub = CSG.subgraph([FIG2_EVI, drug, *extra, *contra])
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_ego_panel(
        ax,
        sub,
        [[FIG2_EVI], [drug, *extra], contra],
        highlighted_edges=highlighted,
        center_id=FIG2_EVI,
        panel_title=f"{node_label(drug, CSG.nodes[drug])} — documented contraindications",
        context_node_ids=set(extra),
    )
    types_present = sorted(
        {sub.nodes[n]["node_type"] for n in sub.nodes()}, key=NODE_TYPE_ORDER.index
    )
    legend = node_legend_handles(types_present)
    if extra:
        legend.append(context_legend_handle(CONTEXT_LABEL))
    legend += [edge_legend_handle(*e) for e in edge_legend]
    fig.suptitle("Figure 2 — Drug safety/usage context", fontsize=12)
    save_static_figure(fig, legend, str(OUT_DIR / "fig2_drug_safety_context"))

    render_interactive_panel(
        sub,
        highlighted,
        OUT_DIR / "fig2_drug_safety_context.html",
        center_id=FIG2_EVI,
        title="Figure 2 — Drug safety/usage context",
        context_node_ids=set(extra),
        context_label=CONTEXT_LABEL if extra else None,
        legend_edges=edge_legend,
    )


# %%
# ---------------------------------------------------------------------------
# Figure 3: disease hierarchy (DIS-DIS, PARENT). Walks the PARENT chain as
# far as it goes — no hardcoded hop ceiling, since the cancer-scoped graph
# (unlike the expanded graph) carries a real multi-level MONDO lineage.
# ---------------------------------------------------------------------------

FIG3_EVI = "CIVICEVI_7864"  # childhood ALL -> childhood leukemia -> leukemia


def build_figure3_disease_hierarchy() -> None:
    if FIG3_EVI not in CSG:
        raise RuntimeError(
            f"Figure 3's vetted example EVI {FIG3_EVI} is missing from the current snapshot."
        )
    dis = CSG.nodes[FIG3_EVI]["disease_node_id"]
    chain = [dis]
    node = dis
    while node in dis_parent_of and dis_parent_of[node]:  # noqa: RUF019
        node = dis_parent_of[node][0]
        chain.append(node)
    if len(chain) < 2:
        raise RuntimeError(
            f"{FIG3_EVI}'s disease has no DIS-DIS PARENT edge in the current snapshot."
        )

    sub = CSG.subgraph([FIG3_EVI, *chain])
    nlist = [[FIG3_EVI]] + [[n] for n in chain]
    highlighted = [
        {"u": chain[i + 1], "v": chain[i], "label": "PARENT"}
        for i in range(len(chain) - 1)
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    draw_ego_panel(
        ax,
        sub,
        nlist,
        highlighted_edges=highlighted,
        center_id=FIG3_EVI,
        force_center_view=False,
        panel_title=" → ".join(
            paper_node_label(n, CSG.nodes[n]) for n in reversed(chain)
        ),
    )
    edge_legend = [
        (HIGHLIGHT_COLOR, "solid", "DIS-DIS: PARENT (from=parent, to=child)")
    ]
    types_present = sorted(
        {sub.nodes[n]["node_type"] for n in sub.nodes()}, key=NODE_TYPE_ORDER.index
    )
    legend = node_legend_handles(types_present) + [
        edge_legend_handle(*e) for e in edge_legend
    ]
    fig.suptitle("Figure 3 — Disease hierarchy", fontsize=12)
    save_static_figure(fig, legend, str(OUT_DIR / "fig3_disease_hierarchy"))

    render_interactive_panel(
        sub,
        highlighted,
        OUT_DIR / "fig3_disease_hierarchy.html",
        center_id=FIG3_EVI,
        title="Figure 3 — Disease hierarchy",
        legend_edges=edge_legend,
    )


# %%
# ---------------------------------------------------------------------------
# Figure 4: gene-disease structural gap (GEN-DIS, origin). Essentially any
# EVI with a resolved gene+disease works (3,927/3,927 per
# find_figure_candidates.py's fig4 search) — picked deterministically
# (sorted EVI id) for reproducibility rather than hardcoding one specific id.
# ---------------------------------------------------------------------------


def build_figure4_gene_disease_gap() -> None:
    evi_candidates = sorted(
        n
        for n, a in CSG.nodes(data=True)
        if a.get("node_type") == "EVI"
        and a.get("gene_node_ids")
        and a.get("disease_node_id")
    )
    candidate = None
    for eid in evi_candidates:
        attrs = CSG.nodes[eid]
        own_dis = attrs["disease_node_id"]
        for gene in (g for g in attrs["gene_node_ids"] if g):
            others = sorted(
                dis
                for dis, origin in gen_dis_origin.get(gene, [])
                if dis != own_dis and origin == ["optimuskg"]
            )
            if others:
                candidate = (eid, gene, own_dis, others[0])
                break
        if candidate:
            break
    if candidate is None:
        raise RuntimeError("No Figure 4 candidate found in the current snapshot.")
    evi, gene, civic_dis, gap_dis = candidate

    highlighted = [
        {"u": gene, "v": civic_dis, "label": "civic"},
        {
            "u": gene,
            "v": gap_dis,
            "label": "optimuskg only",
            "color": GAP_COLOR,
            "style": "dashed",
        },
    ]
    edge_legend = [
        (HIGHLIGHT_COLOR, "solid", 'origin includes "civic" (this EVI\'s own pair)'),
        (GAP_COLOR, "dashed", 'origin == ["optimuskg"] only (structural, uncurated)'),
    ]

    # The EVI's variant(s) and drug(s) — not part of the gene-disease gap
    # story, but shown muted so the figure doesn't look like gene+disease
    # is all this evidence item resolves to.
    extra = other_resolved_nodes(
        CSG.nodes[evi], exclude={gene, civic_dis, gap_dis}, include={"variant", "drug"}
    )

    sub = CSG.subgraph([evi, gene, *extra, civic_dis, gap_dis])
    fig, ax = plt.subplots(figsize=(11, 11))
    draw_ego_panel(
        ax,
        sub,
        [[evi], [gene, *extra], [civic_dis, gap_dis]],
        highlighted_edges=highlighted,
        center_id=evi,
        panel_title=f"{node_label(gene, CSG.nodes[gene])}: curated vs. structural disease association",
        context_node_ids=set(extra),
    )
    types_present = sorted(
        {sub.nodes[n]["node_type"] for n in sub.nodes()}, key=NODE_TYPE_ORDER.index
    )
    legend = node_legend_handles(types_present)
    if extra:
        legend.append(context_legend_handle(CONTEXT_LABEL))
    legend += [edge_legend_handle(*e) for e in edge_legend]
    fig.suptitle("Figure 4 — Gene–disease structural gap", fontsize=12)
    save_static_figure(fig, legend, str(OUT_DIR / "fig4_gene_disease_gap"))

    render_interactive_panel(
        sub,
        highlighted,
        OUT_DIR / "fig4_gene_disease_gap.html",
        center_id=evi,
        title="Figure 4 — Gene–disease structural gap",
        context_node_ids=set(extra),
        context_label=CONTEXT_LABEL if extra else None,
        legend_edges=edge_legend,
    )


# %%
# ---------------------------------------------------------------------------
# Figure 5: drug-drug interaction safety flag (DRG-DRG,
# SYNERGISTIC_INTERACTION). Despite its name, this native relation is
# sourced entirely from DrugBank's drug-drug INTERACTION data, not a
# documented therapeutic-synergy claim — checked directly: 0 of 157,329
# descriptions carry positive/beneficial-synergy language, and every one
# is a pharmacokinetic/toxicity caution ("risk of adverse effects can be
# increased," "serum concentration/metabolism altered," etc.). So this
# figure isn't validating that a CIViC-cited combination is effective —
# it's flagging that OptimusKG's independent pharmacology data documents a
# real interaction *risk* between two drugs this evidence item uses
# together, the same spirit as Figure 2's contraindication story but at
# the drug-pair level. (118 COMBINATION EVIs have a corroborating native
# interaction edge, per find_figure_candidates.py's fig5 search; a further
# 186 EVIs have one too but under CIViC's SUBSTITUTES claim instead —
# interchangeable alternatives, not co-administration — a different,
# not-yet-built story about flagging a risk between drugs that aren't
# meant to be combined at all.)
# ---------------------------------------------------------------------------

FIG5_EVI = "CIVICEVI_6938"  # BRAF V600E melanoma, dabrafenib + trametinib (COMBINATION)


def build_figure5_drug_interaction_safety() -> None:
    if FIG5_EVI not in CSG:
        raise RuntimeError(
            f"Figure 5's vetted example EVI {FIG5_EVI} is missing from the current snapshot."
        )
    attrs = CSG.nodes[FIG5_EVI]
    drugs = [d for d in (attrs.get("drug_node_ids") or []) if d]
    interaction = None
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            interaction = drg_drg_interaction.get(frozenset((drugs[i], drugs[j])))
            if interaction:
                drug1, drug2 = drugs[i], drugs[j]
                break
        if interaction:
            break
    if interaction is None:
        raise RuntimeError(
            f"No native SYNERGISTIC_INTERACTION edge found between {FIG5_EVI}'s drugs in the current snapshot."
        )

    highlighted = [{"u": drug1, "v": drug2, "label": "SYNERGISTIC_INTERACTION"}]
    edge_legend = [
        (
            HIGHLIGHT_COLOR,
            "solid",
            "DRG-DRG: SYNERGISTIC_INTERACTION — DrugBank interaction caution, not an efficacy claim",
        )
    ]

    # The EVI's gene, disease, and variant(s) — not part of the drug-pair
    # story, but shown muted so the figure doesn't look like the two drugs
    # are all this evidence item resolves to.
    extra = other_resolved_nodes(
        attrs, exclude={drug1, drug2}, include={"gene", "disease", "variant"}
    )

    sub = CSG.subgraph([FIG5_EVI, drug1, drug2, *extra])
    fig, ax = plt.subplots(figsize=(11, 11))
    draw_ego_panel(
        ax,
        sub,
        [[FIG5_EVI], [drug1, drug2, *extra]],
        highlighted_edges=highlighted,
        center_id=FIG5_EVI,
        panel_title=interaction["interaction_description"],
        context_node_ids=set(extra),
    )
    types_present = sorted(
        {sub.nodes[n]["node_type"] for n in sub.nodes()}, key=NODE_TYPE_ORDER.index
    )
    legend = node_legend_handles(types_present)
    if extra:
        legend.append(context_legend_handle(CONTEXT_LABEL))
    legend += [edge_legend_handle(*e) for e in edge_legend]
    fig.suptitle("Figure 5 — Drug-drug interaction safety flag", fontsize=12)
    save_static_figure(fig, legend, str(OUT_DIR / "fig5_drug_interaction"))

    render_interactive_panel(
        sub,
        highlighted,
        OUT_DIR / "fig5_drug_interaction.html",
        center_id=FIG5_EVI,
        title="Figure 5 — Drug-drug interaction safety flag",
        context_node_ids=set(extra),
        context_label=CONTEXT_LABEL if extra else None,
        legend_edges=edge_legend,
    )


# %%
# ---------------------------------------------------------------------------
# Build all 5. Each raises loudly if its vetted example (or, for Figure 4,
# the deterministic search) no longer holds against the current
# cancer_scoped_graph snapshot, rather than silently substituting an
# unreviewed example.
# ---------------------------------------------------------------------------

for build_fn in (
    build_figure1_moa,
    build_figure2_drug_safety,
    build_figure3_disease_hierarchy,
    build_figure4_gene_disease_gap,
    build_figure5_drug_interaction_safety,
):
    build_fn()

print("\n[done]")

# %%
