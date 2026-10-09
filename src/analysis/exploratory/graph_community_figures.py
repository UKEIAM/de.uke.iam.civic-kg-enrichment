#!/usr/bin/env python3
"""
3 concrete illustrations of graph_communities.py's structural-gap finding
— that finding is entirely aggregate (community sizes and curated
fractions), this puts real nodes/edges on the page:

1. The largest fully evidence-backed community (id 27, size 8) — every
   Leiden community graph_communities.py flags as "100% curated" is tiny
   (max size 8), so the whole thing fits on one figure.
2. A small, fully-drawable structural-gap community (the largest all-gap
   community with size <= 50) — the same-scale counterpart to (1), for a
   clean apples-to-apples comparison.
3. A "hub neighborhood" inside the single largest gap community (id 4,
   11,549 nodes / 0% evidence-backed) — that community itself is far too
   large to draw as a node-link diagram, so this takes its highest-degree
   node (computed within the community's own induced subgraph) and its
   top-degree neighbors, the same "hub + neighbors" language the 4 EVI
   figures already use.

Reuses graph_communities.py's persisted community membership
(graph_communities_*.json's all_curated_communities/all_gap_communities
"members" lists) rather than rerunning Leiden (~200s) — run
graph_communities.py first if those files predate a graph rebuild.

Inputs:
    - data/graphs/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
    - data/processed/graph_communities/graph_communities_*.json (from graph_communities.py)
Outputs:
    - data/processed/community_examples/community_{curated,gap_small,gap_hub}_example.{svg,pdf,html}
    - data/processed/community_examples/community_examples_<date>.json

Usage:
    python src/analysis/exploratory/graph_community_figures.py
"""

# %%
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from graph_viz import (
    NODE_COLORS,
    NODE_MARKERS,
    NODE_TYPE_ORDER,
    node_legend_handles,
    paper_node_label,
    render_interactive_panel,
    save_static_figure,
)
from load_files import find_latest_file

OUT_DIR = Path("data/processed/community_examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
# Ceiling for "small enough to draw whole" when picking the gap-community
# counterpart to the largest all-curated community (size 8).
SMALL_GAP_MAX_SIZE = 50
# How many of the hub node's own top-degree neighbors to pull into the
# reduced view of the largest (11,549-node) gap community.
HUB_NEIGHBOR_N = 20

# %%
# ---------------------------------------------------------------------------
# 0. Load the cancer-scoped graph and the persisted community membership.
# ---------------------------------------------------------------------------

graph_path = find_latest_file("data/graphs", "cancer_scoped_graph_*.pkl")
print(f"[load] {graph_path}")
with open(graph_path, "rb") as f:
    CSG = pickle.load(f)
print(f"[load] done — {CSG.number_of_nodes()} nodes, {CSG.number_of_edges()} edges")

communities_path = find_latest_file("data/processed/graph_communities", "graph_communities_*.json")
print(f"[load] {communities_path}")
with open(communities_path) as f:
    communities_report = json.load(f)
structural_gap = communities_report["structural_gap"]
all_curated = structural_gap["all_curated_communities"]
all_gap = structural_gap["all_gap_communities"]
if not all(("members" in c) for c in all_curated + all_gap):
    raise RuntimeError(
        f"{communities_path} predates member-list persistence — rerun "
        "graph_communities.py first."
    )
print(f"[setup] {len(all_curated)} all-curated, {len(all_gap)} all-gap communities available")


def deoverlap_positions(pos: dict, min_dist: float = 0.09) -> dict:
    """Two nodes with an *identical* neighbor set (e.g. two variants on
    the same gene cited by the exact same evidence items) sit at a
    perfectly symmetric equilibrium in a force-directed layout — no
    amount of tuning k/iterations separates them, since there's no
    asymmetric force to break the tie. This nudges any pair closer than
    min_dist apart along a deterministic (node-index-derived, not
    randomized) direction, purely for label legibility; it doesn't touch
    the general layout otherwise."""
    nodes = list(pos)
    for i, n1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            n2 = nodes[j]
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            dx, dy = x2 - x1, y2 - y1
            dist = (dx**2 + dy**2) ** 0.5
            if dist < min_dist:
                angle = (i * 37 + j * 17) % 360 * (math.pi / 180)
                ox, oy = (min_dist / 2) * math.cos(angle), (min_dist / 2) * math.sin(angle)
                pos[n1] = (x1 - ox, y1 - oy)
                pos[n2] = (x2 + ox, y2 + oy)
    return pos


def draw_community_panel(ax, sub: nx.DiGraph, panel_title: str | None = None, seed: int = RANDOM_SEED) -> None:
    """Plain community structure view: no highlighted edge (unlike the 4
    EVI figures, there's no single relation to point at — the whole point
    is "this is what a uniformly curated/uncurated neighborhood looks
    like"), node size scaled by within-panel degree so hub structure is
    visible, deterministic spring layout."""
    # Default k (1/sqrt(n)) packs nodes tightly; a higher k gives more
    # breathing room generally. More iterations for better convergence at
    # that higher k. deoverlap_positions() below is the actual fix for
    # exactly-coincident nodes — this alone doesn't resolve that case.
    pos = nx.spring_layout(
        sub, seed=seed, k=2.0 / (sub.number_of_nodes() ** 0.5), iterations=200
    )
    pos = deoverlap_positions(pos)
    ax.set_aspect("equal", adjustable="datalim")
    # Self-loops are real data (e.g. a gene's own GEN-GEN "interacts with
    # itself" homodimer record) but render as a distracting stray loop
    # with nothing to say about community structure — same fix as
    # graph_ego_figures.py's draw_ego_panel.
    context_edges = [(u, v) for u, v in sub.edges() if u != v]
    nx.draw_networkx_edges(
        sub, pos, edgelist=context_edges, ax=ax, edge_color="#B0B0B0", width=1.0,
        arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.05",
    )

    degree = dict(sub.to_undirected().degree())
    for node_type, marker in NODE_MARKERS.items():
        nodes_of_type = [n for n in sub.nodes() if sub.nodes[n].get("node_type") == node_type]
        if not nodes_of_type:
            continue
        sizes = [min(220 + 45 * degree.get(n, 0), 1600) for n in nodes_of_type]
        nx.draw_networkx_nodes(
            sub, pos, nodelist=nodes_of_type, ax=ax,
            node_color=NODE_COLORS.get(node_type, "#828C97"), node_shape=marker,
            node_size=sizes, edgecolors="black", linewidths=0.5,
        )

    label_pos = {n: (x, y + 0.08) for n, (x, y) in pos.items()}
    labels = {n: paper_node_label(n, sub.nodes[n]) for n in sub.nodes()}
    nx.draw_networkx_labels(
        sub, label_pos, labels=labels, ax=ax, font_size=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
    )

    ax.margins(0.2)
    if panel_title:
        ax.set_title(panel_title, fontsize=10, wrap=True)
    ax.axis("off")


def build_and_save(sub: nx.DiGraph, out_name: str, static_title: str, panel_title: str, figsize=(9, 9)) -> dict:
    types_present = sorted({sub.nodes[n]["node_type"] for n in sub.nodes()}, key=NODE_TYPE_ORDER.index)
    legend = node_legend_handles(types_present)

    fig, ax = plt.subplots(figsize=figsize)
    draw_community_panel(ax, sub, panel_title=panel_title)
    fig.suptitle(static_title, fontsize=12)
    save_static_figure(fig, legend, str(OUT_DIR / out_name))

    render_interactive_panel(
        sub, highlighted_edges=[], out_path=OUT_DIR / f"{out_name}.html", title=static_title,
    )

    return {
        "n_nodes": sub.number_of_nodes(),
        "n_edges": sub.number_of_edges(),
        "node_types": {t: sum(1 for n in sub.nodes() if sub.nodes[n]["node_type"] == t) for t in types_present},
    }


# %%
# ---------------------------------------------------------------------------
# 1. Largest fully evidence-backed community — whole thing, it's tiny.
# ---------------------------------------------------------------------------

curated_entry = max(all_curated, key=lambda c: c["size"])
curated_sub = CSG.subgraph(curated_entry["members"])
print(
    f"\n[curated example] community {curated_entry['community_id']} — "
    f"{curated_sub.number_of_nodes()} nodes, {curated_sub.number_of_edges()} edges"
)
curated_stats = build_and_save(
    curated_sub,
    "community_curated_example",
    "Community example — fully evidence-backed",
    f"Community {curated_entry['community_id']} — 100% evidence-backed",
)

# %%
# ---------------------------------------------------------------------------
# 2. Largest structural-gap community that's still small enough to draw
#    whole — the same-scale counterpart to (1).
# ---------------------------------------------------------------------------

small_gap_candidates = [c for c in all_gap if c["size"] <= SMALL_GAP_MAX_SIZE]
if not small_gap_candidates:
    raise RuntimeError(f"No all-gap community with size <= {SMALL_GAP_MAX_SIZE} found.")
gap_small_entry = max(small_gap_candidates, key=lambda c: c["size"])
gap_small_sub = CSG.subgraph(gap_small_entry["members"])
print(
    f"\n[gap example, small] community {gap_small_entry['community_id']} — "
    f"{gap_small_sub.number_of_nodes()} nodes, {gap_small_sub.number_of_edges()} edges"
)
gap_small_stats = build_and_save(
    gap_small_sub,
    "community_gap_small_example",
    "Community example — structural gap (small)",
    f"Community {gap_small_entry['community_id']} — 0% evidence-backed",
    figsize=(15, 15),
)

# %%
# ---------------------------------------------------------------------------
# 3. Hub neighborhood inside the single largest gap community — too big to
#    draw whole (11,549 nodes), so this takes its highest-degree node
#    (within the community's own induced subgraph) and that node's own
#    top-degree neighbors.
# ---------------------------------------------------------------------------

gap_big_entry = max(all_gap, key=lambda c: c["size"])
gap_big_full_und = CSG.subgraph(gap_big_entry["members"]).to_undirected()
degree_within_community = dict(gap_big_full_und.degree())
hub_node = max(degree_within_community, key=degree_within_community.get)
neighbor_degrees = {n: degree_within_community[n] for n in gap_big_full_und.neighbors(hub_node)}
top_neighbors = sorted(neighbor_degrees, key=neighbor_degrees.get, reverse=True)[:HUB_NEIGHBOR_N]
hub_members = [hub_node, *top_neighbors]
gap_hub_sub = CSG.subgraph(hub_members)
hub_label = paper_node_label(hub_node, CSG.nodes[hub_node])
print(
    f"\n[gap example, hub] community {gap_big_entry['community_id']} "
    f"({gap_big_entry['size']} nodes total) — hub = {hub_label} "
    f"(within-community degree {degree_within_community[hub_node]}), "
    f"{len(top_neighbors)} of its top neighbors shown"
)
gap_hub_stats = build_and_save(
    gap_hub_sub,
    "community_gap_hub_example",
    "Community example — structural gap (hub neighborhood)",
    f"Community {gap_big_entry['community_id']} ({gap_big_entry['size']} nodes, 0% evidence-backed) "
    f"— hub: {hub_label}",
    figsize=(11, 11),
)

# %%
# ---------------------------------------------------------------------------
# Save a summary JSON alongside the figures.
# ---------------------------------------------------------------------------

report = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "graph_snapshot": Path(graph_path).name,
    "communities_snapshot": Path(communities_path).name,
    "curated_example": {
        "community_id": curated_entry["community_id"],
        "community_size": curated_entry["size"],
        **curated_stats,
    },
    "gap_small_example": {
        "community_id": gap_small_entry["community_id"],
        "community_size": gap_small_entry["size"],
        **gap_small_stats,
    },
    "gap_hub_example": {
        "community_id": gap_big_entry["community_id"],
        "community_size": gap_big_entry["size"],
        "hub_node_id": hub_node,
        "hub_node_label": hub_label,
        "hub_within_community_degree": degree_within_community[hub_node],
        **gap_hub_stats,
    },
}
report_path = OUT_DIR / f"community_examples_{datetime.now(timezone.utc):%Y-%m-%d}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[save] wrote {report_path}")

print("\n[done]")

# %%
