#!/usr/bin/env python3
"""
Generic ego-network picker: finds 2-3 representative evidence items (degree
in [4,10], diverse neighbor-type signature) and renders each as an
interactive pyvis figure. Runs against evidence_backed_expanded_graph_*.pkl.

Inputs:
    - data/processed/evidence_backed_expanded_graph_*.pkl (from build_expanded_evidence_graph.py)
Outputs:
    - data/processed/ego_networks/<evidence_item_id>.html (one pyvis figure
      per representative evidence item)

Usage:
    python src/analysis/graph_ego_networks.py
"""

# %%
import os
import pickle
import sys
from pathlib import Path

import networkx as nx
from pyvis.network import Network

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from graph_viz import NODE_COLORS, node_label
from load_files import find_latest_file

OUT_DIR = Path("data/processed/ego_networks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPRESENTATIVE = 3
# Degree band for "representative": big enough to show real context, small
# enough that the figure stays readable rather than becoming a hairball.
MIN_DEGREE, MAX_DEGREE = 4, 10


def build_ego_network(G: nx.DiGraph, center_id: str, hops: int = 1) -> nx.DiGraph:
    """Breadth-first expansion from center_id, following edges in EITHER
    direction at each hop (successors AND predecessors), not just
    successors. This schema's edges are semantically mostly undirected even
    though G itself is a DiGraph (see the spec's directionality section and
    this pipeline's own from/to work) — a successors-only walk would
    silently drop any node only reachable via an edge stored pointing the
    other way, which for a hub node like an evidence item is most of its
    neighbors."""
    visited = {center_id}
    frontier = {center_id}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G.successors(node))
            next_frontier.update(G.predecessors(node))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    return G.subgraph(visited).copy()


def render_ego_network(G: nx.DiGraph, center_id: str, out_path: Path) -> None:
    ego = build_ego_network(G, center_id, hops=1)
    net = Network(height="750px", width="100%", directed=True, notebook=False)
    for node_id, attrs in ego.nodes(data=True):
        color = NODE_COLORS.get(attrs.get("node_type"), "#828C97")
        size = 30 if node_id == center_id else 18
        net.add_node(
            node_id,
            label=node_label(node_id, attrs),
            title=f"{attrs.get('node_type')}: {node_id}",
            color=color,
            size=size,
        )
    for u, v, data in ego.edges(data=True):
        net.add_edge(u, v, title=data.get("edge_type", ""))
    net.write_html(str(out_path))


# %%
# ---------------------------------------------------------------------------
# 0. Load, pick representative evidence items.
# ---------------------------------------------------------------------------

graph_path = find_latest_file("data/processed", "evidence_backed_expanded_graph_*.pkl")
print(f"[load] {graph_path}")
with open(graph_path, "rb") as f:
    G = pickle.load(f)
print(f"[load] done — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

evidence_item_ids = [n for n, a in G.nodes(data=True) if a.get("node_type") == "EVI"]
print(f"[setup] {len(evidence_item_ids)} evidence items")


def total_degree(node_id: str) -> int:
    return G.in_degree(node_id) + G.out_degree(node_id)


# Diverse by the neighbor node_types touched, not just degree — an evidence
# item connected only to a gene and a disease looks structurally different
# in the figure than one that also reaches a drug or a variant.
def neighbor_type_signature(node_id: str) -> frozenset:
    neighbor_types = set()
    for n in list(G.successors(node_id)) + list(G.predecessors(node_id)):
        neighbor_types.add(G.nodes[n].get("node_type"))
    return frozenset(neighbor_types)


candidates = [
    (eid, total_degree(eid), neighbor_type_signature(eid))
    for eid in evidence_item_ids
    if MIN_DEGREE <= total_degree(eid) <= MAX_DEGREE
]
print(
    f"[setup] {len(candidates)} candidates with degree in [{MIN_DEGREE}, {MAX_DEGREE}]"
)

seen_signatures = set()
representative = []
for eid, degree, signature in sorted(candidates, key=lambda x: -len(x[2])):
    if signature in seen_signatures:
        continue
    seen_signatures.add(signature)
    representative.append((eid, degree, signature))
    if len(representative) == N_REPRESENTATIVE:
        break

print(f"\n[pick] {len(representative)} representative evidence items:")
for eid, degree, signature in representative:
    print(f"  {eid}: degree={degree}, neighbor types={sorted(signature)}")

# %%
# ---------------------------------------------------------------------------
# 1. Render.
# ---------------------------------------------------------------------------

for eid, _, _ in representative:
    out_path = OUT_DIR / f"{eid}.html"
    render_ego_network(G, eid, out_path)
    print(f"[save] wrote {out_path}")

print("\n[done]")

# %%
