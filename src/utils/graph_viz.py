#!/usr/bin/env python3
"""Shared node-type visual encoding for the ego-network scripts.

graph_ego_networks.py's generic picker (interactive pyvis) and
graph_ego_figures.py's 4 purpose-built paper figures (static matplotlib)
both color/label nodes by node_type the same way — kept here once so PHE
(added for the drug-safety figure) doesn't need adding twice and the two
renderers can't drift out of sync with each other."""

NODE_COLORS = {
    "GEN": "#0E7A6F",
    "DIS": "#33578C",
    "DRG": "#1E8CA6",
    "VAR": "#A9781F",
    "EVI": "#B0502B",
    "PHE": "#7A4FA0",
}


def node_label(node_id: str, attrs: dict) -> str:
    """Prefers a precise, human-readable label over a bare node id —
    hgvs_name (variants), then a descriptive name, falling back to symbol
    fields. Used for the pyvis tooltip/label and the static figures' panel
    titles, where a fuller name (e.g. a gene's full descriptive name
    instead of its short symbol) is fine to show."""
    for key in ("hgvs_name", "name", "symbol", "gene_symbol"):
        if attrs.get(key):
            return str(attrs[key])
    return node_id
