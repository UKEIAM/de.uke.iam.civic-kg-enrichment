#!/usr/bin/env python3
"""
Lightweight Streamlit browser for find_figure_candidates.py's output —
pick a figure type, pick one of its top-N candidate evidence items, see it
rendered as a small interactive graph (st-link-analysis, a Cytoscape.js
component) instead of reading the raw JSON.

Reads only the candidates JSON, not the graph pickle — the candidate rows
already carry short display labels (e.g. "BRAF", not the ENSG node id), so
those labels double as this app's node ids; two entities that happen to
share a label would collide, but that's not a concern at this JSON's
scale (a handful of nodes per candidate).

Figure 3 (disease hierarchy) is a known simplification: the JSON's "chain"
field doesn't record which end is the EVI's own disease or which
direction is genuinely parent->child (see build_fig3 below) — the PARENT
chain is drawn undirected here rather than asserting a direction that
might be backwards. graph_ego_figures.py's own Figure 3 handles this
correctly from the graph itself; this browser trades that precision for
not needing the ~1.3GB graph pickle loaded at all.

An optional "load full graph" toggle additionally (a) draws the other
genes/drugs/disease/variants the selected EVI itself resolves to, in
gray, mirroring graph_ego_figures.py's context_node_ids treatment, and
(b) attaches every real graph attribute onto each resolvable node/edge's
Cytoscape `data`, so st-link-analysis's built-in click-to-inspect info
panel (it renders whatever is in `data`) shows the full record instead of
just a label. Unlike the rest of this app, both of these need the
underlying graph — the candidates JSON only carries the display labels
used to build each figure's story, not the entities' own attributes or
their real node ids — so this is the one place the app loads the full
cancer_scoped_graph_*.pkl, done lazily and cached (st.cache_resource) so
the ~1.3GB pickle is only paid for once per server process, and only if
the toggle is ever used. Resolving a display label back to a real node id
is best-effort (resolve_row_ids/resolve_neighbor_by_label below): the
EVI's own gene/drug/disease/variant ids are exact, everything else
(a bridge gene, a contraindicated phenotype, a hierarchy step, the gap
disease) is found by searching the relevant neighbor's real edges for a
label match — nodes that can't be resolved this way just keep their
minimal JSON-only data.

Inputs:
    - data/processed/find_figure_candidates/find_figure_candidates_*.json
      (from find_figure_candidates.py)
    - data/processed/cancer_scoped_graph_*.pkl (only if "load full graph" is used)

Usage:
    streamlit run src/app/candidate_browser.py
"""

import os
import pickle
import sys

import streamlit as st
from st_link_analysis import EdgeStyle, NodeStyle, st_link_analysis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from graph_viz import NODE_COLORS, NODE_TYPE_LABELS, format_attr_value, paper_node_label  # noqa: E402
from load_files import find_latest_file  # noqa: E402

CONTEXT_COLOR = "#B0B0B0"
HIGHLIGHT_COLOR = "#DC267F"
MAX_CONTEXT_NODES = 12
MAX_ATTR_LEN = 500
RESERVED_NODE_KEYS = {"id", "label", "name"}
RESERVED_EDGE_KEYS = {"id", "label", "source", "target"}

CATEGORY_LABELS = {
    "fig1_hit": "Figure 1 — mechanism of action (hit)",
    "fig1_miss": "Figure 1 — mechanism of action (miss)",
    "fig2": "Figure 2 — drug safety context",
    "fig3": "Figure 3 — disease hierarchy",
    "fig4": "Figure 4 — gene-disease structural gap",
    "fig5_combination": "Figure 5 — drug interaction (COMBINATION)",
    "fig5_substitutes": "Figure 5 — drug interaction (SUBSTITUTES)",
}


@st.cache_data
def load_report() -> dict:
    path = find_latest_file("data/processed/find_figure_candidates", "find_figure_candidates_*.json")
    import json

    with open(path) as f:
        return json.load(f), str(path)


@st.cache_resource(show_spinner="Loading full graph (one-time, ~20s — cached for the rest of this session)...")
def load_graph():
    path = find_latest_file("data/processed", "cancer_scoped_graph_*.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_context_entities(G, evi_id: str, exclude_labels: set, cap: int = MAX_CONTEXT_NODES) -> tuple[list, int]:
    """Other genes/drugs/disease/variants the EVI itself resolves to,
    same fields as graph_ego_figures.py's other_resolved_nodes(). Returns
    (label, node_type, real_node_id) triples — the label doubles as this
    app's node id (see module docstring), the real id lets the caller
    also attach full attributes via attach_graph_attrs() — already-shown
    labels excluded, sorted for determinism, capped at `cap` with the
    overflow count returned separately so the caller can note it was
    truncated."""
    if evi_id not in G:
        return [], 0
    attrs = G.nodes[evi_id]
    candidate_ids = (
        list(attrs.get("gene_node_ids") or [])
        + list(attrs.get("drug_node_ids") or [])
        + ([attrs["disease_node_id"]] if attrs.get("disease_node_id") else [])
        + list(attrs.get("variant_node_ids") or [])
    )
    seen = set(exclude_labels)
    entities = []
    for node_id in candidate_ids:
        if not node_id or node_id not in G:
            continue
        node_attrs = G.nodes[node_id]
        label = paper_node_label(node_id, node_attrs)
        if label in seen:
            continue
        seen.add(label)
        entities.append((label, node_attrs.get("node_type", "?"), node_id))
    entities.sort(key=lambda e: e[0])
    return entities[:cap], max(0, len(entities) - cap)


def resolve_neighbor_by_label(G, anchor_id: str | None, target_label: str) -> str | None:
    """Best-effort real graph node id for `target_label` among
    `anchor_id`'s direct neighbors — resolves display-label-only
    entities (a bridge gene, a contraindicated phenotype, a disease
    hierarchy step, the structural-gap disease) back to a real node.
    `anchor_id`'s degree can be large (a well-studied gene can have
    1,000+ edges) but this is one bounded neighbor scan, not a
    multi-hop search."""
    if not anchor_id or anchor_id not in G:
        return None
    for n in set(G.successors(anchor_id)) | set(G.predecessors(anchor_id)):
        if n in G and paper_node_label(n, G.nodes[n]) == target_label:
            return n
    return None


def resolve_row_ids(G, category: str, row: dict, evi_attrs: dict) -> dict:
    """Real graph node ids for every display label this row uses, where
    resolvable. See module docstring for the resolution strategy.
    Anything not found is simply left out — that node keeps whatever
    minimal data the JSON already gave it."""

    def match_own(candidate_ids, label):
        for nid in candidate_ids:
            if nid in G and paper_node_label(nid, G.nodes[nid]) == label:
                return nid
        return None

    own_genes = list(evi_attrs.get("gene_node_ids") or [])
    own_drugs = list(evi_attrs.get("drug_node_ids") or [])
    own_disease = evi_attrs.get("disease_node_id")
    ids: dict = {}

    if category in ("fig1_hit", "fig1_miss"):
        gene_id = match_own(own_genes, row["gene"])
        drug_id = match_own(own_drugs, row["drug"])
        if gene_id:
            ids[row["gene"]] = gene_id
        if drug_id:
            ids[row["drug"]] = drug_id
        if category == "fig1_miss" and gene_id:
            bridge_id = resolve_neighbor_by_label(G, gene_id, row["bridge_gene"])
            if bridge_id:
                ids[row["bridge_gene"]] = bridge_id
    elif category == "fig2":
        drug_id = match_own(own_drugs, row["drug"])
        if drug_id:
            ids[row["drug"]] = drug_id
            for label in row["contraindications"]:
                phe_id = resolve_neighbor_by_label(G, drug_id, label)
                if phe_id:
                    ids[label] = phe_id
    elif category == "fig3":
        prev_id = own_disease
        for label in row["chain"]:
            if prev_id and prev_id in G and paper_node_label(prev_id, G.nodes[prev_id]) == label:
                resolved = prev_id
            else:
                resolved = resolve_neighbor_by_label(G, prev_id, label)
            if resolved:
                ids[label] = resolved
            prev_id = resolved
    elif category == "fig4":
        gene_id = match_own(own_genes, row["gene"])
        if gene_id:
            ids[row["gene"]] = gene_id
            if own_disease in G and paper_node_label(own_disease, G.nodes[own_disease]) == row["civic_disease"]:
                civic_id = own_disease
            else:
                civic_id = resolve_neighbor_by_label(G, gene_id, row["civic_disease"])
            gap_id = resolve_neighbor_by_label(G, gene_id, row["optimuskg_only_disease"])
            if civic_id:
                ids[row["civic_disease"]] = civic_id
            if gap_id:
                ids[row["optimuskg_only_disease"]] = gap_id
    elif category in ("fig5_combination", "fig5_substitutes"):
        d1 = match_own(own_drugs, row["drug_1"])
        d2 = match_own(own_drugs, row["drug_2"])
        if d1:
            ids[row["drug_1"]] = d1
        if d2:
            ids[row["drug_2"]] = d2

    return ids


def safe_attr(value) -> str:
    s = format_attr_value(value)
    return s if len(s) <= MAX_ATTR_LEN else s[:MAX_ATTR_LEN] + "…"


def attach_graph_attrs(nodes: list, edges: list, G, id_map: dict) -> None:
    """Merge full graph attributes into each node's/edge's Cytoscape
    `data` dict wherever a real node id is known (via id_map), so
    st-link-analysis's built-in info panel shows everything on click —
    it renders whatever is in `data` verbatim. id/label/name (and, for
    edges, source/target) are left alone so the on-canvas caption and
    the graph's wiring don't change; the real attributes may include
    their own "name"-like fields under different keys."""
    for node in nodes:
        real_id = id_map.get(node["data"]["id"])
        if not real_id or real_id not in G:
            continue
        for key, value in G.nodes[real_id].items():
            if key not in RESERVED_NODE_KEYS:
                node["data"][key] = safe_attr(value)
    for edge in edges:
        d = edge["data"]
        u, v = id_map.get(d["source"]), id_map.get(d["target"])
        if not (u and v):
            continue
        edge_data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
        if not edge_data:
            continue
        for key, value in edge_data.items():
            if key not in RESERVED_EDGE_KEYS:
                d[key] = safe_attr(value)


def make_node(node_id: str, node_type: str) -> dict:
    return {"data": {"id": node_id, "label": node_type, "name": node_id}}


def make_edge(edge_id: str, source: str, target: str, kind: str, relation: str | None = None) -> dict:
    data = {"id": edge_id, "label": kind, "source": source, "target": target}
    if relation:
        data["relation"] = relation
    return {"data": data}


def build_fig1_hit(row: dict) -> tuple[list, list]:
    evi, gene, drug = row["evi"], row["gene"], row["drug"]
    nodes = [make_node(evi, "EVI"), make_node(gene, "GEN"), make_node(drug, "DRG")]
    edges = [
        make_edge("e0", evi, gene, "CONTEXT"),
        make_edge("e1", evi, drug, "CONTEXT"),
        make_edge("e2", drug, gene, "HIGHLIGHT", row["relation"]),
    ]
    return nodes, edges


def build_fig1_miss(row: dict) -> tuple[list, list]:
    evi, gene, drug, bridge = row["evi"], row["gene"], row["drug"], row["bridge_gene"]
    nodes = [make_node(evi, "EVI"), make_node(gene, "GEN"), make_node(drug, "DRG"), make_node(bridge, "GEN")]
    edges = [
        make_edge("e0", evi, gene, "CONTEXT"),
        make_edge("e1", evi, drug, "CONTEXT"),
        make_edge("e2", gene, bridge, "HIGHLIGHT", "interacts with"),
        make_edge("e3", drug, bridge, "HIGHLIGHT", row["bridge_relation"]),
    ]
    return nodes, edges


def build_fig2(row: dict) -> tuple[list, list]:
    evi, drug, contra = row["evi"], row["drug"], row["contraindications"]
    nodes = [make_node(evi, "EVI"), make_node(drug, "DRG")] + [make_node(p, "PHE") for p in contra]
    edges = [make_edge("e0", evi, drug, "CONTEXT")]
    edges += [make_edge(f"e{i + 1}", drug, p, "HIGHLIGHT", "CONTRAINDICATION") for i, p in enumerate(contra)]
    return nodes, edges


def build_fig3(row: dict) -> tuple[list, list]:
    # chain is a hierarchy path, but the JSON doesn't record which end is
    # the EVI's own disease or which way is genuinely parent->child (see
    # module docstring) — connect the EVI to chain[0] (correct for the
    # common "walked upward" case) and draw the chain undirected.
    evi, chain = row["evi"], row["chain"]
    nodes = [make_node(evi, "EVI")] + [make_node(d, "DIS") for d in chain]
    edges = [make_edge("e0", evi, chain[0], "CONTEXT")]
    edges += [
        make_edge(f"e{i + 1}", chain[i], chain[i + 1], "CHAIN", "PARENT") for i in range(len(chain) - 1)
    ]
    return nodes, edges


def build_fig4(row: dict) -> tuple[list, list]:
    evi, gene, civic_dis, gap_dis = row["evi"], row["gene"], row["civic_disease"], row["optimuskg_only_disease"]
    nodes = [make_node(evi, "EVI"), make_node(gene, "GEN"), make_node(civic_dis, "DIS"), make_node(gap_dis, "DIS")]
    edges = [
        make_edge("e0", evi, gene, "CONTEXT"),
        make_edge("e1", gene, civic_dis, "HIGHLIGHT", "civic"),
        make_edge("e2", gene, gap_dis, "HIGHLIGHT", "optimuskg only"),
    ]
    return nodes, edges


def build_fig5(row: dict) -> tuple[list, list]:
    evi, d1, d2 = row["evi"], row["drug_1"], row["drug_2"]
    nodes = [make_node(evi, "EVI"), make_node(d1, "DRG"), make_node(d2, "DRG")]
    edges = [
        make_edge("e0", evi, d1, "CONTEXT"),
        make_edge("e1", evi, d2, "CONTEXT"),
        make_edge("e2", d1, d2, "HIGHLIGHT", "SYNERGISTIC_INTERACTION"),
    ]
    return nodes, edges


BUILDERS = {
    "fig1_hit": build_fig1_hit,
    "fig1_miss": build_fig1_miss,
    "fig2": build_fig2,
    "fig3": build_fig3,
    "fig4": build_fig4,
    "fig5_combination": build_fig5,
    "fig5_substitutes": build_fig5,
}


def row_summary(category: str, row: dict) -> str:
    if category in ("fig1_hit",):
        return f"{row['evi']}: {row['gene']} / {row['drug']} — {row['relation']}"
    if category == "fig1_miss":
        return f"{row['evi']}: {row['gene']} / {row['drug']} — bridge {row['bridge_gene']}"
    if category == "fig2":
        return f"{row['evi']}: {row['drug']} — {len(row['contraindications'])} contraindications"
    if category == "fig3":
        return f"{row['evi']}: " + " -> ".join(row["chain"])
    if category == "fig4":
        return f"{row['evi']}: {row['gene']} — civic: {row['civic_disease']}, gap: {row['optimuskg_only_disease']}"
    if category in ("fig5_combination", "fig5_substitutes"):
        return f"{row['evi']}: {row['drug_1']} + {row['drug_2']}"
    return row.get("evi", str(row))


st.set_page_config(page_title="Figure candidate browser", layout="wide")
st.title("Figure candidate browser")

report, report_path = load_report()
st.caption(f"Source: `{report_path}` (graph snapshot: {report.get('graph_snapshot')})")

available = [c for c in CATEGORY_LABELS if c in report and report[c].get("top")]
if not available:
    st.error("No candidate categories with results found in the report.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    category = st.selectbox(
        "Figure type", available, format_func=lambda c: CATEGORY_LABELS[c]
    )
    section = report[category]
    total_key = next((k for k in section if k.startswith("total")), None)
    if total_key:
        st.caption(f"{section[total_key]} total candidates — showing top {len(section['top'])}")

    rows = section["top"]
    row_idx = st.selectbox(
        "Candidate", range(len(rows)), format_func=lambda i: row_summary(category, rows[i])
    )
    selected_row = rows[row_idx]
    st.json(selected_row, expanded=False)

    layout = st.selectbox("Layout", ["cose", "breadthfirst", "circle", "concentric", "grid"], index=0)

    show_context = st.checkbox(
        "Load full graph — adds other resolved entities (gray) and full "
        "attribute details on click",
        value=False,
    )

with col2:
    nodes, edges = BUILDERS[category](selected_row)
    evi_id = selected_row["evi"]

    context_truncated = 0
    context_entities: list = []
    if show_context:
        G = load_graph()
        evi_attrs = G.nodes[evi_id] if evi_id in G else {}
        already_labels = {n["data"]["id"] for n in nodes}
        context_entities, context_truncated = get_context_entities(G, evi_id, already_labels)

        id_map = {evi_id: evi_id, **resolve_row_ids(G, category, selected_row, evi_attrs)}
        for label, node_type, real_id in context_entities:
            nodes.append({"data": {"id": label, "label": "CONTEXT", "name": label, "node_type": node_type}})
            edges.append(make_edge(f"ctx-{label}", evi_id, label, "CONTEXT"))
            id_map[label] = real_id

        attach_graph_attrs(nodes, edges, G, id_map)

    elements = {"nodes": nodes, "edges": edges}

    types_present = sorted({n["data"]["label"] for n in nodes if n["data"]["label"] != "CONTEXT"})
    node_styles = [
        NodeStyle(t, NODE_COLORS.get(t, "#828C97"), caption="name") for t in types_present
    ]
    if any(n["data"]["label"] == "CONTEXT" for n in nodes):
        node_styles.append(NodeStyle("CONTEXT", CONTEXT_COLOR, caption="name"))
    edge_styles = [
        EdgeStyle("CONTEXT", color=CONTEXT_COLOR, directed=False),
        EdgeStyle("HIGHLIGHT", color=HIGHLIGHT_COLOR, caption="relation", directed=True),
        EdgeStyle("CHAIN", color=HIGHLIGHT_COLOR, caption="relation", directed=False),
    ]

    st_link_analysis(
        elements, layout=layout, node_styles=node_styles, edge_styles=edge_styles, height=650,
        key=f"{category}-{row_idx}-{show_context}",
    )
    st.caption(
        "Click a node or edge to open its attribute panel."
        + ("" if show_context else " Enable \"load full graph\" above for the full attribute record.")
    )
    legend = "  ".join(f"🔵 {NODE_TYPE_LABELS.get(t, t)}" for t in types_present)
    if show_context and context_entities:
        legend += "  ⚪ other resolved entity (context)"
    st.caption(legend)
    if context_truncated:
        st.caption(f"+{context_truncated} more directly connected entities not shown (capped at {MAX_CONTEXT_NODES})")
