#!/usr/bin/env python3
"""
Lightweight Streamlit browser for find_figure_candidates.py's output —
pick a figure type, pick one of its top-N candidate evidence items, see it
rendered as a small interactive graph (st-link-analysis).

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

Figure 6 has no EVI to anchor its context on (the story is the absence of
one), so get_fig6_context() builds it from two edge types instead: the
gene's other ClinVar variants that genuinely have matching-type CIViC
coverage for the SAME track as the gap variant (VAR-GEN, filtered on
qualifying for that track AND not themselves being a gap candidate for
it) — not just "has >=1 evidence item of any kind," since a variant can
have an unrelated-type evidence item while still being a genuine gap for
the track being highlighted — plus the gap variant's own disease/
phenotype associations (VAR-DIS). Each is capped and truncation-counted
independently, same as get_context_entities().

Inputs:
    - data/processed/find_figure_candidates/find_figure_candidates_*.json
      (from find_figure_candidates.py)
    - data/graphs/cancer_scoped_graph_*.pkl (only if "load full graph" is used)

Usage:
    streamlit run src/app/candidate_browser.py
"""

import os
import pickle
import sys

import streamlit as st
from st_link_analysis import EdgeStyle, NodeStyle, st_link_analysis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from graph_viz import (  # noqa: E402
    NODE_COLORS,
    NODE_TYPE_LABELS,
    format_attr_value,
    paper_node_label,
)
from load_files import find_latest_file  # noqa: E402

CONTEXT_COLOR = "#B0B0B0"
HIGHLIGHT_COLOR = "#DC267F"
MAX_CONTEXT_NODES = 12
EDGE_LABEL_TEXT_COLOR = "#FFFFFF"
# Cytoscape.js's own default node label font-size is 16px (confirmed in
# the bundled build); bumped up for publication-figure legibility.
NODE_FONT_SIZE = "18px"
# text-wrap requires text-max-width to actually trigger wrapping; its
# wrap algorithm only ever breaks at word boundaries (never mid-word —
# a single word wider than this just overflows its line rather than
# being split), so this is just where multi-word labels start wrapping.
NODE_LABEL_WRAP_WIDTH = "100px"
# EdgeStyle/WhiteLabelEdgeStyle don't expose font-size either — edge
# captions (relation text) default to the same tiny 16px otherwise.
EDGE_FONT_SIZE = "14px"


class WhiteLabelEdgeStyle(EdgeStyle):
    """EdgeStyle doesn't expose edge-label text color (its dump() only
    ever sets line/background/arrow color, never Cytoscape.js's own
    "color" style property for label text) — the pink HIGHLIGHT/CHAIN
    line color makes a dark caption hard to read, so this subclass adds
    white label text on top of whatever EdgeStyle already produces."""

    def dump(self) -> dict:
        style = super().dump()
        style["style"]["color"] = EDGE_LABEL_TEXT_COLOR
        return style


class RawStyle:
    """A literal Cytoscape.js style rule for any selector — node or edge
    — bypassing NodeStyle/EdgeStyle's own label-based selector
    (node[label='...']/edge[label='...']) wherever a property they
    don't expose is needed (e.g. node font-size/text-wrap, or
    edge[label='<kind>'][undirected = "True"] for the bidirectional
    -arrow fix — see its call sites for each one's own reasoning)."""

    def __init__(self, selector: str, style: dict) -> None:
        self.selector = selector
        self.style = style

    def dump(self) -> dict:
        return {"selector": self.selector, "style": self.style}


MAX_ATTR_LEN = 500
RESERVED_NODE_KEYS = {"id", "label", "name"}
RESERVED_EDGE_KEYS = {"id", "label", "source", "target", "relation"}

CATEGORY_LABELS = {
    "fig1_miss": "Figure 1 — mechanism of action (miss)",
    "fig1_hit": "Figure 1 — mechanism of action (hit)",
    "fig2": "Figure 2 — drug safety context",
    "fig3": "Figure 3 — disease hierarchy",
    "fig4_combination": "Figure 4 — drug interaction (COMBINATION)",
    "fig4_substitutes": "Figure 4 — drug interaction (SUBSTITUTES)",
    "fig5": "Figure 5 — gene-disease structural gap",
    "fig6": "Figure 6 — ClinVar curation gap",
}


@st.cache_data
def load_report() -> dict:
    path = find_latest_file(
        "data/processed/find_figure_candidates", "find_figure_candidates_*.json"
    )
    import json

    with open(path) as f:
        return json.load(f), str(path)


@st.cache_resource(
    show_spinner="Loading full graph (one-time, ~20s — cached for the rest of this session)..."
)
def load_graph():
    path = find_latest_file("data/graphs", "cancer_scoped_graph_*.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_context_entities(
    G, evi_id: str, exclude_labels: set, cap: int = MAX_CONTEXT_NODES
) -> tuple[list, int]:
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


def _neighbors_by_edge_type(G, anchor_id: str, edge_type: str) -> set:
    neighbors = set()
    for _, v, d in G.out_edges(anchor_id, data=True):
        if d.get("edge_type") == edge_type:
            neighbors.add(v)
    for u, _, d in G.in_edges(anchor_id, data=True):
        if d.get("edge_type") == edge_type:
            neighbors.add(u)
    return neighbors


def get_fig6_context(
    G,
    gene_id: str,
    variant_id: str,
    track: str,
    exclude_labels: set,
    cap: int = MAX_CONTEXT_NODES,
) -> dict:
    """Figure 6 has no EVI to pull "other resolved entities" from (see
    build_fig6), so its context is assembled from two edge types instead:
    the gene's other ClinVar variants that genuinely have matching-type
    CIViC coverage for the SAME track as the gap variant (VAR-GEN,
    filtered on qualifying for `track` AND not being a `{track}_gap_candidate`
    themselves) — plus the gap variant's own disease/phenotype
    associations (VAR-DIS).

    Deliberately NOT just "has >=1 CIViC evidence item of any kind"
    (the global in_civic_evidence_item flag): a variant can have, say, a
    PREDICTIVE evidence item while still being a genuine PREDISPOSING
    gap — the same type-matching distinction save_graph_nodes.py's
    {track}_gap_candidate fields exist for in the first place (see that
    script's module docstring). Showing such a variant as "covered"
    context would misrepresent it as resolving the same kind of gap the
    highlighted variant has. Requiring membership in `track` as well
    (not just "not a gap for track") also excludes variants that don't
    even carry that classification at all — {track}_gap_candidate is
    trivially False for those, not a meaningful "covered" signal.

    Returns {"other_variants": (entities, truncated), "diseases": (...)},
    same (label, node_type, real_id) shape and per-category cap as
    get_context_entities()."""
    seen = set(exclude_labels)

    def collect(
        anchor_id: str, edge_type: str, require_track_covered: bool = False
    ) -> tuple[list, int]:
        entities = []
        for node_id in _neighbors_by_edge_type(G, anchor_id, edge_type):
            if node_id == variant_id or node_id not in G:
                continue
            attrs = G.nodes[node_id]
            if require_track_covered:
                if track not in (attrs.get("tracks") or []):
                    continue
                if attrs.get(f"{track}_gap_candidate"):
                    continue
            label = paper_node_label(node_id, attrs)
            if label in seen:
                continue
            seen.add(label)
            entities.append((label, attrs.get("node_type", "?"), node_id))
        entities.sort(key=lambda e: e[0])
        return entities[:cap], max(0, len(entities) - cap)

    return {
        "other_variants": collect(gene_id, "VAR-GEN", require_track_covered=True),
        "diseases": collect(variant_id, "VAR-DIS"),
    }


def resolve_neighbor_by_label(
    G, anchor_id: str | None, target_label: str
) -> str | None:
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
    own_variants = list(evi_attrs.get("variant_node_ids") or [])
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
        for variant_label in row.get("all_variants") or []:
            if variant_label not in ids:
                resolved = match_own(own_variants, variant_label)
                if resolved:
                    ids[variant_label] = resolved
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
            if (
                prev_id
                and prev_id in G
                and paper_node_label(prev_id, G.nodes[prev_id]) == label
            ):
                resolved = prev_id
            else:
                resolved = resolve_neighbor_by_label(G, prev_id, label)
            if resolved:
                ids[label] = resolved
            prev_id = resolved
    elif category in ("fig4_combination", "fig4_substitutes"):
        d1 = match_own(own_drugs, row["drug_1"])
        d2 = match_own(own_drugs, row["drug_2"])
        if d1:
            ids[row["drug_1"]] = d1
        if d2:
            ids[row["drug_2"]] = d2
        # all_drugs/all_variants cover every drug/variant the EVI resolves
        # to, not just the highlighted interacting pair — resolve those too
        # so "load full graph" enriches them the same way.
        for drug_label in row.get("all_drugs") or []:
            if drug_label not in ids:
                resolved = match_own(own_drugs, drug_label)
                if resolved:
                    ids[drug_label] = resolved
        for variant_label in row.get("all_variants") or []:
            if variant_label not in ids:
                resolved = match_own(own_variants, variant_label)
                if resolved:
                    ids[variant_label] = resolved
    elif category == "fig5":
        gene_id = match_own(own_genes, row["gene"])
        if gene_id:
            ids[row["gene"]] = gene_id
            if (
                own_disease in G
                and paper_node_label(own_disease, G.nodes[own_disease])
                == row["civic_disease"]
            ):
                civic_id = own_disease
            else:
                civic_id = resolve_neighbor_by_label(G, gene_id, row["civic_disease"])
            if civic_id:
                ids[row["civic_disease"]] = civic_id
            for gap_label in row.get("optimuskg_only_diseases") or []:
                if gap_label not in ids:
                    gap_id = resolve_neighbor_by_label(G, gene_id, gap_label)
                    if gap_id:
                        ids[gap_label] = gap_id
    elif category == "fig6":
        # No EVI here — the variant id IS already a real graph id (unlike
        # every other category's pseudo-id-as-label nodes), so it resolves
        # itself; only the gene label needs resolving, via the variant's
        # own VAR-GEN edge.
        ids[row["variant"]] = row["variant"]
        gene_id = resolve_neighbor_by_label(G, row["variant"], row["gene"])
        if gene_id:
            ids[row["gene"]] = gene_id

    return ids


def safe_attr(value) -> str:
    s = format_attr_value(value)
    return s if len(s) <= MAX_ATTR_LEN else s[:MAX_ATTR_LEN] + "…"


def attach_graph_attrs(nodes: list, edges: list, G, id_map: dict) -> None:
    """Merge full graph attributes into each node's/edge's Cytoscape
    `data` dict wherever a real node id is known (via id_map), so
    st-link-analysis's built-in info panel shows everything on click —
    it renders whatever is in `data` verbatim. id/label/name (and, for
    edges, relation) are left alone so the on-canvas caption doesn't
    change — "relation" specifically stays protected because a few
    figures deliberately set it to a cosmetic/abbreviated or derived
    value (e.g. Figure 4's "SYN_INT", Figure 5's "CIVIC_ONLY"/
    "OPTIMUSKG_CIVIC") rather than the real edge attribute's own
    value; the real attributes may
    include their own "name"/"relation"-like fields under different
    keys, still fully visible on click.

    source/target ARE corrected (swapped) when the real graph edge only
    matched in the reverse direction — some figures build an edge in
    whichever order reads naturally for the figure's story (e.g. Figure
    3's chain direction isn't knowable from the JSON alone — see
    build_fig3), which can disagree with the real edge's own (u, v)
    order. Swapping here means a genuinely directional edge's single
    arrow (the CHAIN undirected=False style rule) always points the
    correct way once the real direction is known, rather than asserting
    whichever direction the figure happened to construct it in. A no-op
    for every edge kind that already points the real way, and visually
    invisible for undirected=True edges (styled with arrows on both
    ends regardless)."""
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
        forward_data = G.get_edge_data(u, v)
        edge_data = forward_data or G.get_edge_data(v, u)
        if not edge_data:
            continue
        if not forward_data:
            d["source"], d["target"] = d["target"], d["source"]
        for key, value in edge_data.items():
            if key not in RESERVED_EDGE_KEYS:
                d[key] = safe_attr(value)


TITLE_CASE_NODE_TYPES = {"DRG", "DIS"}


def display_name(node_id: str, node_type: str) -> str:
    """Drug/disease labels come out of paper_node_label() however
    OptimusKG's own symbol/name field happens to be cased (often all
    caps for drugs) — title-case them for display. id/matching logic
    elsewhere always uses the untouched node_id, never this."""
    return node_id.title() if node_type in TITLE_CASE_NODE_TYPES else node_id


def make_node(node_id: str, node_type: str) -> dict:
    return {
        "data": {
            "id": node_id,
            "label": node_type,
            "name": display_name(node_id, node_type),
        }
    }


def normalize_relation_text(text: str) -> str:
    """Every edge's displayed relation, uppercased with spaces turned
    into underscores (e.g. "interacts with" -> "INTERACTS_WITH") for a
    single consistent style across all figures. Applied centrally here
    rather than at each call site so no edge can slip through
    un-normalized."""
    return text.upper().replace(" ", "_")


def make_edge(
    edge_id: str, source: str, target: str, kind: str, relation: str | None = None
) -> dict:
    data = {"id": edge_id, "label": kind, "source": source, "target": target}
    if relation:
        data["relation"] = normalize_relation_text(relation)
    return {"data": data}


def add_variant_context(
    nodes: list, edges: list, evi: str, variants: list | None, seen: set
) -> None:
    """Gray CONTEXT nodes for every variant the EVI resolves to — variants
    are supporting molecular context in these figures, not the
    highlighted story, so they're styled like get_context_entities()'s
    other gray nodes rather than colored by their own VAR type.
    `seen` is shared with the caller's own node-id set so a variant that
    happens to share a label with an already-added node is skipped."""
    for i, variant in enumerate(variants or []):
        if variant in seen:
            continue
        seen.add(variant)
        nodes.append(
            {
                "data": {
                    "id": variant,
                    "label": "CONTEXT",
                    "name": variant,
                    "node_type": "VAR",
                }
            }
        )
        edges.append(make_edge(f"e-var-{i}", evi, variant, "CONTEXT"))


def build_fig1_hit(row: dict) -> tuple[list, list]:
    evi, gene, drug = row["evi"], row["gene"], row["drug"]
    nodes = [make_node(evi, "EVI"), make_node(gene, "GEN"), make_node(drug, "DRG")]
    edges = [
        make_edge("e0", evi, gene, "CONTEXT"),
        make_edge("e1", evi, drug, "CONTEXT"),
        make_edge("e2", drug, gene, "HIGHLIGHT", row["relation"]),
    ]
    add_variant_context(nodes, edges, evi, row.get("all_variants"), {gene, drug})
    return nodes, edges


def build_fig1_miss(row: dict) -> tuple[list, list]:
    evi, gene, drug, bridge = row["evi"], row["gene"], row["drug"], row["bridge_gene"]
    nodes = [
        make_node(evi, "EVI"),
        make_node(gene, "GEN"),
        make_node(drug, "DRG"),
        make_node(bridge, "GEN"),
    ]
    edges = [
        make_edge("e0", evi, gene, "CONTEXT"),
        make_edge("e1", evi, drug, "CONTEXT"),
        make_edge("e2", gene, bridge, "HIGHLIGHT", "interacts with"),
        make_edge("e3", drug, bridge, "HIGHLIGHT", row["bridge_relation"]),
    ]
    add_variant_context(
        nodes, edges, evi, row.get("all_variants"), {gene, drug, bridge}
    )
    return nodes, edges


def build_fig2(row: dict) -> tuple[list, list]:
    evi, drug, contra = row["evi"], row["drug"], row["contraindications"]
    nodes = [make_node(evi, "EVI"), make_node(drug, "DRG")] + [
        make_node(p, "PHE") for p in contra
    ]
    edges = [make_edge("e0", evi, drug, "CONTEXT")]
    edges += [
        make_edge(f"e{i + 1}", drug, p, "HIGHLIGHT", "CONTRAINDICATION")
        for i, p in enumerate(contra)
    ]
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
        make_edge(f"e{i + 1}", chain[i], chain[i + 1], "CHAIN", "PARENT")
        for i in range(len(chain) - 1)
    ]
    return nodes, edges


def build_fig4(row: dict) -> tuple[list, list]:
    evi, d1, d2 = row["evi"], row["drug_1"], row["drug_2"]
    nodes = [make_node(evi, "EVI"), make_node(d1, "DRG"), make_node(d2, "DRG")]
    edges = [
        make_edge("e0", evi, d1, "CONTEXT"),
        make_edge("e1", evi, d2, "CONTEXT"),
        make_edge(
            "e2", d1, d2, "HIGHLIGHT", "SYN_INT"
        ),  # SYNERGISTIC_INTERACTION, abbreviated for on-canvas readability
    ]
    # all_drugs/all_variants are every drug/variant the EVI itself resolves
    # to (not just the interacting pair highlighted above) — shown as plain
    # CONTEXT nodes so the figure reflects the full regimen/molecular
    # context an evidence item actually represents.
    seen = {d1, d2}
    for i, drug in enumerate(row.get("all_drugs") or []):
        if drug in seen:
            continue
        seen.add(drug)
        nodes.append(make_node(drug, "DRG"))
        edges.append(make_edge(f"e-drug-{i}", evi, drug, "CONTEXT"))
    add_variant_context(nodes, edges, evi, row.get("all_variants"), seen)
    return nodes, edges


def civic_disease_edge_label(origin: list | None) -> str:
    """CIVIC_ONLY if this gene-disease association has never been
    independently corroborated by OptimusKG, OPTIMUSKG_CIVIC if it has
    — derived from the real GEN-DIS edge's own origin value (see
    find_figure_candidates.py's civic_disease_origin()), not a flat
    hardcoded label. Checks "civic" in origin specifically, not just
    "optimuskg" in origin — the rare disease_resolved_via=
    mondo_parent_fallback case (CIViC's exact MONDO term isn't an
    OptimusKG node, so disease_node_id is an ANCESTOR term instead) can
    land on a GEN-DIS edge tagged origin=["optimuskg"] only, with no
    "civic" at all, since the "civic" tag lives on the edge to the more
    specific, pre-fallback term instead — checking "optimuskg in origin"
    alone would wrongly call that OPTIMUSKG_CIVIC. origin is None/
    missing "civic" entirely defaults to CIVIC_ONLY either way, since
    this edge's own gene-disease pair is always CIViC-asserted by
    construction (see find_figure_candidates.py's fig5_candidates) even
    when the graph's own edge-level provenance tag doesn't reflect it."""
    if origin and "civic" in origin and "optimuskg" in origin:
        return "OPTIMUSKG_CIVIC"
    return "CIVIC_ONLY"


def build_fig5(row: dict) -> tuple[list, list]:
    evi, gene, civic_dis = row["evi"], row["gene"], row["civic_disease"]
    nodes = [make_node(evi, "EVI"), make_node(gene, "GEN"), make_node(civic_dis, "DIS")]
    edges = [
        make_edge("e0", evi, gene, "CONTEXT"),
        make_edge(
            "e1",
            gene,
            civic_dis,
            "HIGHLIGHT",
            civic_disease_edge_label(row.get("civic_disease_origin")),
        ),
    ]
    # optimuskg_only_diseases is already the diversity-preferring top-10
    # selected server-side (see find_figure_candidates.py) — every one of
    # them is a highlighted "structural gap" sibling of civic_dis, not a
    # single example, so they're all first-class HIGHLIGHT nodes rather
    # than gray context.
    for i, gap_dis in enumerate(row.get("optimuskg_only_diseases") or []):
        if gap_dis == civic_dis:
            continue
        nodes.append(make_node(gap_dis, "DIS"))
        edges.append(
            make_edge(f"e-gap-{i}", gene, gap_dis, "HIGHLIGHT", "OptimusKG only")
        )
    return nodes, edges


def build_fig6(row: dict) -> tuple[list, list]:
    # No EVI to anchor on — the whole point is the absence of a matching
    # one. Just the variant and its gene, with the gap verdict itself
    # (which track, which classification) carried on the edge.
    variant, gene = row["variant"], row["gene"]
    # Unlike every other figure's variant entries (already pre-formatted
    # as "p.Val600Glu"-style labels by find_figure_candidates.py's
    # label()), fig6's "variant" field is deliberately the real graph id
    # (needed to anchor context/attribute resolution — see module
    # docstring), so its display name needs the same HGVS protein-change
    # extraction applied here instead, from the row's own hgvs_name.
    variant_name = paper_node_label(
        variant, {"node_type": "VAR", "hgvs_name": row.get("hgvs_name")}
    )
    nodes = [
        {"data": {"id": variant, "label": "VAR", "name": variant_name}},
        make_node(gene, "GEN"),
    ]
    # Caption kept short (just the track, e.g. "PREDISPOSING_GAP" after
    # normalize_relation_text) — the classification value and the "why
    # it's a gap" explanation still ride along as separate data fields,
    # visible in the click-to-inspect panel rather than crowding the
    # on-canvas edge label.
    gap_edge = make_edge("e0", variant, gene, "HIGHLIGHT", f"{row['track']}_gap")
    gap_edge["data"]["classification"] = row["classification"]
    gap_edge["data"]["gap_reason"] = (
        "no matching CIViC evidence item for this classification"
    )
    edges = [gap_edge]
    return nodes, edges


BUILDERS = {
    "fig1_miss": build_fig1_miss,
    "fig1_hit": build_fig1_hit,
    "fig2": build_fig2,
    "fig3": build_fig3,
    "fig4_combination": build_fig4,
    "fig4_substitutes": build_fig4,
    "fig5": build_fig5,
    "fig6": build_fig6,
}


def row_summary(category: str, row: dict) -> str:
    if category == "fig1_miss":
        return (
            f"{row['evi']}: {row['gene']} / {row['drug']} — bridge {row['bridge_gene']}"
        )
    if category in ("fig1_hit",):
        return f"{row['evi']}: {row['gene']} / {row['drug']} — {row['relation']}"
    if category == "fig2":
        return f"{row['evi']}: {row['drug']} — {len(row['contraindications'])} contraindications"
    if category == "fig3":
        return f"{row['evi']}: " + " -> ".join(row["chain"])
    if category in ("fig4_combination", "fig4_substitutes"):
        return f"{row['evi']}: {row['drug_1']} + {row['drug_2']}"
    if category == "fig5":
        n_total = row.get(
            "optimuskg_only_diseases_total",
            len(row.get("optimuskg_only_diseases") or []),
        )
        return (
            f"{row['evi']}: {row['gene']} — civic: {row['civic_disease']}, "
            f"gap: {n_total} optimuskg-only diseases"
        )
    if category == "fig6":
        return f"{row['variant']} ({row['gene']}): {row['track']} = {row['classification']}"
    return row.get("evi", str(row))


st.set_page_config(page_title="GRACE - Interactive candidate explorer", layout="wide")
st.title("GRACE - Interactive candidate explorer")

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
    # Prefer a diversified view when the search provides one (fig6: one
    # candidate per gene, since ranking by raw count alone is dominated by
    # a single outlier gene — see find_figure_candidates.py).
    rows = section.get("top_by_gene", section["top"])

    total_key = next((k for k in section if k.startswith("total")), None)
    if total_key:
        caption = f"{section[total_key]} total candidates"
        if "total_distinct_genes" in section:
            caption += f" ({section['total_distinct_genes']} distinct genes)"
        caption += f" — showing top {len(rows)}"
        if "top_by_gene" in section:
            caption += " (one per gene)"
        st.caption(caption)
    row_idx = st.selectbox(
        "Candidate",
        range(len(rows)),
        format_func=lambda i: row_summary(category, rows[i]),
    )
    selected_row = rows[row_idx]
    st.json(selected_row, expanded=False)

    layout = st.selectbox(
        "Layout", ["cose", "breadthfirst", "circle", "concentric", "grid"], index=0
    )

    show_context = st.checkbox(
        "Load full graph — adds other resolved entities (gray) and full "
        "attribute details on click",
        value=False,
    )

with col2:
    nodes, edges = BUILDERS[category](selected_row)
    # Every other category anchors on its EVI (already a real graph id);
    # fig6 has no EVI — the variant plays the same role (also already a
    # real graph id) as the thing context nodes would radiate from.
    anchor_id = selected_row["variant"] if category == "fig6" else selected_row["evi"]

    context_truncated = 0
    context_entities: list = []
    fig6_no_matching_coverage = False
    if show_context:
        G = load_graph()
        id_map = {anchor_id: anchor_id}

        if category == "fig6":
            id_map.update(resolve_row_ids(G, category, selected_row, {}))
            gene_id = id_map.get(selected_row["gene"])
            if gene_id:
                already_labels = {n["data"]["id"] for n in nodes}
                fig6_ctx = get_fig6_context(
                    G, gene_id, anchor_id, selected_row["track"], already_labels
                )
                fig6_no_matching_coverage = not fig6_ctx["other_variants"][0]
                # Context edges must reference each node's Cytoscape id —
                # the gene's canvas id is its display label (selected_row
                # ["gene"]), NOT gene_id (the resolved real graph id, only
                # needed above as a traversal starting point).
                for ctx_key, ctx_anchor in (
                    ("other_variants", selected_row["gene"]),
                    ("diseases", anchor_id),
                ):
                    entities, truncated = fig6_ctx[ctx_key]
                    context_truncated += truncated
                    for label, node_type, real_id in entities:
                        nodes.append(
                            {
                                "data": {
                                    "id": label,
                                    "label": "CONTEXT",
                                    "name": display_name(label, node_type),
                                    "node_type": node_type,
                                }
                            }
                        )
                        edges.append(
                            make_edge(
                                f"ctx-{ctx_key}-{label}", ctx_anchor, label, "CONTEXT"
                            )
                        )
                        id_map[label] = real_id
                        context_entities.append((label, node_type, real_id))
        else:
            anchor_attrs = G.nodes[anchor_id] if anchor_id in G else {}
            already_labels = {n["data"]["id"] for n in nodes}
            context_entities, context_truncated = get_context_entities(
                G, anchor_id, already_labels
            )
            id_map.update(resolve_row_ids(G, category, selected_row, anchor_attrs))
            for label, node_type, real_id in context_entities:
                nodes.append(
                    {
                        "data": {
                            "id": label,
                            "label": "CONTEXT",
                            "name": display_name(label, node_type),
                            "node_type": node_type,
                        }
                    }
                )
                edges.append(make_edge(f"ctx-{label}", anchor_id, label, "CONTEXT"))
                id_map[label] = real_id

        attach_graph_attrs(nodes, edges, G, id_map)

    elements = {"nodes": nodes, "edges": edges}

    types_present = sorted(
        {n["data"]["label"] for n in nodes if n["data"]["label"] != "CONTEXT"}
    )
    node_styles = [
        # EVI styled gray like CONTEXT nodes (not its own NODE_COLORS
        # entry) — every figure's EVI is scaffolding for the story being
        # told, not itself part of the highlighted finding. Overridden
        # here rather than in graph_viz.py's shared NODE_COLORS, since
        # the exploratory scripts' own static figures still want EVI in
        # its own distinct color.
        NodeStyle(
            t,
            CONTEXT_COLOR if t == "EVI" else NODE_COLORS.get(t, "#828C97"),
            caption="name",
        )
        for t in types_present
    ]
    node_kinds_present = list(types_present)
    if any(n["data"]["label"] == "CONTEXT" for n in nodes):
        node_styles.append(NodeStyle("CONTEXT", CONTEXT_COLOR, caption="name"))
        node_kinds_present.append("CONTEXT")
    # A bare "node" selector here was silently losing to the component's
    # own built-in base stylesheet (same issue as edges below — its
    # bundled JS ships its own default node font-size via the exact same
    # bare "node" selector) — label-specific selectors, matching
    # NodeStyle's own node[label='...'] pattern above, apply correctly
    # instead.
    for kind in node_kinds_present:
        node_styles.append(
            RawStyle(
                f'node[label="{kind}"]',
                {
                    "font-size": NODE_FONT_SIZE,
                    "text-wrap": "wrap",
                    "text-max-width": NODE_LABEL_WRAP_WIDTH,
                },
            )
        )
    edge_styles = [
        EdgeStyle("CONTEXT", color=CONTEXT_COLOR, directed=False),
        WhiteLabelEdgeStyle(
            "HIGHLIGHT", color=HIGHLIGHT_COLOR, caption="relation", directed=True
        ),
        WhiteLabelEdgeStyle(
            "CHAIN", color=HIGHLIGHT_COLOR, caption="relation", directed=False
        ),
    ]
    # A bare "edge" selector here was silently losing to the component's
    # own built-in base stylesheet (bundled JS ships its own default
    # edge font-size via the exact same bare "edge" selector) — label
    # -specific selectors, matching the pattern CONTEXT/HIGHLIGHT/CHAIN
    # already use above (and the undirected-arrow fix below, which does
    # visibly work), apply correctly instead.
    for kind in ("CONTEXT", "HIGHLIGHT", "CHAIN"):
        edge_styles.append(
            RawStyle(f'edge[label="{kind}"]', {"font-size": EDGE_FONT_SIZE})
        )
    # Overrides each kind's own arrow above wherever the real edge data
    # says undirected=True (only present at all when "load full graph" is
    # on) — both arrowheads instead of whatever single direction the
    # edge's CONTEXT/HIGHLIGHT/CHAIN kind alone would otherwise draw. One
    # rule per kind (not a single global one) so each arrow's color still
    # matches that kind's own line color.
    for kind, color in (
        ("CONTEXT", CONTEXT_COLOR),
        ("HIGHLIGHT", HIGHLIGHT_COLOR),
        ("CHAIN", HIGHLIGHT_COLOR),
    ):
        edge_styles.append(
            RawStyle(
                f'edge[label="{kind}"][undirected = "True"]',
                {
                    "source-arrow-shape": "triangle",
                    "target-arrow-shape": "triangle",
                    "source-arrow-color": color,
                    "target-arrow-color": color,
                },
            )
        )
    # CHAIN's base style has no arrow at all (direction isn't knowable
    # from the JSON alone — see build_fig3/module docstring), but once
    # "load full graph" resolves the real DIS-DIS/PARENT edge (always
    # undirected=False — genuinely directional), a single correctly
    # -oriented arrow belongs here. attach_graph_attrs() has already
    # swapped source/target so this edge's stored direction always
    # matches the real one, so a plain target-arrow is always correct —
    # never source-arrow, unlike the undirected=True rule above.
    edge_styles.append(
        RawStyle(
            'edge[label="CHAIN"][undirected = "False"]',
            {"target-arrow-shape": "triangle", "target-arrow-color": HIGHLIGHT_COLOR},
        )
    )

    st_link_analysis(
        elements,
        layout=layout,
        node_styles=node_styles,
        edge_styles=edge_styles,
        height=650,
        key=f"{category}-{row_idx}-{show_context}",
    )
    st.caption(
        "Click a node or edge to open its attribute panel."
        + (
            ""
            if show_context
            else ' Enable "load full graph" above for the full attribute record.'
        )
    )
    legend = "  ".join(f"🔵 {NODE_TYPE_LABELS.get(t, t)}" for t in types_present)
    if show_context and context_entities:
        legend += "  ⚪ other resolved entity (context)"
    st.caption(legend)
    if context_truncated:
        st.caption(
            f"+{context_truncated} more directly connected entities not shown (capped at {MAX_CONTEXT_NODES})"
        )
    if fig6_no_matching_coverage:
        st.caption(
            "No other variants on this gene have matching-type CIViC coverage for "
            f"the {selected_row['track']} track — true type-matched coverage is rare "
            "(see graph_statistics.py's clinvar_gap_coverage)."
        )
