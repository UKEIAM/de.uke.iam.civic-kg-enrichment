#!/usr/bin/env python3
"""Shared visual encoding and rendering helpers for this project's graph
figure scripts — graph_ego_networks.py's generic picker, graph_ego_figures.py's
4 EVI-centered figures, and graph_community_figures.py's community
examples all color/label nodes by node_type the same way and share the
same tooltip/legend/export machinery, kept here once so they can't drift
out of sync with each other. Panel-layout logic that's genuinely specific
to one script (e.g. graph_ego_figures.py's shell-layout-by-hop-distance)
stays in that script — only what every consumer needs unchanged lives here."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from pyvis.network import Network

NODE_COLORS = {
    "GEN": "#0E7A6F",
    "DIS": "#33578C",
    "DRG": "#1E8CA6",
    "VAR": "#A9781F",
    "EVI": "#B0502B",
    "PHE": "#7A4FA0",
    # Native passthrough types (BPO/CCO/MFN/PWY/EXP) — never appear in the
    # EVI-centered figures (their include-sets are always GEN/DIS/DRG/VAR/
    # EVI/PHE), but a community figure pulls in whatever's actually in the
    # community, which can be any native node type.
    "BPO": "#4C7A3D",
    "CCO": "#8A4B6B",
    "MFN": "#5C6BC0",
    "PWY": "#2E8B87",
    "EXP": "#6B4226",
}

NODE_TYPE_LABELS = {
    "GEN": "Gene",
    "DIS": "Disease",
    "DRG": "Drug",
    "VAR": "Variant",
    "EVI": "Evidence item",
    "PHE": "Phenotype",
    "BPO": "Biological process",
    "CCO": "Cellular component",
    "MFN": "Molecular function",
    "PWY": "Pathway",
    "EXP": "Exposure",
}

# Shape, not just color, so the node-type encoding survives a grayscale
# printout — matplotlib marker codes.
NODE_MARKERS = {
    "GEN": "o",
    "DIS": "s",
    "DRG": "D",
    "VAR": "^",
    "PHE": "v",
    "EVI": "*",
    "BPO": "P",
    "CCO": "X",
    "MFN": "p",
    "PWY": "h",
    "EXP": "8",
}

# Node types drawn in a fixed, consistent order across every figure's
# legend — callers filter this down to whichever types actually appear.
NODE_TYPE_ORDER = ["EVI", "GEN", "DIS", "DRG", "VAR", "PHE", "BPO", "CCO", "MFN", "PWY", "EXP"]

HIGHLIGHT_COLOR = "#DC267F"  # saturated, distinct from every NODE_COLORS value
CONTEXT_COLOR = "#B0B0B0"  # muted gray for non-highlighted context edges AND nodes
GAP_COLOR = "#7A7A7A"  # dashed for "structural, not curated" encoding


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


def paper_node_label(node_id: str, attrs: dict) -> str:
    """Like node_label(), but prefers the short symbol over the long
    descriptive name — node_label() picks up a gene's full "B-Raf
    proto-oncogene, serine/threonine kinase" over "BRAF"; compact figure
    node labels need the short form. A VAR node's own gene_symbol (e.g.
    "BRAF") isn't a distinguishing label — every variant on the same gene
    would show the same text — so VAR nodes are labeled from their own
    hgvs_name instead, trimmed to just the protein change ("p.Val600Glu")
    when one is present, since the full HGVS string is too long to sit
    next to a node marker. Many ClinVar variants have no protein-level
    annotation at all (non-coding, intronic, splice, UTR) — those fall
    back to just the coding-level change ("c.340+1G>A") instead of the
    full "NM_000551.4(VHL):c.340+1G>A" string, so every variant gets the
    same short "prefix.change" style regardless of which level of HGVS
    notation it actually has."""
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


def format_attr_value(value) -> str:
    """Compact, tooltip-friendly rendering of one attribute value — a
    dict or a list of dicts (e.g. an edge's civic_evidence records, a
    node's sources: {direct, indirect}) becomes compact JSON; a list of
    scalars (e.g. a disease's ancestors) becomes a comma-joined string;
    anything else is just str()'d."""
    if isinstance(value, dict) or (
        isinstance(value, list) and any(isinstance(x, dict) for x in value)
    ):
        return json.dumps(value)
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "[]"
    return str(value)


def node_tooltip(node_id: str, attrs: dict) -> str:
    """Every attribute on the node, one per line, for an interactive
    HTML's hover tooltip — the static figures only ever show a label, so
    this is the one place the full record is visible without opening the
    pickle directly. id and node_type lead; the rest are alphabetical.
    Plain text with \\n line breaks, not HTML — vis-network's node/edge
    title renders as literal text (not innerHTML), so <b>/<br> tags show
    up as literal characters instead of formatting; \\n is what it
    actually treats as a line break."""
    lines = [f"id: {node_id}"]
    ordered_keys = ["node_type"] + sorted(k for k in attrs if k != "node_type")
    for key in ordered_keys:
        if key not in attrs:
            continue
        lines.append(f"{key}: {format_attr_value(attrs[key])}")
    return "\n".join(lines)


def edge_tooltip(u: str, v: str, data: dict, extra: str | None = None) -> str:
    """Every attribute on the edge, one per line — same completeness and
    plain-text-with-\\n formatting as node_tooltip(). extra prepends a
    figure-specific note (e.g. which relation this edge is highlighted
    for) above the raw attributes."""
    lines = []
    if extra:
        lines.append(extra)
    lines.append(f"from: {u}")
    lines.append(f"to: {v}")
    for key in sorted(k for k in data if k not in ("from", "to")):
        lines.append(f"{key}: {format_attr_value(data[key])}")
    return "\n".join(lines)


def node_legend_handles(types: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=NODE_MARKERS[t],
            color="w",
            markerfacecolor=NODE_COLORS[t],
            markeredgecolor="black",
            markersize=9,
            label=NODE_TYPE_LABELS[t],
        )
        for t in types
    ]


def context_legend_handle(label: str) -> Line2D:
    """A node's type-color swatch (node_legend_handles) doesn't distinguish
    a highlighted/in-focus instance of that type from a muted context one
    — e.g. a figure can show one drug in its real DRG color and another,
    unrelated drug in gray. This one extra legend row explains the gray
    encoding itself; label is caller-specific (what "muted" means for this
    figure — an EVI's other resolved entities, or a community's non-focal
    members)."""
    return Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=CONTEXT_COLOR,
        markeredgecolor="black",
        markersize=9,
        label=label,
    )


def edge_legend_handle(color: str, style: str, label: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=color,
        lw=2.5,
        linestyle="--" if style == "dashed" else "-",
        label=label,
    )


def save_static_figure(fig, legend_handles: list[Line2D], out_path_base: str) -> None:
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{out_path_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_path_base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[save] wrote {out_path_base}.svg / .pdf")


def legend_html(
    node_types: list[str],
    edges: list[tuple[str, str, str]],
    context_label: str | None = None,
) -> str:
    """pyvis renders no legend of its own — this builds the same
    node-type/edge-style key a static figure's matplotlib legend shows, as
    a small floating panel injected into the saved HTML so it's visible on
    the page itself, not just discoverable by hovering. context_label, if
    given, adds one row explaining the muted-gray encoding (see
    context_legend_handle())."""
    rows = []
    for t in node_types:
        rows.append(
            '<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{NODE_COLORS[t]};'
            'border:1px solid #333;display:inline-block;flex:none;"></span>'
            f"<span>{NODE_TYPE_LABELS[t]}</span></div>"
        )
    if context_label:
        rows.append(
            '<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{CONTEXT_COLOR};'
            'border:1px solid #333;display:inline-block;flex:none;"></span>'
            f"<span>{context_label}</span></div>"
        )
    for color, style, label in edges:
        border = "dashed" if style == "dashed" else "solid"
        rows.append(
            '<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
            f'<span style="width:22px;border-top:3px {border} {color};display:inline-block;flex:none;"></span>'
            f"<span>{label}</span></div>"
        )
    return (
        '<div style="position:fixed;bottom:12px;left:12px;background:white;'
        "border:1px solid #ccc;border-radius:8px;padding:10px 14px;"
        "font:13px -apple-system,sans-serif;box-shadow:0 1px 6px rgba(0,0,0,.25);"
        'z-index:1000;max-width:280px;">'
        '<div style="font-weight:600;margin-bottom:6px;">Legend</div>'
        + "".join(rows)
        + "</div>"
    )


def title_bar_html(title: str) -> str:
    """A single title bar, injected once into the page — pyvis Network's
    own heading=... parameter renders its default template's <h1> TWICE, so
    callers build the title here instead and never pass heading= to
    Network()."""
    return (
        '<div style="text-align:center;padding:14px 10px 2px;'
        'font:600 20px -apple-system,sans-serif;">'
        f"{title}</div>"
    )


def render_interactive_panel(
    sub: nx.Graph,
    highlighted_edges: list[dict],
    out_path: Path,
    synthetic_edges: list[dict] | None = None,
    center_id: str | None = None,
    title: str | None = None,
    context_node_ids: set | None = None,
    context_label: str | None = None,
    legend_edges: list[tuple[str, str, str]] | None = None,
) -> None:
    """Interactive pyvis rendering of any subgraph — node/edge color by
    type (muted gray for context_node_ids), full-attribute hover tooltips,
    relation labels drawn directly on highlighted edges (not just on
    hover), a single title bar, and an injected legend. center_id (if
    given) just renders larger — pyvis does its own force-directed layout,
    so there's no positional "center" the way the static shell-layout
    figures have one."""
    context_node_ids = context_node_ids or set()
    highlighted_by_pair = {(h["u"], h["v"]): h for h in highlighted_edges}
    highlighted_by_pair.update({(h["v"], h["u"]): h for h in highlighted_edges})

    net = Network(height="750px", width="100%", directed=True, notebook=False)
    for node_id, attrs in sub.nodes(data=True):
        color = (
            CONTEXT_COLOR
            if node_id in context_node_ids
            else NODE_COLORS.get(attrs.get("node_type"), "#828C97")
        )
        size = 32 if node_id == center_id else 18
        net.add_node(
            node_id,
            label=paper_node_label(node_id, attrs),
            title=node_tooltip(node_id, attrs),
            color=color,
            size=size,
        )
    for u, v, data in sub.edges(data=True):
        if u == v:
            continue  # self-loops (real data, e.g. a gene's own homodimer record) add nothing to any figure's story
        h = highlighted_by_pair.get((u, v))
        if h:
            net.add_edge(
                u,
                v,
                color=h.get("color", HIGHLIGHT_COLOR),
                width=3,
                dashes=h.get("style") == "dashed",
                label=h.get("label", ""),
                title=edge_tooltip(u, v, data, extra=h.get("label")),
            )
        else:
            net.add_edge(u, v, color=CONTEXT_COLOR, title=edge_tooltip(u, v, data))

    for s in synthetic_edges or []:
        # Not a real graph edge (e.g. a "no direct edge" absence connector)
        # — both endpoints are already added as nodes above, so pyvis can
        # still draw a line between them.
        net.add_edge(
            s["u"],
            s["v"],
            color=s.get("color", GAP_COLOR),
            width=1.5,
            dashes=True,
            label=s.get("label", ""),
            title=s.get("label", ""),
        )

    net.write_html(str(out_path))

    node_types_present = [
        t for t in NODE_TYPE_ORDER if any(sub.nodes[n].get("node_type") == t for n in sub.nodes())
    ]
    legend = legend_html(node_types_present, legend_edges or [], context_label)
    page_html = out_path.read_text()
    if title:
        page_html = page_html.replace("<body>", "<body>" + title_bar_html(title), 1)
    out_path.write_text(page_html.replace("</body>", legend + "</body>"))

    print(f"[save] wrote {out_path}")
