#!/usr/bin/env python3
"""
Centrality, curated vs. not: does having a CIViC/ClinVar evidence item track
structural importance in the cancer-scoped graph, or is curation independent
of a node's position in the graph? Or, "are curated nodes typically more central"?

BETWEENNESS_SAMPLE_K: controls the k-sample
approximation (nx.betweenness_centrality(..., k=...)); a real analysis run
can raise it at the cost of runtime.

Code flow:
1. build the undirected view (G_und = G.to_undirected())
2. sanity-check that the graph is actually one coherent structure, not scattered islands (betweeness)
3. compute two different flavors of "how structurally important is this node" (raw connection count, and bridge-between-others count)
4. and then, for each entity type, statistically test whether "CIViC curated this" lines up with "this node is structurally important,"
or whether curation is basically independent of a node's position in the network.

Inputs:
    - data/processed/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
Outputs:
    - printed report
    - data/processed/graph_centrality/graph_centrality_<date>.json (same
      figures the report prints, structured for downstream reuse)

Usage:
    python src/analysis/exploratory/graph_centrality.py
"""

# %%
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from load_files import find_latest_file

BETWEENNESS_SAMPLE_K = 100
RANDOM_SEED = 42

OUT_DIR = Path("data/processed/graph_centrality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# 0. Load, build the undirected view.
# ---------------------------------------------------------------------------

graph_path = find_latest_file("data/processed", "cancer_scoped_graph_*.pkl")
print(f"[load] {graph_path}")
t0 = time.time()
with open(graph_path, "rb") as f:
    G = pickle.load(f)
print(
    f"[load] done in {time.time() - t0:.1f}s — {G.number_of_nodes()} nodes, "
    f"{G.number_of_edges()} edges"
)

t0 = time.time()
G_und = G.to_undirected()
print(f"[setup] built undirected view in {time.time() - t0:.1f}s")

# %%
# ---------------------------------------------------------------------------
# 1. Weak connectivity: checked on the raw DiGraph, since
#    nx.connected_components() isn't implemented for it.
# ---------------------------------------------------------------------------

t0 = time.time()
n_wcc = nx.number_weakly_connected_components(G)
largest_wcc = max(nx.weakly_connected_components(G), key=len)
print(
    f"\n[connectivity] {n_wcc} weakly connected components in {time.time() - t0:.1f}s "
    f"; largest has {len(largest_wcc)} / {G.number_of_nodes()} nodes "
    f"({100 * len(largest_wcc) / G.number_of_nodes():.1f}%)"
)

report = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "graph_snapshot": Path(graph_path).name,
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "betweenness_sample_k": BETWEENNESS_SAMPLE_K,
    "random_seed": RANDOM_SEED,
    "connectivity": {
        "n_weakly_connected_components": n_wcc,
        "largest_wcc_size": len(largest_wcc),
        "largest_wcc_fraction": len(largest_wcc) / G.number_of_nodes(),
    },
    "by_node_type": {},
}

# %%
# ---------------------------------------------------------------------------
# 2. Degree (total, undirected) and approximate betweenness.
# ---------------------------------------------------------------------------

t0 = time.time()
degree = dict(G_und.degree())
print(f"\n[centrality] degree computed in {time.time() - t0:.1f}s")

t0 = time.time()
betweenness = nx.betweenness_centrality(G_und, k=BETWEENNESS_SAMPLE_K, seed=RANDOM_SEED)
print(
    f"[centrality] betweenness (k={BETWEENNESS_SAMPLE_K} sample) computed in "
    f"{time.time() - t0:.1f}s"
)

# %%
# ---------------------------------------------------------------------------
# 3. Curated vs. not, for GEN/DIS/DRG/VAR
#    Each node type has an in_civic_evidence_item flag indicating whether it's
#    from a CIViC evidence item. VAR carries the same
#    in_civic_evidence_item flag as GEN/DIS/DRG (set on the existing variants
#    in save_graph_nodes.py, explicitly False on build_cancer_scoped_variants.py's
#    new-only ones).
# ---------------------------------------------------------------------------


def _median(xs):
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


# Betweenness values are tiny fractions at this graph's scale (~192k nodes);
# use scientific notation for anything below the 2-decimal threshold.
def _fmt(x: float) -> str:
    return f"{x:.2e}" if abs(x) < 0.01 and x != 0 else f"{x:.2f}"


def mann_whitney_report(
    label: str, values: dict, curated_ids: set, other_ids: set
) -> dict | None:
    curated_vals = [values[n] for n in curated_ids if n in values]
    other_vals = [values[n] for n in other_ids if n in values]
    if not curated_vals or not other_vals:
        print(f"  [{label}] skipped - one group is empty")
        return None

    result = stats.mannwhitneyu(curated_vals, other_vals, alternative="two-sided")
    u = float(result.statistic)
    p = float(result.pvalue)
    prob_superiority = u / (len(curated_vals) * len(other_vals))

    curated_median = _median(curated_vals)
    curated_mean = sum(curated_vals) / len(curated_vals)
    other_median = _median(other_vals)
    other_mean = sum(other_vals) / len(other_vals)

    print(
        f"  [{label}] curated: n={len(curated_vals)}, median={_fmt(curated_median)}, "
        f"mean={_fmt(curated_mean)}"
    )
    print(
        f"  [{label}] not curated: n={len(other_vals)}, median={_fmt(other_median)}, "
        f"mean={_fmt(other_mean)}"
    )
    print(
        f"  [{label}] Mann-Whitney U p={p:.3e}, U/(n1*n2)={prob_superiority:.3f} "
        f"(probability a random curated node outranks a random non-curated one)"
    )

    return {
        "curated": {
            "n": len(curated_vals),
            "median": curated_median,
            "mean": curated_mean,
        },
        "not_curated": {
            "n": len(other_vals),
            "median": other_median,
            "mean": other_mean,
        },
        "mannwhitneyu": {
            "u_statistic": u,
            "p_value": p,
            "prob_superiority": prob_superiority,
        },
    }


for node_type in ("GEN", "DIS", "DRG", "VAR"):
    type_ids = {n for n, a in G.nodes(data=True) if a.get("node_type") == node_type}
    curated_ids = {n for n in type_ids if G.nodes[n].get("in_civic_evidence_item")}
    other_ids = type_ids - curated_ids
    print(
        f"\n[{node_type}] {len(curated_ids)} curated (in_civic_evidence_item=True), "
        f"{len(other_ids)} not"
    )
    degree_result = mann_whitney_report("degree", degree, curated_ids, other_ids)
    betweenness_result = mann_whitney_report(
        "betweenness", betweenness, curated_ids, other_ids
    )
    report["by_node_type"][node_type] = {
        "n_curated": len(curated_ids),
        "n_not_curated": len(other_ids),
        "degree": degree_result,
        "betweenness": betweenness_result,
    }

# %%
# ---------------------------------------------------------------------------
# 4. Evidence items themselves: which EVI nodes are most central, and what
#    do the entities they directly cite look like? EVI isn't part of the
#    curated-vs-not comparison above (every EVI is definitionally
#    curated), but an EVI's own degree is capped by how many entities it
#    cites (usually a handful) — a high-betweenness EVI is the more
#    informative number: it means this specific evidence item sits on an
#    unusually large share of shortest paths through the graph, i.e. it's
#    a structural bridge, not just a well-connected record.
# ---------------------------------------------------------------------------

TOP_EVI_N = 10
evi_ids = [n for n, a in G.nodes(data=True) if a.get("node_type") == "EVI"]


def node_short_label(node_id: str) -> str:
    attrs = G.nodes[node_id]
    for key in ("symbol", "name", "hgvs_name"):
        if attrs.get(key):
            return str(attrs[key])
    return node_id


def evi_neighbors(eid: str) -> list[dict]:
    """Every node this EVI directly connects to, with its own degree —
    sorted by degree desc, so the most central directly-cited entity is
    listed first."""
    neighbor_ids = set(G.successors(eid)) | set(G.predecessors(eid))
    rows = [
        {
            "id": n,
            "node_type": G.nodes[n].get("node_type"),
            "label": node_short_label(n),
            "degree": degree.get(n, 0),
        }
        for n in neighbor_ids
    ]
    return sorted(rows, key=lambda r: -r["degree"])


top_evi_by_degree = sorted(evi_ids, key=lambda n: -degree.get(n, 0))[:TOP_EVI_N]
top_evi_by_betweenness = sorted(evi_ids, key=lambda n: -betweenness.get(n, 0))[:TOP_EVI_N]

print(f"\n[EVI] {len(evi_ids)} evidence item nodes")
print(f"\n[EVI] top {TOP_EVI_N} by degree:")
for eid in top_evi_by_degree:
    print(f"  {eid}: degree={degree[eid]}")
print(f"\n[EVI] top {TOP_EVI_N} by betweenness:")
for eid in top_evi_by_betweenness:
    print(f"  {eid}: betweenness={betweenness[eid]:.3e}, degree={degree[eid]}")

report["evidence_items"] = {
    "n_evi_nodes": len(evi_ids),
    "top_by_degree": [
        {"evi": eid, "degree": degree[eid], "neighbors": evi_neighbors(eid)}
        for eid in top_evi_by_degree
    ],
    "top_by_betweenness": [
        {
            "evi": eid,
            "betweenness": betweenness[eid],
            "degree": degree[eid],
            "neighbors": evi_neighbors(eid),
        }
        for eid in top_evi_by_betweenness
    ],
}

report_path = OUT_DIR / f"graph_centrality_{datetime.now(timezone.utc):%Y-%m-%d}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[save] wrote {report_path}")

print("\n[done]")

# %%
