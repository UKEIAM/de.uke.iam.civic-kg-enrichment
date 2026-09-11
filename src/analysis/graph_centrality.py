#!/usr/bin/env python3
"""
Centrality, curated vs. not: does having a CIViC/ClinVar evidence item track
structural importance in the cancer-scoped graph, or is curation independent
of a node's position in the graph? Or, "are curated nodes typically more central"?

Direction-sensitive algorithms need an undirected view:
G_und = G.to_undirected()
is built once and used for degree/betweenness; weak connectivity is the one
thing checked on the raw DiGraph (direction genuinely doesn't matter for it
either way, but nx.connected_components() isn't implemented for a DiGraph at
all, so it has to be nx.number_weakly_connected_components()).

Betweenness is exact-infeasible at cancer_scoped_graph's scale (~5M edges,
O(V*E) per node); BETWEENNESS_SAMPLE_K below controls the k-sample
approximation (nx.betweenness_centrality(..., k=...)); a real analysis run
can raise it at the cost of runtime.

Code flow:
1. build the undirected view
2. sanity-check that the graph is actually one coherent structure, not scattered islands (betweeness)
3. compute two different flavors of "how structurally important is this node" (raw connection count, and bridge-between-others count)
4. and then, for each entity type, statistically test whether "CIViC curated this" lines up with "this node is structurally important,"
or whether curation is basically independent of a node's position in the network.

Inputs:
    - data/processed/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
Outputs:
    - printed report only (no file output)

Usage:
    python src/analysis/graph_centrality.py
"""

# %%
import os
import pickle
import sys
import time

import networkx as nx
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

BETWEENNESS_SAMPLE_K = 100
RANDOM_SEED = 42

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


def mann_whitney_report(
    label: str, values: dict, curated_ids: set, other_ids: set
) -> None:
    curated_vals = [values[n] for n in curated_ids if n in values]
    other_vals = [values[n] for n in other_ids if n in values]
    if not curated_vals or not other_vals:
        print(f"  [{label}] skipped - one group is empty")
        return

    result = stats.mannwhitneyu(curated_vals, other_vals, alternative="two-sided")
    u = result.statistic
    p = result.pvalue
    prob_superiority = u / (len(curated_vals) * len(other_vals))

    def _median(xs):
        s = sorted(xs)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

    # Betweenness values are tiny fractions at this graph's scale (~192k
    # nodes); use scientific notation for anything below the 2-decimal threshold.
    def _fmt(x: float) -> str:
        return f"{x:.2e}" if abs(x) < 0.01 and x != 0 else f"{x:.2f}"

    print(
        f"  [{label}] curated: n={len(curated_vals)}, median={_fmt(_median(curated_vals))}, "
        f"mean={_fmt(sum(curated_vals) / len(curated_vals))}"
    )
    print(
        f"  [{label}] not curated: n={len(other_vals)}, median={_fmt(_median(other_vals))}, "
        f"mean={_fmt(sum(other_vals) / len(other_vals))}"
    )
    print(
        f"  [{label}] Mann-Whitney U p={p:.3e}, U/(n1*n2)={prob_superiority:.3f} "
        f"(probability a random curated node outranks a random non-curated one)"
    )


for node_type in ("GEN", "DIS", "DRG", "VAR"):
    type_ids = {n for n, a in G.nodes(data=True) if a.get("node_type") == node_type}
    curated_ids = {n for n in type_ids if G.nodes[n].get("in_civic_evidence_item")}
    other_ids = type_ids - curated_ids
    print(
        f"\n[{node_type}] {len(curated_ids)} curated (in_civic_evidence_item=True), "
        f"{len(other_ids)} not"
    )
    mann_whitney_report("degree", degree, curated_ids, other_ids)
    mann_whitney_report("betweenness", betweenness, curated_ids, other_ids)

print("\n[done]")

# %%
