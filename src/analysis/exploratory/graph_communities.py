#!/usr/bin/env python3
"""
One Leiden pass on the cancer-scoped graph's undirected view: do
evidence-backed nodes cluster together, or scatter across communities? A
sanity-check figure that doubles as a structural-gap signal; a community
that's almost entirely non-evidence-backed is a candidate region CIViC/
ClinVar curation hasn't reached yet, even though it's structurally as
connected as curated regions.

igraph/leidenalg, not networkx's own louvain_communities: compiled C
implementation, needed at this graph's scale (~192k nodes / ~5.1M edges)

Inputs:
    - data/processed/cancer_scoped_graph_*.pkl (from build_cancer_scoped_graph.py)
Outputs:
    - printed report
    - data/processed/graph_communities/graph_communities_<date>.json (same
      figures the report prints, structured for downstream reuse)

Usage:
    python src/analysis/exploratory/graph_communities.py
"""

# %%
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import igraph as ig
import leidenalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
from load_files import find_latest_file

RANDOM_SEED = 42
# A community below this size is too small
MIN_COMMUNITY_SIZE_FOR_FLAGGING = 5

OUT_DIR = Path("data/processed/graph_communities")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# 0. Load, build the undirected view, convert to igraph.
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
g_ig = ig.Graph.from_networkx(G_und)
print(
    f"[setup] undirected view + igraph conversion in {time.time() - t0:.1f}s — "
    f"{g_ig.vcount()} vertices, {g_ig.ecount()} edges"
)

# %%
# ---------------------------------------------------------------------------
# 1. Leiden.
# ---------------------------------------------------------------------------

t0 = time.time()
partition = leidenalg.find_partition(
    g_ig, leidenalg.RBConfigurationVertexPartition, seed=RANDOM_SEED
)
# partition.quality() is RBConfiguration's own objective (sum of within-
# community edges minus the resolution*expected-edges term), NOT normalized
# by 2m — at this graph's scale that's ~2.5M, not the standard modularity
# score. g_ig.modularity() below computes the actual normalized value.
modularity = g_ig.modularity(partition.membership)
print(
    f"\n[leiden] done in {time.time() - t0:.1f}s — {len(partition)} communities, "
    f"modularity={modularity:.4f}"
)

community_sizes = Counter(partition.membership)
print(
    f"[leiden] size distribution: largest={max(community_sizes.values())}, "
    f"smallest={min(community_sizes.values())}, "
    f"median={sorted(community_sizes.values())[len(community_sizes) // 2]}"
)

report = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "graph_snapshot": Path(graph_path).name,
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "random_seed": RANDOM_SEED,
    "min_community_size_for_flagging": MIN_COMMUNITY_SIZE_FOR_FLAGGING,
    "leiden": {
        "n_communities": len(partition),
        "modularity": modularity,
        "size_largest": max(community_sizes.values()),
        "size_smallest": min(community_sizes.values()),
        "size_median": sorted(community_sizes.values())[len(community_sizes) // 2],
    },
}

# %%
# ---------------------------------------------------------------------------
# 2. Per-community evidence-backed fraction. Curated = in_civic_evidence_item
#    for GEN/DIS/DRG/VAR nodes; EVI and the 6 passthrough native types
#    (BPO/PHE/MFN/CCO/PWY/EXP) don't carry that flag at all and are excluded
#    from the fraction (they'd silently read as "not curated" otherwise).
# ---------------------------------------------------------------------------

node_ids = g_ig.vs["_nx_name"]
CURATED_ELIGIBLE_TYPES = {"GEN", "DIS", "DRG", "VAR"}

# Every node id per community — persisted below for the flagged (all-
# curated/all-gap) communities so a later script can pull the real
# nodes/edges for a specific community without rerunning Leiden (which
# itself needs the ~170s igraph conversion above every time).
community_members: dict[int, list[str]] = {}
community_curated = Counter()
community_eligible = Counter()
for vertex_idx, node_id in enumerate(node_ids):
    comm = partition.membership[vertex_idx]
    community_members.setdefault(comm, []).append(node_id)
    attrs = G.nodes[node_id]
    if attrs.get("node_type") not in CURATED_ELIGIBLE_TYPES:
        continue
    community_eligible[comm] += 1
    if attrs.get("in_civic_evidence_item"):
        community_curated[comm] += 1

all_curated = []
all_gap = []
mixed_fractions = []
for comm, size in community_sizes.items():
    if size < MIN_COMMUNITY_SIZE_FOR_FLAGGING:
        continue
    eligible = community_eligible.get(comm, 0)
    if eligible == 0:
        continue
    curated = community_curated.get(comm, 0)
    fraction = curated / eligible
    mixed_fractions.append(fraction)
    if fraction == 1.0:
        all_curated.append((comm, size, eligible))
    elif fraction == 0.0:
        all_gap.append((comm, size, eligible))

print(
    f"\n[structural gap] of {len(mixed_fractions)} communities with >= "
    f"{MIN_COMMUNITY_SIZE_FOR_FLAGGING} nodes and >=1 GEN/DIS/DRG/VAR member:"
)
print(f"  {len(all_curated)} are 100% evidence-backed")
print(f"  {len(all_gap)} are 0% evidence-backed (structural-gap candidates)")
print(
    f"  mean curated fraction across all of them: "
    f"{sum(mixed_fractions) / len(mixed_fractions):.3f}"
)

print(
    "\n[structural gap] 10 largest all-gap communities (comm_id, size, eligible_nodes):"
)
top_all_gap = sorted(all_gap, key=lambda x: -x[1])[:10]
for comm, size, eligible in top_all_gap:
    print(f"  {comm}: size={size}, eligible={eligible}")

report["structural_gap"] = {
    "n_communities_considered": len(mixed_fractions),
    "n_all_curated": len(all_curated),
    "n_all_gap": len(all_gap),
    "mean_curated_fraction": sum(mixed_fractions) / len(mixed_fractions),
    "all_curated_communities": [
        {
            "community_id": comm, "size": size, "eligible": eligible,
            "members": community_members[comm],
        }
        for comm, size, eligible in all_curated
    ],
    "all_gap_communities": [
        {
            "community_id": comm, "size": size, "eligible": eligible,
            "members": community_members[comm],
        }
        for comm, size, eligible in all_gap
    ],
    "top_10_largest_all_gap_communities": [
        {"community_id": comm, "size": size, "eligible": eligible}
        for comm, size, eligible in top_all_gap
    ],
}

report_path = OUT_DIR / f"graph_communities_{datetime.now(timezone.utc):%Y-%m-%d}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n[save] wrote {report_path}")

print("\n[done]")

# %%
