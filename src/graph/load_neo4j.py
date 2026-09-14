#!/usr/bin/env python3
"""
One-way Cypher load of evidence_backed_expanded_graph_*.pkl into Neo4j,
the exploration/Bloom copy, not a second source of truth. networkx/igraph
stays the system of record for all analysis (graph_centrality.py,
graph_communities.py); this script only ever reads from the pickle and
writes to Neo4j, never the reverse. Rebuilding the graph means rerunning
this too, nothing keeps the two in sync automatically.

Source is the EXPANDED graph specifically (not the cancer-scoped one).

Node/edge property dicts are flattened for Neo4j's flat-primitives-or-
arrays-only property model: struct-valued attrs (e.g. OptimusKG's own
sources={direct:[...], indirect:[...]}) are JSON-stringified into a
"<key>_json" property rather than dropped.

The actual write loop only runs
if NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD are set (via .env — see
.env.example), and is otherwise skipped.

Inputs:
    - data/processed/evidence_backed_expanded_graph_*.pkl (from build_expanded_evidence_graph.py)
    - .env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
Outputs:
    - writes directly to the configured Neo4j database (or, with no
      instance configured, a dry-run summary to stdout only)

Usage:
    python src/graph/load_neo4j.py
"""

# %%
import json
import os
import pickle
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from load_files import find_latest_file

load_dotenv()

BATCH_SIZE = 1000

NODE_LABELS = {
    "GEN": "Gene",
    "DIS": "Disease",
    "DRG": "Drug",
    "VAR": "Variant",
    "EVI": "EvidenceItem",
    "ANA": "AnatomicalStructure",
    "BPO": "BiologicalProcess",
    "CCO": "CellularComponent",
    "EXP": "Exposure",
    "MFN": "MolecularFunction",
    "PHE": "Phenotype",
    "PWY": "Pathway",
}

# edge_type -> fixed relationship type, for the Direct types where there's
# exactly one kind of fact and no per-record splitting.
FIXED_RELATIONSHIP_TYPES = {
    "GEN-EVI": "HAS_EVIDENCE",
    "DIS-EVI": "HAS_EVIDENCE",
    "DRG-EVI": "HAS_EVIDENCE",
    "VAR-EVI": "HAS_EVIDENCE",
    "VAR-GEN": "LOCATED_IN",
}

# Merged types: one relationship per record in a list field, typed by that
# record's own origin/kind rather than sharing one edge_type-wide type.
ORIGIN_TO_RELATIONSHIP_TYPE = {
    "clinvar_phenotype": "CLINVAR_PHENOTYPE",
    "civic_evidence": "CIVIC_EVIDENCE",
    "civic": "CIVIC_EVIDENCE",
}
MERGED_ORIGIN_FIELD_EDGE_TYPES = {"VAR-DIS"}
MERGED_RECORD_LIST_EDGE_TYPES = {
    "VAR-DRG"
}  # always CIVIC_EVIDENCE, one per evidence_items entry

# Shared/namespaced types: the row's own OptimusKG relation value IS the
# type, plus a parallel CIVIC_EVIDENCE relationship per civic_evidence entry.
SHARED_NAMESPACED_EDGE_TYPES = {"DRG-GEN", "DRG-DIS"}

# Native pass-through types: OptimusKG's own relation value IS the type,
# one relationship per edge, no CIViC overlay to split out. ANA-GEN was
# missing from the spec's own table (an oversight caught here, not in
# review) — the expanded graph carries 86,141 of them, so omitting it would
# crash on the first one.
NATIVE_PASSTHROUGH_EDGE_TYPES = {
    "GEN-GEN",
    "ANA-GEN",
    "PHE-GEN",
    "BPO-GEN",
    "CCO-GEN",
    "MFN-GEN",
    "PWY-GEN",
    "DRG-PHE",
    "DIS-DIS",
    "DIS-PHE",
    "EXP-GEN",
    "EXP-DIS",
    "DRG-BPO",
}

# %%
# ---------------------------------------------------------------------------
# Pure translation functions — no I/O, no driver. These are what the
# self-test section below exercises without a live instance.
# ---------------------------------------------------------------------------


def flatten_properties(attrs: dict, exclude: set) -> dict:
    """Neo4j properties are flat primitives or arrays of primitives only —
    no nested objects, and no null elements inside an array either (Neo4j
    rejects "collections containing null values" outright). A struct-valued
    attr (dict) gets JSON-stringified into "<key>_json" rather than dropped;
    a list of dicts likewise; a list of primitives is kept as a native
    array with any None entries filtered out (e.g. an EvidenceItem's
    drug_node_ids can be [None] for an unresolved reference — the same
    "omit rather than store null" rule scalars already get below, applied
    per-element). None scalar values are omitted entirely (a missing
    property already reads as null in Cypher, so there's nothing to
    store)."""
    props = {}
    for key, value in attrs.items():
        if key in exclude or value is None:
            continue
        if isinstance(value, dict):
            props[f"{key}_json"] = json.dumps(value)
        elif isinstance(value, list):
            if any(isinstance(item, dict) for item in value):
                props[f"{key}_json"] = json.dumps(value)
            else:
                cleaned = [item for item in value if item is not None]
                if cleaned:
                    props[key] = cleaned
        else:
            props[key] = value
    return props


def edge_to_relationships(edge_type: str, attrs: dict) -> list[tuple[str, dict]]:
    """The single place the spec's Direct/Merged/Shared/native-pass-through
    typing rules live. Returns a list of (relationship_type, properties)
    pairs — usually one, sometimes several for a Merged edge whose origin/
    evidence_items list has more than one entry, occasionally zero (a
    Merged/Shared edge that's purely structural on the CIViC side with no
    origin/civic_evidence entries at all shouldn't happen in this pipeline's
    own tables, but the caller should not assume exactly one)."""
    base_exclude = {"edge_type", "from", "to"}

    if edge_type in FIXED_RELATIONSHIP_TYPES:
        rel_type = FIXED_RELATIONSHIP_TYPES[edge_type]
        return [(rel_type, flatten_properties(attrs, base_exclude))]

    if edge_type in MERGED_ORIGIN_FIELD_EDGE_TYPES:
        base_props = flatten_properties(attrs, base_exclude | {"origin"})
        return [
            (ORIGIN_TO_RELATIONSHIP_TYPE[origin], base_props)
            for origin in attrs.get("origin", [])
        ]

    if edge_type in MERGED_RECORD_LIST_EDGE_TYPES:
        base_props = flatten_properties(
            attrs, base_exclude | {"origin", "evidence_items", "n_evidence_items"}
        )
        return [
            ("CIVIC_EVIDENCE", {**base_props, **record})
            for record in attrs.get("evidence_items", [])
        ]

    if edge_type in SHARED_NAMESPACED_EDGE_TYPES:
        relation = attrs.get("relation")
        base_exclude_shared = base_exclude | {
            "relation",
            "civic_evidence",
            "n_evidence_items",
        }
        base_props = flatten_properties(attrs, base_exclude_shared)
        relationships = []
        if relation:
            relationships.append((relation.upper(), base_props))
        for record in attrs.get("civic_evidence", []):
            relationships.append(("CIVIC_EVIDENCE", {**base_props, **record}))
        return relationships

    if edge_type in NATIVE_PASSTHROUGH_EDGE_TYPES:
        relation = attrs.get("relation", edge_type)
        props = flatten_properties(attrs, base_exclude | {"relation"})
        return [(relation.upper(), props)]

    raise ValueError(f"No relationship-typing rule for edge_type {edge_type!r}")


# %%
# ---------------------------------------------------------------------------
# Self-test: pure-function assertions against real sample edges pulled from
# the pickle, runs regardless of whether a live Neo4j instance exists.
# ---------------------------------------------------------------------------


def run_self_test(G) -> None:
    samples_by_type = {}
    for u, v, data in G.edges(data=True):
        et = data.get("edge_type")
        if et not in samples_by_type:
            samples_by_type[et] = (u, v, data)
        if len(samples_by_type) == len(
            FIXED_RELATIONSHIP_TYPES.keys()
            | MERGED_ORIGIN_FIELD_EDGE_TYPES
            | MERGED_RECORD_LIST_EDGE_TYPES
            | SHARED_NAMESPACED_EDGE_TYPES
            | NATIVE_PASSTHROUGH_EDGE_TYPES
        ):
            break

    checked = 0
    for et, (u, v, data) in samples_by_type.items():
        rels = edge_to_relationships(et, data)
        assert isinstance(rels, list), f"{et}: expected a list"
        for rel_type, props in rels:
            assert isinstance(rel_type, str) and rel_type == rel_type.upper(), (
                f"{et}: relationship type {rel_type!r} should be an uppercase string"
            )
            for key, val in props.items():
                assert not isinstance(val, dict), f"{et}: {key} is still a dict"
                if isinstance(val, list):
                    assert not any(isinstance(x, dict) for x in val), (
                        f"{et}: {key} is a list containing dicts"
                    )
        checked += 1
        print(f"  [self-test] {et}: {[r[0] for r in rels]} — OK")

    assert checked == len(samples_by_type)
    print(f"[self-test] {checked} edge types checked, all OK")

    node_sample = next(iter(G.nodes(data=True)))
    node_id, node_attrs = node_sample
    label = NODE_LABELS[node_attrs["node_type"]]
    props = flatten_properties(node_attrs, {"node_type"})
    assert not any(isinstance(v, dict) for v in props.values())
    print(
        f"  [self-test] node {node_id} -> label={label}, {len(props)} properties — OK"
    )


# %%
# ---------------------------------------------------------------------------
# Batched Neo4j writes. Only imported/used if a live instance is configured.
# ---------------------------------------------------------------------------


def write_nodes(driver, label: str, rows: list[dict]) -> None:
    query = f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props"
    with driver.session() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            session.execute_write(lambda tx, b=batch: tx.run(query, rows=b))


def write_relationships(driver, rel_type: str, rows: list[dict]) -> None:
    """MERGE on (a, b, type) alone would collapse every row sharing the
    same (from, to) under this type into one relationship — wrong for
    CIVIC_EVIDENCE specifically, where a single node pair can legitimately
    carry several distinct evidence citations (one DRG-GEN sample edge had
    19). Each row gets a merge_key that MERGE actually keys on: the
    citation's own id (evidence_items/civic_evidence entries always carry
    one) where present, so distinct citations become distinct
    relationships; falling back to "from->to" for the handful of types
    that can only ever produce one relationship per pair anyway (safe —
    nothing to disambiguate there), so reruns stay idempotent."""
    prepared = [
        {**row, "merge_key": row["props"].get("id", f"{row['from']}->{row['to']}")}
        for row in rows
    ]
    query = (
        f"UNWIND $rows AS row "
        f"MATCH (a {{id: row.from}}), (b {{id: row.to}}) "
        f"MERGE (a)-[r:{rel_type} {{_merge_key: row.merge_key}}]->(b) "
        f"SET r += row.props"
    )
    with driver.session() as session:
        for i in range(0, len(prepared), BATCH_SIZE):
            batch = prepared[i : i + BATCH_SIZE]
            session.execute_write(lambda tx, b=batch: tx.run(query, rows=b))


def ensure_constraints(driver) -> None:
    with driver.session() as session:
        for label in NODE_LABELS.values():
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )


def get_existing_counts(driver) -> tuple[dict[str, int], dict[str, int]]:
    """Resume support: if a previous run was interrupted mid-load (writes
    are batched, so a kill mid-batch leaves earlier batches durably
    committed), skip any label/relationship type whose live count already
    matches what this run would write, rather than resending it. Every
    write is a MERGE, so redoing a partially-loaded type is still correct,
    just slower — this is purely a speed-up, not a correctness requirement."""
    with driver.session() as session:
        node_counts = {
            r["label"]: r["c"]
            for r in session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS c"
            )
        }
        rel_counts = {
            r["t"]: r["c"]
            for r in session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c")
        }
    return node_counts, rel_counts


# %%
# ---------------------------------------------------------------------------
# Load the graph, self-test, then either write to Neo4j or print a dry run.
# ---------------------------------------------------------------------------

graph_path = find_latest_file("data/processed", "evidence_backed_expanded_graph_*.pkl")
print(f"[load] {graph_path}")
with open(graph_path, "rb") as f:
    G = pickle.load(f)
print(f"[load] done — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\n[self-test] checking translation functions against real sample edges:")
run_self_test(G)

nodes_by_label: dict[str, list[dict]] = {}
for node_id, attrs in G.nodes(data=True):
    label = NODE_LABELS[attrs["node_type"]]
    props = flatten_properties(attrs, {"node_type"})
    nodes_by_label.setdefault(label, []).append({"id": node_id, "props": props})

rels_by_type: dict[str, list[dict]] = {}
for u, v, data in G.edges(data=True):
    for rel_type, props in edge_to_relationships(data["edge_type"], data):
        rels_by_type.setdefault(rel_type, []).append(
            {"from": u, "to": v, "props": props}
        )

n_nodes_total = sum(len(v) for v in nodes_by_label.values())
n_rels_total = sum(len(v) for v in rels_by_type.values())
print(
    f"\n[translate] {n_nodes_total} nodes across {len(nodes_by_label)} labels, "
    f"{n_rels_total} relationships across {len(rels_by_type)} types"
)
for label, rows in sorted(nodes_by_label.items(), key=lambda x: -len(x[1])):
    print(f"  :{label}: {len(rows)}")
for rel_type, rows in sorted(rels_by_type.items(), key=lambda x: -len(x[1])):
    print(f"  :{rel_type}: {len(rows)}")

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
    print(
        "\n[dry run] NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not set (copy .env.example "
        "to .env and fill them in) — nothing written. The counts above are exactly "
        "what a real run would write."
    )
else:
    from neo4j import GraphDatabase

    print(f"\n[connect] {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[connect] OK")

    ensure_constraints(driver)
    print("[write] uniqueness constraints ensured")

    existing_node_counts, existing_rel_counts = get_existing_counts(driver)
    print(
        f"[resume] {len(existing_node_counts)} labels / {len(existing_rel_counts)} "
        f"relationship types already present on the live instance"
    )

    t0 = time.time()
    for label, rows in nodes_by_label.items():
        if existing_node_counts.get(label) == len(rows):
            print(f"[skip] :{label}: {len(rows)} nodes already loaded")
            continue
        write_nodes(driver, label, rows)
        print(f"[write] :{label}: {len(rows)} nodes")
    for rel_type, rows in rels_by_type.items():
        if existing_rel_counts.get(rel_type) == len(rows):
            print(f"[skip] :{rel_type}: {len(rows)} relationships already loaded")
            continue
        write_relationships(driver, rel_type, rows)
        print(f"[write] :{rel_type}: {len(rows)} relationships")
    print(f"[write] done in {time.time() - t0:.1f}s")

    driver.close()

print("\n[done]")

# %%
