# GRACE: A Graph-Based Resource for Aligned Cancer Evidence That Enables Guided Clinical Exploration and Mechanistic Insights

Builds a cancer-focused knowledge graph by enriching OptimusKG (the
`optimuskg` package, see `requirements.txt`) with curated evidence from
[CIViC](https://civicdb.org/) and [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/). 
contextualizes curated evidence with drug mechanisms, safety information and disease 
hierarchies, and exposes gene-disease and variant-level curation gap candidates.
Sripts to populate a Neo4j AuraDB instance and a setup a (local) web application
are provided.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline overview

The pipeline fetches raw data, resolves CIViC entities against OptimusKG and
ClinVar, then assembles three progressively different graph artifacts. Each
is a `networkx.DiGraph` pickle under `data/graphs/`, built by its own script
in `src/graph/`, and used for a different purpose downstream:

- **`evidence_graph_<date>.pkl`** (from `build_evidence_graph.py`), the
  "evidence-backed" graph: exactly the 5 node types (gene, disease, drug,
  variant, evidence item) and 11 edge types this project adds on top of
  OptimusKG, with no other native OptimusKG entities folded in. This is the
  base layer every other graph below builds on.

- **`cancer_scoped_graph_<date>.pkl`** (from `build_cancer_scoped_graph.py`), 
  the evidence-backed graph scoped down to cancer-disease relevance: only
  genes/drugs/diseases reachable from the cancer disease seed, plus OptimusKG's
  native edges restricted to that scope. This is the graph the analysis and
  app layers actually use: `src/analysis/graph_statistics.py`,
  `src/analysis/find_figure_candidates.py`, and `src/app/candidate_browser.py`
  (via its "load full graph" toggle) all read the latest
  `cancer_scoped_graph_*.pkl`.

- **`evidence_backed_expanded_graph_<date>.pkl`** (from
  `build_expanded_evidence_graph.py`), the evidence-backed graph expanded
  back out with OptimusKG's full native context: its other 7 node types
  (biological process, phenotype, anatomy, molecular function, cellular
  component, pathway, exposure) and native edges, plus every variant and
  every disease connected to a variant, not just the ones with direct CIViC
  evidence. This is **not** cancer-scoped. It's the richer, exploratory
  graph used for the Neo4j/Bloom load (`src/graph/load_neo4j.py`).

## Running the pipeline

Three orchestration scripts in `scripts/`, run in order:

```bash
bash scripts/fetch_data.sh                    # CIViC, ClinVar, OptimusKG raw data
bash scripts/build_cancer_scoped_graph.sh     # alignment -> cancer_scoped_graph_<date>.pkl
bash scripts/build_expanded_evidence_graph.sh # evidence_backed_expanded_graph_<date>.pkl
```

## Exploring candidates with the Streamlit app

```bash
streamlit run src/app/candidate_browser.py
```

Opens at `http://localhost:8501`. Reads
`data/processed/find_figure_candidates/find_figure_candidates_*.json` (from
`python src/analysis/find_figure_candidates.py` (run this first if that
file doesn't exist yet) and, only if the "load full graph" toggle is used,
the latest `cancer_scoped_graph_*.pkl` as well.

## Loading into Neo4j

Before running `src/graph/load_neo4j.py`, manually setting up a free AuraDB instance is needed.

### Setting up a free AuraDB instance

1. Go to [console.neo4j.io](https://console.neo4j.io) and sign up or log in
2. Click **Create instance** → **Free**.
3. Give the instance a name and pick a cloud region, then **Create**.
4. On the instance's connection screen, copy the generated credentials
5. Copy `.env.example` to `.env` and fill in the three values:
   ```
   NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=<your generated password>
   ```
6. Select "http API" as the connection method and enter your credentials 
7. Wait for the instance's status to show "Running" 

### Populating the AuraDB instance

```bash
python src/graph/load_neo4j.py
```

With `.env` filled in, this writes directly to the configured AuraDB
instance in batches. With no instance configured, it instead prints a
dry-run summary to stdout only, so it's safe to run without credentials set
up yet. This loading might take some time (a few hours).

## Acknowledgments

Interactive graph visualization in the Streamlit app is powered by
**st-link-analysis**:

> AlrasheedA. st-link-analysis: a custom Streamlit component for link
> analysis \[software]. GitHub; 2024 \[cited 2026 Oct 8]. Available from:
> https://github.com/AlrasheedA/st-link-analysis

## Development environment

This repository also includes a GPU-ready devcontainer (Python 3.10,
PyTorch, CUDA/cuBLAS/cuDNN/NCCL, RAPIDS, TensorRT); not required for the
pipeline above, but available for GPU-accelerated experimentation. Opening
the project in VS Code configures everything automatically; to build and
run it manually:

```bash
docker build -t <YOUR PROJECT NAME>:0.1 .
docker run -p <YOUR LOCAL PORT>:22 --rm --gpus all <YOUR PROJECT NAME>:0.1
```

Add any extra Python packages to `requirements.txt`; more complex
environment customization goes into `Dockerfile`.
