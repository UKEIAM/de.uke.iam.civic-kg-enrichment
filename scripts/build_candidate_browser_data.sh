#!/usr/bin/env bash
# Builds data/processed/find_figure_candidates/find_figure_candidates_<date>.json
# — the one file src/app/candidate_browser.py needs to start. The app's
# optional "load full graph" toggle additionally reads the latest
# data/graphs/cancer_scoped_graph_*.pkl directly, which this script also
# requires as find_figure_candidates.py's own input, so nothing else needs
# building beyond it.
#
# Requires cancer_scoped_graph_*.pkl to already exist in data/graphs/ — run
# scripts/fetch_data.sh and scripts/build_cancer_scoped_graph.sh first if
# it doesn't.
#
# Usage:
#   ./scripts/build_candidate_browser_data.sh
#   streamlit run src/app/candidate_browser.py

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! compgen -G "data/graphs/cancer_scoped_graph_*.pkl" > /dev/null; then
    echo "error: no data/graphs/cancer_scoped_graph_*.pkl found." >&2
    echo "Run scripts/fetch_data.sh and scripts/build_cancer_scoped_graph.sh first." >&2
    exit 1
fi

echo "=== Finding figure candidates (input for the Streamlit candidate browser) ==="
python3 src/analysis/find_figure_candidates.py

echo
echo "=== build_candidate_browser_data.sh done ==="
echo "Start the app with: streamlit run src/app/candidate_browser.py"
