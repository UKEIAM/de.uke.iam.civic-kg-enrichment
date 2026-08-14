#!/usr/bin/env python3
"""Script to fetch the largest connected component of the OptimusKG graph and save it to a pickle file.
It will be used for downstream processing.

Usage:
    python src/fetch_data/save_optimuskg_pickle.py
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

import optimuskg

OUTPUT_DIR = Path("data/raw/optimuskg/")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pickle_path = OUTPUT_DIR / f"optimuskg_lcc_graph_{date_str}.pkl"

    def save_optimuskg_to_pickle():
        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                return pickle.load(f)
        print("Fetching OptimusKG graph and saving to Pickle...")
        G = optimuskg.load_networkx(lcc=True)

        # Save pickle file
        with open(pickle_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        return G

    save_optimuskg_to_pickle()

    print(f"✅ Saved raw OptimusKG graph to Pickle in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
