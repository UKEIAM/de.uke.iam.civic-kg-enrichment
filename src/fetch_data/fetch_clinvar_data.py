#!/usr/bin/env python3
"""Fetches ClinVar's variant_summary.txt.gz from the NCBI FTP server and
decompresses it to a local file, unmodified (native tab-delimited format).

No pandas round-trip here on purpose: the source file is large (millions of
rows) and nothing about its content needs to change before it lands on disk,
so it's streamed and decompressed directly rather than loaded into a
DataFrame just to be written back out unchanged.

Usage:
    python src/fetch_data/fetch_clinvar_data.py
"""

# %%
import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

CLINVAR_FTP = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
OUTPUT_DIR = Path("data/raw/clinvar/")


# %%
def fetch_clinvar_data(output_file: Path) -> str | None:
    """Streams ClinVar's variant_summary.txt.gz and decompresses it to
    output_file. Returns the server's Last-Modified header for the .gz
    response — ClinVar's own release date for this snapshot, as opposed to
    today's download date the output filename is stamped with — or None if
    the server didn't send one."""
    print(f"Fetching ClinVar variant summary from {CLINVAR_FTP} ...")
    with requests.get(CLINVAR_FTP, stream=True) as response:
        response.raise_for_status()
        last_modified = response.headers.get("Last-Modified")
        with gzip.GzipFile(fileobj=response.raw) as gz_in, open(
            output_file, "wb"
        ) as f_out:
            shutil.copyfileobj(gz_in, f_out)
    return last_modified


# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
output_file = OUTPUT_DIR / f"clinvar_variant_summary_{date_str}.tsv"
last_modified = fetch_clinvar_data(output_file)
print(f"✅ Saved ClinVar data to {output_file}.")

metadata_file = OUTPUT_DIR / f"clinvar_variant_summary_{date_str}_metadata.json"
with open(metadata_file, "w") as f:
    json.dump(
        {
            "source_url": CLINVAR_FTP,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "clinvar_last_modified": last_modified,
        },
        f,
        indent=2,
    )
print(f"✅ Saved fetch metadata (incl. ClinVar's own release date) to {metadata_file}.")