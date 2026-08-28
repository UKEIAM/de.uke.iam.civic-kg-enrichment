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
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

CLINVAR_FTP = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
OUTPUT_DIR = Path("data/raw/clinvar/")


# %%
def fetch_clinvar_data(output_file: Path) -> None:
    """Streams ClinVar's variant_summary.txt.gz and decompresses it to output_file."""
    print(f"Fetching ClinVar variant summary from {CLINVAR_FTP} ...")
    with requests.get(CLINVAR_FTP, stream=True) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as gz_in, open(
            output_file, "wb"
        ) as f_out:
            shutil.copyfileobj(gz_in, f_out)


# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
output_file = OUTPUT_DIR / f"clinvar_variant_summary_{date_str}.tsv"
fetch_clinvar_data(output_file)
print(f"✅ Saved ClinVar data to {output_file}.")