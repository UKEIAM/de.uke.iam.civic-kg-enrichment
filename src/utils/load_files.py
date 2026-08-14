#!/usr/bin/env python3
"""Utility function to find the latest file in a directory."""

from pathlib import Path


def find_latest_file(dir: str, file_name_pattern: str) -> Path:
    """Find the most recent file in a specific directory.
    The ISO date format sorts chronologically as text, so the last glob match is
    the newest run -- no need to hardcode the date."""
    candidates = sorted(Path(dir).glob(file_name_pattern))
    if not candidates:
        raise FileNotFoundError(f"No {file_name_pattern} file found in {dir}")
    return candidates[-1]
