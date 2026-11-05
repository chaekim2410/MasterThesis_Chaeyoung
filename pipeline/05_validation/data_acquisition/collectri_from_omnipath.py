#!/usr/bin/env python3
"""Download mouse CollecTRI TF-target interactions from the OmniPath API.

The validation notebook uses a cleaned two-column CSV. This script documents
the public request that can be used to recreate that input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests


API = "https://omnipathdb.org/interactions"


def download(output: Path, timeout: int = 120) -> pd.DataFrame:
    params = {
        "datasets": "collectri",
        "organisms": "10090",  # Mus musculus
        "genesymbols": "yes",
    }
    response = requests.get(API, params=params, timeout=timeout)
    response.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    return pd.read_csv(output, sep="\t")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = download(args.output)
    print(f"Saved {len(table):,} OmniPath rows to {args.output}")
    print("Inspect the returned columns before making a cleaned TF/target table.")


if __name__ == "__main__":
    main()
