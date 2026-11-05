#!/usr/bin/env python3
"""Retrieve Allen Mouse Brain Atlas ISH regional expression by API.

This is the acquisition step used before the broad-region validation notebook.
It queries the Allen API for a mouse gene, its coronal experiment, and the
StructureUnionize expression energy for the requested structures.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


API = "http://api.brain-map.org/api/v2/data/query.json"
REGION_TO_STRUCTURE_ID = {
    "Isocortex": 315,
    "STR": 477,
    "CTXsp": 703,
    "HY": 1097,
    "TH": 549,
    "OLF": 698,
    "PAL": 803,
    "HPF": 1089,
}


def query(params: dict, timeout: int = 30) -> list[dict]:
    response = requests.get(API, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload["msg"]


def gene_id(symbol: str) -> int | None:
    rows = query({"criteria": f"model::Gene,rma::criteria,[acronym$eq'{symbol}'][organism_id$eq2]", "num_rows": 5})
    return rows[0]["id"] if rows else None


def coronal_experiment_id(symbol: str) -> int | None:
    criteria = (
        "model::SectionDataSet,rma::criteria,[failed$eq'false'],products[id$eq1],"
        f"genes[acronym$eq'{symbol}'],plane_of_section[name$eq'coronal']"
    )
    rows = query({"criteria": criteria, "num_rows": 5})
    return rows[0]["id"] if rows else None


def structure_energy(dataset_id: int) -> dict[str, float]:
    ids = ",".join(str(v) for v in REGION_TO_STRUCTURE_ID.values())
    criteria = f"model::StructureUnionize,rma::criteria,section_data_set[id$eq{dataset_id}],structure[id$in{ids}]"
    rows = query({"criteria": criteria, "include": "structure", "num_rows": 100})
    return {row["structure"]["acronym"]: row["expression_energy"] for row in rows}


def download(tf_list: list[str], output: Path, delay: float = 0.2) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for symbol in tf_list:
        gid = gene_id(symbol)
        dataset_id = coronal_experiment_id(symbol) if gid is not None else None
        if dataset_id is not None:
            rows[symbol] = structure_energy(dataset_id)
        time.sleep(delay)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "TF"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tf_list = [line.strip() for line in args.tf_list.read_text().splitlines() if line.strip() and not line.startswith("#")]
    table = download(tf_list, args.output)
    print(f"Saved Allen ISH expression matrix for {len(table):,} TFs to {args.output}")


if __name__ == "__main__":
    main()
