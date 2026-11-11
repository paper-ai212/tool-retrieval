"""Collect target-ecosystem tool documents for DOVE view generation. No labels used."""

import glob
import json
import os

from datasets import load_dataset

CACHE_DIR = os.environ["UNMIX_CACHE_DIR"]
TOOLS_DIR = os.environ["UNMIX_TOOL_DIR"]
DOVE_DIR = os.environ["UNMIX_DOVE_DIR"]
DOMAINS = ["code", "customized"]


def main():
    # Collect target documents without query labels.
    rows = []
    for domain in DOMAINS:
        parquet_paths = sorted(glob.glob(f"{TOOLS_DIR}/{domain}/tools-*.parquet"))
        dataset = load_dataset(
            "parquet", data_files=parquet_paths, cache_dir=CACHE_DIR
        )["train"]
        for tool in dataset:
            rows.append(
                {"id": tool["id"], "domain": domain, "doc": tool["documentation"]}
            )
        print(domain, len(dataset), flush=True)
    with open(f"{DOVE_DIR}/full_set.jsonl", "w") as output_file:
        for tool in rows:
            output_file.write(json.dumps(tool, ensure_ascii=False) + "\n")
    print("total rows:", len(rows))
    json.dump(
        {"n": len(rows), "domains": DOMAINS},
        open(f"{DOVE_DIR}/full_meta.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
