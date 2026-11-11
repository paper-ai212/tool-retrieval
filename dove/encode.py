"""Encode DOVE views and prepare query metadata for UNMIX."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


def last_token_pool(hidden_states, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_positions = (attention_mask * positions).max(dim=1).values
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device),
        last_positions,
    ]


CACHE_DIR = os.environ["UNMIX_CACHE_DIR"]
QUERY_DIR = os.environ["UNMIX_QUERY_DIR"]
DOVE_DIR = os.environ["UNMIX_DOVE_DIR"]


def encode(model, tokenizer, texts, batch_size, max_length=8192):
    embeddings = []
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            hidden_states = model(**tokens).last_hidden_state.float()
            batch_embeddings = last_token_pool(hidden_states, tokens["attention_mask"])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu().numpy()
        embeddings.append(batch_embeddings.astype(np.float32))
        if (batch_start // batch_size) % 100 == 0:
            print(
                "  enc %d/%d" % (batch_start + len(batch_texts), len(texts)), flush=True
            )
    return np.concatenate(embeddings, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--bs", type=int, default=8)
    args = parser.parse_args()

    tool_rows = [json.loads(line) for line in open(f"{DOVE_DIR}/full_set.jsonl")]
    rewrites = json.load(open(f"{DOVE_DIR}/rewrites.json"))
    # Keep view embeddings aligned with their tool IDs.
    invalid_ids = [
        tool["id"]
        for tool in tool_rows
        if tool["id"] in rewrites
        and rewrites[tool["id"]] != "__FAIL__"
        and (
            not isinstance(rewrites[tool["id"]], dict)
            or not isinstance(rewrites[tool["id"]].get("user_queries"), list)
            or len(rewrites[tool["id"]]["user_queries"]) != 3
        )
    ]
    if invalid_ids:
        raise ValueError(
            f"Expected exactly 3 queries per rewrite; rerun generate.py for "
            f"{len(invalid_ids)} invalid tools (example: {invalid_ids[0]})"
        )
    rewritten_tools = [
        tool
        for tool in tool_rows
        if tool["id"] in rewrites and rewrites[tool["id"]] != "__FAIL__"
    ]
    print(
        "rows=%d rewritten-ok=%d fail=%d"
        % (len(tool_rows), len(rewritten_tools), len(tool_rows) - len(rewritten_tools)),
        flush=True,
    )
    view_tool_ids = [tool["id"] for tool in rewritten_tools]

    v1_cache_path = f"{DOVE_DIR}/enc_v1.npy"
    v2_cache_path = f"{DOVE_DIR}/enc_v2.npy"
    metadata_path = f"{DOVE_DIR}/enc_meta.json"
    if not os.path.exists(v1_cache_path) or not os.path.exists(v2_cache_path):
        if args.gpu < 0:
            sys.exit("--gpu required for encode")
        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["UNMIX_MODEL"], trust_remote_code=True, padding_side="right"
        )
        model = AutoModel.from_pretrained(
            os.environ["UNMIX_MODEL"], torch_dtype=torch.float16, trust_remote_code=True
        ).to(f"cuda:{args.gpu}")
        model.eval()
        print("model loaded on gpu", args.gpu, flush=True)
        # Encode the rewrite and pseudo-queries into separate caches.
        if not os.path.exists(v1_cache_path):
            v1_texts = [
                json.dumps(rewrites[tool_id]["doc"], ensure_ascii=False)
                for tool_id in view_tool_ids
            ]
            print("encoding V1 (%d)..." % len(v1_texts), flush=True)
            v1_embeddings = encode(model, tokenizer, v1_texts, args.bs)
            np.save(v1_cache_path, v1_embeddings)
            print("V1 saved", flush=True)
        if not os.path.exists(v2_cache_path):
            v2_texts = [
                query
                for tool_id in view_tool_ids
                for query in rewrites[tool_id]["user_queries"]
            ]
            print("encoding V2 (%d)..." % len(v2_texts), flush=True)
            v2_embeddings = encode(model, tokenizer, v2_texts, args.bs)
            np.save(v2_cache_path, v2_embeddings)
            print("V2 saved", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()
        print("GPU released", flush=True)
    # Preserve query labels and view order for retrieval.
    if not os.path.exists(metadata_path):
        query_metadata = []
        tasks = sorted(os.listdir(QUERY_DIR))
        for task in tasks:
            if not os.path.isdir(f"{QUERY_DIR}/{task}"):
                continue
            dataset = load_dataset_parquet(task)
            for query in dataset:
                labels = query["labels"]
                if isinstance(labels, str):
                    labels = json.loads(labels)
                gold_ids = [
                    label["id"]
                    for label in labels
                    if isinstance(label, dict) and label.get("relevance", 0) >= 1
                ]
                query_metadata.append(
                    {
                        "task": task,
                        "qid": str(query["id"]),
                        "category": str(query.get("category") or ""),
                        "golds": gold_ids,
                    }
                )
        json.dump(
            {
                "qmeta": query_metadata,
                "ids_order": view_tool_ids,
                "lens_v2": [
                    len(rewrites[tool_id]["user_queries"])
                    for tool_id in view_tool_ids
                ],
            },
            open(metadata_path, "w"),
            ensure_ascii=False,
        )
        print("query metadata saved", flush=True)


def load_dataset_parquet(task):
    return load_dataset(
        "parquet",
        data_files=f"{QUERY_DIR}/{task}/queries-*.parquet",
        cache_dir=CACHE_DIR,
    )["train"]


if __name__ == "__main__":
    main()
