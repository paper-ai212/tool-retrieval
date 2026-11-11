"""Encode frozen and BORE queries using the same instruction template."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

QUERY_MAX_LENGTH = 512


def format_query(query, instruction=""):
    return f"Instruct: {instruction or ''}\nQuery:{query or ''}".strip()


def last_token_pool(hidden_states, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_positions = (attention_mask * positions).max(dim=1).values
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device),
        last_positions,
    ]


ENCODING_PROFILES = {
    "0.6b": {
        "batch_size": 32,
    },
    "4b": {
        "batch_size": 64,
    },
}


def load_query_texts(
    query_metadata, query_dir, cache_dir, skip_failed_tasks=False
):
    query_lookup = {}
    for task_name in sorted(os.listdir(query_dir)):
        task_dir = os.path.join(query_dir, task_name)
        if not os.path.isdir(task_dir):
            continue
        try:
            if skip_failed_tasks:
                dataset = load_dataset(
                    "parquet",
                    data_files=f"{task_dir}/queries-*.parquet",
                    cache_dir=cache_dir,
                )["train"]
            else:
                dataset = load_dataset(
                    "parquet", data_files=f"{task_dir}/queries-*.parquet"
                )["train"]
        except Exception as error:
            if not skip_failed_tasks:
                raise
            print("skip", task_name, error, flush=True)
            continue
        for record in dataset:
            query_lookup[str(record["id"])] = (
                str(record.get("instruction") or ""),
                str(record.get("query") or ""),
            )
    texts = []
    for record in query_metadata:
        instruction, query = query_lookup[record["qid"]]
        texts.append(format_query(query, instruction))
    return texts


def encode_queries(texts, model_path, device, adapter_path, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    if adapter_path and adapter_path != "-":
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    embeddings = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=QUERY_MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            hidden_states = model(**tokens).last_hidden_state.float()
        embeddings.append(
            F.normalize(
                last_token_pool(hidden_states, tokens["attention_mask"]), p=2, dim=1
            )
            .cpu()
            .numpy()
        )
    del model
    torch.cuda.empty_cache()
    return np.concatenate(embeddings, 0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-size", choices=tuple(ENCODING_PROFILES), default="0.6b"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--frozen-cache")
    args = parser.parse_args()
    profile = ENCODING_PROFILES[args.model_size]
    batch_size = (
        args.batch_size if args.batch_size is not None else profile["batch_size"]
    )
    dove_dir = os.environ["UNMIX_DOVE_DIR"]
    output_dir = os.environ["UNMIX_BORE_DIR"]
    model_path = os.environ["UNMIX_MODEL"]
    device = os.environ["UNMIX_DEVICE"]
    cache_dir = os.environ.get("UNMIX_CACHE_DIR")
    metadata = json.load(open(f"{dove_dir}/enc_meta.json"))
    texts = load_query_texts(
        metadata["qmeta"],
        os.environ["UNMIX_QUERY_DIR"],
        cache_dir,
        skip_failed_tasks=args.model_size == "4b",
    )
    frozen_cache = args.frozen_cache or os.environ.get(
        "UNMIX_FROZEN_QUERIES", f"{output_dir}/frozen_queries.npy"
    )
    # Encode both branches with the same query template.
    branches = [("frozen", None, frozen_cache)]
    branches.append(
        (
            "bore",
            os.environ["UNMIX_ADAPTER"],
            os.environ.get("UNMIX_BORE_QUERIES", f"{output_dir}/bore_queries.npy"),
        )
    )

    print(
        f"Encoding profile={args.model_size}, template=instruct, max_length={QUERY_MAX_LENGTH}, "
        f"batch_size={batch_size}, padding=right",
        flush=True,
    )
    for branch_name, adapter_path, cache_path in branches:
        embeddings = encode_queries(texts, model_path, device, adapter_path, batch_size)
        np.save(cache_path, embeddings)
        print("saved", branch_name, embeddings.shape, flush=True)


if __name__ == "__main__":
    main()
