"""Build BORE caches, optionally reusing an existing tool order and data split."""

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
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


def json_text(value):
    return json.dumps(value, ensure_ascii=False)


def render_document(document, view_name):
    tool_profile = document.get("tool_profile") or {}
    tool_profile = tool_profile if isinstance(tool_profile, dict) else {}
    if view_name == "raw":
        return json_text(document)
    if view_name == "nl":
        parts = ["Tool name: " + str(document.get("name", ""))]
        if tool_profile.get("function"):
            parts.append("Function: " + str(tool_profile["function"]))
        if tool_profile.get("tags"):
            parts.append("Tags: " + ", ".join(map(str, tool_profile["tags"])))
        for field_name in ("functionality", "framework", "domain"):
            if document.get(field_name):
                parts.append(f"{field_name.capitalize()}: " + str(document[field_name]))
        if document.get("description"):
            parts.append("Description: " + str(document["description"]))
        if document.get("parameters"):
            parts.append("Parameters: " + json_text(document["parameters"]))
        return " . ".join(parts)
    if view_name == "params":
        return json_text(document.get("parameters", {}))
    raise ValueError(view_name)


def encode(model, tokenizer, texts, label, batch_size=64, max_length=1024):
    embedding_batches = []
    start_time = time.time()
    for batch_start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[batch_start : batch_start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            hidden_states = model(**tokens).last_hidden_state.float()
            batch_embeddings = torch.nn.functional.normalize(
                last_token_pool(hidden_states, tokens["attention_mask"]),
                p=2,
                dim=1,
            )
        embedding_batches.append(batch_embeddings.cpu().numpy().astype(np.float16))
        if (batch_start // batch_size) % 40 == 0:
            encoded_count = batch_start + len(tokens["attention_mask"])
            elapsed = time.time() - start_time
            print(
                f"  [{label}] {encoded_count}/{len(texts)} {elapsed:.0f}s "
                f"{encoded_count / max(elapsed, 1e-9):.0f}/s",
                flush=True,
            )
    print(
        f"  [{label}] done {len(texts)} in {time.time() - start_time:.0f}s", flush=True
    )
    return np.concatenate(embedding_batches, 0)


def build_source_split(gold_names_by_query):
    query_gold_pairs = list(gold_names_by_query.items())
    split_rng = np.random.default_rng(42)
    permutation = split_rng.permutation(len(query_gold_pairs))
    query_gold_pairs = [query_gold_pairs[index] for index in permutation]
    training_count = min(10000, int(len(query_gold_pairs) * 0.85))
    training_queries = [
        {
            "qid": f"w{index}",
            "text": query[1],
            "instruction": query[0],
            "golds": sorted(golds),
        }
        for index, (query, golds) in enumerate(query_gold_pairs[:training_count])
    ]
    validation_queries = [
        {
            "qid": f"v{index}",
            "text": query[1],
            "instruction": query[0],
            "golds": sorted(golds),
        }
        for index, (query, golds) in enumerate(query_gold_pairs[training_count:])
    ]
    return training_queries, validation_queries


def build_target_queries(query_dir, dove_dir, target_tool_ids):
    target_tool_set = set(target_tool_ids)
    metadata_path = f"{dove_dir}/enc_meta.json"
    if os.path.exists(metadata_path):
        query_order = [
            record["qid"] for record in json.load(open(metadata_path))["qmeta"]
        ]
    else:
        metadata_path = f"{dove_dir}/instruction_queries_meta.json"
        if not os.path.exists(metadata_path):
            metadata_path = f"{dove_dir}/enc_q_plain_meta.json"
        query_order = json.load(open(metadata_path))["order"]
    query_positions = {query_id: index for index, query_id in enumerate(query_order)}
    evaluation_queries = []
    for task_name in sorted(os.listdir(query_dir)):
        task_dir = os.path.join(query_dir, task_name)
        if not os.path.isdir(task_dir):
            continue
        try:
            dataset = load_dataset(
                "parquet", data_files=f"{task_dir}/queries-*.parquet"
            )["train"]
        except Exception as error:
            print("skip", task_name, error)
            continue
        for record in dataset:
            if str(record.get("category")) not in ("code", "customized"):
                continue
            labels = record["labels"]
            labels = json.loads(labels) if isinstance(labels, str) else labels
            gold_tool_ids = [str(label["id"]) for label in labels]
            gold_tool_ids = [
                tool_id for tool_id in gold_tool_ids if tool_id in target_tool_set
            ]
            if not gold_tool_ids:
                continue
            query_id = str(record["id"])
            if query_id not in query_positions:
                continue
            evaluation_queries.append(
                {
                    "qid": query_id,
                    "task": task_name,
                    "category": str(record["category"]),
                    "pindex": query_positions[query_id],
                    "golds": gold_tool_ids,
                }
            )
    return evaluation_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=os.environ.get("UNMIX_BORE_SOURCE_DIR"),
        help="Reuse tool order, source split and target metadata from an existing BORE cache.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"))
    parser.add_argument("--batch-size", type=int, help="Document encoding batch size.")
    args = parser.parse_args()
    reuse_split = bool(args.source_dir)
    dtype_name = args.dtype or ("bfloat16" if reuse_split else "float16")
    batch_size = (
        args.batch_size if args.batch_size is not None else (32 if reuse_split else 64)
    )
    source_pairs_path = os.environ.get("UNMIX_SOURCE_PAIRS") or (
        f"{os.environ['UNMIX_ROOT']}/datasets/tool-rex-retriever-50k/Tool-DE_train_retriever_50k.json"
    )
    tool_dir = os.environ.get("UNMIX_TOOL_DIR") or (
        f"{os.environ['UNMIX_ROOT']}/datasets/tool-rex-tools"
    )
    output_dir = os.environ["UNMIX_BORE_DIR"]
    model_path = os.environ["UNMIX_MODEL"]
    os.makedirs(output_dir, exist_ok=True)
    gpu_index = int(os.environ.get("GPU", "0"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype_name),
            trust_remote_code=True,
        )
        .to(f"cuda:{gpu_index}")
        .eval()
    )
    print(
        f"model loaded on cuda:{gpu_index}; dtype={dtype_name}; reuse_split={reuse_split}",
        flush=True,
    )

    training_pairs = json.load(open(source_pairs_path))
    documents_by_name = {}
    gold_names_by_query = {}
    for pair in training_pairs:
        try:
            document = json.loads(pair["response"])
        except Exception:
            continue
        if document.get("name"):
            documents_by_name.setdefault(document["name"], document)
        if not reuse_split:
            query_key = (str(pair.get("instruction") or ""), str(pair["query"]))
            positive_names = gold_names_by_query.setdefault(query_key, set())
            if document.get("name"):
                positive_names.add(document["name"])

    # Reuse the cached split when requested; otherwise build one from source pairs.
    if reuse_split:
        tool_names = json.load(open(f"{args.source_dir}/train_doc_names.json"))["names"]
        training_queries = json.load(open(f"{args.source_dir}/train_queries.json"))
        validation_queries = json.load(open(f"{args.source_dir}/val_queries.json"))
        source_views = ("raw", "params", "nl")
    else:
        tool_names = sorted(documents_by_name)
        training_queries, validation_queries = build_source_split(gold_names_by_query)
        source_views = ("raw", "nl", "params")
    json.dump({"names": tool_names}, open(f"{output_dir}/train_doc_names.json", "w"))
    for view_name in source_views:
        view_embeddings = encode(
            model,
            tokenizer,
            [
                render_document(documents_by_name[name], view_name)
                for name in tool_names
            ],
            f"source:{view_name}",
            batch_size=batch_size,
        )
        np.save(f"{output_dir}/train_doc_{view_name}.npy", view_embeddings)
    json.dump(training_queries, open(f"{output_dir}/train_queries.json", "w"))
    json.dump(validation_queries, open(f"{output_dir}/val_queries.json", "w"))
    query_embeddings = encode(
        model,
        tokenizer,
        [
            format_query(record["text"], record.get("instruction"))
            for record in training_queries
        ],
        "source:train",
        batch_size=32,
        max_length=QUERY_MAX_LENGTH,
    )
    np.save(f"{output_dir}/train_q.npy", query_embeddings)

    # Cache target documents and evaluation metadata in a consistent order.
    target_tool_ids, target_documents = [], []
    for ecosystem in ("code", "customized"):
        frame = pd.read_parquet(
            f"{tool_dir}/{ecosystem}/tools-00000-of-00001.parquet"
        )
        for tool_id, documentation in zip(frame["id"], frame["documentation"]):
            try:
                document = json.loads(documentation)
                if reuse_split and not isinstance(document, dict):
                    document = {"description": str(documentation)}
            except Exception:
                document = (
                    {"description": str(documentation)}
                    if reuse_split
                    else {"name": str(tool_id)}
                )
            target_tool_ids.append(str(tool_id))
            target_documents.append(document)
    if reuse_split:
        assert (
            target_tool_ids
            == json.load(open(f"{args.source_dir}/tgt_doc_ids.json"))["tids"]
        ), "target id order mismatch"
        evaluation_queries = json.load(open(f"{args.source_dir}/tgt_eval.json"))
    else:
        evaluation_queries = build_target_queries(
            os.environ["UNMIX_QUERY_DIR"],
            os.environ["UNMIX_DOVE_DIR"],
            target_tool_ids,
        )
    json.dump({"tids": target_tool_ids}, open(f"{output_dir}/tgt_doc_ids.json", "w"))
    np.save(
        f"{output_dir}/tgt_doc_raw.npy",
        encode(
            model,
            tokenizer,
            [render_document(document, "raw") for document in target_documents],
            "target:raw",
            batch_size=batch_size,
        ),
    )
    json.dump(evaluation_queries, open(f"{output_dir}/tgt_eval.json", "w"))
    print(
        f"source train={len(training_queries)} val={len(validation_queries)} tools={len(tool_names)}; "
        f"target queries={len(evaluation_queries)} tools={len(target_tool_ids)}; "
        f"categories={dict(Counter(record['category'] for record in evaluation_queries))}",
        flush=True,
    )
    print("ALL DONE ->", output_dir, flush=True)


if __name__ == "__main__":
    main()
