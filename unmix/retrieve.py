"""Shared UNMIX retrieval using the original scoring and promotion-only fusion."""

import argparse
import json
import math
import os

import numpy as np
import torch


DOVE_BETA = 0.5
FUSION_WEIGHT = 0.8
POOL_DEPTH = 100


def build_scores(
    query_embeddings,
    document_embeddings,
    v1_embeddings,
    v2_embeddings,
    view_columns,
    block_size=256,
):
    query_count = len(query_embeddings)
    original_scores = np.empty(
        (query_count, len(document_embeddings)), dtype=np.float32
    )
    view_gains = np.empty((query_count, len(view_columns)), dtype=np.float32)
    for start in range(0, query_count, block_size):
        end = min(start + block_size, query_count)
        query_batch = query_embeddings[start:end]
        batch_scores = query_batch @ document_embeddings.T
        # Select the best score across the rewrite and three pseudo-queries.
        best_view_scores = np.maximum(
            query_batch @ v1_embeddings.T, query_batch @ v2_embeddings[:, 0, :].T
        )
        np.maximum(
            best_view_scores,
            query_batch @ v2_embeddings[:, 1, :].T,
            out=best_view_scores,
        )
        np.maximum(
            best_view_scores,
            query_batch @ v2_embeddings[:, 2, :].T,
            out=best_view_scores,
        )
        view_gains[start:end] = best_view_scores - batch_scores[:, view_columns]
        original_scores[start:end] = batch_scores
    return original_scores, view_gains


def apply_dove(original_scores, view_gains, view_columns, beta=DOVE_BETA):
    # Views may promote a document but cannot lower its score.
    scores = original_scores.copy()
    scores[:, view_columns] += beta * np.maximum(view_gains, 0.0)
    return scores


def top_indices(scores, count=POOL_DEPTH):
    indices = np.argpartition(-scores, count, axis=1)[:, :count]
    values = np.take_along_axis(scores, indices, 1)
    return np.take_along_axis(indices, np.argsort(-values, axis=1), 1)


def promotion_only_fusion(frozen_scores, bore_scores, weight=FUSION_WEIGHT):
    frozen_top = top_indices(frozen_scores)
    bore_top = top_indices(bore_scores)
    fused_scores = np.full_like(frozen_scores, -1e9)
    candidate_pools = []
    for query_index in range(len(frozen_scores)):
        # Standardize both branches over their shared candidate pool.
        candidates = np.union1d(frozen_top[query_index], bore_top[query_index])
        candidate_pools.append(candidates)
        frozen_values = frozen_scores[query_index, candidates]
        bore_values = bore_scores[query_index, candidates]
        frozen_z = (frozen_values - frozen_values.mean()) / (frozen_values.std() + 1e-6)
        bore_z = (bore_values - bore_values.mean()) / (bore_values.std() + 1e-6)
        fused_scores[query_index, candidates] = frozen_z + weight * np.maximum(
            bore_z - frozen_z, 0.0
        )
    return fused_scores, candidate_pools


def retrieval_metrics(scores, candidate_pools, gold_indices, count=POOL_DEPTH):
    metrics = {name: np.zeros(len(scores)) for name in ("ndcg10", "cov10", "c10")}
    for query_index, candidates in enumerate(candidate_pools):
        if not len(candidates):
            continue
        order = np.argsort(-scores[query_index, candidates])[:count]
        top_ten = candidates[order[:10]]
        gold_set = set(int(index) for index in gold_indices[query_index])
        if not gold_set:
            continue
        hit_count = sum(1 for index in top_ten if int(index) in gold_set)
        metrics["cov10"][query_index] = hit_count / len(gold_set)
        metrics["c10"][query_index] = 1.0 if hit_count == len(gold_set) else 0.0
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, index in enumerate(top_ten, 1)
            if int(index) in gold_set
        )
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), 10) + 1)
        )
        metrics["ndcg10"][query_index] = dcg / ideal_dcg if ideal_dcg else 0.0
    return metrics


def macro_average(metrics, query_tasks, query_categories):
    rows = {
        category: {
            name: 100.0
            * float(
                np.mean(
                    [
                        values[query_tasks == task].mean()
                        for task in sorted(
                            set(query_tasks[query_categories == category])
                        )
                    ]
                )
            )
            for name, values in metrics.items()
        }
        for category in sorted(set(query_categories))
    }
    rows["Avg"] = {
        name: float(np.mean([row[name] for row in rows.values()])) for name in metrics
    }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("0.6b", "4b"), default="0.6b")
    parser.add_argument("--document-format", choices=("torch", "numpy"))
    args = parser.parse_args()

    cache_dir = os.environ["UNMIX_CACHE_DIR"]
    dove_dir = os.environ["UNMIX_DOVE_DIR"]
    output_dir = os.environ["UNMIX_BORE_DIR"]
    document_path = os.environ.get("UNMIX_DOCUMENT_EMBEDDINGS")
    if document_path is None:
        document_path = f"{cache_dir}/document_embeddings.npy"
        if not os.path.exists(document_path) and args.model_size == "0.6b":
            document_path = f"{cache_dir}/tool_emb_0.6b.npy"
    document_format = args.document_format or (
        "torch" if os.path.basename(document_path) == "tool_emb_0.6b.npy" else "numpy"
    )
    if document_format == "torch":
        document_embeddings = (
            torch.load(document_path, map_location="cpu").numpy().astype(np.float32)
        )
    else:
        document_embeddings = np.load(document_path).astype(np.float32)

    tool_ids_path = os.environ.get("UNMIX_TOOL_IDS", f"{cache_dir}/tool_ids.json")
    if "UNMIX_TOOL_IDS" not in os.environ and not os.path.exists(tool_ids_path):
        tool_ids_path = f"{cache_dir}/tool_ids_0.6b.json"
    tool_ids = json.load(open(tool_ids_path))
    tool_columns = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    metadata = json.load(open(f"{dove_dir}/enc_meta.json"))
    queries = metadata["qmeta"]
    viewed_ids = metadata["ids_order"]
    view_columns = np.array(
        [tool_columns[tool_id] for tool_id in viewed_ids], dtype=np.int64
    )
    v1_embeddings = np.load(f"{dove_dir}/enc_v1.npy").astype(np.float32)
    v2_embeddings = (
        np.load(f"{dove_dir}/enc_v2.npy")
        .astype(np.float32)
        .reshape(len(viewed_ids), 3, -1)
    )
    # Map labels and view IDs to the cached document order.
    gold_indices = [
        np.array(
            [
                tool_columns[tool_id]
                for tool_id in query["golds"]
                if tool_id in tool_columns
            ],
            dtype=np.int64,
        )
        for query in queries
    ]
    query_tasks = np.array([query["task"] for query in queries])
    query_categories = np.array([query["category"] for query in queries])

    frozen_original, frozen_gains = build_scores(
        np.load(os.environ.get("UNMIX_FROZEN_QUERIES", f"{output_dir}/frozen_queries.npy")),
        document_embeddings,
        v1_embeddings,
        v2_embeddings,
        view_columns,
    )
    bore_queries = np.load(os.environ.get("UNMIX_BORE_QUERIES", f"{output_dir}/bore_queries.npy"))
    bore_scores = bore_queries @ document_embeddings.T
    frozen_scores = apply_dove(frozen_original, frozen_gains, view_columns)
    fused_scores, candidate_pools = promotion_only_fusion(frozen_scores, bore_scores)
    rows = macro_average(
        retrieval_metrics(fused_scores, candidate_pools, gold_indices),
        query_tasks,
        query_categories,
    )
    for category, values in rows.items():
        print(
            f"{category:12s} N@10 {values['ndcg10']:.2f} "
            f"R@10 {values['cov10']:.2f} C@10 {values['c10']:.2f}",
            flush=True,
        )
    suffix = "06b" if args.model_size == "0.6b" else "4b"
    json.dump(
        {"UNMIX": rows},
        open(f"{output_dir}/unmix_{suffix}.json", "w"),
        indent=1,
        default=float,
    )


if __name__ == "__main__":
    main()
