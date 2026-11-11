import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
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

BORE_DIR = os.environ["UNMIX_BORE_DIR"]
MODEL = os.environ["UNMIX_MODEL"]
QUERY_DIR = os.environ["UNMIX_QUERY_DIR"]


class QueryEncoder:
    def __init__(self, model, tokenizer, max_length=QUERY_MAX_LENGTH):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, texts, grad=True):
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)
        if grad:
            hidden_states = self.model(**tokens).last_hidden_state.float()
        else:
            with torch.no_grad():
                hidden_states = self.model(**tokens).last_hidden_state.float()
        return F.normalize(
            last_token_pool(hidden_states, tokens["attention_mask"]), p=2, dim=1
        )


def report_metrics(score_matrix, gold_sets, label, document_vectors=None):
    score_matrix = np.asarray(score_matrix)
    hit_rates = []
    reciprocal_ranks = []
    complete_rates = []
    coverages = []
    selected_redundancies = []
    gold_count_groups = {}
    for query_index, gold_indices in enumerate(gold_sets):
        if not len(gold_indices):
            continue
        query_scores = score_matrix[query_index]
        top_indices = np.argpartition(-query_scores, 300)[:300]
        top_indices = top_indices[np.argsort(-query_scores[top_indices])]
        rank_by_tool = {
            int(tool_index): rank_index + 1
            for rank_index, tool_index in enumerate(top_indices)
        }
        first_gold_rank = min(
            (rank_by_tool.get(int(tool_index), 10**9) for tool_index in gold_indices)
        )
        top10_indices = top_indices[:10]
        coverage = len(
            set((int(tool_index) for tool_index in top10_indices))
            & set((int(tool_index) for tool_index in gold_indices))
        ) / float(len(gold_indices))
        query_metrics = (
            1.0 if first_gold_rank <= 10 else 0.0,
            1.0 / first_gold_rank if first_gold_rank <= 10 else 0.0,
            1.0 if coverage == 1.0 else 0.0,
            coverage,
        )
        hit_rates.append(query_metrics[0])
        reciprocal_ranks.append(query_metrics[1])
        complete_rates.append(query_metrics[2])
        coverages.append(query_metrics[3])
        gold_count_groups.setdefault(min(len(gold_indices), 4), []).append(
            query_metrics
        )
        if document_vectors is not None and len(top10_indices) > 1:
            selected_vectors = document_vectors[top10_indices]
            pairwise_similarity = selected_vectors @ selected_vectors.T
            upper_pairs = np.triu_indices(len(top10_indices), 1)
            selected_redundancies.append(float(pairwise_similarity[upper_pairs].mean()))
    redundancy_log = (
        f"  selred {np.mean(selected_redundancies):.4f}"
        if selected_redundancies
        else ""
    )
    print(
        f"{label:24s} R@10 {np.mean(hit_rates):.4f}  MRR@10 {np.mean(reciprocal_ranks):.4f}  C@10 {np.mean(complete_rates):.4f}  cov@10 {np.mean(coverages):.4f}{redundancy_log}",
        flush=True,
    )
    for rank_index in sorted(gold_count_groups):
        group_metrics = np.array(gold_count_groups[rank_index])
        print(
            f"    |gold|={rank_index}  n={len(gold_count_groups[rank_index]):5d}  R@10 {group_metrics[:, 0].mean():.4f}  C@10 {group_metrics[:, 2].mean():.4f}  cov@10 {group_metrics[:, 3].mean():.4f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=300)
    parser.add_argument("--tag", type=str, default="bore")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inv_k", type=int, default=10)
    # Full BORE reward configuration from the paper.
    parser.set_defaults(
        w_ndcg=0.3,
        w_red=0.2,
        beta_kl=0.02,
        w_inv=0.1,
        inv_views="params,nl",
        w_keep=0.5,
        bias_gamma=1.0,
    )
    parser.add_argument(
        "--margin_hi", type=float, default=3.0, help="低 margin 组过采样倍率"
    )
    parser.add_argument("--margin_thr", type=float, default=0.02)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, padding_side="right"
    )
    base_model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    bore_model = get_peft_model(
        base_model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="FEATURE_EXTRACTION",
        ),
    ).to(device)
    bore_model.train()
    encode_queries = QueryEncoder(bore_model, tokenizer)
    print(
        f"[grpo] trainable {sum((parameter.numel() for parameter in bore_model.parameters() if parameter.requires_grad)) / 1000000.0:.2f}M  tag={args.tag} bias_gamma={args.bias_gamma}",
        flush=True,
    )
    source_document_vectors = np.load(f"{BORE_DIR}/train_doc_raw.npy").astype(
        np.float32
    )
    source_document_embeddings = torch.from_numpy(source_document_vectors).to(device)
    frozen_query_vectors = np.load(f"{BORE_DIR}/train_q.npy").astype(np.float32)
    # Fixed source-query mean; keep the original reward coordinates.
    q_bar = frozen_query_vectors.mean(0)
    document_bias = torch.from_numpy(source_document_vectors @ q_bar).to(device)
    if args.bias_gamma > 0:
        print(
            f"[BORE bias] ON gamma={args.bias_gamma}  ||qbar||={float(np.linalg.norm(q_bar)):.4f}  c(d) mean {float(document_bias.mean()):.4f} sd {float(document_bias.std()):.4f}",
            flush=True,
        )
    else:
        print("[BORE bias] DISABLED", flush=True)
    source_tool_names = json.load(open(f"{BORE_DIR}/train_doc_names.json"))["names"]
    source_tool_columns = {
        tool_name: query_index
        for query_index, tool_name in enumerate(source_tool_names)
    }
    field_view_names = [
        view_name for view_name in args.inv_views.split(",") if view_name
    ]
    field_view_embeddings = {}
    for view_name in field_view_names:
        field_cache_path = f"{BORE_DIR}/train_doc_{view_name}.npy"
        if os.path.exists(field_cache_path):
            field_view_embeddings[view_name] = torch.from_numpy(
                np.load(field_cache_path).astype(np.float32)
            ).to(device)
        else:
            raise FileNotFoundError(f"Required field-view cache is missing: {field_cache_path}")
    if args.w_inv > 0:
        print(
            f"[inv] ON views={list(field_view_embeddings)} w_inv={args.w_inv} k={args.inv_k}",
            flush=True,
        )
    else:
        print("[inv] DISABLED", flush=True)
    training_queries = json.load(open(f"{BORE_DIR}/train_queries.json"))
    validation_queries = json.load(open(f"{BORE_DIR}/val_queries.json"))
    training_golds = [
        np.array(
            [
                source_tool_columns[tool_id]
                for tool_id in query_record["golds"]
                if tool_id in source_tool_columns
            ],
            dtype=np.int64,
        )
        for query_record in training_queries
    ]
    validation_golds = [
        np.array(
            [
                source_tool_columns[tool_id]
                for tool_id in query_record["golds"]
                if tool_id in source_tool_columns
            ],
            dtype=np.int64,
        )
        for query_record in validation_queries
    ]
    valid_query_indices = [
        query_index
        for query_index, gold_indices in enumerate(training_golds)
        if len(gold_indices) > 0
    ]
    if len(valid_query_indices) < len(training_golds):
        print(
            f"[data] drop {len(training_golds) - len(valid_query_indices)} train queries with 0 in-pool gold",
            flush=True,
        )
        training_queries = [
            training_queries[query_index] for query_index in valid_query_indices
        ]
        training_golds = [
            training_golds[query_index] for query_index in valid_query_indices
        ]
    num_source_tools = source_document_vectors.shape[0]
    distinctiveness_width = max((len(gold_indices) for gold_indices in training_golds))
    gold_distinctiveness = np.ones(
        (len(training_queries), distinctiveness_width), dtype=np.float32
    )
    for query_index, gold_indices in enumerate(training_golds):
        gold_count = len(gold_indices)
        if gold_count < 2:
            continue
        gold_vectors = source_document_vectors[gold_indices]
        gold_similarity = gold_vectors @ gold_vectors.T
        off_diagonal_similarity_sum = gold_similarity.sum(1) - 1.0
        gold_distinctiveness[query_index, :gold_count] = (
            1.0 - off_diagonal_similarity_sum / (gold_count - 1.0)
        ).astype(np.float32)
    gold_distinctiveness = np.clip(gold_distinctiveness, 0.05, 1.0)
    max_gold_count = max((len(gold_indices) for gold_indices in training_golds))
    padded_gold_indices = np.zeros(
        (len(training_queries), max_gold_count), dtype=np.int64
    )
    gold_presence_mask = np.zeros(
        (len(training_queries), max_gold_count), dtype=np.float32
    )
    for query_index, gold_indices in enumerate(training_golds):
        padded_gold_indices[query_index, : len(gold_indices)] = gold_indices
        gold_presence_mask[query_index, : len(gold_indices)] = 1.0
    print(
        f"[BORE coverage] full-gold: max|gold|={max_gold_count} (was capped at 4 by NG)",
        flush=True,
    )
    frozen_margins = np.zeros(len(training_queries), dtype=np.float32)
    with torch.no_grad():
        reference_scores = (
            torch.from_numpy(frozen_query_vectors).to(device)
            @ source_document_embeddings.T
        )
        if args.bias_gamma > 0:
            reference_scores = (
                reference_scores - args.bias_gamma * document_bias[None, :]
            )
        for query_index, gold_indices in enumerate(training_golds):
            if not len(gold_indices):
                continue
            query_scores = reference_scores[query_index]
            gold_tool_mask = torch.zeros(
                num_source_tools, dtype=torch.bool, device=device
            )
            gold_tool_mask[torch.from_numpy(gold_indices).to(device)] = True
            nongold_scores = query_scores.clone()
            nongold_scores[gold_tool_mask] = -2.0
            frozen_margins[query_index] = float(
                query_scores[torch.from_numpy(gold_indices).to(device)].min()
                - nongold_scores.max()
            )
        del reference_scores
    low_margin_mask = frozen_margins < args.margin_thr
    print(
        f"[BORE margin] frozen margin: p25 {np.percentile(frozen_margins, 25):.4f} p50 {np.median(frozen_margins):.4f} low(<{args.margin_thr}) n={int(low_margin_mask.sum())} ({100 * low_margin_mask.mean():.1f}%)",
        flush=True,
    )
    gold_duplicate_fraction = np.zeros(len(training_queries), dtype=np.float32)
    for query_index, gold_indices in enumerate(training_golds):
        if len(gold_indices) < 2:
            continue
        gold_vectors = source_document_vectors[gold_indices]
        gold_similarity = gold_vectors @ gold_vectors.T
        upper_pairs = np.triu_indices(len(gold_indices), 1)
        gold_duplicate_fraction[query_index] = float(
            (gold_similarity[upper_pairs] > 0.9).mean()
        )
    frozen_gold_hits = np.zeros_like(gold_presence_mask)
    if args.w_keep > 0:
        with torch.no_grad():
            frozen_training_queries = torch.from_numpy(frozen_query_vectors)
            for chunk_start in range(0, len(training_golds), 1024):
                chunk_end = min(chunk_start + 1024, len(training_golds))
                frozen_top10 = (
                    torch.topk(
                        frozen_training_queries[chunk_start:chunk_end].to(device)
                        @ source_document_embeddings.T,
                        10,
                        dim=1,
                    )
                    .indices.cpu()
                    .numpy()
                )
                for chunk_query_index in range(chunk_end - chunk_start):
                    query_gold_indices = training_golds[chunk_start + chunk_query_index]
                    if not len(query_gold_indices):
                        continue
                    frozen_top_tools = set(
                        (int(tool_id) for tool_id in frozen_top10[chunk_query_index])
                    )
                    for gold_position, gold_tool_id in enumerate(query_gold_indices):
                        if int(gold_tool_id) in frozen_top_tools:
                            frozen_gold_hits[
                                chunk_start + chunk_query_index, gold_position
                            ] = 1.0
            del frozen_training_queries

    else:
        print("[KEEP] DISABLED", flush=True)
    gold_count_buckets = np.array(
        [min(len(gold_indices), 4) for gold_indices in training_golds]
    )
    target_document_vectors = np.load(f"{BORE_DIR}/tgt_doc_raw.npy").astype(np.float32)
    target_tool_ids = json.load(open(f"{BORE_DIR}/tgt_doc_ids.json"))["tids"]
    target_tool_columns = {
        list_position: query_index
        for query_index, list_position in enumerate(target_tool_ids)
    }
    target_queries = json.load(open(f"{BORE_DIR}/tgt_eval.json"))
    target_document_embeddings = torch.from_numpy(target_document_vectors).to(device)
    target_golds = [
        np.array(
            [
                target_tool_columns[tool_id]
                for tool_id in target_record["golds"]
                if tool_id in target_tool_columns
            ],
            dtype=np.int64,
        )
        for target_record in target_queries
    ]
    validation_texts = [
        format_query(record["text"], record.get("instruction"))
        for record in validation_queries
    ]
    query_directory = QUERY_DIR
    query_text_by_id = {}
    for task_name in sorted(os.listdir(query_directory)):
        task_directory = os.path.join(query_directory, task_name)
        if not os.path.isdir(task_directory):
            continue
        try:
            task_queries = load_dataset(
                "parquet", data_files=f"{task_directory}/queries-*.parquet"
            )["train"]
        except Exception:
            continue
        for query_record in task_queries:
            query_text_by_id[str(query_record["id"])] = format_query(
                query_record.get("query"), query_record.get("instruction")
            )
    target_texts = [
        query_text_by_id.get(target_record["qid"], format_query(""))
        for target_record in target_queries
    ]
    category_query_indices = {}
    for query_index, target_record in enumerate(target_queries):
        category_query_indices.setdefault(target_record["category"], []).append(
            query_index
        )
    print(
        f"web train={len(training_queries)} val={len(validation_queries)} tools={num_source_tools} | target eval={len(target_queries)} tools={target_document_vectors.shape[0]} cats={ {category_name: len(group_indices) for category_name, group_indices in category_query_indices.items()} }",
        flush=True,
    )

    def evaluate(label):
        bore_model.eval()
        with torch.no_grad():

            def encode_batches(text_batch, batch_size=64):
                encoded_batches = []
                for batch_start in range(0, len(text_batch), batch_size):
                    encoded_batches.append(
                        encode_queries(
                            text_batch[batch_start : batch_start + batch_size],
                            grad=False,
                        )
                    )
                return torch.cat(encoded_batches, 0)

            validation_scores = (
                (encode_batches(validation_texts) @ source_document_embeddings.T)
                .float()
                .cpu()
                .numpy()
            )
            target_scores = (
                (encode_batches(target_texts) @ target_document_embeddings.T)
                .float()
                .cpu()
                .numpy()
            )
            report_metrics(
                validation_scores,
                validation_golds,
                f"[web] {label}",
                source_document_vectors,
            )
            for category_name, category_indices in sorted(
                category_query_indices.items()
            ):
                report_metrics(
                    target_scores[category_indices],
                    [target_golds[query_index] for query_index in category_indices],
                    f"[tgt:{category_name}] {label}",
                    target_document_vectors,
                )
            report_metrics(
                target_scores,
                target_golds,
                f"[tgt:ALL] {label}",
                target_document_vectors,
            )
        bore_model.train()

    print("=== frozen ===", flush=True)
    evaluate("frozen")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in bore_model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    training_started = time.time()
    for step in range(args.steps + 1):
        # Balance training queries by margin, gold count, and diversity.
        sampling_weights = np.ones(len(training_queries), dtype=np.float64)
        sampling_weights[low_margin_mask] = args.margin_hi
        for bucket_id in (1, 2, 3, 4):
            bucket_mask = gold_count_buckets == bucket_id
            if bucket_mask.sum():
                sampling_weights[bucket_mask] *= (
                    0.25 * len(training_queries) / max(bucket_mask.sum(), 1)
                )
        multi_gold_mask = gold_count_buckets >= 2
        sampling_weights[multi_gold_mask] *= (
            1.0 - 0.5 * gold_duplicate_fraction[multi_gold_mask]
        )
        sampling_weights /= sampling_weights.sum()
        batch_query_indices = np.random.choice(
            len(training_queries), args.bs, replace=False, p=sampling_weights
        )
        batch_texts = [
            format_query(
                training_queries[query_index]["text"],
                training_queries[query_index].get("instruction"),
            )
            for query_index in batch_query_indices
        ]
        batch_gold_indices = torch.from_numpy(
            padded_gold_indices[batch_query_indices]
        ).to(device)
        batch_gold_mask = torch.from_numpy(gold_presence_mask[batch_query_indices]).to(
            device
        )
        safe_gold_indices = batch_gold_indices.clamp(min=0)
        q_bore = encode_queries(batch_texts, grad=True)
        policy_scores = q_bore @ source_document_embeddings.T
        if args.bias_gamma > 0:
            policy_scores = policy_scores - args.bias_gamma * document_bias[None, :]
        policy_scores = policy_scores / args.tau
        # Form the centered-score candidate pool and inject gold tools.
        top_candidate_indices = torch.topk(
            policy_scores.detach(), args.M, dim=1
        ).indices
        candidate_indices = top_candidate_indices.clone()
        retained_gold_mask = batch_gold_mask.clone()
        for gold_position in range(max_gold_count):
            valid_gold_rows = batch_gold_mask[:, gold_position] > 0
            if valid_gold_rows.any():
                candidate_indices[valid_gold_rows, gold_position] = (
                    batch_gold_indices[valid_gold_rows, gold_position]
                )
        # Preserve the original gold injection and candidate ordering.
        candidate_logits = policy_scores.gather(1, candidate_indices)
        batch_size = len(batch_query_indices)
        candidate_count = candidate_indices.shape[1]
        with torch.no_grad():
            # Sample ordered lists without replacement over candidate positions.
            sampling_logits = (
                candidate_logits.detach()[:, None, :]
                .expand(batch_size, args.G, candidate_count)
                .clone()
            )
            sampled_positions = torch.zeros(
                batch_size, args.G, args.L, dtype=torch.long, device=device
            )
            for list_position in range(args.L):
                action_distribution = torch.distributions.Categorical(
                    logits=sampling_logits
                )
                sampled_action = action_distribution.sample()
                sampled_positions[:, :, list_position] = sampled_action
                sampling_logits = sampling_logits.scatter(
                    2,
                    sampled_action[:, :, None],
                    torch.full_like(
                        sampled_action[:, :, None],
                        -1000000000.0,
                        dtype=sampling_logits.dtype,
                    ),
                )
            sampled_tool_indices = torch.gather(
                candidate_indices[:, None, :].expand(
                    batch_size, args.G, candidate_count
                ),
                2,
                sampled_positions,
            )
            gold_matches = (
                sampled_tool_indices[:, :, None, :]
                == safe_gold_indices[:, None, :, None]
            )
            gold_retrieved = gold_matches.any(-1)
            gold_ranks = gold_matches.float().argmax(-1)
            batch_gold_weights = torch.from_numpy(
                gold_distinctiveness[batch_query_indices]
            ).to(device)
            coverage_denominator = (
                retained_gold_mask * batch_gold_weights
            ).sum(1, keepdim=True).clamp(min=1e-06)
            coverage = (
                gold_retrieved.float()
                * retained_gold_mask[:, None, :]
                * batch_gold_weights[:, None, :]
            ).sum(-1) / coverage_denominator
            gold_rank_gains = torch.where(
                gold_retrieved & (gold_ranks < args.L),
                1.0 / torch.log2(gold_ranks.float() + 2.0),
                torch.zeros_like(gold_ranks, dtype=torch.float),
            )
            discounted_coverage = (
                gold_rank_gains
                * retained_gold_mask[:, None, :]
                * batch_gold_weights[:, None, :]
            ).sum(-1) / coverage_denominator
            candidate_embeddings = source_document_embeddings[candidate_indices]
            selected_embeddings = torch.gather(
                candidate_embeddings[:, None, :, :].expand(
                    batch_size, args.G, candidate_count, candidate_embeddings.shape[-1]
                ),
                2,
                sampled_positions[:, :, :, None].expand(
                    batch_size, args.G, args.L, candidate_embeddings.shape[-1]
                ),
            )
            selected_similarity = selected_embeddings @ selected_embeddings.transpose(
                -1, -2
            )
            pair_mask = torch.triu(torch.ones(args.L, args.L, device=device), 1)
            redundancy = (selected_similarity * pair_mask).sum(
                (-1, -2)
            ) / pair_mask.sum().clamp(min=1)
            redundancy_weight = args.w_red * (1.0 - (coverage >= 0.999).float())
            # Combine coverage, ranking, redundancy, and optional rewards.
            rewards = (
                coverage
                + args.w_ndcg * discounted_coverage
                - redundancy_weight * redundancy
            )
            if args.w_keep > 0:
                protected_gold_mask = torch.from_numpy(
                    frozen_gold_hits[batch_query_indices]
                ).to(device)
                keep_reward = (
                    gold_retrieved.float()
                    * retained_gold_mask[:, None, :]
                    * protected_gold_mask[:, None, :]
                ).sum(-1) / protected_gold_mask.sum(1, keepdim=True).clamp(min=1)
                rewards = rewards + args.w_keep * keep_reward
            if args.w_inv > 0 and field_view_embeddings:
                invariance_reward = torch.zeros_like(rewards)
                for (
                    view_name,
                    view_document_embeddings,
                ) in field_view_embeddings.items():
                    with torch.no_grad():
                        field_scores = q_bore.detach() @ view_document_embeddings.T
                    candidate_field_scores = field_scores.gather(1, candidate_indices)
                    field_top_k = min(args.inv_k, candidate_count)
                    field_top_positions = (
                        torch.topk(candidate_field_scores.detach(), field_top_k, dim=1)
                        .indices[:, None, :]
                        .expand(batch_size, args.G, field_top_k)
                    )
                    invariance_reward = invariance_reward + (
                        sampled_positions[:, :, :, None]
                        == field_top_positions[:, :, None, :]
                    ).any(-1).float().mean(-1)
                invariance_reward = invariance_reward / len(field_view_embeddings)
                rewards = rewards + args.w_inv * invariance_reward
            advantages = (rewards - rewards.mean(1, keepdim=True)) / (
                rewards.std(1, keepdim=True) + 1e-06
            )
            advantages = advantages * (retained_gold_mask.sum(1) > 0).float()[:, None]
        # Recompute list log probabilities for the policy gradient.
        policy_logits = candidate_logits[:, None, :].expand(
            batch_size, args.G, candidate_count
        )
        list_log_probability = torch.zeros(batch_size, args.G, device=device)
        for list_position in range(args.L):
            action_distribution = torch.distributions.Categorical(logits=policy_logits)
            list_log_probability = list_log_probability + action_distribution.log_prob(
                sampled_positions[:, :, list_position]
            )
            policy_logits = policy_logits.scatter(
                2,
                sampled_positions[:, :, list_position, None],
                torch.full_like(
                    sampled_positions[:, :, list_position, None],
                    -1000000000.0,
                    dtype=policy_logits.dtype,
                ),
            )
        policy_loss = -(advantages.detach() * list_log_probability).sum(-1).mean()
        with torch.no_grad():
            base_model.eval()
            with bore_model.disable_adapter():
                q_frozen = encode_queries(batch_texts, grad=False)
            base_model.train()
        # Regularize the policy against the frozen encoder.
        reference_policy_scores = q_frozen @ source_document_embeddings.T
        if args.bias_gamma > 0:
            reference_policy_scores = (
                reference_policy_scores - args.bias_gamma * document_bias[None, :]
            )
        reference_logits = (
            (reference_policy_scores / args.tau).gather(1, candidate_indices).detach()
        )
        # Preserve the original categorical KL regularizer.
        policy_log_probabilities = F.log_softmax(candidate_logits, 1)
        reference_log_probabilities = F.log_softmax(reference_logits, 1)
        kl_divergence = (
            (
                policy_log_probabilities.exp()
                * (policy_log_probabilities - reference_log_probabilities)
            )
            .sum(1)
            .mean()
        )
        loss = policy_loss + args.beta_kl * kl_divergence
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in bore_model.parameters()
                if parameter.requires_grad
            ],
            1.0,
        )
        optimizer.step()
        if step % 50 == 0:
            print(
                f"  step {step} loss {loss.item():.5f} cov {coverage.mean().item():.4f} dmean {(q_bore.mean(0) - q_frozen.mean(0)).norm().item():.4f} C@10 {(coverage >= 0.999).float().mean().item():.4f} adv_sd {advantages.std().item():.3f} red {redundancy.mean().item():.3f} ({time.time() - training_started:.0f}s)",
                flush=True,
            )
        if args.eval_every and step > 0 and (step % args.eval_every == 0):
            print(f"=== eval @ {step} ===", flush=True)
            evaluate(f"grpo{step}")
            bore_model.save_pretrained(
                f"{BORE_DIR}/bore_grpo_{args.tag}_s{step}"
            )
            print(f"[ckpt] saved bore_grpo_{args.tag}_s{step}", flush=True)
    print("=== final ===", flush=True)
    evaluate("grpo-final")
    bore_model.save_pretrained(f"{BORE_DIR}/bore_grpo_{args.tag}_final")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
