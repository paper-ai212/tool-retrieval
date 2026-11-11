"""Encode the full tool corpus using the original normalized JSON representation."""

import json
import os
import time
import glob

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def last_token_pool(hidden_states, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_positions = (attention_mask * positions).max(dim=1).values
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device),
        last_positions,
    ]


MODEL_PATH = os.environ["UNMIX_MODEL"]
cache_dir = os.environ["UNMIX_CACHE_DIR"]
tool_dir = os.environ.get("UNMIX_TOOL_DIR") or f"{os.environ['UNMIX_ROOT']}/datasets/tool-rex-tools"
OUTPUT_PATH = os.environ.get("UNMIX_DOCUMENT_EMBEDDINGS", f"{cache_dir}/document_embeddings.npy")
if os.path.exists(OUTPUT_PATH):
    print("already exists, skip")
    raise SystemExit
tool_ids_path = os.environ.get("UNMIX_TOOL_IDS", f"{cache_dir}/tool_ids.json")
if "UNMIX_TOOL_IDS" not in os.environ and not os.path.exists(tool_ids_path):
    tool_ids_path = f"{cache_dir}/tool_ids_0.6b.json"
tool_ids = json.load(open(tool_ids_path))
# Align the original documents with the cached tool order.
documentation_by_id = {}
for ecosystem in ["web", "code", "customized"]:
    for parquet_path in sorted(
        glob.glob(f"{tool_dir}/{ecosystem}/tools-*.parquet")
    ):
        tool_frame = pd.read_parquet(parquet_path)
        for tool_id, documentation in zip(
            tool_frame["id"], tool_frame["documentation"]
        ):
            documentation_by_id[str(tool_id)] = documentation
print("docs loaded", len(documentation_by_id))

document_texts = []
for tool_id in tool_ids:
    original_documentation = documentation_by_id[tool_id]
    try:
        document = json.loads(original_documentation)
        if not isinstance(document, dict):
            document = {"description": str(original_documentation)}
    except Exception:
        document = {"description": str(original_documentation)}
    document_texts.append(json.dumps(document, ensure_ascii=False))
print("texts", len(document_texts), "| ex:", document_texts[0][:80])
GPU_INDEX = int(os.environ.get("GPU", "0"))
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, padding_side="right"
)
model = (
    AutoModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    .to(f"cuda:{GPU_INDEX}")
    .eval()
)
print("embedding model loaded", flush=True)


# Encode normalized document embeddings in batches.
embedding_batches = []
start_time = time.time()
batch_size = 64
with torch.no_grad():
    for batch_start in range(0, len(document_texts), batch_size):
        batch_texts = document_texts[batch_start : batch_start + batch_size]
        tokenized_batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)
        hidden_states = model(**tokenized_batch).last_hidden_state.float()
        batch_embeddings = torch.nn.functional.normalize(
            last_token_pool(hidden_states, tokenized_batch["attention_mask"]),
            p=2,
            dim=1,
        )
        embedding_batches.append(batch_embeddings.cpu().numpy().astype(np.float16))
        if batch_start // batch_size % 50 == 0:
            encoded_count = batch_start + len(batch_texts)
            elapsed_seconds = time.time() - start_time
            print(
                "  %d/%d %.0fs %.1f/s"
                % (
                    encoded_count,
                    len(document_texts),
                    elapsed_seconds,
                    encoded_count / max(elapsed_seconds, 1e-09),
                ),
                flush=True,
            )
document_embeddings = np.concatenate(embedding_batches, 0)
np.save(OUTPUT_PATH, document_embeddings)
print(
    "saved",
    document_embeddings.shape,
    document_embeddings.dtype,
    OUTPUT_PATH,
    flush=True,
)
