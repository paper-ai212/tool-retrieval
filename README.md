# UNMIX

This repository contains an implementation of UNMIX for cross-ecosystem tool retrieval. It combines document-side views from DOVE (Dialect-Oriented View Ensemble), query-side adaptation with BORE (Bias-Orthogonal Reward), and promotion-only fusion.

## Project layout

- `dove/anchors.py` selects source-ecosystem style anchors for DOVE.
- `dove/build_corpus.py` collects target-ecosystem tool documentation.
- `dove/generate.py` generates one V1 source-dialect rewrite and three V2 doc2query views per tool.
- `dove/encode.py` encodes the DOVE views and prepares query metadata.
- `bore/cache.py` prepares embeddings and data for BORE training and evaluation.
- `bore/train.py` trains the query encoder's LoRA adapter with BORE.
- `unmix/encode_documents.py` encodes the original tool documents.
- `unmix/encode_queries.py` encodes queries with the frozen model and the adapter.
- `unmix/retrieve.py` scores documents, fuses the branches, and reports retrieval metrics.

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```sh
pip install -r requirements.txt
```

The expected data includes source query-document pairs, tool documentation in Parquet files, and query datasets with relevance labels from [Tool-REX](https://github.com/EIT-NLP/Tool-REX), as well as an ordered JSON list of tool IDs for the full retrieval corpus. The model, data, and caches must be prepared separately.

## Workflow

Run from the repository root after configuring the required paths and preparing the ordered tool-ID list:

```sh
python dove/anchors.py
python dove/build_corpus.py
python dove/generate.py
python dove/encode.py --gpu 0
python bore/cache.py
python bore/train.py --gpu 0
python unmix/encode_documents.py
python unmix/encode_queries.py
python unmix/retrieve.py
```

After training, configure the intended saved adapter checkpoint before encoding queries. Keep the model, adapter, tool order, query metadata, and embedding caches consistent across stages. Existing cache files may be reused or skipped by individual scripts, so regenerate them when their inputs change. Review each script's options and defaults before running; for example, generation can resume from an existing output file and BORE can reuse a prior data split.

The retrieval script reports NDCG@10, recall/coverage@10, and complete coverage@10 by category and as an average. Training also reports target evaluation metrics; using those metrics to select checkpoints or tune settings affects a source-only evaluation protocol.

## Acknowledgments and citation

We thank the authors of [Tool-REX](https://github.com/EIT-NLP/Tool-REX) for releasing the benchmark, datasets, and models used in this work.
