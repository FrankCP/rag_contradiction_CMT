# Replicating the experiments

Contradiction detection in retrieved passages on synthetic data, VitaminC and WikiContradict.

## 1. Setup

Requirements: Python 3.10 or newer, Ollama installed and running, and a GPU for the NLI model.

1. Create and activate a virtual environment in this folder.
2. Install the packages: `pip install -r requirements.txt`.
3. To rebuild the synthetic data only: install `requirements-synthetic.txt` and download the spaCy model `en_core_web_md`.
4. Download the Ollama models with `ollama pull`: `bge-m3` and `qwen2.5:7b`.

The versions in `requirements.txt` are minimums. If `pip` installs the CPU build of PyTorch, install the CUDA build from pytorch.org. If a recent `datasets` release cannot load `tals/vitaminc`, install an older one, for example `datasets<4`.

| Component | Value |
|---|---|
| NLI model | `cross-encoder/nli-deberta-v3-base` |
| Embeddings | `bge-m3` through Ollama |
| LLM | `qwen2.5:7b` through Ollama, temperature 0, zero-shot and chain-of-thought |
| Datasets | `tals/vitaminc` (test split), `ibm-research/Wikipedia_contradict_benchmark` |

## 2. Folder layout

    contradiction_exp_v2/
      src/        scripts
      data/       synthetic_dev/test.jsonl, vitaminc_main/nei.jsonl, sentences.txt
      config/     frozen_config.json, frozen_config_v2.json
      results/    results_*.json, predictions_*.jsonl, lock files
      cache/      llm_cache.jsonl

| Group | Scripts in `src/` |
|---|---|
| Library | `lib.py`, `lib2.py`, `paths.py` |
| Data building | `fetch_sentences.py`, `make_synthetic.py`, `make_vitaminc.py` |
| Tuning | `tune.py`, `tune_v2.py` |
| Runs | `run_pairs.py`, `evaluate_wikicontradict.py`, `run_wikicontradict_v2.py` |
| Analysis | `recount_v1.py`, `significance.py` |

The files in `results/` are the reference outputs. A new run overwrites them, so copy them first. The embedding cache is rebuilt on the first run.

## 3. Rules

- Do not edit the files in `config/` after tuning.
- The VitaminC run and the WikiContradict runs write a lock file in `results/` and refuse to run twice. Do not delete it to run again.
- If the synthetic test run shows a bug, fix it and rerun that step only. Do not change thresholds because of results.

## 4. Quick check without models

    python src/recount_v1.py
    python src/significance.py

Expected: `nli_match` found 60 and missed 69 on WikiContradict; VitaminC `nli_sem` vs `nli_match` ΔF1 +0.054 [+0.044; +0.066], McNemar p < 0.001.

## 5. Full replication

Run from the root folder.

**Step 0. Data (optional, the files are included)**

    python src/fetch_sentences.py
    python src/make_synthetic.py --n 1000

**Step 1. Tune and freeze**

    python src/tune.py
    python src/tune_v2.py

Frozen v2 values: NLI threshold 0.05; semantic gate = word overlap ≥ 0.10 or bge-m3 cosine ≥ 0.50; conformal threshold 0.0017 (α = 0.05, 500 dev negatives).

**Step 2. Synthetic test**

    python src/run_pairs.py --data data/synthetic_test.jsonl --llm-n 500

**Step 3. VitaminC**

    python src/make_vitaminc.py --inspect
    python src/make_vitaminc.py
    python src/run_pairs.py --data data/vitaminc_main.jsonl --single-look --llm-n 1000
    python src/run_pairs.py --data data/vitaminc_nei.jsonl --llm-n 0

**Step 4. WikiContradict**

    python src/evaluate_wikicontradict.py --show-columns
    python src/evaluate_wikicontradict.py
    python src/recount_v1.py
    python src/run_wikicontradict_v2.py

**Step 5. Significance**

    python src/significance.py

## 6. Expected results

F1 on synthetic and VitaminC; recall / false-alarm rate on WikiContradict.

| Method | Synthetic | VitaminC | WikiContradict |
|---|---|---|---|
| Rules | 0.99 | 0.12 | 0.04 / 0.00 |
| NLI | 0.90 | 0.61 | 0.54 / 0.61 |
| NLI + word matching | 0.98 | 0.64 | 0.47 / 0.01 |
| NLI + semantic matching | 0.98 | 0.69 | 0.61 / 0.01 |
| + conformal threshold | 0.96 | 0.74 | 0.83 / 0.01 |
| LLM zero-shot (subset) | 0.97 | 0.73 | 0.46 / 0.01 |
| LLM chain-of-thought (subset) | 0.94 | 0.76 | 0.61 / 0.00 |
| Cascade NLI → LLM | 0.95 | 0.65 | 0.37 / 0.01 |

Time per pair: NLI about 0.011 s, LLM about 2.4 s (zero-shot) and 3.3 s (chain-of-thought). Small differences are possible with other hardware or library versions.

## 7. Troubleshooting

- `The documented second run was already done`: a lock file exists. Use a fresh copy of the folder.
- `frozen_config_v2.json is missing`: run `python src/tune_v2.py` or restore the file in `config/`.
- Ollama connection errors: check that Ollama is running and both models are pulled.
- Wrong column names on WikiContradict: run with `--show-columns`, then pass `--col1` and `--col2`.
