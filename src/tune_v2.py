"""Tune the v2 methods on synthetic dev and write frozen_config_v2.json."""
import json, argparse, itertools, hashlib, datetime
import numpy as np
from lib import NLI, read_jsonl, prf
from lib2 import Embedder, features, sem_gate, conformal_threshold
from paths import DATA, CONFIG

parser = argparse.ArgumentParser()
parser.add_argument("--dev", default=DATA / "synthetic_dev.jsonl")
parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
parser.add_argument("--embed-model", default="bge-m3")
parser.add_argument("--llm-model", default="qwen2.5:7b")
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--out", default=CONFIG / "frozen_config_v2.json")
args = parser.parse_args()

rows = read_jsonl(args.dev)
pairs = [(r["s1"], r["s2"]) for r in rows]
labels = np.array([r["label"] for r in rows])
feats = features(pairs, NLI(args.model), Embedder(args.embed_model))

THRESHOLD_GRID = sorted(set(np.round(np.r_[np.arange(0.05, 0.95, 0.05), np.arange(0.95, 0.995, 0.01)], 2)))
WORD_GATE_GRID = np.round(np.arange(0.0, 0.65, 0.05), 2)
COS_GATE_GRID = np.round(np.arange(0.50, 0.96, 0.05), 2)

best, best_key = None, None
for threshold, word_gate, cos_gate in itertools.product(THRESHOLD_GRID, WORD_GATE_GRID, COS_GATE_GRID):
    gate = (feats["ov"] >= word_gate) | (feats["cos"] >= cos_gate)
    f1 = prf(labels, (gate & (feats["p"] >= threshold)).astype(int))[2]
    sort_key = (round(f1, 4), threshold, -word_gate, -cos_gate)  # ties: higher threshold, then lower gates
    if best_key is None or sort_key > best_key:
        best_key, best = sort_key, (float(threshold), float(word_gate), float(cos_gate), f1)

threshold, word_gate, cos_gate, f1 = best
cfg = {"nli_model": args.model, "embed_model": args.embed_model, "llm_model": args.llm_model,
       "sem_threshold": threshold, "sem_word_gate": word_gate, "sem_cos_gate": cos_gate}
scores = sem_gate(feats, cfg) * feats["p"]
cfg["alpha"] = args.alpha
cfg["conf_threshold"] = conformal_threshold(scores[labels == 0], args.alpha)
cfg["conf_n_negatives"] = int((labels == 0).sum())
cfg["dev_sha1"] = hashlib.sha1(open(args.dev, "rb").read()).hexdigest()
cfg["frozen_at"] = datetime.datetime.now().isoformat(timespec="seconds")
cfg["note"] = ("Frozen on synthetic dev before the VitaminC single run and the documented WikiContradict re-run. "
               "Do not edit.")

conf_pred = (scores > cfg["conf_threshold"]).astype(int)
P, R, conf_f1 = prf(labels, conf_pred)
print(f"nli_sem       threshold={threshold:.2f} word_gate={word_gate:.2f} cos_gate={cos_gate:.2f}  dev F1={f1:.3f}")
print(f"nli_sem_conf  alpha={args.alpha} conformal threshold={cfg['conf_threshold']:.4f} "
      f"(n negatives={cfg['conf_n_negatives']})  dev F1={conf_f1:.3f}  dev FPR={conf_pred[labels == 0].mean():.3f}")
json.dump(cfg, open(args.out, "w"), indent=2)
print("saved", args.out)
