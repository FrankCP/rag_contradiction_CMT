"""Tune the v1 methods on synthetic dev and write frozen_config.json."""
import json, argparse, itertools
import numpy as np
from lib import NLI, read_jsonl, prf, predict_methods
from paths import DATA, CONFIG

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default=DATA / "synthetic_dev.jsonl")
ap.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
ap.add_argument("--out", default=CONFIG / "frozen_config.json")
args = ap.parse_args()

rows = read_jsonl(args.dev)
pairs = [(r["s1"], r["s2"]) for r in rows]
labels = [r["label"] for r in rows]
probs = NLI(args.model).contradiction(pairs)

THRESHOLD_GRID = sorted(set(np.round(np.r_[np.arange(0.05, 0.95, 0.05), np.arange(0.95, 0.995, 0.01)], 2)))
GATE_GRID = np.round(np.arange(0.0, 0.65, 0.05), 2)
cfg = {"nli_model": args.model}


def f1(method, params):
    return prf(labels, predict_methods(pairs, probs, params)[method])[2]


for m in ("nli", "hybrid"):
    # ties: take the higher threshold
    best = max(THRESHOLD_GRID, key=lambda t: (round(f1(m, {**cfg, "nli_threshold": t, "hybrid_threshold": t}), 4), t))
    cfg[f"{m}_threshold"] = float(best)
    print(f"{m:13s} threshold={best:.2f}             dev F1={f1(m, {**cfg, 'nli_threshold': best, 'hybrid_threshold': best}):.3f}")

for m in ("nli_match", "hybrid_match"):
    def score(candidate):
        threshold, gate = candidate
        params = {**cfg, "nli_match_threshold": threshold, "nli_match_gate": gate, "hybrid_match_threshold": threshold, "hybrid_match_gate": gate}
        return f1(m, params)
    # ties: higher threshold, then lower gate
    (t, g) = max(itertools.product(THRESHOLD_GRID, GATE_GRID), key=lambda tg: (round(score(tg), 4), tg[0], -tg[1]))
    cfg[f"{m}_threshold"], cfg[f"{m}_gate"] = float(t), float(g)
    print(f"{m:13s} threshold={t:.2f} gate={g:.2f}   dev F1={score((t, g)):.3f}")

cfg["note"] = "Frozen on synthetic dev. Do not change before the single WikiContradict run."
json.dump(cfg, open(args.out, "w"), indent=2)
print("saved", args.out)
