"""Evaluate all methods on a pair file with the frozen configs.
  python src/run_pairs.py --data data/synthetic_test.jsonl --llm-n 500
  python src/run_pairs.py --data data/vitaminc_main.jsonl --single-look --llm-n 1000"""
import os, json, argparse, random, time
import numpy as np
from lib import NLI, read_jsonl, write_jsonl, prf, boot_ci, predict_methods, jaccard, content
from lib2 import Embedder, LLMJudge, features, v2_methods
from paths import CONFIG, RESULTS

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--config1", default=CONFIG / "frozen_config.json")
parser.add_argument("--config2", default=CONFIG / "frozen_config_v2.json")
parser.add_argument("--llm-n", type=int, default=0)
parser.add_argument("--single-look", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

name = os.path.splitext(os.path.basename(args.data))[0]
lock = RESULTS / f"{name}_run.lock"
if args.single_look and os.path.exists(lock) and not args.force:
    raise SystemExit(f"Single-look rule: {args.data} was already evaluated ({lock}).")

cfg1, cfg2 = json.load(open(args.config1)), json.load(open(args.config2))
rows = read_jsonl(args.data)
pairs = [(rec["s1"], rec["s2"]) for rec in rows]
labels = np.array([rec["label"] for rec in rows])
types = np.array([rec.get("type", "?") for rec in rows])

feats = features(pairs, NLI(cfg1["nli_model"]), Embedder(cfg2["embed_model"]))
preds = {k: np.asarray(v) for k, v in predict_methods(pairs, feats["p"].tolist(), cfg1).items()}
preds.update(v2_methods(feats, cfg2))

rng = random.Random(0)
preds["baseline_random"] = np.array([rng.randint(0, 1) for _ in rows])
preds["baseline_majority"] = np.array([int(labels.mean() >= 0.5)] * len(rows))
norm = lambda s: " ".join(s.lower().split())
preds["baseline_string"] = np.array([int(norm(text_a) != norm(text_b) and jaccard(content(text_a), content(text_b)) >= 0.6)
                                     for text_a, text_b in pairs])

subset_idx = np.arange(len(rows))
llm_stats = {}
if args.llm_n >= 0:
    if 0 < args.llm_n < len(rows):
        subset_idx = np.sort(np.random.default_rng(0).choice(len(rows), args.llm_n, replace=False))
    subset_pairs = [pairs[i] for i in subset_idx]
    for mode in ("zero", "cot"):
        judge = LLMJudge(cfg2["llm_model"], mode)
        t0 = time.time()
        preds_llm = np.asarray(judge.judge(subset_pairs))
        llm_stats[mode] = {**judge.stats(), "wall_sec": round(time.time() - t0, 1)}
        preds[f"llm_{mode}"] = preds_llm
    flagged = preds["nli_sem"][subset_idx] == 1                     # cascade: LLM confirms what NLI flagged
    preds["cascade"] = preds["nli_sem"][subset_idx] * preds["llm_zero"]
    llm_stats["cascade_llm_calls"] = int(flagged.sum())


def report(sel, methods, title):
    y_sel, types_sel = labels[sel], types[sel]
    print(f"\n== {title}: {len(y_sel)} pairs, {int(y_sel.sum())} conflicts ==")
    print(f"{'method':18s} {'P':>6s} {'R':>6s} {'F1':>6s}  F1 95% CI        FPR    FPR by negative type")
    out = {}
    for method in methods:
        pred = preds[method] if len(preds[method]) == len(y_sel) else preds[method][sel]
        P, R, F1 = prf(y_sel, pred)
        lo, hi = boot_ci(y_sel, pred)
        fpr = float(pred[y_sel == 0].mean()) if (y_sel == 0).any() else None
        neg_t = {t: float(pred[(types_sel == t) & (y_sel == 0)].mean()) for t in sorted(set(types_sel[y_sel == 0].tolist()))}
        rec_t = {t: float(pred[(types_sel == t) & (y_sel == 1)].mean()) for t in sorted(set(types_sel[y_sel == 1].tolist()))}
        out[method] = dict(P=P, R=R, F1=F1, F1_ci=[lo, hi], FPR=fpr, FPR_by_type=neg_t, recall_by_type=rec_t)
        print(f"{method:18s} {P:6.3f} {R:6.3f} {F1:6.3f}  [{lo:.3f}; {hi:.3f}]  "
              f"{'' if fpr is None else f'{fpr:.3f}'}  " + ", ".join(f"{k}={v:.2f}" for k, v in neg_t.items()))
    return out


base = ["baseline_random", "baseline_majority", "baseline_string", "rules", "nli", "hybrid",
        "nli_match", "hybrid_match", "nli_sem", "nli_sem_conf"]
llm = [method for method in ("llm_zero", "llm_cot", "cascade") if method in preds]
results = {"data": os.path.basename(args.data), "n": len(rows), "nli_sec_per_pair": round(feats["nli_sec_per_pair"], 4), "llm": llm_stats}
if len(subset_idx) == len(rows):
    results["all"] = report(np.arange(len(rows)), base + llm, "all pairs")
else:
    results["all"] = report(np.arange(len(rows)), base, "all pairs (no LLM)")
    results["subset"] = report(subset_idx, base + llm, f"LLM subset (same {len(subset_idx)} pairs for every method)")
print(f"\nNLI: {results['nli_sec_per_pair']} s/pair;  LLM: {json.dumps(llm_stats)}")
json.dump(results, open(RESULTS / f"results_{name}.json", "w"), indent=2)

out_rows = []
subset_pos = {int(i): k for k, i in enumerate(subset_idx)}
for k, rec in enumerate(rows):
    row = dict(rec, nli_prob=round(float(feats["p"][k]), 4), overlap=round(float(feats["ov"][k]), 3),
               cos=round(float(feats["cos"][k]), 3))
    for method in base[3:]:
        row[method] = int(preds[method][k])
    for method in llm:
        if k in subset_pos:
            row[method] = int(preds[method][subset_pos[k]])
    out_rows.append(row)
write_jsonl(RESULTS / f"predictions_{name}.jsonl", out_rows)
print(f"saved results_{name}.json and predictions_{name}.jsonl")
if args.single_look:
    open(lock, "w").write("done\n")
