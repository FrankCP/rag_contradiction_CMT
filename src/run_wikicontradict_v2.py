"""Second run on WikiContradict with the v2 methods."""
import os, re, json, argparse, random
from collections import Counter
import numpy as np
from datasets import load_dataset
from lib import NLI, rules_predict, content, jaccard, boot_ci, overlap, write_jsonl
from lib2 import Embedder, LLMJudge
from paths import CONFIG, RESULTS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ibm-research/Wikipedia_contradict_benchmark")
parser.add_argument("--split", default="train")
parser.add_argument("--config1", default=CONFIG / "frozen_config.json")
parser.add_argument("--config2", default=CONFIG / "frozen_config_v2.json")
parser.add_argument("--top-m", type=int, default=5)
parser.add_argument("--no-llm", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

LOCK = RESULTS / "wikicontradict_v2_run.lock"
if os.path.exists(LOCK) and not args.force:
    raise SystemExit("The documented second run was already done (wikicontradict_v2_run.lock).")
if not os.path.exists(args.config2):
    raise SystemExit("frozen_config_v2.json is missing: run tune_v2.py first.")
cfg1, cfg2 = json.load(open(args.config1)), json.load(open(args.config2))
print("*** SECOND, DOCUMENTED RUN ON WIKICONTRADICT (methods frozen on synthetic dev before this run) ***")


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.replace("\n", " ")) if len(s.split()) >= 4]


dataset = load_dataset(args.dataset, split=args.split)
raw = [r for r in dataset if r.get("context1") and r.get("context2")]
seen, items = set(), []
for r in raw:
    key = (r["context1"].strip(), r["context2"].strip())
    if frozenset(key) in seen:
        continue
    seen.add(frozenset(key))
    items.append(dict(c1=key[0], c2=key[1], type=r.get("contradictType", "?"),
                      article=r.get("WikipediaArticleTitle", ""), qid=r.get("question_ID")))
rng = random.Random(0)
neg_pairs = []
for it in items:
    others = [j for j in range(len(items)) if items[j]["article"] != it["article"]]
    neg_pairs.append((it["c1"], items[rng.choice(others)]["c2"]))
pos_pairs = [(i["c1"], i["c2"]) for i in items]
print(f"unique conflicting pairs: {len(items)}; negatives: {len(neg_pairs)}")

nli, emb = NLI(cfg1["nli_model"]), Embedder(cfg2["embed_model"])


def classify(p1, p2):
    S1, S2 = split_sents(p1), split_sents(p2)
    allp = [(x, z) for x in S1 for z in S2]
    if not allp:
        return None
    ov_all = [jaccard(content(x), content(z)) for x, z in allp]
    v1 = [allp[k] for k in np.argsort(-np.asarray(ov_all), kind="stable")[:args.top_m]]
    E1, E2 = emb.encode(S1), emb.encode(S2)
    cos_m = E1 @ E2.T
    cos_all = [float(cos_m[i, j]) for i in range(len(S1)) for j in range(len(S2))]
    by_cos = [allp[k] for k in np.argsort(-np.asarray(cos_all), kind="stable")[:args.top_m]]
    v2 = list(dict.fromkeys(v1 + by_cos))
    cos_of = {allp[k]: cos_all[k] for k in range(len(allp))}

    pr1 = nli.contradiction(v1)
    rl = [rules_predict(x, z)[0] for x, z in v1]
    ov1 = [overlap(x, z) for x, z in v1]
    o = {"rules": int(any(r == 1 for r in rl)),
         "nli": int(any(p >= cfg1["nli_threshold"] for p in pr1)),
         "hybrid": int(any((r == 1) if r is not None else (p >= cfg1["hybrid_threshold"]) for r, p in zip(rl, pr1))),
         "nli_match": int(any(g >= cfg1["nli_match_gate"] and p >= cfg1["nli_match_threshold"]
                              for g, p in zip(ov1, pr1))),
         "hybrid_match": int(any((r == 1) if r is not None else
                                 (g >= cfg1["hybrid_match_gate"] and p >= cfg1["hybrid_match_threshold"])
                                 for r, g, p in zip(rl, ov1, pr1))),
         "matched_word": int(any(g >= cfg1["nli_match_gate"] for g in ov1))}
    pr2 = nli.contradiction(v2)
    gate2 = [(overlap(x, z) >= cfg2["sem_word_gate"]) or (cos_of[(x, z)] >= cfg2["sem_cos_gate"]) for x, z in v2]
    o["nli_sem"] = int(any(g and p >= cfg2["sem_threshold"] for g, p in zip(gate2, pr2)))
    o["nli_sem_conf"] = int(any(g and p > cfg2["conf_threshold"] for g, p in zip(gate2, pr2)))
    o["matched_sem"] = int(any(gate2))
    o["max_p_v1"] = round(float(max(pr1)), 4)
    return o


METHODS = ["rules", "nli", "hybrid", "nli_match", "hybrid_match", "nli_sem", "nli_sem_conf"]
empty = {m: 0 for m in METHODS} | {"matched_word": 0, "matched_sem": 0, "max_p_v1": 0.0}
pos = [classify(*p) or dict(empty) for p in pos_pairs]
neg = [classify(*p) or dict(empty) for p in neg_pairs]

if not args.no_llm:
    jz, jc = LLMJudge(cfg2["llm_model"], "zero"), LLMJudge(cfg2["llm_model"], "cot")
    for lst, prs in ((pos, pos_pairs), (neg, neg_pairs)):
        for o, z, c in zip(lst, jz.judge(prs), jc.judge(prs)):
            o["llm_zero"], o["llm_cot"] = z, c
            o["cascade"] = o["nli_sem"] * z
    METHODS += ["llm_zero", "llm_cot", "cascade"]

types = np.array([i["type"] for i in items])
ones = np.ones(len(pos))
res = {"run": "second, documented", "n_pairs": len(items), "contradictType": dict(Counter(types.tolist()))}
print(f"\n{'method':14s} recall  95% CI          FPR    P      F1     recall by type")
for m in METHODS:
    rec = np.array([p[m] for p in pos]); fp = np.array([n[m] for n in neg])
    tp, fpn = rec.sum(), fp.sum()
    P = tp / (tp + fpn) if tp + fpn else 0.0
    R = rec.mean()
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    ci = boot_ci(ones, rec, fn=lambda y, p: p.mean())
    by_t = {t: float(rec[types == t].mean()) for t in sorted(set(types.tolist()))}
    res[m] = dict(recall=float(R), recall_ci=ci, FPR=float(fp.mean()), P=float(P), F1=float(F1), recall_by_type=by_t)
    print(f"{m:14s} {R:.3f}  [{ci[0]:.3f}; {ci[1]:.3f}]  {fp.mean():.3f}  {P:.3f}  {F1:.3f}  "
          + ", ".join(f"{t[:10]}={v:.2f}" for t, v in by_t.items()))

print("\nStage decomposition on the conflicting pairs")
for m, gate in (("nli_match", "matched_word"), ("nli_sem", "matched_sem")):
    n = len(pos); mt = sum(p[gate] for p in pos); dt = sum(p[m] for p in pos)
    res[f"decomp_{m}"] = dict(n=n, matched=mt, detected=dt, cut_by_gate=n - mt, missed_by_model=mt - dt,
                              p_matched=mt / n, p_detected_given_matched=dt / mt if mt else 0.0)
    print(f"{m:10s} n={n}  cut by gate={n - mt}  missed by model={mt - dt}  detected={dt}  "
          f"-> {mt / n:.3f} x {dt / mt if mt else 0:.3f} = {dt / n:.3f}")
if not args.no_llm:
    res["llm"] = {"zero": jz.stats(), "cot": jc.stats()}
json.dump(res, open(RESULTS / "results_wikicontradict_v2.json", "w"), indent=2)
write_jsonl(RESULTS / "predictions_wikicontradict_v2.jsonl",
            [{**it, "label": 1, **p} for it, p in zip(items, pos)] +
            [{"c1": c1, "c2": c2, "label": 0, **p} for (c1, c2), p in zip(neg_pairs, neg)])
open(LOCK, "w").write("done\n")
print("saved results_wikicontradict_v2.json and predictions_wikicontradict_v2.jsonl")
