"""Single-look run of the v1 methods on WikiContradict."""
import os, re, json, argparse, random
import numpy as np
from lib import NLI, rules_predict, content, jaccard, boot_ci, overlap
from paths import CONFIG, RESULTS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ibm-research/Wikipedia_contradict_benchmark")
parser.add_argument("--split", default="train")
parser.add_argument("--col1", default="context1")
parser.add_argument("--col2", default="context2")
parser.add_argument("--config", default=CONFIG / "frozen_config.json")
parser.add_argument("--top-m", type=int, default=5, help="candidate sentence pairs per passage pair")
parser.add_argument("--show-columns", action="store_true")
parser.add_argument("--inspect", action="store_true", help="describe the data structure only (no methods are run)")
parser.add_argument("--force", action="store_true")
parser.add_argument("--no-nli", action="store_true")
parser.add_argument("--out", default=RESULTS / "results_wikicontradict.json")
args = parser.parse_args()

def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.replace("\n", " ")) if len(s.split()) >= 4]


from datasets import load_dataset
dataset = load_dataset(args.dataset, split=args.split)
if args.show_columns:
    print(dataset.column_names); print(dataset[0]); raise SystemExit

from collections import Counter, defaultdict
raw = [r for r in dataset if r.get(args.col1) and r.get(args.col2)]
seen, items = set(), []
for r in raw:
    key = (r[args.col1].strip(), r[args.col2].strip())
    ukey = frozenset(key)
    if ukey in seen:
        continue
    seen.add(ukey)
    items.append(dict(c1=key[0], c2=key[1], type=r.get("contradictType", "?"),
                      article=r.get("WikipediaArticleTitle", ""), qid=r.get("question_ID")))

if args.inspect:
    print("rows in file:", len(raw))
    print("distinct question_ID:", len({r.get('question_ID') for r in raw}))
    print("distinct (context1, context2) pairs:", len(items))
    print("distinct articles:", len({i['article'] for i in items}))
    print("contradictType:", dict(Counter(i['type'] for i in items)))
    print("samepassage:", dict(Counter(r.get('samepassage') for r in raw)))
    rep = Counter((r[args.col1].strip(), r[args.col2].strip()) for r in raw)
    print("how often a pair repeats:", dict(Counter(rep.values())))
    by_q = defaultdict(list)
    for r in raw:
        by_q[(r[args.col1].strip(), r[args.col2].strip())].append(r.get("question"))
    ex = [qs for qs in by_q.values() if len(qs) > 1][:2]
    print("example of a repeated pair - its questions:", ex)
    raise SystemExit
LOCK = RESULTS / "wikicontradict_run.lock"
if os.path.exists(LOCK) and not args.force and not args.no_nli:
    raise SystemExit("Single-look rule: WikiContradict was already evaluated. Delete the lock only for a documented re-run.")

cfg = {} if args.no_nli else json.load(open(args.config))
nli = None if args.no_nli else NLI(cfg["nli_model"])


def candidates(p1, p2):
    S1, S2 = split_sents(p1), split_sents(p2)
    sc = [(jaccard(content(x), content(y)), x, y) for x in S1 for y in S2]
    sc.sort(key=lambda t: -t[0])
    return [(x, y) for _, x, y in sc[:args.top_m]]


METHODS = ["rules"] if args.no_nli else ["rules", "nli", "hybrid", "nli_match", "hybrid_match"]


def classify(p1, p2):
    cand = candidates(p1, p2)
    if not cand:
        return {m: 0 for m in METHODS}
    rl = [rules_predict(x, y)[0] for x, y in cand]
    out = {"rules": int(any(r == 1 for r in rl))}
    if nli:
        pr = nli.contradiction(cand)
        out["nli"] = int(any(p >= cfg["nli_threshold"] for p in pr))
        out["hybrid"] = int(any((r == 1) if r is not None else (p >= cfg["hybrid_threshold"])
                                for r, p in zip(rl, pr)))
        ov = [overlap(x, y) for x, y in cand]
        out["nli_match"] = int(any(o >= cfg["nli_match_gate"] and p >= cfg["nli_match_threshold"]
                                   for o, p in zip(ov, pr)))
        out["hybrid_match"] = int(any((r == 1) if r is not None else
                                      (o >= cfg["hybrid_match_gate"] and p >= cfg["hybrid_match_threshold"])
                                      for r, o, p in zip(rl, ov, pr)))
    return out


rng = random.Random(0)
pos = [classify(i["c1"], i["c2"]) for i in items]
neg, neg_pairs = [], []
for k, it in enumerate(items):
    others = [j for j in range(len(items)) if items[j]["article"] != it["article"]]
    j = rng.choice(others)
    neg_pairs.append((it["c1"], items[j]["c2"]))
    neg.append(classify(it["c1"], items[j]["c2"]))

types = np.array([i["type"] for i in items])
res = {"n_pairs": len(items), "rows_in_file": len(raw), "contradictType": dict(Counter(types.tolist()))}
print(f"distinct conflicting pairs: {len(items)} (rows in file: {len(raw)})")
for m in METHODS:
    rec = np.array([p[m] for p in pos]); fp = np.array([n[m] for n in neg])
    ones = np.ones(len(rec))
    res[m] = dict(recall=float(rec.mean()), recall_ci=boot_ci(ones, rec, fn=lambda y, p: p.mean()),
                  false_positive_rate=float(fp.mean()), fpr_ci=boot_ci(ones, fp, fn=lambda y, p: p.mean()),
                  recall_by_type={t: float(rec[types == t].mean()) for t in sorted(set(types.tolist()))})
    print(f"{m:13s} recall={rec.mean():.3f} CI=[{res[m]['recall_ci'][0]:.3f}; {res[m]['recall_ci'][1]:.3f}]"
          f"  FPR={fp.mean():.3f}  by type: " + ", ".join(f"{t[:20]}={v:.2f}" for t, v in res[m]["recall_by_type"].items()))
json.dump(res, open(args.out, "w"), indent=2)

with open(RESULTS / "predictions_wikicontradict.jsonl", "w", encoding="utf-8") as f:
    for it, p in zip(items, pos):
        f.write(json.dumps({**it, "label": 1, **p}, ensure_ascii=False) + "\n")
    for (c1, c2), p in zip(neg_pairs, neg):
        f.write(json.dumps({"c1": c1, "c2": c2, "label": 0, **p}, ensure_ascii=False) + "\n")
if not args.no_nli:
    open(LOCK, "w").write("done\n")