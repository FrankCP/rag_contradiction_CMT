"""Build vitaminc_main.jsonl and vitaminc_nei.jsonl from the VitaminC test split."""
import random, argparse
from collections import Counter
from datasets import load_dataset
from lib import write_jsonl
from paths import DATA

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="test")
parser.add_argument("--revision-type", default="real", help="value of the revision_type field to keep ('' = all)")
parser.add_argument("--n-per-class", type=int, default=1500)
parser.add_argument("--n-nei", type=int, default=500)
parser.add_argument("--inspect", action="store_true")
parser.add_argument("--seed", type=int, default=13)
args = parser.parse_args()

dataset = load_dataset("tals/vitaminc", split=args.split)
print("rows:", len(dataset))
print("revision_type:", dict(Counter(dataset["revision_type"])))
print("label:", dict(Counter(dataset["label"])))
if args.inspect:
    print("example:", dataset[0])
    raise SystemExit

keep = [record for record in dataset if not args.revision_type or record["revision_type"] == args.revision_type]
if not keep:
    raise SystemExit(f"No rows with revision_type == {args.revision_type!r}; pick one of the values printed above.")

seen, buckets = set(), {"refutes": [], "supports": [], "nei": []}
for record in keep:
    pair_key = (record["claim"].strip(), record["evidence"].strip())
    if pair_key in seen or not pair_key[0] or not pair_key[1]:
        continue
    seen.add(pair_key)
    label_text = record["label"].upper()
    bucket = "refutes" if "REFUTE" in label_text else "supports" if "SUPPORT" in label_text else "nei"
    buckets[bucket].append(dict(id=record["unique_id"], case_id=record.get("case_id"), page=record.get("page"),
                           s1=record["evidence"].strip(), s2=record["claim"].strip(),
                           label=1 if bucket == "refutes" else 0, type=bucket))
print("after filter and de-duplication:", {pair_key: len(v) for pair_key, v in buckets.items()})

rng = random.Random(args.seed)
for bucket in buckets.values():
    rng.shuffle(bucket)
n_pairs = min(args.n_per_class, len(buckets["refutes"]), len(buckets["supports"]))
main = buckets["refutes"][:n_pairs] + buckets["supports"][:n_pairs]
rng.shuffle(main)
nei = buckets["nei"][:args.n_nei]
write_jsonl(DATA / "vitaminc_main.jsonl", main)
write_jsonl(DATA / "vitaminc_nei.jsonl", nei)
mean_words = lambda xs: sum(len(record.split()) for record in xs) / max(len(xs), 1)
print(f"vitaminc_main.jsonl: {len(main)} pairs ({n_pairs} conflicts / {n_pairs} no-conflict); "
      f"vitaminc_nei.jsonl: {len(nei)} pairs")
print(f"average length: evidence {mean_words([record['s1'] for record in main]):.1f} words, claim {mean_words([record['s2'] for record in main]):.1f} words")
