"""McNemar test and paired bootstrap of F1 differences, from saved predictions."""
import json, math
import numpy as np
from lib import read_jsonl, prf
from paths import RESULTS

COMPARISONS = [
    ("predictions_vitaminc_main.jsonl", "nli_sem", "nli_match", "VitaminC, all 3000 pairs"),
    ("predictions_vitaminc_main.jsonl", "llm_cot", "nli_sem_conf", "VitaminC, LLM subset (1000)"),
    ("predictions_vitaminc_main.jsonl", "llm_cot", "nli_sem", "VitaminC, LLM subset (1000)"),
    ("predictions_vitaminc_main.jsonl", "llm_cot", "llm_zero", "VitaminC, LLM subset (1000)"),
    ("predictions_wikicontradict_v2.jsonl", "nli_sem", "nli_match", "WikiContradict, 129 + 129"),
    ("predictions_synthetic_test.jsonl", "llm_cot", "llm_zero", "synthetic, LLM subset (500)"),
]


def mcnemar_exact(a_correct, b_correct):
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    smaller = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / 2 ** n
    return a_only, b_only, min(1.0, 2 * tail)


def boot_diff(labels, pred_a, pred_b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sample = rng.integers(0, len(labels), len(labels))
        diffs.append(prf(labels[sample], pred_a[sample])[2] - prf(labels[sample], pred_b[sample])[2])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


out = []
print(f"{'data':30s} {'A':13s} {'B':13s} {'F1 A':>6s} {'F1 B':>6s} {'diff':>7s}  {'95% CI of diff':18s} "
      f"{'A>B':>4s} {'B>A':>4s} {'McNemar p':>9s}")
for path, method_a, method_b, description in COMPARISONS:
    try:
        rows = read_jsonl(RESULTS / path)
    except FileNotFoundError:
        print(f"{description:30s} file {path} not found, skipped")
        continue
    rows = [row for row in rows if method_a in row and method_b in row]
    if not rows:
        print(f"{description:30s} no pairs with both {method_a} and {method_b}, skipped")
        continue
    labels = np.array([row["label"] for row in rows])
    pred_a = np.array([row[method_a] for row in rows]); pred_b = np.array([row[method_b] for row in rows])
    f1_a, f1_b = prf(labels, pred_a)[2], prf(labels, pred_b)[2]
    ci_low, ci_high = boot_diff(labels, pred_a, pred_b)
    a_only, b_only, p_value = mcnemar_exact(pred_a == labels, pred_b == labels)
    out.append(dict(data=description, n=len(rows), A=method_a, B=method_b, F1_A=f1_a, F1_B=f1_b, diff=f1_a - f1_b, diff_ci=[ci_low, ci_high],
                    A_right_B_wrong=a_only, A_wrong_B_right=b_only, mcnemar_p=p_value))
    print(f"{description:30s} {method_a:13s} {method_b:13s} {f1_a:6.3f} {f1_b:6.3f} {f1_a - f1_b:+7.3f}  [{ci_low:+.3f}; {ci_high:+.3f}]    "
          f"{a_only:4d} {b_only:4d} {p_value:9.4f}")
json.dump(out, open(RESULTS / "results_significance.json", "w"), indent=2)
print("\nA difference is significant at 5% if the CI of the difference excludes 0 (and McNemar p < 0.05).")
print("saved results_significance.json")
