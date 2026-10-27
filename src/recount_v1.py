"""Recount v1 WikiContradict errors from predictions_wikicontradict.jsonl."""
import json
from lib import read_jsonl
from paths import RESULTS

rows = read_jsonl(RESULTS / "predictions_wikicontradict.jsonl")
conflicts = [row for row in rows if row["label"] == 1]
negatives = [row for row in rows if row["label"] == 0]
print(f"conflicts: {len(conflicts)}   negatives: {len(negatives)}")
for method in ("rules", "nli", "hybrid", "nli_match", "hybrid_match"):
    if method not in conflicts[0]:
        continue
    found = sum(row[method] for row in conflicts)
    false_alarms = sum(row[method] for row in negatives)
    print(f"{method:13s} found={found:3d}  missed={len(conflicts) - found:3d}  recall={found / len(conflicts):.3f}  "
          f"false alarms={false_alarms:3d}  FPR={false_alarms / len(negatives):.3f}")
both = [row for row in conflicts if row.get("nli") == 1 and row.get("nli_match") == 0]
print(f"\nfound by plain NLI but not by NLI+matching: {len(both)}  (part of the 'cut by the gate' count)")
print("The exact split 'cut by gate' vs 'missed by model' is printed by run_wikicontradict_v2.py.")
