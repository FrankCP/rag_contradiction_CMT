"""Shared helpers: rules, NLI wrapper, metrics."""
import re, json
import numpy as np

STOP = set("the a an of and or in on at to for with by from is was are were be been as that this it its his her their has have had which who also".split())
NEG = {"not", "no", "never", "neither", "nor", "cannot"}
PREFIXES = ["According to the article, ", "It is known that ", "As reported, "]


def words(text):
    return re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text)


def content(text):
    out = set()
    for word in words(text):
        lowered = word.lower()
        if lowered in STOP or lowered in NEG or lowered.endswith("n't"):
            continue
        out.add(lowered)
    return out


def jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b) if (set_a | set_b) else 0.0


def numbers(text):
    vals = []
    for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text):
        try:
            vals.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return sorted(vals)


def neg_count(text):
    return sum(1 for w in words(text) if w.lower() in NEG or w.lower().endswith("n't"))


def caps(text):
    tokens = words(text)
    return {word for word in tokens[1:] if word[0].isupper()}


def rules_predict(text_a, text_b):
    """Return (label, reason): 1 conflict, 0 none, None undecided."""
    overlap_score = jaccard(content(text_a), content(text_b))
    if overlap_score < 0.5:
        return None, "low_overlap"
    if numbers(text_a) != numbers(text_b):
        return 1, "number"
    if neg_count(text_a) % 2 != neg_count(text_b) % 2:
        return 1, "negation"
    caps_a, caps_b = caps(text_a), caps(text_b)
    if (caps_a - caps_b) and (caps_b - caps_a):
        return 1, "name"
    if overlap_score >= 0.8:
        return 0, "same"
    return None, "unsure"


class NLI:

    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base", device=None, batch=32):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch = batch
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.idx = [k for k, v in labels.items() if "contra" in v][0]

    def _probs(self, texts_a, texts_b):
        out = []
        for i in range(0, len(texts_a), self.batch):
            enc = self.tok(texts_a[i:i + self.batch], texts_b[i:i + self.batch], truncation=True,
                           padding=True, max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[:, self.idx]
            out.extend(probs.cpu().tolist())
        return out

    def contradiction(self, pairs):
        texts_a = [a for a, _ in pairs]
        texts_b = [b for _, b in pairs]
        forward, backward = self._probs(texts_a, texts_b), self._probs(texts_b, texts_a)
        return [max(x, y) for x, y in zip(forward, backward)]


def prf(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def boot_ci(y_true, y_pred, fn=lambda y, p: prf(y, p)[2], n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        sample = rng.integers(0, len(y_true), len(y_true))
        scores.append(fn(y_true[sample], y_pred[sample]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def overlap(text_a, text_b):
    return jaccard(content(text_a), content(text_b))


def predict_methods(pairs, nli_probs, cfg):
    rule_labels = [rules_predict(a, b)[0] for a, b in pairs]
    out = {"rules": [1 if r == 1 else 0 for r in rule_labels]}
    if nli_probs is None:
        return out
    overlaps = [overlap(a, b) for a, b in pairs]
    out["nli"] = [int(p >= cfg["nli_threshold"]) for p in nli_probs]
    out["hybrid"] = [r if r is not None else int(p >= cfg["hybrid_threshold"]) for r, p in zip(rule_labels, nli_probs)]
    if "nli_match_gate" in cfg:
        gate, threshold = cfg["nli_match_gate"], cfg["nli_match_threshold"]
        out["nli_match"] = [int(o >= gate and p >= threshold) for o, p in zip(overlaps, nli_probs)]
        gate, threshold = cfg["hybrid_match_gate"], cfg["hybrid_match_threshold"]
        out["hybrid_match"] = [r if r is not None else int(o >= gate and p >= threshold)
                               for r, o, p in zip(rule_labels, overlaps, nli_probs)]
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
