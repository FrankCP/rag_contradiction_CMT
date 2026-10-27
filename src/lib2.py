"""Embeddings, LLM judge, semantic gate and conformal threshold (v2)."""
import hashlib, json, math, os, time
import numpy as np
import requests
from lib import rules_predict, overlap
from paths import CACHE, RESULTS

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def _key(*parts):
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


class JsonlCache:

    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self.d[record["k"]] = record["v"]

    def get(self, key):
        return self.d.get(key)

    def put(self, key, value):
        self.d[key] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


class Embedder:

    def __init__(self, model="bge-m3", cache=CACHE / "emb_cache.jsonl", batch=32):
        self.model, self.batch, self.cache = model, batch, JsonlCache(cache)

    def encode(self, texts):
        keys = [_key(self.model, text) for text in texts]
        todo = {}
        for key, text in zip(keys, texts):
            if self.cache.get(key) is None:
                todo.setdefault(key, text)
        items = list(todo.items())
        for start in range(0, len(items), self.batch):
            chunk = items[start:start + self.batch]
            vecs = self._post([text for _, text in chunk])
            if vecs is None:
                vecs = [self._single(text) for _, text in chunk]
            for (key, _), vec in zip(chunk, vecs):
                vec = np.asarray(vec, dtype=np.float32)
                if not np.all(np.isfinite(vec)):
                    vec = np.zeros_like(vec)
                vec /= (np.linalg.norm(vec) or 1.0)
                self.cache.put(key, [round(float(x), 5) for x in vec])
        return np.asarray([self.cache.get(key) for key in keys], dtype=np.float32)

    def _post(self, texts):
        r = requests.post(f"{OLLAMA}/api/embed",
                          json={"model": self.model, "input": texts, "truncate": True}, timeout=600)
        if r.status_code != 200:
            self.last_error = f"{r.status_code}: {r.text[:300]}"
            return None
        return r.json()["embeddings"]

    def _single(self, text):
        clean = " ".join(text.split())[:1500]
        for attempt in ([text, clean] if clean and clean != text else [text]):
            out = self._post([attempt])
            if out is not None:
                if attempt != text:
                    self._log(text, "embedded after cleaning/truncation")
                return out[0]
        self._log(text, "FAILED, zero vector used (cosine = 0, only the word gate applies)")
        return [0.0] * 1024

    def _log(self, text, what):
        self.problems = getattr(self, "problems", 0) + 1
        msg = f"[embed] {what}; Ollama said {getattr(self, 'last_error', '?')}; text={text[:120]!r}"
        print(msg)
        with open(RESULTS / "embed_errors.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def cosine(self, pairs):
        A = self.encode([a for a, _ in pairs])
        B = self.encode([b for _, b in pairs])
        return (A * B).sum(1).tolist()


PROMPTS = {
    "zero": ("Two text passages were retrieved by a search system. Decide whether they contradict each other: "
             "at least one statement in Passage A and one statement in Passage B cannot both be true at the same time. "
             "Passages about different topics, or one passage giving extra details, are not a contradiction.\n"
             'Answer ONLY with JSON: {{"conflict": true}} or {{"conflict": false}}.\n\n'
             "Passage A:\n{a}\n\nPassage B:\n{b}"),
    "cot": ("Two text passages were retrieved by a search system. Decide whether they contradict each other: "
            "at least one statement in Passage A and one statement in Passage B cannot both be true at the same time. "
            "Passages about different topics, or one passage giving extra details, are not a contradiction.\n"
            "Think step by step: find the facts that both passages state about the same entity, compare them, "
            "then decide.\n"
            'Answer ONLY with JSON: {{"reasoning": "<two or three sentences>", "conflict": true or false}}.\n\n'
            "Passage A:\n{a}\n\nPassage B:\n{b}"),
}


class LLMJudge:
    def __init__(self, model="qwen2.5:7b", mode="zero", cache=CACHE / "llm_cache.jsonl", max_chars=2000):
        assert mode in PROMPTS
        self.model, self.mode, self.max_chars = model, mode, max_chars
        self.cache = JsonlCache(cache)
        self.invalid, self.calls, self.seconds = 0, 0, 0.0

    @staticmethod
    def _parse(txt):
        try:
            v = json.loads(txt).get("conflict")
        except Exception:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return int(v.strip().lower() == "true")
        return None

    def _ask(self, passage_a, passage_b):
        prompt = PROMPTS[self.mode].format(a=passage_a[:self.max_chars], b=passage_b[:self.max_chars])
        r = requests.post(f"{OLLAMA}/api/generate",
                          json={"model": self.model, "prompt": prompt, "format": "json", "stream": False,
                                "options": {"temperature": 0, "num_predict": 400 if self.mode == "cot" else 30}},
                          timeout=600)
        r.raise_for_status()
        return r.json()["response"]

    def judge(self, pairs):
        out = []
        for passage_a, passage_b in pairs:
            key = _key(self.model, self.mode, passage_a, passage_b)
            verdict = self.cache.get(key)
            if verdict is None:
                t0 = time.time()
                verdict = self._parse(self._ask(passage_a, passage_b))
                if verdict is None:
                    verdict = self._parse(self._ask(passage_a, passage_b))
                self.seconds += time.time() - t0
                self.calls += 1
                if verdict is None:
                    self.invalid += 1
                    verdict = -1
                self.cache.put(key, verdict)
                if self.calls % 25 == 0:
                    print(f"[llm {self.mode}] {len(out) + 1}/{len(pairs)} pairs, "
                          f"{self.seconds / self.calls:.1f} s per call", flush=True)
            out.append(1 if verdict == 1 else 0)
        return out

    def stats(self):
        return dict(model=self.model, mode=self.mode, new_calls=self.calls, invalid=self.invalid,
                    sec_per_call=round(self.seconds / self.calls, 3) if self.calls else None)


def features(pairs, nli, emb):
    start_time = time.time()
    nli_probs = np.asarray(nli.contradiction(pairs), dtype=float)
    nli_sec = (time.time() - start_time) / max(len(pairs), 1)
    return dict(p=nli_probs,
                ov=np.asarray([overlap(a, b) for a, b in pairs], dtype=float),
                cos=np.asarray(emb.cosine(pairs), dtype=float),
                rule=[rules_predict(a, b)[0] for a, b in pairs],
                nli_sec_per_pair=nli_sec)


def sem_gate(feats, cfg2):
    return (feats["ov"] >= cfg2["sem_word_gate"]) | (feats["cos"] >= cfg2["sem_cos_gate"])


def v2_methods(feats, cfg2):
    gate = sem_gate(feats, cfg2)
    return {"nli_sem": (gate & (feats["p"] >= cfg2["sem_threshold"])).astype(int),
            "nli_sem_conf": ((gate * feats["p"]) > cfg2["conf_threshold"]).astype(int)}


def conformal_threshold(neg_scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest no-conflict score."""
    sorted_scores = np.sort(np.asarray(neg_scores, dtype=float))
    n = len(sorted_scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    return 1.0 if rank > n else float(sorted_scores[rank - 1])