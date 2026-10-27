"""Build synthetic dev and test pairs from sentences.txt (needs spaCy en_core_web_md)."""
import re, random, argparse
from collections import Counter, defaultdict
from lib import write_jsonl, PREFIXES
from paths import DATA

parser = argparse.ArgumentParser()
parser.add_argument("--sentences", default=DATA / "sentences.txt")
parser.add_argument("--n", type=int, default=1000, help="pairs per split")
parser.add_argument("--seed", type=int, default=13)
parser.add_argument("--spacy-model", default="en_core_web_md", help="en_core_web_md / _lg give better names")
parser.add_argument("--min-count", type=int, default=3, help="how often a name must be seen with the same kind")
parser.add_argument("--max-sents", type=int, default=15000, help="sentences per split passed to spaCy")
args = parser.parse_args()
rng = random.Random(args.seed)

try:
    import spacy
    nlp = spacy.load(args.spacy_model, disable=["lemmatizer"])
except Exception as e:
    raise SystemExit(f"spaCy model missing. Run:\n  pip install spacy\n  python -m spacy download {args.spacy_model}\n" + str(e))

ENT_KINDS = {"PERSON", "GPE", "LOC", "NORP"}
GOOD_ENT = re.compile(r"^[A-Z][A-Za-z'.\- ]*[A-Za-z]$")


def kind(e):
    if e.label_ == "NORP":
        return "NORP_pl" if e.text.endswith("s") else "NORP_adj"
    return e.label_


def good_ent(e):
    t = e.text
    return e.label_ in ENT_KINDS and len(t) > 2 and len(t.split()) <= 4 and bool(GOOD_ENT.match(t))
NEG_RE = re.compile(r"\b(is|was|are|were|has|have|had)\b(?! not)")
LOWER_OK = {"The", "A", "An", "This", "These", "That", "Those", "In", "It", "On", "At", "For", "By", "During",
            "After", "Before", "Its", "His", "Her", "Their", "There", "Some", "Many", "Most", "When", "As",
            "He", "She", "They", "We", "Starting", "Since", "Although", "However", "While"}


def is_clean(s):
    ws = s.split()
    if not s.rstrip().endswith((".", "!", "?")):
        return False
    if re.search(r"\b(References|External links|See also|Further reading|ISBN)\b", s):
        return False
    lower = sum(1 for w in ws if w[:1].islower())
    if lower < 5 or lower / len(ws) < 0.45:
        return False
    if "  " in s:
        return False
    return True


def date_mut(s):
    m = re.search(r"(?<![\w-])(1[0-9]{3}|20[0-3][0-9])(?![\w-])", s)
    if not m:
        return None
    y = int(m.group(1)); d = rng.choice([-1, 1]) * rng.randint(1, 25)
    return s[:m.start()] + str(y + d) + s[m.end():]


def number_mut(s):
    for m in re.finditer(r"(?<![\w.,-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?=(?:st|nd|rd|th|%)?(?![\w-]))", s):
        raw = m.group(1)
        v = float(raw.replace(",", ""))
        if v == 0 or (raw.isdigit() and 1000 <= v <= 2039):
            continue
        before = s[:m.start()].split()
        if before and before[-1][:1].isupper() and len(before) > 1:
            continue
        nv = v
        while nv == v:
            nv = max(1, round(v * rng.choice([0.5, 0.8, 1.2, 1.5, 2, 3])) + rng.randint(1, 3))
        if "." in raw:
            new = f"{v * rng.choice([0.5, 1.5, 2]):.1f}"
        elif "," in raw:
            new = f"{int(nv):,}"
        else:
            new = str(int(nv))
        return s[:m.start()] + new + s[m.end():]
    return None


def negation_mut(s):
    m = NEG_RE.search(s)
    return s[:m.end()] + " not" + s[m.end():] if m else None


def name_mut(s, doc, pool):
    ents = [e for e in doc.ents if good_ent(e) and e.start_char > 0
            and s[e.start_char - 1] not in "-–" and s[e.end_char:e.end_char + 1] not in ("-", "–")]
    rng.shuffle(ents)
    for e in ents:
        if e.text not in pool[kind(e)]:
            continue
        options = [x for x in pool[kind(e)] if x != e.text and x not in s]
        if options:
            return s[:e.start_char] + rng.choice(options) + s[e.end_char:]
    return None


def paraphrase(s):
    first = s.split(" ", 1)[0].strip(",.;:")
    body = s[0].lower() + s[1:] if first in LOWER_OK else s
    return rng.choice(PREFIXES) + body


sents = [l.strip() for l in open(args.sentences, encoding="utf-8") if l.strip()]
before = len(sents)
sents = [s for s in sents if is_clean(s)]
print(f"clean sentences: {len(sents)} of {before}")
rng.shuffle(sents)
half = len(sents) // 2
splits = {"dev": sents[:half], "test": sents[half:]}
KINDS = ["date", "number", "name", "negation"]

for name, ss in splits.items():
    ss = ss[:args.max_sents]
    docs = list(nlp.pipe(ss, batch_size=256))
    seen = defaultdict(Counter)
    for d in docs:
        for e in d.ents:
            if good_ent(e):
                seen[e.text][kind(e)] += 1
    pool = defaultdict(list)
    for text, labs in seen.items():
        lab, c = labs.most_common(1)[0]
        if c >= args.min_count and c / sum(labs.values()) >= 0.8:
            pool[lab].append(text)
    for k in pool:
        pool[k].sort()
    print(f"{name}: name pool sizes", {k: len(v) for k, v in pool.items()})

    pos_target = args.n // 2
    quota = {k: pos_target // len(KINDS) for k in KINDS}
    for k in KINDS[: pos_target - sum(quota.values())]:
        quota[k] += 1
    got = {k: 0 for k in KINDS}
    rows, i, used = [], 0, set()

    for s, doc in zip(ss, docs):
        if all(got[k] >= quota[k] for k in KINDS):
            break
        open_kinds = sorted([k for k in KINDS if got[k] < quota[k]], key=lambda k: got[k] / quota[k])
        for k in open_kinds:
            new = {"date": lambda: date_mut(s), "number": lambda: number_mut(s),
                   "name": lambda: name_mut(s, doc, pool), "negation": lambda: negation_mut(s)}[k]()
            if new and new != s:
                rows.append(dict(id=f"{name}-{i}", s1=s, s2=new, label=1, type=k)); i += 1
                got[k] += 1; used.add(s)
                break
    short = {k: quota[k] - got[k] for k in KINDS if got[k] < quota[k]}
    if short:
        print(f"WARNING {name}: not enough sentences for {short}; raise --max-sents or fetch more articles")

    rest = [s for s in ss if s not in used]
    for j in range(args.n - pos_target):
        s = rng.choice(rest)
        if j % 2 == 0:
            rows.append(dict(id=f"{name}-{i}", s1=s, s2=paraphrase(s), label=0, type="paraphrase"))
        else:
            rows.append(dict(id=f"{name}-{i}", s1=s, s2=rng.choice(rest), label=0, type="unrelated"))
        i += 1
    rng.shuffle(rows)
    write_jsonl(DATA / f"synthetic_{name}.jsonl", rows)
    print(name, len(rows), dict(Counter(r["type"] for r in rows)))