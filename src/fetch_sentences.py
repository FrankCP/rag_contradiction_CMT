"""Write sentences.txt from English Wikipedia."""
import re, sys, argparse
from datasets import load_dataset
from paths import DATA

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=DATA / "sentences.txt")
parser.add_argument("--articles", type=int, default=3000)
parser.add_argument("--config", default="20231101.en")
args = parser.parse_args()

dataset = load_dataset("wikimedia/wikipedia", args.config, split="train", streaming=True)
kept = 0
with open(args.out, "w", encoding="utf-8") as out_file:
    for i, row in enumerate(dataset):
        if i >= args.articles:
            break
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", row["text"][:6000].replace("\n", " ")):
            n_words = len(sentence.split())
            if 10 <= n_words <= 40 and (re.search(r"\d", sentence) or len(re.findall(r"\b[A-Z][a-z]+", sentence)) >= 3):
                out_file.write(sentence.strip() + "\n")
                kept += 1
print("sentences kept:", kept)
