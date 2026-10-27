from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, CONFIG, RESULTS, CACHE = ROOT / "data", ROOT / "config", ROOT / "results", ROOT / "cache"
for folder in (DATA, CONFIG, RESULTS, CACHE):
    folder.mkdir(exist_ok=True)
