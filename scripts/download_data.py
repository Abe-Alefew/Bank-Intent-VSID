"""Download Banking77 (real customer-support queries) into ./data  (public PolyAI GitHub repo)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vsid.config import Config
from vsid.data import load_banking77
tr, te, classes = load_banking77(Config().data_dir)
print(f"train={len(tr)}  test={len(te)}  intents={len(classes)}")
print(tr.sample(5, random_state=0).to_string())
