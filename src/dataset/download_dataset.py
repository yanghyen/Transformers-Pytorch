"""데이터셋만 미리 다운로드 (학습 없이)."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from datasets import load_dataset
from config.config import get_config

if __name__ == "__main__":
    config = get_config()
    print(f"Downloading {config['datasource']} ({config['lang_src']}-{config['lang_tgt']})...")
    ds = load_dataset(config["datasource"], f"{config['lang_src']}-{config['lang_tgt']}", split="train")
    print(f"Done. Loaded {len(ds)} examples.")
