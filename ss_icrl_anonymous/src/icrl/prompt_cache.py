# =============================================================
# File: src/icrl/prompt_cache.py
# =============================================================
"""
Very small in‑memory / on‑disk prompt cache so that the same
prompt+round combination will not be regenerated twice inside a
single run (useful when batch size >1 and dataset may contain
duplicate questions).

Interface:
    cache = PromptCache(dir_path)
    cached_text = cache.get(hash_key)
    cache.put(hash_key, prompt_str)
"""
from pathlib import Path
import json, hashlib
from typing import Optional

class PromptCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()  # fixed‑length name
        return self.root / f"{h}.json"

    def get(self, key: str) -> Optional[str]:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())["prompt"]
        return None

    def put(self, key: str, prompt: str):
        self._path(key).write_text(json.dumps({"prompt": prompt}, ensure_ascii=False))
