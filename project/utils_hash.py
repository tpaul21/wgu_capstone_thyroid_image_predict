# File: project/utils_hash.py
from pathlib import Path
import hashlib

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_hash_for(ckpt: Path) -> None:
    ckpt = Path(ckpt)
    (ckpt.parent / (ckpt.name + ".sha256")).write_text(sha256_file(ckpt))

def verify_hash(ckpt: Path) -> bool:
    ckpt = Path(ckpt)
    ref = ckpt.parent / (ckpt.name + ".sha256")
    return ref.exists() and ref.read_text().strip() == sha256_file(ckpt)
