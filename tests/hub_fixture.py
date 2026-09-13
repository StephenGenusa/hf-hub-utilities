"""Build a fake HF hub cache in a temp dir."""
import hashlib
import os
from pathlib import Path


def folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def add_repo(hub: Path, repo_id: str, files: dict[str, bytes], commit: str | None = None, main: bool = True) -> str:
    commit = commit or hashlib.sha1(f"{repo_id}:{sorted(files)}".encode()).hexdigest()
    root = hub / folder(repo_id)
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snap = root / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        h = hashlib.sha256(content).hexdigest()
        blob = root / "blobs" / h
        if not blob.exists():
            blob.write_bytes(content)
        link = snap / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        depth = len(Path(rel).parts) - 1
        if not link.is_symlink():
            link.symlink_to(os.path.join(*([".."] * (2 + depth)), "blobs", h))
    if main:
        (root / "refs").mkdir(exist_ok=True)
        (root / "refs" / "main").write_text(commit)
    return commit
