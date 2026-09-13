"""Read-only scan of the HF hub cache for GGUF files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from hfhub.hfu import _SHARD_SUFFIX, is_mmproj, pick_mmproj

_HEX = set("0123456789abcdef")


@dataclass(frozen=True)
class GgufEntry:
    repo_id: str
    relpath: str
    blob: Path
    sha256: str
    size: int
    is_mmproj: bool
    is_current: bool
    is_shard: bool

    @property
    def key(self) -> str:
        return f"{self.repo_id}:{self.relpath}"

    @property
    def basename(self) -> str:
        return os.path.basename(self.relpath)


def repo_folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def repo_id_from_folder(name: str) -> str:
    return name[len("models--"):].replace("--", "/")


def _is_sha(name: str) -> bool:
    return len(name) == 64 and set(name) <= _HEX


def _snapshots_in_order(repo_root: Path) -> list[tuple[Path, bool]]:
    """(snapshot dir, is_main) with the refs/main snapshot first, then newest first."""
    snaps = repo_root / "snapshots"
    if not snaps.is_dir():
        return []
    main = None
    ref = repo_root / "refs" / "main"
    if ref.is_file():
        main = ref.read_text().strip()
    dirs = sorted((d for d in snaps.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    ordered = [(d, d.name == main) for d in dirs]
    ordered.sort(key=lambda t: not t[1])
    return ordered


def scan(hub: Path) -> list[GgufEntry]:
    out: list[GgufEntry] = []
    for repo_root in sorted(hub.glob("models--*")):
        repo_id = repo_id_from_folder(repo_root.name)
        seen: set[str] = set()
        for snap, is_main in _snapshots_in_order(repo_root):
            for link in sorted(snap.rglob("*.gguf")):
                rel = str(link.relative_to(snap))
                if rel in seen or not link.is_symlink():
                    continue
                blob = Path(os.path.realpath(link))
                if not _is_sha(blob.name) or not blob.is_file():
                    continue
                seen.add(rel)
                out.append(GgufEntry(
                    repo_id=repo_id, relpath=rel, blob=blob, sha256=blob.name,
                    size=blob.stat().st_size, is_mmproj=is_mmproj(rel), is_current=is_main,
                    is_shard=bool(_SHARD_SUFFIX.search(os.path.basename(rel)[:-len(".gguf")])),
                ))
    return out


def unlinked(hub: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for repo_root in sorted(hub.glob("models--*")):
        blobs = repo_root / "blobs"
        if not blobs.is_dir():
            continue
        linked = set()
        for snap, _ in _snapshots_in_order(repo_root):
            for link in snap.rglob("*"):
                if link.is_symlink():
                    linked.add(os.path.realpath(link))
        for b in sorted(blobs.iterdir()):
            if _is_sha(b.name) and str(b.resolve()) not in linked:
                out.append((repo_id_from_folder(repo_root.name), b))
    return out


def weights(entries: list[GgufEntry]) -> list[GgufEntry]:
    return [e for e in entries if not e.is_mmproj]


def mmproj_for(entry: GgufEntry, entries: list[GgufEntry]) -> GgufEntry | None:
    candidates = {e.relpath: e for e in entries if e.repo_id == entry.repo_id and e.is_mmproj}
    if not candidates:
        return None
    chosen = pick_mmproj(list(candidates), None)
    return candidates[chosen[0]] if chosen else None
