"""LM Studio view: <root>/<org>/<name>/<basename>.gguf symlinks into the HF cache."""
from __future__ import annotations

import os
from pathlib import Path

from hfhub.cache import GgufEntry, mmproj_for, weights
from hfhub.state import State
from hfhub.views.base import Desired, ForeignItem, Presence


def remove_empty_parents(root: Path, path: Path) -> None:
    """Remove empty directories from path's parent up to (not including) root."""
    d = path.parent
    while d != root and d.is_relative_to(root):
        try:
            d.rmdir()
        except OSError:
            return
        d = d.parent


class LmStudioView:
    name = "lmstudio"

    def __init__(self, root: Path):
        self.root = root
        self.warnings: list[str] = []

    def desired(self, entries: list[GgufEntry]) -> dict[str, Desired]:
        self.warnings = []
        out: dict[str, Desired] = {}
        taken: dict[str, GgufEntry] = {}
        for e in sorted(weights(entries), key=lambda e: (not e.is_current, e.relpath.count("/"), e.relpath)):
            rel = f"{e.repo_id}/{e.basename}"
            if rel in taken:
                self.warnings.append(f"{e.key}: basename collides with {taken[rel].key}; skipped")
                continue
            taken[rel] = e
            links = {rel: e.blob}
            mm = mmproj_for(e, entries)
            if mm is not None:
                links[f"{e.repo_id}/{mm.basename}"] = mm.blob
            out[e.key] = Desired(key=e.key, sha256=e.sha256, links=links)
        return out

    def _status(self, rel: str, target: Path) -> str:
        p = self.root / rel
        if p.is_symlink():
            return "ok" if os.readlink(p) == str(target) else "wrong"
        return "wrong" if p.exists() else "missing"

    def present(self, d: Desired) -> Presence:
        statuses = {self._status(rel, t) for rel, t in d.links.items()}
        if "wrong" in statuses:
            return Presence.WRONG
        return Presence.CORRECT if statuses == {"ok"} else Presence.ABSENT

    def create(self, d: Desired) -> list[str]:
        for rel, target in d.links.items():
            p = self.root / rel
            if p.is_symlink() and os.readlink(p) == str(target):
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.symlink_to(target)
        return list(d.links)

    def remove(self, paths: list[str]) -> None:
        for rel in paths:
            p = self.root / rel
            if p.is_symlink() or p.exists():
                p.unlink()
            remove_empty_parents(self.root, p)

    def foreign(self, state: State) -> list[ForeignItem]:
        owned = {p for o in state.owned.values() for p in o.paths}
        out: list[ForeignItem] = []
        if not self.root.is_dir():
            return out
        for p in sorted(self.root.rglob("*.gguf")):
            rel = str(p.relative_to(self.root))
            if rel in owned:
                continue
            if p.is_symlink() or p.is_file():
                out.append(ForeignItem(key=f"lmstudio:{rel}", path=p))
        return out
