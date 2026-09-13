"""Find foreign files that are probably the same model as a cache entry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hfhub import adopt, cache, config as cfg, sync
from hfhub.gguf_header import file_type_name, read_header

Out = Callable[[str], None]


@dataclass(frozen=True)
class Signature:
    architecture: str
    file_type: int
    params: int


@dataclass(frozen=True)
class Dupe:
    foreign_key: str
    cache_key: str
    reason: str


def signature(path: Path) -> Signature:
    h = read_header(path, max_array_items=1)
    ft = h.kv.get("general.file_type")
    return Signature(str(h.kv.get("general.architecture", "?")), int(ft) if isinstance(ft, int) else -1, h.param_count)


def find_dupes(foreign: list[tuple[str, Signature]], entries: list[tuple[str, Signature]], tolerance: float = 0.02) -> list[Dupe]:
    out: list[Dupe] = []
    for fk, fs in foreign:
        for ck, cs in entries:
            if fs.architecture != cs.architecture or fs.file_type != cs.file_type or cs.params == 0:
                continue
            if abs(fs.params - cs.params) / cs.params <= tolerance:
                out.append(Dupe(fk, ck, f"{fs.architecture}, {file_type_name(fs.file_type)}, "
                                        f"{fs.params / 1e9:.2f}B vs {cs.params / 1e9:.2f}B"))
    return out


def run(config: cfg.Config, view_names: list[str], out: Out = print) -> None:
    foreign = []
    for name, item in adopt.find_foreign(config, view_names, out=out):
        try:
            foreign.append((item.key, signature(item.path)))
        except (OSError, ValueError) as e:
            out(f"skip {item.key}: {e}")
    if not foreign:
        out("no foreign files")
        return
    entries = []
    for e in cache.weights(cache.scan(sync.hub_dir())):
        try:
            entries.append((e.key, signature(e.blob)))
        except (OSError, ValueError):
            pass
    found = find_dupes(foreign, entries)
    if not found:
        out("no likely duplicates")
    for d in found:
        out(f"{d.foreign_key}  ≈  {d.cache_key}  ({d.reason})")
    out("remove one with: hf-xfer view remove --foreign <foreign key> --execute")
