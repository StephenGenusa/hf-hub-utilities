"""Maintenance operations on the HF hub cache itself: report, repair, dedupe, prune, thin.

Every operation takes ``execute`` and writes nothing unless it is True. They act
only on the cache; after a deleting operation, run ``hf-xfer sync --execute`` so
the views drop the removed entries.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from hfhub import cache

Out = Callable[[str], None]
GB = 1e9

_QUANT_BITS = re.compile(r"(?i)(?:^|[-._])(?:UD-|CD-)?(?:IQ|Q|TQ)(\d)")
_FULL_PREC = re.compile(r"(?i)(?:^|[-._])(BF16|F16|F32)$")


@dataclass
class Result:
    actions: int = 0
    bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self, verb: str, execute: bool) -> str:
        mode = "applied" if execute else "dry run"
        return f"[cache] {mode}: {verb} {self.actions} item(s), {self.bytes / GB:.1f} GB"


# ---------------------------------------------------------------- helpers
def _repo_id(repo_dir: Path) -> str:
    return cache.repo_id_from_folder(repo_dir.name)


def _is_sha(name: str) -> bool:
    return len(name) == 64 and all(c in "0123456789abcdef" for c in name)


def _blobs(repo_dir: Path) -> list[Path]:
    d = repo_dir / "blobs"
    return [b for b in d.iterdir() if _is_sha(b.name)] if d.is_dir() else []


def _links(repo_dir: Path):
    """Yield (snapshot_dir, link_path, blob_name) for every symlink under snapshots/."""
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return
    for snap in snaps.iterdir():
        if not snap.is_dir():
            continue
        for l in snap.rglob("*"):
            if l.is_symlink():
                yield snap, l, os.path.basename(os.readlink(l))


def _main_ref(repo_dir: Path) -> str | None:
    p = repo_dir / "refs" / "main"
    return p.read_text().strip() if p.is_file() else None


def _rel_target(path_in_snapshot: str, sha: str) -> str:
    depth = len(Path(path_in_snapshot).parts)
    return os.path.join(*([".."] * (1 + depth)), "blobs", sha)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _rmdir_empty(d: Path) -> None:
    if d.is_dir() and not any(d.rglob("*")):
        shutil.rmtree(d)


def quant_bits(relpath: str) -> int | None:
    """Bits per weight implied by a GGUF filename: 16 for BF16/F16/F32, else the quant digit."""
    stem = os.path.basename(relpath)[:-len(".gguf")] if relpath.lower().endswith(".gguf") else relpath
    if _FULL_PREC.search(stem):
        return 16
    m = _QUANT_BITS.search(stem)
    return int(m.group(1)) if m else None


def parse_min_quant(text: str) -> int:
    """'IQ4_XS' -> 4, 'Q5_K_M' -> 5, '8' -> 8."""
    if text.isdigit():
        return int(text)
    b = quant_bits(text)
    if b is None:
        raise ValueError(f"cannot read a bit width from {text!r}")
    return b


def parse_size(text: str) -> int:
    """'23G' / '23GB' / '500M' / plain bytes -> bytes."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?)B?\s*", text, re.I)
    if not m:
        raise ValueError(f"cannot parse size {text!r}")
    mult = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def is_draft(entry: cache.GgufEntry) -> bool:
    """MTP draft heads: not standalone models, never counted as quants."""
    parts = entry.relpath.split("/")
    return "MTP" in parts[:-1] or entry.basename.lower().startswith("mtp-")


# ---------------------------------------------------------------- hub lookup
def hub_tree(repo_id: str) -> tuple[str, dict[str, tuple[str, int]]]:
    """Current commit plus {sha256: (path, size)} for every LFS file in the repo."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type="model", files_metadata=True)
    by_sha: dict[str, tuple[str, int]] = {}
    for s in info.siblings or []:
        sha = s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
        if sha:
            by_sha[sha] = (s.rfilename, s.size or 0)
    return info.sha, by_sha


# ---------------------------------------------------------------- report
def report(hub: Path, out: Out = print) -> None:
    sup, xrepo_bytes, unlinked, partial, multi = 0, 0, 0, 0, []
    seen_inodes: dict[str, set[int]] = defaultdict(set)
    for repo in sorted(hub.glob("models--*")):
        blobs = _blobs(repo)
        if not blobs and not (repo / "blobs").is_dir():
            continue
        linked_by: dict[str, set[str]] = defaultdict(set)
        for snap, _, b in _links(repo):
            linked_by[b].add(snap.name)
        ref = _main_ref(repo)
        main_paths = {}
        if ref and (repo / "snapshots" / ref).is_dir():
            main_paths = {str(l.relative_to(repo / "snapshots" / ref)): b for s, l, b in _links(repo) if s.name == ref}
        for b in blobs:
            sz = b.stat().st_size
            seen_inodes[b.name].add(b.stat().st_ino)
            if len(seen_inodes[b.name]) > 1:
                xrepo_bytes += sz
            if b.name not in linked_by:
                unlinked += sz
            elif ref and ref not in linked_by[b.name]:
                # superseded only if some path in main now points at a different blob
                for s, l, bb in _links(repo):
                    if bb == b.name and str(l.relative_to(s)) in main_paths and main_paths[str(l.relative_to(s))] != b.name:
                        sup += sz
                        break
        for p in (repo / "blobs").glob("*.incomplete"):
            partial += p.stat().st_size
    entries = cache.scan(hub)
    by_repo: dict[str, list[cache.GgufEntry]] = defaultdict(list)
    for e in entries:
        if not e.is_mmproj and not e.is_shard and not is_draft(e):
            by_repo[e.repo_id].append(e)
    for r, es in by_repo.items():
        if len(es) >= 4:
            multi.append((sum(e.size for e in es), len(es), r))
    out(f"superseded revisions (prune-superseded): {sup / GB:.1f} GB")
    out(f"identical blobs across repos (dedupe):   {xrepo_bytes / GB:.1f} GB")
    out(f"unlinked blobs (repair):                 {unlinked / GB:.1f} GB")
    out(f"partial downloads (*.incomplete):        {partial / GB:.1f} GB")
    out("repos with 4+ quants of one model (thin):")
    for sz, n, r in sorted(multi, reverse=True):
        out(f"  {sz / GB:7.1f} GB  {n:2d} quants  {r}")


# ---------------------------------------------------------------- repair
def repair(hub: Path, execute: bool, out: Out = print,
           tree_fn: Callable[[str], tuple[str, dict[str, tuple[str, int]]]] = hub_tree) -> Result:
    """Recreate snapshot links for repos whose blobs are present but unreferenced."""
    res = Result()
    for repo in sorted(hub.glob("models--*")):
        blobs = _blobs(repo)
        if not blobs:
            continue
        linked = {b for _, _, b in _links(repo)}
        orphans = [b for b in blobs if b.name not in linked]
        if not orphans:
            continue
        rid = _repo_id(repo)
        try:
            commit, by_sha = tree_fn(rid)
        except Exception as e:  # network / 404 / gated
            res.notes.append(f"{rid}: Hub lookup failed ({e}); left alone")
            continue
        snap = repo / "snapshots" / commit
        made = 0
        for b in orphans:
            hit = by_sha.get(b.name)
            if not hit or hit[1] != b.stat().st_size:
                res.notes.append(f"{rid}: blob {b.name[:12]} ({b.stat().st_size / GB:.1f} GB) is not in the current revision; left alone")
                continue
            path = hit[0]
            link = snap / path
            if link.is_symlink() or link.exists():
                continue
            made += 1
            res.bytes += b.stat().st_size
            if execute:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(_rel_target(path, b.name))
        if made:
            res.actions += made
            out(f"  {'relinked' if execute else 'would relink'} {made} file(s) in {rid} at revision {commit[:8]}")
            if execute:
                (repo / "refs").mkdir(exist_ok=True)
                if _main_ref(repo) is None or not (repo / "snapshots" / _main_ref(repo)).is_dir():
                    (repo / "refs" / "main").write_text(commit)
                for d in (repo / "snapshots").iterdir():
                    if d != snap:
                        _rmdir_empty(d)
    for n in res.notes:
        out("  note: " + n)
    out(res.summary("relink", execute))
    return res


# ---------------------------------------------------------------- dedupe
def dedupe(hub: Path, execute: bool, out: Out = print) -> Result:
    """Hardlink identical blobs across repos; turn plain-file copies in snapshots into links."""
    res = Result()
    owners: dict[str, list[Path]] = defaultdict(list)
    for repo in hub.glob("models--*"):
        for b in _blobs(repo):
            owners[b.name].append(b)
    for sha, paths in owners.items():
        paths.sort(key=lambda p: (-p.stat().st_nlink, str(p)))
        canon = paths[0]
        st = canon.stat()
        for dup in paths[1:]:
            d = dup.stat()
            if d.st_ino == st.st_ino or d.st_dev != st.st_dev or d.st_size != st.st_size:
                continue
            res.actions += 1
            res.bytes += d.st_size
            if execute:
                tmp = dup.with_name(dup.name + ".hl")
                os.link(canon, tmp)
                os.replace(tmp, dup)
    # plain files inside snapshot folders
    for repo in sorted(hub.glob("models--*")):
        snaps = repo / "snapshots"
        if not snaps.is_dir():
            continue
        for f in snaps.rglob("*"):
            if f.is_symlink() or not f.is_file() or f.stat().st_size < 1_000_000:
                continue  # small config/json files are legitimately plain in some layouts
            sha = file_sha256(f)
            blob = repo / "blobs" / sha
            rel = str(f.relative_to(snaps / f.relative_to(snaps).parts[0]))
            res.actions += 1
            if blob.exists():
                res.bytes += f.stat().st_size
                out(f"  {'replaced' if execute else 'would replace'} copy with link: {_repo_id(repo)}:{rel}")
                if execute:
                    f.unlink()
                    f.symlink_to(_rel_target(rel, sha))
            else:
                out(f"  {'moved' if execute else 'would move'} copy into blobs: {_repo_id(repo)}:{rel}")
                if execute:
                    blob.parent.mkdir(exist_ok=True)
                    shutil.move(str(f), str(blob))
                    f.symlink_to(_rel_target(rel, sha))
    out(res.summary("dedupe", execute))
    return res


# ---------------------------------------------------------------- prune superseded
def prune_superseded(hub: Path, execute: bool, out: Out = print) -> Result:
    """Delete blobs that an older snapshot links where main now links a different blob at the same path."""
    res = Result()
    for repo in sorted(hub.glob("models--*")):
        ref = _main_ref(repo)
        snaps = repo / "snapshots"
        if not ref or not (snaps / ref).is_dir():
            continue
        main = {str(l.relative_to(snaps / ref)): b for s, l, b in _links(repo) if s.name == ref}
        main_blobs = set(main.values())
        victims: dict[str, list[Path]] = defaultdict(list)
        for s, l, b in _links(repo):
            if s.name == ref or b in main_blobs:
                continue
            rel = str(l.relative_to(s))
            if rel in main:
                victims[b].append(l)
        for b, links in victims.items():
            blob = repo / "blobs" / b
            sz = blob.stat().st_size if blob.exists() else 0
            res.actions += 1
            res.bytes += sz
            rel = str(links[0].relative_to(snaps / links[0].relative_to(snaps).parts[0]))
            out(f"  {'deleted' if execute else 'would delete'} {sz / GB:5.1f} GB  {_repo_id(repo)}:{rel}")
            if execute:
                for l in links:
                    l.unlink()
                if blob.exists():
                    blob.unlink()
        if execute:
            for d in snaps.iterdir():
                if d.name != ref:
                    _rmdir_empty(d)
    out(res.summary("prune", execute))
    if execute and res.actions:
        out("run `hf-xfer sync --execute` to prune the views")
    return res


# ---------------------------------------------------------------- thin
def thin(hub: Path, execute: bool, min_bits: int | None = None, max_size: int | None = None,
         repos: list[str] | None = None, allow_empty: bool = False, out: Out = print) -> Result:
    """Delete quants below ``min_bits`` or larger than ``max_size`` in repos that keep at least one quant."""
    res = Result()
    entries = cache.scan(hub)
    by_repo: dict[str, list[cache.GgufEntry]] = defaultdict(list)
    for e in entries:
        if e.is_mmproj or e.is_shard or is_draft(e):
            continue
        if repos and e.repo_id not in repos:
            continue
        by_repo[e.repo_id].append(e)
    for rid, es in sorted(by_repo.items()):
        victims = []
        for e in es:
            bits = quant_bits(e.relpath)
            too_small = min_bits is not None and bits is not None and bits < min_bits
            too_big = max_size is not None and e.size > max_size
            if too_small or too_big:
                victims.append(e)
        if not victims:
            continue
        if len(victims) == len(es) and not allow_empty:
            out(f"  skipped {rid}: every quant matches the rule; pass --allow-empty to remove the model entirely")
            continue
        for e in sorted(victims, key=lambda e: -e.size):
            res.actions += 1
            res.bytes += e.size
            out(f"  {'deleted' if execute else 'would delete'} {e.size / GB:5.1f} GB  {e.key}")
            if execute:
                repo = hub / cache.repo_folder(rid)
                for _, l, b in list(_links(repo)):
                    if b == e.sha256:
                        l.unlink()
                if e.blob.exists():
                    e.blob.unlink()
    out(res.summary("thin", execute))
    if execute and res.actions:
        out("run `hf-xfer sync --execute` to prune the views")
    return res
