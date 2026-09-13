"""Adopt a foreign (LM Studio real file / Ollama registry model) into the HF cache."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hfhub import cache, config as cfg, state as st, sync
from hfhub.ollama_registry import MT_MODEL, MT_PROJECTOR
from hfhub.views.base import ForeignItem
from hfhub.views.ollama import REGISTRY_CACHE_DIR, blob_path
from hfhub.xfer import RepoMap, _place_file, _rel_symlink_target, map_from_hub

Out = Callable[[str], None]


@dataclass
class AdoptPlan:
    repo_id: str
    relpath: str
    commit: str
    etag: str
    hub_backed: bool
    note: str = ""


def hub_lookup(repo_id: str) -> RepoMap | None:
    try:
        return map_from_hub(repo_id, "model", "main")
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _foreign_filename(item: ForeignItem) -> str:
    if item.key.startswith("ollama:"):
        name_tag = item.key.split("/")[-1]           # "llama3.1:8b"
        name, _, tag = name_tag.partition(":")
        return f"{name}-{tag or 'latest'}.gguf"
    return os.path.basename(item.path)


def _ollama_alias(key: str) -> str:
    """The Ollama name the key was pulled under: `ollama:<registry>/<ns>/<name>:<tag>`.

    The namespace has to survive: dropping it turns `myorg/mymodel:8b` into
    `library/mymodel/8b` on disk, which is a different model as far as Ollama is
    concerned, and the name the user has been typing stops resolving.
    """
    parts = key.removeprefix("ollama:").split("/")[1:]   # drop the registry host
    name, _, tag = parts[-1].partition(":")
    ns = parts[:-1]
    full = "/".join(ns + [name]) if ns != ["library"] else name
    return f"{full}:{tag}" if tag else full


def plan_adopt(item: ForeignItem, repo_id: str, sha256: str,
               hub: Callable[[str], RepoMap | None] | None = None) -> AdoptPlan:
    """Where this file would land in the cache: Hub-verified when the Hub knows the hash.

    A hash the repo's current revision lists gives us the real filename and the
    real commit, so the entry is indistinguishable from a downloaded one. Anything
    else gets a revision synthesised from repo_id+hash: deterministic, so adopting
    the same bytes twice lands on the same snapshot instead of piling up revisions.
    """
    hub = hub or hub_lookup
    rmap = hub(repo_id)
    if rmap is not None and rmap.commit_hash:
        for rel, etag in rmap.etags.items():
            if etag == sha256:
                return AdoptPlan(repo_id, rel, rmap.commit_hash, sha256, True)
        note = f"hash not in {repo_id}@main; synthesised revision"
    else:
        note = f"{repo_id} not on the Hub; synthesised revision"
    commit = hashlib.sha256(f"{repo_id}:{sha256}".encode()).hexdigest()[:40]
    return AdoptPlan(repo_id, _foreign_filename(item), commit, sha256, False, note)


def execute_adopt(item: ForeignItem, plan: AdoptPlan, hub_dir: Path, move: bool) -> Path:
    """Put the file in the cache as blob + snapshot symlink (+ refs/main when free).

    refs/main is never moved off a revision that is already there: it may be the
    real Hub revision of a repo we are only adding one loose file to, and a
    redownload updates that snapshot, not ours. `cache.scan` walks every snapshot,
    so the adopted file is found either way - just not flagged `is_current`.
    """
    repo_root = hub_dir / cache.repo_folder(plan.repo_id)
    blob = repo_root / "blobs" / plan.etag
    _place_file(item.path, blob, "move" if move else "copy")
    snap = repo_root / "snapshots" / plan.commit / plan.relpath
    snap.parent.mkdir(parents=True, exist_ok=True)
    target = _rel_symlink_target(blob, snap)
    if not (snap.is_symlink() and os.readlink(snap) == target):
        if snap.is_symlink() or snap.exists():
            snap.unlink()
        snap.symlink_to(target)
    (repo_root / "refs").mkdir(exist_ok=True)
    main = repo_root / "refs" / "main"
    if not main.is_file() or main.read_text().strip() == plan.commit:
        main.write_text(plan.commit)
    return snap


def find_foreign(config: cfg.Config, view_names: list[str], out: Out = print) -> list[tuple[str, ForeignItem]]:
    items: list[tuple[str, ForeignItem]] = []
    for name in view_names:
        view = sync.build_view(name, config, offline=True)
        if view is None:
            continue
        state = sync._load_state(view, out)
        if state is None:          # corrupt state file: reported, this view is skipped
            continue
        for f in view.foreign(state):
            items.append((name, f))
    return items


def _config_digest(view_root: Path, item: ForeignItem) -> str | None:
    """The digest of the foreign manifest's config blob, read from the manifest."""
    try:
        m = json.loads((view_root / item.extra["manifest"]).read_text())
        return m["config"]["digest"].removeprefix("sha256:")
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return None


def _config_bytes(view_root: Path, item: ForeignItem) -> bytes:
    """The foreign manifest's config blob, so create() need not fetch one."""
    digest = _config_digest(view_root, item)
    if digest is None:
        return b"{}"
    try:
        return (view_root / blob_path(digest)).read_bytes()
    except OSError:
        return b"{}"


def _seed_registry_cache(view_root: Path, plan: AdoptPlan, item: ForeignItem) -> None:
    """Keep the foreign manifest's template/params/license layers for the new hf.co manifest.

    Both halves of the view's registry cache are written: without the config blob
    the next sync would go to huggingface.co for a repo that is usually not there,
    and the entry would be skipped instead of built from what Ollama already had.
    A model with nothing but a model layer gets the `missing` marker instead, which
    sends create() straight to the synthesised manifest - still no network.
    """
    layers = [l for l in item.extra.get("layers", []) if l["mediaType"] not in (MT_MODEL, MT_PROJECTOR)]
    fake = {"schemaVersion": 2, "config": {"digest": "sha256:" + "0" * 64, "size": 0},
            "layers": [{"digest": "sha256:" + plan.etag, "mediaType": MT_MODEL, "size": 0}] + layers}
    base = view_root / REGISTRY_CACHE_DIR / plan.repo_id
    base.mkdir(parents=True, exist_ok=True)
    name = os.path.basename(plan.relpath)       # matches OllamaView._registry_cache
    (base / (name + ".json")).write_text(json.dumps(fake if layers else {"missing": True}))
    (base / (name + ".config.json")).write_bytes(_config_bytes(view_root, item))


def run(config: cfg.Config, key: str, repo_id: str, view_names: list[str], move: bool, execute: bool,
        out: Out = print) -> None:
    match = [(n, f) for n, f in find_foreign(config, view_names, out) if f.key == key]
    if not match:
        out(f"no foreign item with key {key}")
        return
    view_name, item = match[0]
    if item.path.name.startswith("sha256-"):
        sha = item.path.name[len("sha256-"):]
    else:
        out(f"hashing {item.path} ...")
        sha = file_sha256(item.path)
    plan = plan_adopt(item, repo_id, sha)
    out(f"adopt {key} -> {plan.repo_id}:{plan.relpath} ({'hub-backed' if plan.hub_backed else plan.note})")
    if not execute:
        return
    hub = sync.hub_dir()
    view = sync.build_view(view_name, config)
    # Move the file first: everything after it is repairable by re-running, while
    # a config alias or a rewritten manifest pointing at a file we never adopted
    # is not. Nothing the foreign side already has is destroyed here - the old
    # Ollama manifest stays put and is overwritten in place by the sync below,
    # which is what makes it an owned alias rather than a leftover.
    execute_adopt(item, plan, hub, move)
    main = hub / cache.repo_folder(plan.repo_id) / "refs" / "main"
    current = main.read_text().strip() if main.is_file() else plan.commit
    if current != plan.commit:
        out(f"refs/main left at {current} (real Hub revision); "
            f"adopted file lives in snapshot {plan.commit}")
    if view_name == "ollama":
        _seed_registry_cache(view.root, plan, item)
        config.views["ollama"].aliases[_ollama_alias(key)] = f"{plan.repo_id}:{plan.relpath}"
        cfg.save(config)
        view = sync.build_view(view_name, config)   # pick up the new alias
    entries = cache.scan(hub)
    plan_, _ = sync.sync_view(view, entries, execute=True)
    out(f"[{view_name}] synced: " + ", ".join(f"{k}={v}" for k, v in sorted(plan_.summary().items())))
    for w in getattr(view, "warnings", []):
        out(f"  warning   {w}")
    if view_name == "lmstudio" and not move:
        out(f"  {item.path} was copied, not moved: it stays a real file and keeps showing up "
            f"as foreign until you remove it or re-adopt with --move")


def remove_foreign(config: cfg.Config, key: str, view_names: list[str], execute: bool, out: Out = print) -> None:
    """Delete a foreign item. The one path that removes something we did not create."""
    match = [(n, f) for n, f in find_foreign(config, view_names, out) if f.key == key]
    if not match:
        out(f"no foreign item with key {key}")
        return
    view_name, item = match[0]
    view = sync.build_view(view_name, config, offline=True)
    targets = [item.path]
    if view_name == "ollama":
        manifest = view.root / item.extra["manifest"]
        targets = [manifest]
        referenced = view._referenced_digests(exclude=manifest)
        digests = [l["digest"].removeprefix("sha256:") for l in item.extra.get("layers", [])]
        config_digest = _config_digest(view.root, item)
        if config_digest is not None:
            digests.append(config_digest)
        for digest in dict.fromkeys(digests):
            if digest not in referenced:
                targets.append(view.root / blob_path(digest))
    for t in targets:
        out(f"delete {t}")
    if not execute:
        return
    from hfhub.views.lmstudio import remove_empty_parents
    for t in targets:
        if t.is_symlink() or t.exists():
            t.unlink()
        remove_empty_parents(view.root, t)
