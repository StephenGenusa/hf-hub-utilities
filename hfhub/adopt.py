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
    """Put the file in the cache as blob + snapshot symlink + refs/main."""
    repo_root = hub_dir / cache.repo_folder(plan.repo_id)
    blob = repo_root / "blobs" / plan.etag
    _place_file(item.path, blob, "move" if move else "copy")
    snap = repo_root / "snapshots" / plan.commit / plan.relpath
    snap.parent.mkdir(parents=True, exist_ok=True)
    if not snap.is_symlink():
        snap.symlink_to(_rel_symlink_target(blob, snap))
    (repo_root / "refs").mkdir(exist_ok=True)
    (repo_root / "refs" / "main").write_text(plan.commit)
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


def _config_bytes(view_root: Path, item: ForeignItem) -> bytes:
    """The foreign manifest's config blob, so create() need not fetch one."""
    try:
        m = json.loads((view_root / item.extra["manifest"]).read_text())
        digest = m["config"]["digest"].removeprefix("sha256:")
    except (KeyError, OSError, TypeError, ValueError):
        return b"{}"
    p = view_root / blob_path(digest)
    try:
        return p.read_bytes()
    except OSError:
        return b"{}"


def _seed_registry_cache(view_root: Path, plan: AdoptPlan, item: ForeignItem) -> None:
    """Keep the foreign manifest's template/params/license layers for the new hf.co manifest.

    Both halves of the view's registry cache are written: without the config blob
    the next sync would go to huggingface.co for a repo that is usually not there,
    and the entry would be skipped instead of built from what Ollama already had.
    """
    layers = [l for l in item.extra.get("layers", []) if l["mediaType"] not in (MT_MODEL, MT_PROJECTOR)]
    if not layers:
        return
    fake = {"schemaVersion": 2, "config": {"digest": "sha256:" + "0" * 64, "size": 0},
            "layers": [{"digest": "sha256:" + plan.etag, "mediaType": MT_MODEL, "size": 0}] + layers}
    base = view_root / REGISTRY_CACHE_DIR / plan.repo_id
    base.mkdir(parents=True, exist_ok=True)
    (base / (plan.relpath + ".json")).write_text(json.dumps(fake))
    (base / (plan.relpath + ".config.json")).write_bytes(_config_bytes(view_root, item))


def run(config: cfg.Config, key: str, repo_id: str, view_names: list[str], move: bool, execute: bool,
        out: Out = print) -> None:
    match = [(n, f) for n, f in find_foreign(config, view_names, out) if f.key == key]
    if not match:
        out(f"no foreign item with key {key}")
        return
    view_name, item = match[0]
    sha = item.path.name[len("sha256-"):] if item.path.name.startswith("sha256-") else file_sha256(item.path)
    plan = plan_adopt(item, repo_id, sha)
    out(f"adopt {key} -> {plan.repo_id}:{plan.relpath} ({'hub-backed' if plan.hub_backed else plan.note})")
    if not execute:
        return
    hub = sync.hub_dir()
    view = sync.build_view(view_name, config)
    if view_name == "ollama":
        # Seed the registry cache before the old manifest goes: it is the only
        # record of the template/params layers this model was published with.
        _seed_registry_cache(view.root, plan, item)
        name_tag = key.split("/")[-1]
        config.views["ollama"].aliases[name_tag] = f"{plan.repo_id}:{plan.relpath}"
        cfg.save(config)
        view = sync.build_view(view_name, config)   # pick up the new alias
        old_manifest = view.root / item.extra["manifest"]
        if old_manifest.exists():
            old_manifest.unlink()
    execute_adopt(item, plan, hub, move)
    entries = cache.scan(hub)
    plan_, _ = sync.sync_view(view, entries, execute=True)
    out(f"[{view_name}] synced: " + ", ".join(f"{k}={v}" for k, v in sorted(plan_.summary().items())))


def _referenced_elsewhere(view_root: Path, manifest: Path) -> set[str]:
    """Every digest any manifest other than `manifest` still points at."""
    referenced: set[str] = set()
    mroot = view_root / "manifests"
    if not mroot.is_dir():
        return referenced
    for p in mroot.rglob("*"):
        if not p.is_file() or p == manifest:
            continue
        try:
            m = json.loads(p.read_text())
            digests = [l["digest"] for l in m["layers"]]
        except (OSError, TypeError, ValueError, KeyError):
            continue
        config = m.get("config") if isinstance(m, dict) else None
        if isinstance(config, dict) and isinstance(config.get("digest"), str):
            digests.append(config["digest"])
        referenced.update(d.removeprefix("sha256:") for d in digests if isinstance(d, str))
    return referenced


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
        referenced = _referenced_elsewhere(view.root, manifest)
        for l in item.extra.get("layers", []):
            digest = l["digest"].removeprefix("sha256:")
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
