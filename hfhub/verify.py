"""`hf-xfer verify`: check the cache and the views against each other. Read-only.

It looks at the filesystem rather than at sync's state file, so it also catches what
the state cannot know about: links broken by hand, blobs deleted under a view, store
space nothing uses any more. With --hub it also compares every cached GGUF with the
file the Hub serves now.

Problems are things that are wrong (a broken link, an Ollama manifest pointing at a
missing blob); notes are things worth knowing that need no repair (a ref left by a
--local-dir download, a file someone placed in a view by hand).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable

from hfhub import blobs as hfblobs, cache, config as cfg, sync
from hfhub.views.base import Presence
from hfhub.views.ollama import blob_path

Out = Callable[[str], None]
GB = 1e9


class Report:
    def __init__(self, out: Out):
        self.out = out
        self.problems = 0

    def problem(self, msg: str) -> None:
        self.problems += 1
        self.out(f"  problem  {msg}")

    def note(self, msg: str) -> None:
        self.out(f"  note     {msg}")


# ------------------------------------------------------------------ cache
def check_cache(hub: Path, r: Report) -> set[Path]:
    """Snapshot links, refs and the shared store. Returns every repo blob a snapshot links."""
    r.out(f"[cache] {hub}")
    linked: set[Path] = set()
    for repo in sorted(hub.glob("models--*")):
        rid = cache.repo_id_from_folder(repo.name)
        snaps = repo / "snapshots"
        has_snaps = snaps.is_dir() and any(snaps.iterdir())
        refs = repo / "refs"
        for ref in sorted(refs.rglob("*")) if refs.is_dir() else []:
            if not ref.is_file():
                continue
            commit = ref.read_text().strip()
            if (snaps / commit).is_dir():
                continue
            name = ref.relative_to(refs).as_posix()
            if has_snaps:
                r.problem(f"{rid}: refs/{name} -> {commit[:12]} has no snapshot")
            else:
                r.note(f"{rid}: refs/{name} only, no files (downloaded with --local-dir elsewhere?)")
        for p in sorted(snaps.rglob("*")) if snaps.is_dir() else []:
            if not p.is_symlink():
                continue
            if not p.exists():
                r.problem(f"{rid}: broken snapshot link {p.relative_to(repo).as_posix()}")
                continue
            linked.add(hfblobs.repo_blob(p))
        for inc in (repo / "blobs").glob("*.incomplete") if (repo / "blobs").is_dir() else []:
            r.note(f"{rid}: partial download {inc.name} ({inc.stat().st_size / GB:.1f} GB; "
                   f"finish or delete it)")
    leaked = hfblobs.unreferenced_store_blobs(hub)
    if leaked:
        r.problem(f"{len(leaked)} shared-store file(s), {sum(leaked.values()) / GB:.1f} GB, "
                  f"no repo uses (`hf-xfer cache gc --execute`)")
    return linked


# ------------------------------------------------------------------ views
def _into_hub(link: Path, hub: Path) -> Path | None:
    """The cache path a view link points at, or None when it points outside the cache."""
    where = Path(os.path.normpath(os.path.join(os.path.dirname(link), hfblobs.read_link(link))))
    for root in (Path(os.path.normpath(hub)), Path(os.path.realpath(hub))):
        if where.is_relative_to(root):
            return where
    return None


def check_lmstudio(view, hub: Path, linked: set[Path], r: Report) -> None:
    resolved_linked = {Path(os.path.realpath(b)) for b in linked}
    for p in sorted(view.root.rglob("*.gguf")):
        rel = p.relative_to(view.root).as_posix()
        if not p.is_symlink():
            r.note(f"{rel}: a real file, not a link into the cache (`hf-xfer view adopt` it)")
            continue
        target = _into_hub(p, hub)
        if target is None:
            r.note(f"{rel}: links outside the HF cache ({hfblobs.read_link(p)})")
        elif not p.exists():
            r.problem(f"{rel}: broken link -> {target} (`hf-xfer sync --execute`, then `hf-xfer view clean --execute`)")
        elif Path(os.path.realpath(p)) not in resolved_linked:
            r.problem(f"{rel}: links a cache file no snapshot uses any more")


def check_ollama(view, hub: Path, linked: set[Path], r: Report) -> None:
    referenced: set[str] = set()
    resolved_linked = {Path(os.path.realpath(x)) for x in linked}
    mroot = view.root / "manifests"
    manifests = sorted(p for p in mroot.rglob("*") if p.is_file()) if mroot.is_dir() else []
    for m in manifests:
        rel = m.relative_to(mroot).as_posix()
        try:
            data = json.loads(m.read_text())
            layers = list(data["layers"]) + ([data["config"]] if isinstance(data.get("config"), dict) else [])
        except (OSError, ValueError, KeyError, TypeError):
            r.problem(f"{rel}: unreadable manifest")
            continue
        for layer in layers:
            digest = str(layer.get("digest", "")).removeprefix("sha256:")
            if not digest:
                continue
            referenced.add(digest)
            b = view.root / blob_path(digest)
            if not (b.is_symlink() or b.exists()):
                r.problem(f"{rel}: blob {digest[:12]} missing")
            elif b.is_symlink():
                target = _into_hub(b, hub)
                if not b.exists():
                    r.problem(f"{rel}: blob {digest[:12]} is a broken link (`hf-xfer sync --execute`, then `hf-xfer view clean --execute`)")
                elif target is not None and target.name != digest:
                    r.problem(f"{rel}: blob {digest[:12]} links a file named {target.name[:12]} "
                              f"(written before the shared-store fix; `hf-xfer sync --execute` relinks it)")
                elif target is not None and Path(os.path.realpath(b)) not in resolved_linked:
                    r.problem(f"{rel}: blob {digest[:12]} links a cache file no snapshot uses any more")
                elif isinstance(layer.get("size"), int) and b.stat().st_size != layer["size"]:
                    r.problem(f"{rel}: blob {digest[:12]} is {b.stat().st_size} bytes, manifest says {layer['size']}")
    blobs_dir = view.root / "blobs"
    for b in sorted(blobs_dir.glob("sha256-*")) if blobs_dir.is_dir() else []:
        if b.name.removeprefix("sha256-") not in referenced:
            kind = "link" if b.is_symlink() else f"{b.stat().st_size / GB:.1f} GB file"
            r.problem(f"blob {b.name[7:19]} ({kind}) no manifest uses")
    r.out(f"  {len(manifests)} manifest(s)")


def check_pending(view, entries, r: Report) -> None:
    """What `hf-xfer sync --execute` would still change in this view."""
    state = sync._load_state(view, r.out)
    if state is None:
        r.problem("state file unreadable")
        return
    desired = view.desired(entries)
    for key, d in sorted(desired.items()):
        if key in state.tombstones:
            continue
        pres = view.present(d)
        if pres is Presence.CORRECT:
            continue
        owned = key in state.owned
        is_stale = getattr(view, "is_stale", None)
        if pres is Presence.WRONG and owned and is_stale is not None and is_stale(d, state.owned[key]):
            r.problem(f"{key}: links an older cache file (`hf-xfer sync --execute` relinks it)")
        elif pres is Presence.WRONG:
            r.problem(f"{key}: something else sits where its link belongs")
        elif pres is Presence.ABSENT and not owned:
            r.problem(f"{key}: not in the view yet (`hf-xfer sync --execute`)")
        elif pres is Presence.ABSENT:
            r.note(f"{key}: deleted outside sync; the next sync records a tombstone")
        else:
            r.problem(f"{key}: incomplete (`hf-xfer sync --execute` fills it in)")
    for key in sorted(set(state.owned) - set(desired)):
        r.problem(f"{key}: no longer in the cache (`hf-xfer sync --execute` removes it)")


# ------------------------------------------------------------------ hub
def check_hub(entries, r: Report, info_fn=None) -> None:
    """Every cached GGUF against the file the Hub's main revision serves now."""
    if info_fn is None:
        from huggingface_hub import HfApi
        api = HfApi()

        def info_fn(repo_id):
            return api.model_info(repo_id, files_metadata=True)
    r.out("[hub] current files on huggingface.co")
    by_repo: dict[str, list] = defaultdict(list)
    for e in entries:
        by_repo[e.repo_id].append(e)
    for repo_id, es in sorted(by_repo.items()):
        try:
            info = info_fn(repo_id)
        except Exception as e:  # 404, gated, network
            r.note(f"{repo_id}: no Hub lookup ({type(e).__name__})")
            continue
        hub_sha = {}
        for s in info.siblings or []:
            lfs = s.lfs
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if sha:
                hub_sha[s.rfilename] = sha
        lower = {k.lower(): k for k in hub_sha}
        for e in sorted(es, key=lambda e: e.relpath):
            want = hub_sha.get(e.relpath)
            if want is None:
                renamed = lower.get(e.relpath.lower())
                if renamed:
                    r.problem(f"{e.key}: renamed on the Hub to {renamed}")
                else:
                    r.note(f"{e.key}: no longer on the Hub")
            elif want != e.sha256:
                r.problem(f"{e.key}: outdated (local {e.sha256[:12]}, Hub {want[:12]})")


# ------------------------------------------------------------------ entry point
def run(config: cfg.Config, view_names: list[str], hub: bool = False, out: Out = print) -> int:
    """Print the report; return the number of problems found."""
    r = Report(out)
    hub_dir = sync.hub_dir()
    if not hub_dir.is_dir():
        out(f"[cache] not found: {hub_dir}")
        return 1
    linked = check_cache(hub_dir, r)
    entries = cache.scan(hub_dir)
    for name in view_names:
        view = sync.build_view(name, config, offline=True)
        if view is None:
            continue
        out(f"[{name}] {view.root}")
        if not view.root.is_dir():
            r.problem("root does not exist")
            continue
        if name == "lmstudio":
            check_lmstudio(view, hub_dir, linked, r)
        else:
            check_ollama(view, hub_dir, linked, r)
        check_pending(view, entries, r)
    if hub:
        check_hub(entries, r)
    out(f"\n{r.problems} problem(s)")
    return r.problems
