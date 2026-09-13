"""Drive reconcile/apply for each configured view."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from hfhub import cache, config as cfg, ollama_registry as reg, state as st
from hfhub.views.base import Plan, Presence, View, apply, reconcile
from hfhub.views.lmstudio import LmStudioView
from hfhub.views.ollama import OllamaView
from hfhub.xfer import resolve_cache_dir

Out = Callable[[str], None]


def hub_dir() -> Path:
    path, _ = resolve_cache_dir(None)
    return Path(path)


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for p in (Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "token",):
        if p.is_file():
            return p.read_text().strip()
    return None


def build_view(name: str, config: cfg.Config, offline: bool = False) -> View | None:
    vc = config.views.get(name)
    if vc is None or vc.root is None:
        return None
    if name == "lmstudio":
        return LmStudioView(vc.root)
    if name == "ollama":
        return OllamaView(vc.root, aliases=vc.aliases, offline=offline, token=_token(),
                          fetch_manifest=reg.fetch_manifest, fetch_blob=reg.fetch_blob)
    raise ValueError(name)


def sync_view(view: View, entries: list[cache.GgufEntry], execute: bool) -> tuple[Plan, st.State]:
    state = st.load(view.root)
    desired = view.desired(entries)
    presence = {k: view.present(d) for k, d in desired.items()}
    plan = reconcile(desired, presence, state)
    if execute:
        view.root.mkdir(parents=True, exist_ok=True)
        state = apply(plan, view, state, execute=True, desired=desired)
        st.save(view.root, state)
    return plan, state


def _report(view: View, plan: Plan, execute: bool, out: Out) -> None:
    mode = "applied" if execute else "dry run"
    summary = ", ".join(f"{k}={v}" for k, v in sorted(plan.summary().items()))
    out(f"[{view.name}] {mode}: " + (summary or "nothing"))
    for a in plan.changes():
        out(f"  {a.kind:<9} {a.key}" + (f"  ({a.note})" if a.note else ""))
    for w in getattr(view, "warnings", []):
        out(f"  warning   {w}")


def run(config: cfg.Config, view_names: list[str], execute: bool, offline: bool, out: Out = print) -> dict[str, Plan]:
    entries = cache.scan(hub_dir())
    plans: dict[str, Plan] = {}
    for name in view_names:
        view = build_view(name, config, offline)
        if view is None:
            out(f"[{name}] skipped: no root configured in {config.path}")
            continue
        try:
            plan, _ = sync_view(view, entries, execute)
        except st.StateError as e:
            out(f"[{name}] aborted: {e}")
            continue
        except PermissionError as e:
            out(f"[{name}] aborted: permission denied ({e})")
            continue
        plans[name] = plan
        _report(view, plan, execute, out)
    return plans


def status(config: cfg.Config, view_names: list[str], out: Out = print) -> None:
    hub = hub_dir()
    entries = cache.scan(hub)
    for name in view_names:
        view = build_view(name, config, offline=True)
        if view is None:
            out(f"[{name}] no root configured")
            continue
        state = st.load(view.root)
        desired = view.desired(entries)
        missing = [k for k in state.tombstones]
        foreign = view.foreign(state)
        out(f"[{name}] root={view.root}")
        out(f"  owned: {len(state.owned)}   desired: {len(desired)}   tombstoned: {len(missing)}   foreign: {len(foreign)}")
        for k in missing:
            out(f"  tombstoned {k}")
        for f in foreign:
            out(f"  foreign    {f.key}  ({f.path})")
    unl = cache.unlinked(hub)
    if unl:
        out(f"[cache] unlinked blobs (no snapshot link): {len(unl)}")
        for repo_id, blob in unl:
            out(f"  {repo_id}  {blob.name[:12]}  {blob.stat().st_size / 1e9:.1f} GB")


def view_add(config: cfg.Config, key: str, view_names: list[str], execute: bool, out: Out = print) -> None:
    entries = cache.scan(hub_dir())
    for name in view_names:
        view = build_view(name, config)
        if view is None:
            continue
        state = st.load(view.root)
        if key in state.tombstones:
            out(f"[{name}] clearing tombstone for {key}")
            if execute:
                del state.tombstones[key]
                st.save(view.root, state)
        desired = view.desired(entries)
        if key not in desired:
            out(f"[{name}] {key} is not in the cache; nothing to add")
            continue
        # A cached 404 would keep us on the synthesised manifest forever; an
        # explicit `view add` is the user asking for a fresh registry lookup.
        if execute and hasattr(view, "_registry_cache"):
            cache_file = view._registry_cache(desired[key])
            if cache_file.is_file() and "missing" in cache_file.read_text():
                cache_file.unlink()
        plan, _ = sync_view(view, entries, execute)
        _report(view, Plan([a for a in plan.actions if a.key == key]), execute, out)


def view_remove(config: cfg.Config, key: str, view_names: list[str], execute: bool, out: Out = print) -> None:
    for name in view_names:
        view = build_view(name, config)
        if view is None:
            continue
        state = st.load(view.root)
        owned = state.owned.get(key)
        if owned is None:
            out(f"[{name}] {key} is not owned by sync")
            continue
        others = {p for k, o in state.owned.items() if k != key for p in o.paths}
        paths = [p for p in owned.paths if p not in others]
        out(f"[{name}] remove {key}: " + ", ".join(paths))
        if execute:
            view.remove(paths)
            del state.owned[key]
            from datetime import datetime
            state.tombstones[key] = datetime.now().replace(microsecond=0).isoformat()
            st.save(view.root, state)
