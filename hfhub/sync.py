"""Drive reconcile/apply for each configured view."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

from hfhub import cache, config as cfg, ollama_registry as reg, state as st
from hfhub.views.base import Plan, View, apply, reconcile
from hfhub.views.lmstudio import LmStudioView
from hfhub.views.ollama import OllamaView
from hfhub.xfer import resolve_cache_dir

Out = Callable[[str], None]


class ViewSkipped(Exception):
    """This view cannot be synced right now; the message is the notice to print."""


class MissingRoot(ViewSkipped):
    pass


class MassRemoval(ViewSkipped):
    """Applying the plan would empty the view of everything sync owns there."""


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


def _check_root(view: View) -> None:
    """A view root the user has not created is not ours to create: sync skips it.

    Creating it would turn a typo, or a drive that failed to mount, into an empty
    view that the next run happily fills - or, worse, a view whose real contents
    are elsewhere and whose state file is therefore missing.
    """
    if not view.root.is_dir():
        raise MissingRoot(f"[{view.name}] skipped: root does not exist: {view.root}")


def _mass_removal(plan: Plan, state: st.State) -> int | None:
    """The number of owned entries when the plan removes every single one, else None."""
    if not state.owned:
        return None
    removed = {a.key for a in plan.actions if a.kind in ("prune", "tombstone")}
    return len(state.owned) if set(state.owned) <= removed else None


def sync_view(view: View, entries: list[cache.GgufEntry], execute: bool,
              allow_mass_removal: bool = False) -> tuple[Plan, st.State]:
    _check_root(view)
    state = st.load(view.root)
    desired = view.desired(entries)
    presence = {k: view.present(d) for k, d in desired.items()}
    plan = reconcile(desired, presence, state)
    if execute:
        n = _mass_removal(plan, state)
        if n is not None and not allow_mass_removal:
            raise MassRemoval(f"[{view.name}] refused: plan would remove every owned entry ({n}); "
                              f"pass --allow-mass-removal if this is intended")
        state = apply(plan, view, state, execute=True, desired=desired)
        st.save(view.root, state)
    return plan, state


def _cache_present(out: Out) -> bool:
    """False, with a notice, when the HF cache directory is not there at all.

    An absent cache scans as zero entries, which is indistinguishable from a cache
    the user emptied: every owned entry would be pruned out of every view.
    """
    hub = hub_dir()
    if not hub.is_dir():
        out(f"[cache] not found: {hub}; nothing done")
        return False
    return True


def _cached_as_missing(cache_file: Path) -> bool:
    """True when the cached registry response is the `missing` (404) marker."""
    try:
        data = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        return False
    return bool(isinstance(data, dict) and data.get("missing"))


def _abort(view: View, e: Exception, out: Out) -> None:
    detail = f"permission denied ({e})" if isinstance(e, PermissionError) else str(e)
    out(f"[{view.name}] aborted: {detail}")


def _load_state(view: View, out: Out) -> st.State | None:
    """The view's state, or None after reporting why this view has to be skipped.

    A corrupt state file or an unreadable root is a problem with one view, not
    with the run: every entry point reports it and moves on to the next view.
    """
    try:
        return st.load(view.root)
    except (st.StateError, PermissionError) as e:
        _abort(view, e, out)
        return None


def _report(view: View, plan: Plan, execute: bool, out: Out, mark: str = "") -> None:
    mode = "applied" if execute else "dry run"
    summary = ", ".join(f"{k}={v}" for k, v in sorted(plan.summary().items()))
    out(f"[{view.name}] {mode}: " + (summary or "nothing"))
    for a in plan.changes():
        lead = "* " if mark and a.key == mark else "  "
        out(f"{lead}{a.kind:<9} {a.key}" + (f"  ({a.note})" if a.note else ""))
    for w in getattr(view, "warnings", []):
        out(f"  warning   {w}")


def run(config: cfg.Config, view_names: list[str], execute: bool, offline: bool, out: Out = print,
        allow_mass_removal: bool = False) -> dict[str, Plan]:
    if not _cache_present(out):
        return {}
    entries = cache.scan(hub_dir())
    plans: dict[str, Plan] = {}
    for name in view_names:
        view = build_view(name, config, offline)
        if view is None:
            out(f"[{name}] skipped: no root configured in {config.path}")
            continue
        try:
            plan, _ = sync_view(view, entries, execute, allow_mass_removal)
        except (st.StateError, PermissionError) as e:
            _abort(view, e, out)
            continue
        except ViewSkipped as e:
            out(str(e))
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
        try:
            _check_root(view)
        except ViewSkipped as e:
            out(str(e))
            continue
        state = _load_state(view, out)
        if state is None:
            continue
        try:
            desired = view.desired(entries)
            foreign = view.foreign(state)
        except PermissionError as e:
            _abort(view, e, out)
            continue
        missing = [k for k in state.tombstones]
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
    if not _cache_present(out):
        return
    entries = cache.scan(hub_dir())
    for name in view_names:
        view = build_view(name, config)
        if view is None:
            continue
        try:
            _check_root(view)
        except ViewSkipped as e:
            out(str(e))
            continue
        state = _load_state(view, out)
        if state is None:
            continue
        try:
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
                if cache_file.is_file() and _cached_as_missing(cache_file):
                    cache_file.unlink()
            plan, _ = sync_view(view, entries, execute)
        except (st.StateError, PermissionError) as e:
            _abort(view, e, out)
            continue
        except ViewSkipped as e:
            out(str(e))
            continue
        # `view add` syncs the whole view, so the whole plan is reported; the
        # requested key's line is starred to separate it from the rest.
        _report(view, plan, execute, out, mark=key)


def view_remove(config: cfg.Config, key: str, view_names: list[str], execute: bool, out: Out = print) -> None:
    for name in view_names:
        view = build_view(name, config)
        if view is None:
            continue
        try:
            _check_root(view)
        except ViewSkipped as e:
            out(str(e))
            continue
        state = _load_state(view, out)
        if state is None:
            continue
        owned = state.owned.get(key)
        if owned is None:
            out(f"[{name}] {key} is not owned by sync")
            continue
        others = {p for k, o in state.owned.items() if k != key for p in o.paths}
        paths = [p for p in owned.paths if p not in others]
        out(f"[{name}] remove {key}: " + ", ".join(paths))
        if execute:
            try:
                view.remove(paths)
                del state.owned[key]
                state.tombstones[key] = datetime.now().replace(microsecond=0).isoformat()
                st.save(view.root, state)
            except PermissionError as e:
                _abort(view, e, out)
                continue
