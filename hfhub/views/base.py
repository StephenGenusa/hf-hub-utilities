"""View protocol plus the pure reconcile() and the single writing path apply()."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from hfhub.cache import GgufEntry
from hfhub.state import Owned, State


class Presence(Enum):
    ABSENT = "absent"
    CORRECT = "correct"
    WRONG = "wrong"


class SkipEntry(Exception):
    """Raised by View.create when an entry cannot be created right now (e.g. offline)."""


@dataclass(frozen=True)
class Desired:
    key: str
    sha256: str
    links: dict[str, Path]          # view-relative path -> absolute symlink target
    extra: dict = field(default_factory=dict)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.links))


@dataclass(frozen=True)
class ForeignItem:
    key: str
    path: Path
    extra: dict = field(default_factory=dict)


@dataclass
class Action:
    kind: str            # create | adopt | tombstone | prune | foreign | skip | noop
    key: str
    paths: tuple[str, ...] = ()
    note: str = ""


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.actions:
            out[a.kind] = out.get(a.kind, 0) + 1
        return out

    def changes(self) -> list[Action]:
        return [a for a in self.actions if a.kind != "noop"]


class View(Protocol):
    name: str
    root: Path

    def desired(self, entries: list[GgufEntry]) -> dict[str, Desired]: ...
    def present(self, d: Desired) -> Presence: ...
    def create(self, d: Desired) -> list[str]: ...
    def remove(self, paths: list[str]) -> None: ...
    def foreign(self, state: State) -> list[ForeignItem]: ...


def reconcile(desired: dict[str, Desired], presence: dict[str, Presence], state: State) -> Plan:
    plan = Plan()
    for key, d in sorted(desired.items()):
        owned = key in state.owned
        pres = presence[key]
        if key in state.tombstones:
            plan.actions.append(Action("skip", key, d.paths, "tombstoned"))
        elif pres is Presence.WRONG:
            plan.actions.append(Action("foreign", key, d.paths,
                                       "owned path replaced by something else" if owned else "exists with a different target"))
        elif owned and pres is Presence.CORRECT:
            plan.actions.append(Action("noop", key, d.paths))
        elif owned and pres is Presence.ABSENT:
            plan.actions.append(Action("tombstone", key, d.paths, "deleted outside sync"))
        elif pres is Presence.CORRECT:
            plan.actions.append(Action("adopt", key, d.paths))
        else:
            plan.actions.append(Action("create", key, d.paths))
    for key, o in sorted(state.owned.items()):
        if key not in desired:
            plan.actions.append(Action("prune", key, tuple(o.paths), "no longer in cache"))
    return plan


def apply(plan: Plan, view: View, state: State, execute: bool, desired: dict[str, Desired] | None = None) -> State:
    if not execute:
        return state
    desired = desired or {}
    for a in plan.actions:
        if a.kind == "create":
            d = desired.get(a.key) or Desired(a.key, "", {p: Path() for p in a.paths})
            try:
                paths = view.create(d)
            except SkipEntry as e:
                a.kind, a.note = "skip", str(e)
                continue
            state.owned[a.key] = Owned(paths=sorted(set(paths)), sha256=d.sha256)
        elif a.kind == "adopt":
            d = desired.get(a.key)
            state.owned[a.key] = Owned(paths=list(a.paths), sha256=d.sha256 if d else "")
        elif a.kind == "tombstone":
            state.owned.pop(a.key, None)
            state.tombstones[a.key] = datetime.now().replace(microsecond=0).isoformat()
        elif a.kind == "prune":
            others = {p for k, o in state.owned.items() if k != a.key for p in o.paths}
            view.remove([p for p in a.paths if p not in others])
            state.owned.pop(a.key, None)
    return state
