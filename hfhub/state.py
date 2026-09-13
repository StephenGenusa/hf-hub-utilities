"""Per-view ownership and tombstone record, stored at <view root>/.hfhub-state.json."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_FILE = ".hfhub-state.json"


class StateError(Exception):
    pass


@dataclass
class Owned:
    paths: list[str]
    sha256: str


@dataclass
class State:
    owned: dict[str, Owned] = field(default_factory=dict)
    tombstones: dict[str, str] = field(default_factory=dict)
    version: int = 1

    def owners_of(self, path: str) -> list[str]:
        return [k for k, o in self.owned.items() if path in o.paths]


def load(root: Path) -> State:
    p = root / STATE_FILE
    if not p.exists():
        return State()
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise StateError(f"{p}: unreadable state file ({e}); move it aside to continue") from e
    if not isinstance(data, dict) or data.get("version") != 1:
        raise StateError(f"{p}: unsupported state version {data.get('version') if isinstance(data, dict) else '?'}")
    raw_owned = data.get("owned", {})
    if not isinstance(raw_owned, dict):
        raise StateError(f"{p}: malformed state file (owned must be an object)")
    owned: dict[str, Owned] = {}
    for k, v in raw_owned.items():
        if not isinstance(v, dict):
            raise StateError(f"{p}: malformed state file (owned[{k!r}] must be an object)")
        paths = v.get("paths")
        if not isinstance(paths, list) or not all(isinstance(x, str) for x in paths):
            raise StateError(f"{p}: malformed state file (owned[{k!r}].paths must be a list of strings)")
        sha256 = v.get("sha256")
        if not isinstance(sha256, str):
            raise StateError(f"{p}: malformed state file (owned[{k!r}].sha256 must be a string)")
        owned[k] = Owned(paths=list(paths), sha256=sha256)
    raw_tombstones = data.get("tombstones", {})
    if not isinstance(raw_tombstones, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in raw_tombstones.items()
    ):
        raise StateError(f"{p}: malformed state file (tombstones must be an object of string to string)")
    return State(owned=owned, tombstones=dict(raw_tombstones))


def save(root: Path, state: State) -> None:
    p = root / STATE_FILE
    tmp = p.with_name(p.name + ".tmp")
    payload = {"version": state.version,
               "owned": {k: asdict(v) for k, v in sorted(state.owned.items())},
               "tombstones": dict(sorted(state.tombstones.items()))}
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, p)
