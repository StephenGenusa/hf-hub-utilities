# Consumer Views (LM Studio + Ollama) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the HF hub cache the single store of GGUF bytes and keep two derived views, an LM Studio symlink tree and an Ollama store with hand-written manifests, in sync with it, with tombstones for user deletions, explicit adoption of foreign files into the cache, and duplicate detection.

**Architecture:** A small `hfhub` package. `cache.py` scans the HF cache into `GgufEntry` records. Each view (`views/lmstudio.py`, `views/ollama.py`) turns entries into `Desired` items and knows how to create, check, and remove them. A pure `reconcile()` in `views/base.py` compares desired items, on-disk presence, and the per-view `State` file and emits a `Plan`; `apply()` is the only code path that writes. `ollama_registry.py` fetches Ollama manifest layers from `huggingface.co/v2`. `adopt.py` and `dupes.py` handle foreign files. `cli.py` wires `hf-xfer` and `hfu`.

**Tech Stack:** Python 3.13 (stdlib `tomllib`, `urllib`, `hashlib`, `struct`), `huggingface_hub` (Hub lookups), pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-consumer-views-design.md`

## Global Constraints

- Python `>=3.13`; dependencies stay `huggingface-hub>=1.20`, `hf-xet`; dev `pytest`. No new runtime dependencies.
- Console script names stay `hfu` and `hf-xfer`. `python hfu.py` and `python hf_xfer.py` keep working via shims.
- Dry run is the default for every command that writes; `--execute` performs writes.
- The manager never deletes or overwrites a path it does not own, except `view remove --foreign <key>` which requires the key spelled out.
- `cache.py` is read-only. Nothing writes into the HF hub cache except `adopt` (through the import machinery).
- Blob SHA-256 is taken from the HF blob filename, never recomputed, except for foreign LM Studio real files during adoption.
- State file `.hfhub-state.json` at each view root, written atomically. Corrupt state aborts that view.
- Commit after every task with the message shown. No `Co-Authored-By` or `Claude-Session` trailers (per `~/.claude/CLAUDE.md`).
- Run tests with `cd hf-hub-utilities && .venv/bin/python -m pytest -q` (create the venv in Task 1 if missing).

---

### Task 1: Package restructure with shims

**Files:**
- Create: `hfhub/__init__.py`, `hfhub/views/__init__.py`, `hfhub/xfer.py` (moved from `hf_xfer.py`), `hfhub/hfu.py` (moved from `hfu.py`), `tests/__init__.py`, `tests/test_hfu.py` (moved)
- Modify: `pyproject.toml`, `hfu.py`, `hf_xfer.py` (become shims)

**Interfaces:**
- Produces: importable modules `hfhub.xfer` (functions `resolve_cache_dir`, `_place_file`, `_rel_symlink_target`, `map_from_hub`, `RepoMap`, `main`) and `hfhub.hfu` (functions `group_quants`, `pick_mmproj`, `RepoFile`, `is_mmproj`, `_stem`, `_SHARD_SUFFIX`, `main`).

- [ ] **Step 1: Move files with git so history follows**

```bash
cd ~/PycharmProjects/HuggingfaceHub/hf-hub-utilities
mkdir -p hfhub/views tests
git mv hf_xfer.py hfhub/xfer.py
git mv hfu.py hfhub/hfu.py
git mv test_hfu.py tests/test_hfu.py 2>/dev/null || mv test_hfu.py tests/test_hfu.py
touch hfhub/__init__.py hfhub/views/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write shims**

`hfu.py`:
```python
#!/usr/bin/env python3
"""Shim: the implementation lives in hfhub.hfu."""
from hfhub.hfu import main

if __name__ == "__main__":
    main()
```

`hf_xfer.py`:
```python
#!/usr/bin/env python3
"""Shim: the implementation lives in hfhub.cli (import/export delegate to hfhub.xfer)."""
import sys
from hfhub.cli import xfer_main

if __name__ == "__main__":
    sys.exit(xfer_main())
```

`hfhub/cli.py` (minimal for now; extended in Task 10):
```python
"""Console entry points for hfu and hf-xfer."""
import sys

from hfhub import hfu as _hfu
from hfhub import xfer as _xfer


def hfu_main() -> None:
    _hfu.main()


def xfer_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _xfer.main(argv)
```

- [ ] **Step 3: Update pyproject.toml**

```toml
[project]
name = "huggingfacehub"
version = "0.2.0"
requires-python = ">=3.13"
dependencies = [
    "huggingface-hub>=1.20",
    "hf-xet",
]

[dependency-groups]
dev = ["pytest"]

[tool.setuptools]
packages = ["hfhub", "hfhub.views"]

[project.scripts]
hfu = "hfhub.cli:hfu_main"
hf-xfer = "hfhub.cli:xfer_main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Fix the moved test's import**

In `tests/test_hfu.py` replace `import hfu` with `from hfhub import hfu` and `from hfu import RepoFile` with `from hfhub.hfu import RepoFile`.

- [ ] **Step 5: Install editable and run the existing tests**

```bash
cd ~/PycharmProjects/HuggingfaceHub/hf-hub-utilities
[ -x .venv/bin/python ] || uv venv .venv
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest -q
.venv/bin/hfu --help | head -3
.venv/bin/hf-xfer --help | head -3
```
Expected: all existing tests pass; both help texts print.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Restructure into hfhub package with shims"
```

---

### Task 2: GGUF header reader

**Files:**
- Create: `hfhub/gguf_header.py`, `tests/gguf_fixture.py`, `tests/test_gguf_header.py`

**Interfaces:**
- Produces: `read_header(path: Path, max_array_items: int = 64) -> GgufHeader` with fields `version: int`, `n_tensors: int`, `kv: dict[str, object]`, `param_count: int`; `file_type_name(ft: int) -> str`; `size_label(param_count: int) -> str` (e.g. `"4.65B"`).
- Produces (tests): `write_gguf(path, kv: dict[str, object], tensors: list[tuple[str, list[int]]]) -> None`.

- [ ] **Step 1: Write the test fixture writer**

`tests/gguf_fixture.py`:
```python
"""Minimal GGUF v3 writer for tests: header + KV + tensor infos, no tensor data."""
import struct
from pathlib import Path

_STR, _U32, _ARR, _U64, _F32 = 8, 4, 9, 10, 6


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", 7) + struct.pack("<?", v)
    if isinstance(v, int):
        return struct.pack("<I", _U32) + struct.pack("<I", v) if v < 2**32 else struct.pack("<I", _U64) + struct.pack("<Q", v)
    if isinstance(v, float):
        return struct.pack("<I", _F32) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", _STR) + _s(v)
    if isinstance(v, list):
        out = struct.pack("<I", _ARR) + struct.pack("<I", _STR) + struct.pack("<Q", len(v))
        return out + b"".join(_s(x) for x in v)
    raise TypeError(type(v))


def write_gguf(path: Path, kv: dict[str, object], tensors: list[tuple[str, list[int]]]) -> None:
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, v in kv.items():
        out += _s(k) + _value(v)
    for name, dims in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<I", 0) + struct.pack("<Q", 0)  # dtype F32, offset 0
    path.write_bytes(out)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_gguf_header.py`:
```python
from pathlib import Path

import pytest

from hfhub.gguf_header import file_type_name, read_header, size_label
from tests.gguf_fixture import write_gguf


def test_reads_scalar_and_string_kv(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"general.architecture": "qwen35", "general.file_type": 15}, [])
    h = read_header(p)
    assert h.version == 3
    assert h.kv["general.architecture"] == "qwen35"
    assert h.kv["general.file_type"] == 15


def test_param_count_sums_tensor_elements(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"general.architecture": "llama"}, [("a", [4, 8]), ("b", [10])])
    assert read_header(p).param_count == 42
    assert read_header(p).n_tensors == 2


def test_long_string_arrays_are_truncated(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"tokenizer.ggml.tokens": [f"t{i}" for i in range(100)]}, [])
    h = read_header(p, max_array_items=8)
    assert h.kv["tokenizer.ggml.tokens"] == [f"t{i}" for i in range(8)]


def test_rejects_non_gguf(tmp_path: Path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(ValueError):
        read_header(p)


def test_file_type_names():
    assert file_type_name(15) == "Q4_K_M"
    assert file_type_name(1) == "F16"
    assert file_type_name(999) == "unknown"


def test_size_label():
    assert size_label(4_650_000_000) == "4.7B"
    assert size_label(137_000_000) == "137.0M"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gguf_header.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'hfhub.gguf_header'`

- [ ] **Step 4: Implement the reader**

`hfhub/gguf_header.py`:
```python
"""Read the GGUF header (KV metadata + tensor infos) without touching tensor data."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
           6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_STRING, _ARRAY = 8, 9

_FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
               11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
               17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
               23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
               29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
               38: "MXFP4_MOE"}


@dataclass(frozen=True)
class GgufHeader:
    version: int
    n_tensors: int
    kv: dict[str, object]
    param_count: int


class _Reader:
    def __init__(self, f, max_array_items: int):
        self.f = f
        self.max_array_items = max_array_items

    def raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t: int):
        if t == _STRING:
            return self.string()
        if t == _ARRAY:
            et, n = self.u32(), self.u64()
            keep = min(n, self.max_array_items)
            items = [self.value(et) for _ in range(keep)]
            for _ in range(n - keep):
                self.value(et)
            return items
        fmt, size = _SCALAR[t]
        return struct.unpack(fmt, self.raw(size))[0]


def read_header(path: Path, max_array_items: int = 64) -> GgufHeader:
    with open(path, "rb") as f:
        r = _Reader(f, max_array_items)
        if r.raw(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = r.u32()
        n_tensors, n_kv = r.u64(), r.u64()
        kv: dict[str, object] = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.u32())
        params = 0
        for _ in range(n_tensors):
            r.string()
            ndim = r.u32()
            count = 1
            for _ in range(ndim):
                count *= r.u64()
            r.u32()
            r.u64()
            params += count
    return GgufHeader(version=version, n_tensors=n_tensors, kv=kv, param_count=params)


def file_type_name(ft: int) -> str:
    return _FILE_TYPES.get(ft, "unknown")


def size_label(param_count: int) -> str:
    if param_count >= 1_000_000_000:
        return f"{param_count / 1e9:.1f}B"
    return f"{param_count / 1e6:.1f}M"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gguf_header.py -q`
Expected: 6 passed

- [ ] **Step 6: Sanity check against a real file**

```bash
.venv/bin/python -c "
from pathlib import Path; from hfhub.gguf_header import *
import glob,os
p=glob.glob(os.environ['HF_HOME']+'/hub/models--ibm-granite--granite-docling-258M-GGUF/snapshots/*/granite-docling-258M-BF16.gguf')[0]
h=read_header(Path(p)); print(h.kv['general.architecture'], file_type_name(h.kv['general.file_type']), size_label(h.param_count))"
```
Expected: an architecture name, `BF16`, and a size near `258.0M` (may differ slightly; only sanity).

- [ ] **Step 7: Commit**

```bash
git add hfhub/gguf_header.py tests/gguf_fixture.py tests/test_gguf_header.py
git commit -m "Add stdlib GGUF header reader"
```

---

### Task 3: Config loading and saving

**Files:**
- Create: `hfhub/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `ViewConfig(root: Path | None, aliases: dict[str, str])`, `Config(path: Path, views: dict[str, ViewConfig])`, `load(path: Path | None = None) -> Config`, `dump(config: Config) -> str`, `save(config: Config) -> None`, `default_path() -> Path`, `VIEW_NAMES = ("lmstudio", "ollama")`.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
from pathlib import Path

from hfhub import config as cfg


TOML = '''
[views.lmstudio]
root = "~/lm"

[views.ollama]
root = "/data/ollama"

[views.ollama.aliases]
"qwen3.6:27b" = "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf"
'''


def test_load_parses_roots_and_aliases(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(TOML)
    c = cfg.load(p)
    assert c.views["lmstudio"].root == Path("~/lm").expanduser()
    assert c.views["ollama"].root == Path("/data/ollama")
    assert c.views["ollama"].aliases["qwen3.6:27b"].startswith("unsloth/")
    assert c.views["lmstudio"].aliases == {}


def test_missing_file_gives_disabled_views(tmp_path: Path):
    c = cfg.load(tmp_path / "nope.toml")
    assert c.views["lmstudio"].root is None
    assert c.views["ollama"].root is None


def test_env_override_selects_path(tmp_path: Path, monkeypatch):
    p = tmp_path / "x.toml"
    p.write_text('[views.lmstudio]\nroot = "/a"\n')
    monkeypatch.setenv("HFHUB_CONFIG", str(p))
    assert cfg.load().views["lmstudio"].root == Path("/a")


def test_dump_roundtrips(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(TOML)
    c = cfg.load(p)
    c.views["ollama"].aliases["llama3.1:8b"] = "ollama/llama3.1:llama3.1-8b.gguf"
    cfg.save(c)
    again = cfg.load(p)
    assert again.views["ollama"].aliases["llama3.1:8b"] == "ollama/llama3.1:llama3.1-8b.gguf"
    assert again.views["lmstudio"].root == Path("~/lm").expanduser()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`hfhub/config.py`:
```python
"""~/.config/hfhub/config.toml: view roots and Ollama aliases."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VIEW_NAMES = ("lmstudio", "ollama")


@dataclass
class ViewConfig:
    root: Path | None = None
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    path: Path
    views: dict[str, ViewConfig]


def default_path() -> Path:
    env = os.environ.get("HFHUB_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path("~/.config/hfhub/config.toml").expanduser()


def load(path: Path | None = None) -> Config:
    path = path or default_path()
    views = {name: ViewConfig() for name in VIEW_NAMES}
    if path.is_file():
        data = tomllib.loads(path.read_text())
        for name in VIEW_NAMES:
            section = data.get("views", {}).get(name, {})
            root = section.get("root")
            views[name] = ViewConfig(
                root=Path(root).expanduser() if root else None,
                aliases=dict(section.get("aliases", {})),
            )
    return Config(path=path, views=views)


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dump(config: Config) -> str:
    lines: list[str] = []
    for name in VIEW_NAMES:
        v = config.views[name]
        if v.root is None and not v.aliases:
            continue
        lines.append(f"[views.{name}]")
        if v.root is not None:
            lines.append(f"root = {_q(str(v.root))}")
        lines.append("")
        if v.aliases:
            lines.append(f"[views.{name}.aliases]")
            for alias, target in sorted(v.aliases.items()):
                lines.append(f"{_q(alias)} = {_q(target)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save(config: Config) -> None:
    config.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.path.with_suffix(".tmp")
    tmp.write_text(dump(config))
    os.replace(tmp, config.path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add hfhub/config.py tests/test_config.py
git commit -m "Add hfhub config loading and saving"
```

---

### Task 4: Per-view state file

**Files:**
- Create: `hfhub/state.py`, `tests/test_state.py`

**Interfaces:**
- Produces: `Owned(paths: list[str], sha256: str)`, `State(owned: dict[str, Owned], tombstones: dict[str, str], version: int = 1)`, `StateError`, `STATE_FILE = ".hfhub-state.json"`, `load(root: Path) -> State`, `save(root: Path, state: State) -> None`, `State.owners_of(path: str) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_state.py`:
```python
import json
from pathlib import Path

import pytest

from hfhub import state as st


def test_missing_file_is_empty(tmp_path: Path):
    s = st.load(tmp_path)
    assert s.owned == {} and s.tombstones == {}


def test_save_and_load_roundtrip(tmp_path: Path):
    s = st.State(owned={"a/b:c.gguf": st.Owned(paths=["a/b/c.gguf"], sha256="x" * 64)},
                 tombstones={"a/b:d.gguf": "2026-09-13T00:00:00"})
    st.save(tmp_path, s)
    assert st.load(tmp_path) == s
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_file_raises(tmp_path: Path):
    (tmp_path / st.STATE_FILE).write_text("{not json")
    with pytest.raises(st.StateError):
        st.load(tmp_path)


def test_wrong_version_raises(tmp_path: Path):
    (tmp_path / st.STATE_FILE).write_text(json.dumps({"version": 99, "owned": {}, "tombstones": {}}))
    with pytest.raises(st.StateError):
        st.load(tmp_path)


def test_owners_of_path():
    s = st.State(owned={"k1": st.Owned(["p1", "shared"], "a"), "k2": st.Owned(["shared"], "b")}, tombstones={})
    assert sorted(s.owners_of("shared")) == ["k1", "k2"]
    assert s.owners_of("p1") == ["k1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_state.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`hfhub/state.py`:
```python
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
    owned = {k: Owned(paths=list(v["paths"]), sha256=v["sha256"]) for k, v in data.get("owned", {}).items()}
    return State(owned=owned, tombstones=dict(data.get("tombstones", {})))


def save(root: Path, state: State) -> None:
    p = root / STATE_FILE
    tmp = p.with_name(p.name + ".tmp")
    payload = {"version": state.version,
               "owned": {k: asdict(v) for k, v in sorted(state.owned.items())},
               "tombstones": dict(sorted(state.tombstones.items()))}
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_state.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add hfhub/state.py tests/test_state.py
git commit -m "Add per-view state file with ownership and tombstones"
```

---

### Task 5: HF cache scanner

**Files:**
- Create: `hfhub/cache.py`, `tests/hub_fixture.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: `hfhub.hfu.is_mmproj`, `hfhub.hfu._SHARD_SUFFIX`, `hfhub.hfu.pick_mmproj`.
- Produces: `GgufEntry(repo_id, relpath, blob: Path, sha256, size, is_mmproj, is_current, is_shard)` with property `key -> "repo_id:relpath"`; `scan(hub: Path) -> list[GgufEntry]`; `unlinked(hub: Path) -> list[tuple[str, Path]]`; `mmproj_for(entry, entries) -> GgufEntry | None`; `weights(entries) -> list[GgufEntry]` (non-mmproj); `repo_folder(repo_id) -> str`; `repo_id_from_folder(name) -> str`.
- Produces (tests): `add_repo(hub, repo_id, files: dict[str, bytes], commit=None, main=True) -> str` returning the commit used.

- [ ] **Step 1: Write the hub fixture**

`tests/hub_fixture.py`:
```python
"""Build a fake HF hub cache in a temp dir."""
import hashlib
import os
from pathlib import Path


def folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def add_repo(hub: Path, repo_id: str, files: dict[str, bytes], commit: str | None = None, main: bool = True) -> str:
    commit = commit or hashlib.sha1(f"{repo_id}:{sorted(files)}".encode()).hexdigest()
    root = hub / folder(repo_id)
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snap = root / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        h = hashlib.sha256(content).hexdigest()
        blob = root / "blobs" / h
        if not blob.exists():
            blob.write_bytes(content)
        link = snap / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        depth = len(Path(rel).parts) - 1
        if not link.is_symlink():
            link.symlink_to(os.path.join(*([".."] * (2 + depth)), "blobs", h))
    if main:
        (root / "refs").mkdir(exist_ok=True)
        (root / "refs" / "main").write_text(commit)
    return commit
```

- [ ] **Step 2: Write the failing tests**

`tests/test_cache.py`:
```python
from pathlib import Path

from hfhub import cache
from tests.hub_fixture import add_repo


def test_scan_finds_ggufs_with_sha_and_currency(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "org/Model-GGUF", {"m-Q4_K_M.gguf": b"aaa", "mmproj-F16.gguf": b"bbb", "README.md": b"x"})
    entries = cache.scan(hub)
    keys = {e.key: e for e in entries}
    assert set(keys) == {"org/Model-GGUF:m-Q4_K_M.gguf", "org/Model-GGUF:mmproj-F16.gguf"}
    w = keys["org/Model-GGUF:m-Q4_K_M.gguf"]
    assert w.sha256 == "9834876dcfb05cb167a5c24953eba58c4ac89b1adf57f28f2f9d09af107ee8f0"
    assert w.blob == hub / "models--org--Model-GGUF" / "blobs" / w.sha256
    assert w.size == 3 and w.is_current and not w.is_mmproj and not w.is_shard
    assert keys["org/Model-GGUF:mmproj-F16.gguf"].is_mmproj


def test_refs_main_snapshot_wins_over_older(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"old"}, commit="a" * 40, main=False)
    add_repo(hub, "o/r", {"m.gguf": b"new"}, commit="b" * 40, main=True)
    (e,) = cache.scan(hub)
    assert e.blob.read_bytes() == b"new" and e.is_current


def test_shards_are_flagged(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"BF16/m-BF16-00001-of-00002.gguf": b"1", "BF16/m-BF16-00002-of-00002.gguf": b"2"})
    assert all(e.is_shard for e in cache.scan(hub))
    assert {e.relpath for e in cache.scan(hub)} == {"BF16/m-BF16-00001-of-00002.gguf", "BF16/m-BF16-00002-of-00002.gguf"}


def test_unlinked_blobs_reported(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"linked"})
    orphan = hub / "models--o--r" / "blobs" / ("f" * 64)
    orphan.write_bytes(b"orphan")
    assert cache.unlinked(hub) == [("o/r", orphan)]


def test_mmproj_for_prefers_f16(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"m", "mmproj-BF16.gguf": b"1", "mmproj-F16.gguf": b"2"})
    add_repo(hub, "o/other", {"mmproj-F16.gguf": b"3"})
    entries = cache.scan(hub)
    w = next(e for e in entries if e.relpath == "m.gguf")
    assert cache.mmproj_for(w, entries).relpath == "mmproj-F16.gguf"
    assert cache.mmproj_for(w, entries).repo_id == "o/r"


def test_repo_folder_roundtrip():
    assert cache.repo_folder("unsloth/Qwen3.6-27B-GGUF") == "models--unsloth--Qwen3.6-27B-GGUF"
    assert cache.repo_id_from_folder("models--unsloth--Qwen3.6-27B-GGUF") == "unsloth/Qwen3.6-27B-GGUF"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cache.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement**

`hfhub/cache.py`:
```python
"""Read-only scan of the HF hub cache for GGUF files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from hfhub.hfu import _SHARD_SUFFIX, _stem, is_mmproj, pick_mmproj

_HEX = set("0123456789abcdef")


@dataclass(frozen=True)
class GgufEntry:
    repo_id: str
    relpath: str
    blob: Path
    sha256: str
    size: int
    is_mmproj: bool
    is_current: bool
    is_shard: bool

    @property
    def key(self) -> str:
        return f"{self.repo_id}:{self.relpath}"

    @property
    def basename(self) -> str:
        return os.path.basename(self.relpath)


def repo_folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def repo_id_from_folder(name: str) -> str:
    return name[len("models--"):].replace("--", "/")


def _is_sha(name: str) -> bool:
    return len(name) == 64 and set(name) <= _HEX


def _snapshots_in_order(repo_root: Path) -> list[tuple[Path, bool]]:
    """(snapshot dir, is_main) with the refs/main snapshot first, then newest first."""
    snaps = repo_root / "snapshots"
    if not snaps.is_dir():
        return []
    main = None
    ref = repo_root / "refs" / "main"
    if ref.is_file():
        main = ref.read_text().strip()
    dirs = sorted((d for d in snaps.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    ordered = [(d, d.name == main) for d in dirs]
    ordered.sort(key=lambda t: not t[1])
    return ordered


def scan(hub: Path) -> list[GgufEntry]:
    out: list[GgufEntry] = []
    for repo_root in sorted(hub.glob("models--*")):
        repo_id = repo_id_from_folder(repo_root.name)
        seen: set[str] = set()
        for snap, is_main in _snapshots_in_order(repo_root):
            for link in sorted(snap.rglob("*.gguf")):
                rel = str(link.relative_to(snap))
                if rel in seen or not link.is_symlink():
                    continue
                blob = Path(os.path.realpath(link))
                if not _is_sha(blob.name) or not blob.is_file():
                    continue
                seen.add(rel)
                out.append(GgufEntry(
                    repo_id=repo_id, relpath=rel, blob=blob, sha256=blob.name,
                    size=blob.stat().st_size, is_mmproj=is_mmproj(rel), is_current=is_main,
                    is_shard=bool(_SHARD_SUFFIX.search(os.path.basename(rel)[:-len(".gguf")])),
                ))
    return out


def unlinked(hub: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for repo_root in sorted(hub.glob("models--*")):
        blobs = repo_root / "blobs"
        if not blobs.is_dir():
            continue
        linked = set()
        for snap, _ in _snapshots_in_order(repo_root):
            for link in snap.rglob("*"):
                if link.is_symlink():
                    linked.add(os.path.realpath(link))
        for b in sorted(blobs.iterdir()):
            if _is_sha(b.name) and str(b.resolve()) not in linked:
                out.append((repo_id_from_folder(repo_root.name), b))
    return out


def weights(entries: list[GgufEntry]) -> list[GgufEntry]:
    return [e for e in entries if not e.is_mmproj]


def mmproj_for(entry: GgufEntry, entries: list[GgufEntry]) -> GgufEntry | None:
    candidates = {e.relpath: e for e in entries if e.repo_id == entry.repo_id and e.is_mmproj}
    if not candidates:
        return None
    chosen = pick_mmproj(list(candidates), None)
    return candidates[chosen[0]] if chosen else None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cache.py -q`
Expected: 6 passed

- [ ] **Step 6: Run against the real cache and eyeball**

```bash
.venv/bin/python -c "
from pathlib import Path; import os; from hfhub import cache
hub=Path(os.environ['HF_HOME'])/'hub'; es=cache.scan(hub)
print(len(es),'entries', sum(e.size for e in es)/1e9,'GB'); print(len(cache.unlinked(hub)),'unlinked blobs')"
```
Expected: roughly 150+ entries, about 1180 GB, and 30+ unlinked blobs (the 12 orphan repos).

- [ ] **Step 7: Commit**

```bash
git add hfhub/cache.py tests/hub_fixture.py tests/test_cache.py
git commit -m "Add read-only HF cache scanner"
```

---

### Task 6: Reconcile and apply (views/base.py)

**Files:**
- Create: `hfhub/views/base.py`, `tests/test_reconcile.py`

**Interfaces:**
- Consumes: `hfhub.state.State`, `Owned`.
- Produces: `Desired(key, sha256, links: dict[str, Path], extra: dict)` with property `paths -> tuple[str, ...]` (sorted link paths; views may return more from `create`); `Presence` enum `ABSENT | CORRECT | WRONG`; `Action(kind, key, paths, note)` with `kind` in `create|adopt|tombstone|prune|foreign|skip|noop`; `Plan(actions)` with `summary() -> dict[str, int]` and `changes() -> list[Action]` (everything except noop); `SkipEntry(Exception)`; `View` Protocol (`name`, `root`, `desired(entries)`, `present(d)`, `create(d) -> list[str]`, `remove(paths)`, `foreign(state) -> list[ForeignItem]`); `ForeignItem(key, path: Path, extra: dict)`; `reconcile(desired, presence, state) -> Plan`; `apply(plan, view, state, execute) -> State`.

- [ ] **Step 1: Write the failing tests**

`tests/test_reconcile.py`:
```python
from pathlib import Path

from hfhub.state import Owned, State
from hfhub.views import base
from hfhub.views.base import Desired, Presence, SkipEntry


def d(key: str, sha: str = "s") -> Desired:
    return Desired(key=key, sha256=sha, links={f"{key}.gguf": Path("/blob")}, extra={})


def kinds(plan):
    return {a.key: a.kind for a in plan.actions}


def test_table_rows():
    desired = {k: d(k) for k in ["new", "adoptable", "ok", "deleted", "tomb", "wrong"]}
    presence = {"new": Presence.ABSENT, "adoptable": Presence.CORRECT, "ok": Presence.CORRECT,
                "deleted": Presence.ABSENT, "tomb": Presence.ABSENT, "wrong": Presence.WRONG}
    state = State(owned={"ok": Owned(["ok.gguf"], "s"), "deleted": Owned(["deleted.gguf"], "s"),
                         "gone": Owned(["gone.gguf"], "s")},
                  tombstones={"tomb": "t"})
    plan = base.reconcile(desired, presence, state)
    assert kinds(plan) == {"new": "create", "adoptable": "adopt", "ok": "noop", "deleted": "tombstone",
                           "tomb": "skip", "wrong": "foreign", "gone": "prune"}


def test_owned_but_wrong_is_reported_not_touched():
    plan = base.reconcile({"k": d("k")}, {"k": Presence.WRONG}, State(owned={"k": Owned(["k.gguf"], "s")}))
    assert kinds(plan) == {"k": "foreign"}


class FakeView:
    name = "fake"
    root = Path("/view")

    def __init__(self):
        self.created: list[str] = []
        self.removed: list[str] = []
        self.fail: set[str] = set()

    def desired(self, entries):
        return {}

    def present(self, d):
        return Presence.ABSENT

    def create(self, d):
        if d.key in self.fail:
            raise SkipEntry("offline")
        self.created.append(d.key)
        return list(d.paths) + ["shared/small"]

    def remove(self, paths):
        self.removed += paths

    def foreign(self, state):
        return []


def test_apply_dry_run_writes_nothing():
    v = FakeView()
    plan = base.reconcile({"k": d("k")}, {"k": Presence.ABSENT}, State())
    state = base.apply(plan, v, State(), execute=False)
    assert v.created == [] and state.owned == {}


def test_apply_execute_creates_and_records_paths():
    v = FakeView()
    plan = base.reconcile({"k": d("k")}, {"k": Presence.ABSENT}, State())
    state = base.apply(plan, v, State(), execute=True)
    assert v.created == ["k"]
    assert state.owned["k"].paths == ["k.gguf", "shared/small"]


def test_apply_prune_keeps_paths_owned_by_others():
    v = FakeView()
    state = State(owned={"a": Owned(["a.gguf", "shared/small"], "s"), "b": Owned(["b.gguf", "shared/small"], "s")})
    plan = base.reconcile({"b": d("b")}, {"b": Presence.CORRECT}, state)
    state = base.apply(plan, v, state, execute=True)
    assert v.removed == ["a.gguf"]
    assert "a" not in state.owned and "b" in state.owned


def test_apply_tombstone_records_timestamp():
    v = FakeView()
    state = State(owned={"k": Owned(["k.gguf"], "s")})
    plan = base.reconcile({"k": d("k")}, {"k": Presence.ABSENT}, state)
    state = base.apply(plan, v, state, execute=True)
    assert "k" not in state.owned and "k" in state.tombstones


def test_apply_skip_on_create_failure_leaves_state_clean():
    v = FakeView()
    v.fail.add("k")
    plan = base.reconcile({"k": d("k")}, {"k": Presence.ABSENT}, State())
    state = base.apply(plan, v, State(), execute=True)
    assert state.owned == {}
    assert [a.kind for a in plan.actions] == ["skip"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reconcile.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`hfhub/views/base.py`:
```python
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
```

Note for the implementer: `apply` needs the `desired` dict to hand full `Desired` objects to `create`; the driver in Task 10 passes it. The tests above call `apply` without it for the FakeView, which only needs `paths`; keep the fallback.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reconcile.py -q`
Expected: 7 passed. If `test_apply_execute_creates_and_records_paths` fails because `create` received the fallback `Desired`, change the test to pass `desired={"k": d("k")}` to `apply`; the driver always passes it.

- [ ] **Step 5: Commit**

```bash
git add hfhub/views/base.py tests/test_reconcile.py
git commit -m "Add view protocol with pure reconcile and single apply path"
```

---

### Task 7: LM Studio view

**Files:**
- Create: `hfhub/views/lmstudio.py`, `tests/test_lmstudio.py`

**Interfaces:**
- Consumes: `cache.GgufEntry`, `cache.weights`, `cache.mmproj_for`, `base.Desired`, `base.Presence`, `base.ForeignItem`, `state.State`.
- Produces: `LmStudioView(root: Path)` implementing `View`, with `warnings: list[str]` collected during `desired()`; `remove_empty_parents(root: Path, path: Path) -> None` helper (also used by the Ollama view).

- [ ] **Step 1: Write the failing tests**

`tests/test_lmstudio.py`:
```python
import os
from pathlib import Path

from hfhub import cache
from hfhub.state import Owned, State
from hfhub.views.base import Presence
from hfhub.views.lmstudio import LmStudioView
from tests.hub_fixture import add_repo


def setup(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "org/M-GGUF", {"M-Q4_K_M.gguf": b"w", "mmproj-F16.gguf": b"p", "mmproj-BF16.gguf": b"q"})
    return hub, cache.scan(hub), LmStudioView(tmp_path / "lm")


def test_desired_links_weight_and_mmproj_in_repo_folder(tmp_path: Path):
    hub, entries, view = setup(tmp_path)
    desired = view.desired(entries)
    (d,) = desired.values()
    assert d.key == "org/M-GGUF:M-Q4_K_M.gguf"
    assert set(d.links) == {"org/M-GGUF/M-Q4_K_M.gguf", "org/M-GGUF/mmproj-F16.gguf"}
    assert d.links["org/M-GGUF/M-Q4_K_M.gguf"].name == d.sha256


def test_create_present_remove_cycle(tmp_path: Path):
    hub, entries, view = setup(tmp_path)
    (d,) = view.desired(entries).values()
    assert view.present(d) is Presence.ABSENT
    paths = view.create(d)
    assert sorted(paths) == sorted(d.links)
    link = view.root / "org/M-GGUF/M-Q4_K_M.gguf"
    assert link.is_symlink() and os.readlink(link) == str(d.links["org/M-GGUF/M-Q4_K_M.gguf"])
    assert view.present(d) is Presence.CORRECT
    view.remove(paths)
    assert not (view.root / "org").exists()
    assert view.root.exists()


def test_real_file_at_path_is_wrong(tmp_path: Path):
    hub, entries, view = setup(tmp_path)
    (d,) = view.desired(entries).values()
    p = view.root / "org/M-GGUF/M-Q4_K_M.gguf"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"real")
    assert view.present(d) is Presence.WRONG


def test_partial_links_count_as_absent_and_create_fills(tmp_path: Path):
    hub, entries, view = setup(tmp_path)
    (d,) = view.desired(entries).values()
    p = view.root / "org/M-GGUF/mmproj-F16.gguf"
    p.parent.mkdir(parents=True)
    p.symlink_to(d.links["org/M-GGUF/mmproj-F16.gguf"])
    assert view.present(d) is Presence.ABSENT
    view.create(d)
    assert view.present(d) is Presence.CORRECT


def test_basename_collision_prefers_current_and_warns(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"x.gguf": b"cur", "MTP/x.gguf": b"mtp"})
    view = LmStudioView(tmp_path / "lm")
    desired = view.desired(cache.scan(hub))
    assert list(desired) == ["o/r:x.gguf"]
    assert any("MTP/x.gguf" in w for w in view.warnings)


def test_foreign_lists_real_ggufs_and_outside_symlinks_only(tmp_path: Path):
    hub, entries, view = setup(tmp_path)
    (d,) = view.desired(entries).values()
    paths = view.create(d)
    state = State(owned={d.key: Owned(paths, d.sha256)})
    (view.root / "o2/r2").mkdir(parents=True)
    (view.root / "o2/r2/real.gguf").write_bytes(b"r")
    (view.root / "o2/r2/config.json").write_text("{}")
    (view.root / "o2/r2/ext.gguf").symlink_to(tmp_path / "elsewhere.gguf")
    keys = sorted(f.key for f in view.foreign(state))
    assert keys == ["lmstudio:o2/r2/ext.gguf", "lmstudio:o2/r2/real.gguf"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lmstudio.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`hfhub/views/lmstudio.py`:
```python
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
        for e in sorted(weights(entries), key=lambda e: (not e.is_current, e.relpath)):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_lmstudio.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add hfhub/views/lmstudio.py tests/test_lmstudio.py
git commit -m "Add LM Studio symlink view"
```

---

### Task 8: Ollama registry client and manifest builders

**Files:**
- Create: `hfhub/ollama_registry.py`, `tests/test_ollama_registry.py`, `tests/fixtures/registry_manifest.json`

**Interfaces:**
- Consumes: `gguf_header.GgufHeader`, `file_type_name`, `size_label`.
- Produces: constants `MT_MODEL`, `MT_PROJECTOR`, `MT_TEMPLATE`, `MT_PARAMS`, `MT_LICENSE`, `MT_CONFIG`; `Layer = tuple[str, int]` meaning `(sha256 hex, size)`; `fetch_manifest(org, name, tag, token=None, timeout=30) -> dict | None`; `fetch_blob(org, name, digest, token=None, timeout=60) -> bytes`; `small_layers(manifest) -> list[dict]` (every layer except model/projector); `patch_manifest(manifest, config: dict, model: Layer, projector: Layer | None) -> tuple[dict, bytes]`; `synth_manifest(header: GgufHeader, model: Layer, projector: Layer | None) -> tuple[dict, bytes]`; `sha256_bytes(b) -> str`.

- [ ] **Step 1: Capture a fixture from the live endpoint**

```bash
mkdir -p tests/fixtures
curl -s "https://huggingface.co/v2/unsloth/gemma-4-E2B-it-GGUF/manifests/UD-Q4_K_XL" -o tests/fixtures/registry_manifest.json
python3 -m json.tool tests/fixtures/registry_manifest.json | head -5
```
Expected: `"schemaVersion": 2` and four layers (model, template, projector, params).

- [ ] **Step 2: Write the failing tests**

`tests/test_ollama_registry.py`:
```python
import hashlib
import json
import os
from pathlib import Path

import pytest

from hfhub import ollama_registry as reg
from hfhub.gguf_header import read_header
from tests.gguf_fixture import write_gguf

FIX = Path(__file__).parent / "fixtures" / "registry_manifest.json"


def test_small_layers_exclude_model_and_projector():
    m = json.loads(FIX.read_text())
    kinds = {l["mediaType"] for l in reg.small_layers(m)}
    assert reg.MT_MODEL not in kinds and reg.MT_PROJECTOR not in kinds
    assert reg.MT_TEMPLATE in kinds


def test_patch_manifest_swaps_digests_and_rehashes_config():
    m = json.loads(FIX.read_text())
    config = {"model_format": "gguf", "model_family": "gemma4", "rootfs": {"type": "layers", "diff_ids": ["sha256:old"]}}
    new_m, cfg_bytes = reg.patch_manifest(m, config, model=("a" * 64, 10), projector=("b" * 64, 20))
    layers = {l["mediaType"]: l for l in new_m["layers"]}
    assert layers[reg.MT_MODEL]["digest"] == "sha256:" + "a" * 64 and layers[reg.MT_MODEL]["size"] == 10
    assert layers[reg.MT_PROJECTOR]["digest"] == "sha256:" + "b" * 64
    cfg = json.loads(cfg_bytes)
    assert cfg["rootfs"]["diff_ids"] == [l["digest"] for l in new_m["layers"]]
    assert new_m["config"]["digest"] == "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()
    assert new_m["config"]["size"] == len(cfg_bytes)
    assert m["layers"][0]["digest"] != new_m["layers"][0]["digest"]  # input not mutated


def test_patch_manifest_drops_projector_when_none():
    m = json.loads(FIX.read_text())
    new_m, _ = reg.patch_manifest(m, {"rootfs": {}}, model=("a" * 64, 1), projector=None)
    assert reg.MT_PROJECTOR not in {l["mediaType"] for l in new_m["layers"]}


def test_patch_manifest_adds_projector_when_registry_lacked_one():
    m = json.loads(FIX.read_text())
    m["layers"] = [l for l in m["layers"] if l["mediaType"] != reg.MT_PROJECTOR]
    new_m, _ = reg.patch_manifest(m, {"rootfs": {}}, model=("a" * 64, 1), projector=("b" * 64, 2))
    assert reg.MT_PROJECTOR in {l["mediaType"] for l in new_m["layers"]}


def test_synth_manifest_from_header(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"general.architecture": "llama", "general.file_type": 15}, [("t", [1000, 1000])])
    m, cfg_bytes = reg.synth_manifest(read_header(p), model=("c" * 64, 5), projector=None)
    cfg = json.loads(cfg_bytes)
    assert cfg["model_family"] == "llama" and cfg["file_type"] == "Q4_K_M" and cfg["model_type"] == "1.0M"
    assert m["schemaVersion"] == 2 and [l["mediaType"] for l in m["layers"]] == [reg.MT_MODEL]
    assert m["config"]["digest"] == "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()


@pytest.mark.skipif(not os.environ.get("HFHUB_LIVE"), reason="set HFHUB_LIVE=1 to hit huggingface.co")
def test_live_fetch_manifest_and_blob():
    m = reg.fetch_manifest("unsloth", "gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-UD-Q4_K_XL.gguf")
    assert m and m["schemaVersion"] == 2
    t = next(l for l in m["layers"] if l["mediaType"] == reg.MT_TEMPLATE)
    blob = reg.fetch_blob("unsloth", "gemma-4-E2B-it-GGUF", t["digest"])
    assert len(blob) == t["size"]
    assert reg.fetch_manifest("nobody", "does-not-exist-GGUF", "Q4_K_M") is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ollama_registry.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement**

`hfhub/ollama_registry.py`:
```python
"""Ollama-registry protocol against huggingface.co/v2 plus manifest building."""
from __future__ import annotations

import copy
import hashlib
import json
import urllib.error
import urllib.request

from hfhub.gguf_header import GgufHeader, file_type_name, size_label

BASE = "https://huggingface.co/v2"
MT_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
MT_CONFIG = "application/vnd.docker.container.image.v1+json"
MT_MODEL = "application/vnd.ollama.image.model"
MT_PROJECTOR = "application/vnd.ollama.image.projector"
MT_TEMPLATE = "application/vnd.ollama.image.template"
MT_PARAMS = "application/vnd.ollama.image.params"
MT_LICENSE = "application/vnd.ollama.image.license"

Layer = tuple[str, int]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _request(url: str, token: str | None, timeout: int, accept: str | None = None) -> bytes | None:
    req = urllib.request.Request(url)
    if accept:
        req.add_header("Accept", accept)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_manifest(org: str, name: str, tag: str, token: str | None = None, timeout: int = 30) -> dict | None:
    body = _request(f"{BASE}/{org}/{name}/manifests/{tag}", token, timeout, accept=MT_MANIFEST)
    return json.loads(body) if body is not None else None


def fetch_blob(org: str, name: str, digest: str, token: str | None = None, timeout: int = 60) -> bytes:
    body = _request(f"{BASE}/{org}/{name}/blobs/{digest}", token, timeout)
    if body is None:
        raise FileNotFoundError(f"{org}/{name} blob {digest} not found")
    return body


def small_layers(manifest: dict) -> list[dict]:
    return [l for l in manifest["layers"] if l["mediaType"] not in (MT_MODEL, MT_PROJECTOR)]


def _finish(manifest: dict, config: dict) -> tuple[dict, bytes]:
    config = copy.deepcopy(config)
    config.setdefault("rootfs", {})
    config["rootfs"]["type"] = "layers"
    config["rootfs"]["diff_ids"] = [l["digest"] for l in manifest["layers"]]
    cfg_bytes = json.dumps(config, separators=(",", ":")).encode()
    manifest["config"] = {"digest": "sha256:" + sha256_bytes(cfg_bytes), "mediaType": MT_CONFIG, "size": len(cfg_bytes)}
    return manifest, cfg_bytes


def patch_manifest(manifest: dict, config: dict, model: Layer, projector: Layer | None) -> tuple[dict, bytes]:
    m = copy.deepcopy(manifest)
    layers = []
    seen_proj = False
    for l in m["layers"]:
        if l["mediaType"] == MT_MODEL:
            layers.append({"digest": "sha256:" + model[0], "mediaType": MT_MODEL, "size": model[1]})
        elif l["mediaType"] == MT_PROJECTOR:
            seen_proj = True
            if projector is not None:
                layers.append({"digest": "sha256:" + projector[0], "mediaType": MT_PROJECTOR, "size": projector[1]})
        else:
            layers.append(dict(l))
    if projector is not None and not seen_proj:
        layers.insert(1, {"digest": "sha256:" + projector[0], "mediaType": MT_PROJECTOR, "size": projector[1]})
    m["layers"] = layers
    return _finish(m, config)


def synth_manifest(header: GgufHeader, model: Layer, projector: Layer | None) -> tuple[dict, bytes]:
    arch = str(header.kv.get("general.architecture", "unknown"))
    ft = header.kv.get("general.file_type")
    config = {
        "model_format": "gguf", "model_family": arch, "model_families": [arch],
        "model_type": size_label(header.param_count),
        "file_type": file_type_name(int(ft)) if isinstance(ft, int) else "unknown",
        "architecture": "amd64", "os": "linux",
    }
    layers = [{"digest": "sha256:" + model[0], "mediaType": MT_MODEL, "size": model[1]}]
    if projector is not None:
        layers.append({"digest": "sha256:" + projector[0], "mediaType": MT_PROJECTOR, "size": projector[1]})
    m = {"schemaVersion": 2, "mediaType": MT_MANIFEST, "layers": layers}
    return _finish(m, config)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ollama_registry.py -q` then `HFHUB_LIVE=1 .venv/bin/python -m pytest tests/test_ollama_registry.py -q -k live`
Expected: 5 passed, 1 skipped; then the live test passes.

- [ ] **Step 6: Commit**

```bash
git add hfhub/ollama_registry.py tests/test_ollama_registry.py tests/fixtures/registry_manifest.json
git commit -m "Add Ollama registry client and manifest builders"
```

---

### Task 9: Ollama view

**Files:**
- Create: `hfhub/views/ollama.py`, `tests/test_ollama.py`

**Interfaces:**
- Consumes: everything from Tasks 5, 6, 8; `hfu.group_quants`, `hfu.RepoFile`; `lmstudio.remove_empty_parents`; `gguf_header.read_header`.
- Produces: `OllamaView(root: Path, aliases: dict[str, str], offline: bool = False, token: str | None = None, fetch_manifest=ollama_registry.fetch_manifest, fetch_blob=ollama_registry.fetch_blob)`; pure helpers `derive_tags(entries_of_repo: list[GgufEntry]) -> dict[str, str]` (key -> tag), `alias_manifest_path(alias: str) -> str`, `manifest_path(repo_id, tag) -> str`, `blob_path(sha256) -> str`; `REGISTRY_CACHE_DIR = ".hfhub-registry"`; `warnings: list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ollama.py`:
```python
import json
import os
from pathlib import Path

from hfhub import cache, ollama_registry as reg
from hfhub.state import Owned, State
from hfhub.views.base import Presence, SkipEntry
from hfhub.views.ollama import OllamaView, alias_manifest_path, derive_tags
from tests.gguf_fixture import write_gguf
from tests.hub_fixture import add_repo

FIX = json.loads((Path(__file__).parent / "fixtures" / "registry_manifest.json").read_text())


def gguf_bytes(tmp_path: Path, arch="gemma4", ft=15) -> bytes:
    p = tmp_path / "tmp.gguf"
    write_gguf(p, {"general.architecture": arch, "general.file_type": ft}, [("t", [10])])
    return p.read_bytes()


def make(tmp_path: Path, files: dict[str, bytes] | None = None, aliases=None, manifest=FIX, blobs=None, offline=False):
    hub = tmp_path / "hub"
    files = files or {"M-Q4_K_M.gguf": gguf_bytes(tmp_path), "mmproj-F16.gguf": b"proj"}
    add_repo(hub, "org/M-GGUF", files)
    calls = {"manifest": [], "blob": []}

    def fm(org, name, tag, token=None, timeout=30):
        calls["manifest"].append((org, name, tag))
        return manifest

    def fb(org, name, digest, token=None, timeout=60):
        calls["blob"].append(digest)
        return (blobs or {}).get(digest, b"{}" if digest.endswith(FIX["config"]["digest"][-6:]) else b"tmpl")

    view = OllamaView(tmp_path / "ol", aliases=aliases or {}, offline=offline, fetch_manifest=fm, fetch_blob=fb)
    return hub, cache.scan(hub), view, calls


def test_derive_tags_uses_quant_name_else_filename(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"r-Q4_K_M.gguf": b"1", "r-Q8_0.gguf": b"2", "weird name.gguf": b"3", "mmproj-F16.gguf": b"p"})
    tags = derive_tags(cache.scan(hub))
    assert tags["o/r:r-Q4_K_M.gguf"] == "Q4_K_M"
    assert tags["o/r:r-Q8_0.gguf"] == "Q8_0"
    assert tags["o/r:weird name.gguf"] == "weird_name.gguf"
    assert "o/r:mmproj-F16.gguf" not in tags


def test_desired_has_blob_links_and_manifest(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path)
    (d,) = view.desired(entries).values()
    w = next(e for e in entries if not e.is_mmproj)
    mm = next(e for e in entries if e.is_mmproj)
    assert d.links == {f"blobs/sha256-{w.sha256}": w.blob, f"blobs/sha256-{mm.sha256}": mm.blob}
    assert d.extra["manifest"] == "manifests/hf.co/org/M-GGUF/Q4_K_M"
    assert d.extra["tag"] == "Q4_K_M"


def test_create_writes_manifest_from_registry_and_caches_response(tmp_path: Path):
    hub, entries, view, calls = make(tmp_path)
    (d,) = view.desired(entries).values()
    paths = view.create(d)
    m = json.loads((view.root / d.extra["manifest"]).read_text())
    layers = {l["mediaType"]: l for l in m["layers"]}
    w = next(e for e in entries if not e.is_mmproj)
    assert layers[reg.MT_MODEL]["digest"] == "sha256:" + w.sha256
    assert (view.root / "blobs" / ("sha256-" + layers[reg.MT_TEMPLATE]["digest"][7:])).is_file()
    assert (view.root / "blobs" / ("sha256-" + m["config"]["digest"][7:])).is_file()
    assert d.extra["manifest"] in paths and all(l in paths for l in d.links)
    assert (view.root / ".hfhub-registry" / "org" / "M-GGUF" / "M-Q4_K_M.gguf.json").is_file()
    assert calls["manifest"] == [("org", "M-GGUF", "M-Q4_K_M.gguf")]
    assert view.present(d) is Presence.CORRECT
    view.create(d)  # second call uses the cached response
    assert len(calls["manifest"]) == 1


def test_create_synthesizes_when_registry_404(tmp_path: Path):
    hub, entries, view, calls = make(tmp_path, manifest=None)
    (d,) = view.desired(entries).values()
    view.create(d)
    m = json.loads((view.root / d.extra["manifest"]).read_text())
    assert {l["mediaType"] for l in m["layers"]} == {reg.MT_MODEL, reg.MT_PROJECTOR}
    cfg = json.loads((view.root / "blobs" / ("sha256-" + m["config"]["digest"][7:])).read_bytes())
    assert cfg["model_family"] == "gemma4" and cfg["file_type"] == "Q4_K_M"
    assert any("embedded" in w for w in view.warnings)


def test_offline_without_cached_response_skips(tmp_path: Path):
    hub, entries, view, calls = make(tmp_path, offline=True)
    (d,) = view.desired(entries).values()
    try:
        view.create(d)
        assert False, "expected SkipEntry"
    except SkipEntry:
        pass
    assert calls["manifest"] == []


def test_aliases_write_second_manifest(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path, aliases={"m:latest": "org/M-GGUF:M-Q4_K_M.gguf", "ns/m:q4": "org/M-GGUF:M-Q4_K_M.gguf"})
    (d,) = view.desired(entries).values()
    paths = view.create(d)
    assert "manifests/registry.ollama.ai/library/m/latest" in paths
    assert "manifests/registry.ollama.ai/ns/m/q4" in paths
    a = json.loads((view.root / "manifests/registry.ollama.ai/library/m/latest").read_text())
    b = json.loads((view.root / d.extra["manifest"]).read_text())
    assert a == b


def test_alias_manifest_path():
    assert alias_manifest_path("qwen3.6:27b") == "manifests/registry.ollama.ai/library/qwen3.6/27b"
    assert alias_manifest_path("llama3.1") == "manifests/registry.ollama.ai/library/llama3.1/latest"
    assert alias_manifest_path("me/x:t") == "manifests/registry.ollama.ai/me/x/t"


def test_present_wrong_when_manifest_points_elsewhere(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path)
    (d,) = view.desired(entries).values()
    view.create(d)
    p = view.root / d.extra["manifest"]
    m = json.loads(p.read_text())
    m["layers"][0]["digest"] = "sha256:" + "0" * 64
    p.write_text(json.dumps(m))
    assert view.present(d) is Presence.WRONG


def test_existing_real_blob_is_adopted_not_replaced(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path)
    (d,) = view.desired(entries).values()
    w = next(e for e in entries if not e.is_mmproj)
    real = view.root / "blobs" / f"sha256-{w.sha256}"
    real.parent.mkdir(parents=True)
    real.write_bytes(w.blob.read_bytes())
    view.create(d)
    assert real.is_file() and not real.is_symlink()
    assert view.present(d) is Presence.CORRECT


def test_foreign_lists_manifests_with_real_model_blob(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path)
    (d,) = view.desired(entries).values()
    paths = view.create(d)
    state = State(owned={d.key: Owned(paths, d.sha256)})
    mp = view.root / "manifests/registry.ollama.ai/library/llama3.1/8b"
    mp.parent.mkdir(parents=True)
    blob = view.root / "blobs" / ("sha256-" + "1" * 64)
    blob.write_bytes(gguf_bytes(tmp_path, arch="llama"))
    mp.write_text(json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "2" * 64},
                              "layers": [{"digest": "sha256:" + "1" * 64, "mediaType": reg.MT_MODEL, "size": 1}]}))
    (f,) = view.foreign(state)
    assert f.key == "ollama:registry.ollama.ai/library/llama3.1:8b"
    assert f.path == blob
    assert f.extra["manifest"] == "manifests/registry.ollama.ai/library/llama3.1/8b"


def test_remove_cleans_dirs_but_keeps_shared_blobs_decision_to_apply(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path)
    (d,) = view.desired(entries).values()
    paths = view.create(d)
    view.remove(paths)
    assert not (view.root / "manifests").exists()
    assert not any((view.root / "blobs").iterdir())


def test_shards_are_skipped_with_warning(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"r-BF16-00001-of-00002.gguf": b"1", "r-BF16-00002-of-00002.gguf": b"2"})
    view = OllamaView(tmp_path / "ol", aliases={}, fetch_manifest=lambda *a, **k: None, fetch_blob=lambda *a, **k: b"")
    assert view.desired(cache.scan(hub)) == {}
    assert any("shard" in w for w in view.warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ollama.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`hfhub/views/ollama.py`:
```python
"""Ollama view: blob symlinks + hand-written manifests under an OLLAMA_MODELS directory."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from hfhub import ollama_registry as reg
from hfhub.cache import GgufEntry, mmproj_for, weights
from hfhub.gguf_header import read_header
from hfhub.hfu import RepoFile, group_quants
from hfhub.state import State
from hfhub.views.base import Desired, ForeignItem, Presence, SkipEntry
from hfhub.views.lmstudio import remove_empty_parents

REGISTRY_CACHE_DIR = ".hfhub-registry"
_TAG_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def blob_path(sha256: str) -> str:
    return f"blobs/sha256-{sha256}"


def manifest_path(repo_id: str, tag: str) -> str:
    return f"manifests/hf.co/{repo_id}/{tag}"


def alias_manifest_path(alias: str) -> str:
    name, _, tag = alias.partition(":")
    tag = tag or "latest"
    ns, _, short = name.rpartition("/")
    return f"manifests/registry.ollama.ai/{ns or 'library'}/{short}/{tag}"


def _safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def derive_tags(entries: list[GgufEntry]) -> dict[str, str]:
    """key -> Ollama tag for every non-shard weight entry, per repo."""
    out: dict[str, str] = {}
    by_repo: dict[str, list[GgufEntry]] = {}
    for e in weights(entries):
        if not e.is_shard:
            by_repo.setdefault(e.repo_id, []).append(e)
    for repo_id, es in by_repo.items():
        by_rel = {e.relpath: e for e in es}
        quants = group_quants([RepoFile(e.relpath, e.size) for e in es])
        for q in quants:
            if len(q.files) != 1:
                for f in q.files:
                    out[by_rel[f].key] = _safe_tag(f.replace("/", "_"))
                continue
            e = by_rel[q.files[0]]
            out[e.key] = q.name if _TAG_OK.match(q.name) and len(quants) >= 1 else _safe_tag(e.basename)
        used: dict[str, int] = {}
        for e in es:
            t = out[e.key]
            used[t] = used.get(t, 0) + 1
        for e in es:
            if used[out[e.key]] > 1:
                out[e.key] = _safe_tag(e.relpath.replace("/", "_"))
    return out


class OllamaView:
    name = "ollama"

    def __init__(self, root: Path, aliases: dict[str, str], offline: bool = False, token: str | None = None,
                 fetch_manifest=reg.fetch_manifest, fetch_blob=reg.fetch_blob):
        self.root = root
        self.aliases = dict(aliases)
        self.offline = offline
        self.token = token
        self._fetch_manifest = fetch_manifest
        self._fetch_blob = fetch_blob
        self.warnings: list[str] = []

    # ---- desired -------------------------------------------------------
    def desired(self, entries: list[GgufEntry]) -> dict[str, Desired]:
        self.warnings = []
        tags = derive_tags(entries)
        out: dict[str, Desired] = {}
        for e in weights(entries):
            if e.is_shard:
                self.warnings.append(f"{e.key}: sharded GGUF not supported by Ollama; skipped")
                continue
            links = {blob_path(e.sha256): e.blob}
            mm = mmproj_for(e, entries)
            if mm is not None:
                links[blob_path(mm.sha256)] = mm.blob
            tag = tags[e.key]
            extra = {"tag": tag, "manifest": manifest_path(e.repo_id, tag), "repo_id": e.repo_id,
                     "filename": e.basename, "model": (e.sha256, e.size),
                     "projector": (mm.sha256, mm.size) if mm else None, "blob": str(e.blob),
                     "aliases": [a for a, t in self.aliases.items() if t == e.key]}
            out[e.key] = Desired(key=e.key, sha256=e.sha256, links=links, extra=extra)
        return out

    # ---- presence ------------------------------------------------------
    def _link_status(self, rel: str, target: Path) -> str:
        p = self.root / rel
        if p.is_symlink():
            return "ok" if os.readlink(p) == str(target) else "wrong"
        if p.is_file():
            return "ok"  # Ollama downloaded identical bytes; adopt
        return "missing"

    def _manifest_status(self, d: Desired) -> str:
        p = self.root / d.extra["manifest"]
        if not p.is_file():
            return "missing"
        try:
            m = json.loads(p.read_text())
            layers = {l["mediaType"]: l["digest"] for l in m["layers"]}
        except (ValueError, KeyError, TypeError):
            return "wrong"
        model_ok = layers.get(reg.MT_MODEL) == "sha256:" + d.extra["model"][0]
        proj = d.extra["projector"]
        proj_ok = layers.get(reg.MT_PROJECTOR) == ("sha256:" + proj[0] if proj else None)
        return "ok" if model_ok and proj_ok else "wrong"

    def present(self, d: Desired) -> Presence:
        statuses = {self._link_status(rel, t) for rel, t in d.links.items()}
        statuses.add(self._manifest_status(d))
        for a in d.extra.get("aliases", []):
            statuses.add("ok" if (self.root / alias_manifest_path(a)).is_file() else "missing")
        if "wrong" in statuses:
            return Presence.WRONG
        return Presence.CORRECT if statuses == {"ok"} else Presence.ABSENT

    # ---- create --------------------------------------------------------
    def _registry_cache(self, d: Desired) -> Path:
        return self.root / REGISTRY_CACHE_DIR / d.extra["repo_id"] / (d.extra["filename"] + ".json")

    def _registry_manifest(self, d: Desired) -> dict | None:
        cache_file = self._registry_cache(d)
        if cache_file.is_file():
            data = json.loads(cache_file.read_text())
            return None if data.get("missing") else data
        if self.offline:
            raise SkipEntry("offline and no cached registry response")
        org, name = d.extra["repo_id"].split("/", 1)
        m = self._fetch_manifest(org, name, d.extra["filename"], token=self.token)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(m if m is not None else {"missing": True}))
        return m

    def _write_blob(self, sha256: str, data: bytes) -> str:
        rel = blob_path(sha256)
        p = self.root / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return rel

    def create(self, d: Desired) -> list[str]:
        created: list[str] = []
        for rel, target in d.links.items():
            p = self.root / rel
            if not (p.is_symlink() or p.exists()):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.symlink_to(target)
            created.append(rel)
        registry = self._registry_manifest(d)
        if registry is not None:
            org, name = d.extra["repo_id"].split("/", 1)
            config = json.loads(self._fetch_blob(org, name, registry["config"]["digest"], token=self.token) or b"{}")
            for layer in reg.small_layers(registry):
                data = self._fetch_blob(org, name, layer["digest"], token=self.token)
                created.append(self._write_blob(layer["digest"][7:], data))
            manifest, cfg_bytes = reg.patch_manifest(registry, config, d.extra["model"], d.extra["projector"])
        else:
            header = read_header(Path(d.extra["blob"]))
            manifest, cfg_bytes = reg.synth_manifest(header, d.extra["model"], d.extra["projector"])
            self.warnings.append(f"{d.key}: not on the Hub registry; model relies on its embedded chat template")
        created.append(self._write_blob(manifest["config"]["digest"][7:], cfg_bytes))
        text = json.dumps(manifest, separators=(",", ":"))
        for rel in [d.extra["manifest"]] + [alias_manifest_path(a) for a in d.extra.get("aliases", [])]:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            created.append(rel)
        return created

    # ---- remove / foreign ---------------------------------------------
    def remove(self, paths: list[str]) -> None:
        for rel in paths:
            p = self.root / rel
            if p.is_symlink() or p.exists():
                p.unlink()
            remove_empty_parents(self.root, p)

    def foreign(self, state: State) -> list[ForeignItem]:
        owned = {p for o in state.owned.values() for p in o.paths}
        out: list[ForeignItem] = []
        mroot = self.root / "manifests"
        if not mroot.is_dir():
            return out
        for p in sorted(x for x in mroot.rglob("*") if x.is_file()):
            rel = str(p.relative_to(self.root))
            if rel in owned:
                continue
            try:
                m = json.loads(p.read_text())
                model = next(l for l in m["layers"] if l["mediaType"] == reg.MT_MODEL)
            except (ValueError, KeyError, StopIteration, TypeError):
                continue
            blob = self.root / blob_path(model["digest"][7:])
            if blob.is_file() and not blob.is_symlink():
                parts = p.relative_to(mroot).parts  # registry, ns, name, tag
                key = "ollama:" + "/".join(parts[:-1]) + ":" + parts[-1]
                out.append(ForeignItem(key=key, path=blob, extra={"manifest": rel, "layers": m["layers"]}))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ollama.py -q`
Expected: 12 passed. If `test_derive_tags_uses_quant_name_else_filename` fails on `weird name.gguf`, check what `group_quants` returns as `q.name` for it; the fallback `_safe_tag(e.basename)` must produce `weird_name.gguf`.

- [ ] **Step 5: Commit**

```bash
git add hfhub/views/ollama.py tests/test_ollama.py
git commit -m "Add Ollama view with registry-backed manifests and aliases"
```

---

### Task 10: Sync driver and CLI (sync, view status/add/remove), hfu hook

**Files:**
- Create: `hfhub/sync.py`, `tests/test_sync.py`, `tests/test_cli.py`
- Modify: `hfhub/cli.py`, `hfhub/hfu.py` (`_OWN_FLAGS`, `parse_args`, `main`)

**Interfaces:**
- Consumes: Tasks 3 to 9.
- Produces: `sync.build_view(name, config, offline) -> View | None`; `sync.hub_dir() -> Path` (HF_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub, via `xfer.resolve_cache_dir(None)`); `sync.sync_view(view, entries, execute) -> tuple[Plan, State]`; `sync.run(config, view_names, execute, offline, out=print) -> dict[str, Plan]`; `sync.status(config, view_names, out) -> None`; `sync.view_add(config, key, view_names, execute, out)`; `sync.view_remove(config, key, view_names, execute, out)`; `cli.xfer_main(argv) -> int` handling `sync`, `view status|add|remove`, delegating `import|export` to `xfer.main`; `hfu --no-sync`.

- [ ] **Step 1: Write the failing sync tests**

`tests/test_sync.py`:
```python
from pathlib import Path

from hfhub import config as cfg, state as st, sync
from hfhub.views.base import Presence
from tests.hub_fixture import add_repo
from tests.test_ollama import gguf_bytes, FIX


def make(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hub"
    add_repo(hub, "org/M-GGUF", {"M-Q4_K_M.gguf": gguf_bytes(tmp_path), "mmproj-F16.gguf": b"p"})
    monkeypatch.setattr(sync, "hub_dir", lambda: hub)
    monkeypatch.setattr(sync.reg, "fetch_manifest", lambda *a, **k: FIX)
    monkeypatch.setattr(sync.reg, "fetch_blob", lambda *a, **k: b"{}")
    c = cfg.Config(path=tmp_path / "c.toml", views={
        "lmstudio": cfg.ViewConfig(root=tmp_path / "lm"),
        "ollama": cfg.ViewConfig(root=tmp_path / "ol", aliases={"m:q4": "org/M-GGUF:M-Q4_K_M.gguf"})})
    return hub, c


def test_dry_run_plans_but_writes_nothing(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    plans = sync.run(c, ["lmstudio", "ollama"], execute=False, offline=False, out=lambda *_: None)
    assert plans["lmstudio"].summary() == {"create": 1}
    assert plans["ollama"].summary() == {"create": 1}
    assert not (tmp_path / "lm").exists() and not (tmp_path / "ol").exists()


def test_execute_then_second_run_is_noop(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio", "ollama"], execute=True, offline=False, out=lambda *_: None)
    assert (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").is_symlink()
    assert (tmp_path / "ol/manifests/hf.co/org/M-GGUF/Q4_K_M").is_file()
    assert (tmp_path / "ol/manifests/registry.ollama.ai/library/m/q4").is_file()
    plans = sync.run(c, ["lmstudio", "ollama"], execute=True, offline=False, out=lambda *_: None)
    assert plans["lmstudio"].changes() == [] and plans["ollama"].changes() == []
    assert st.load(tmp_path / "lm").owned and st.load(tmp_path / "ol").owned


def test_user_deletion_is_tombstoned_and_view_add_restores(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").unlink()
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    assert plans["lmstudio"].summary() == {"tombstone": 1}
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    assert plans["lmstudio"].summary() == {"skip": 1}
    sync.view_add(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lambda *_: None)
    assert (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").is_symlink()
    assert "org/M-GGUF:M-Q4_K_M.gguf" not in st.load(tmp_path / "lm").tombstones


def test_view_remove_tombstones_and_prunes(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    sync.view_remove(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lambda *_: None)
    assert not (tmp_path / "lm/org").exists()
    assert "org/M-GGUF:M-Q4_K_M.gguf" in st.load(tmp_path / "lm").tombstones


def test_cache_removal_prunes(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    import shutil
    shutil.rmtree(hub / "models--org--M-GGUF")
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    assert plans["lmstudio"].summary() == {"prune": 1}
    assert not (tmp_path / "lm/org").exists()


def test_disabled_view_is_skipped(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    c.views["ollama"].root = None
    msgs = []
    plans = sync.run(c, ["lmstudio", "ollama"], execute=False, offline=False, out=msgs.append)
    assert "ollama" not in plans and any("ollama" in m and "no root" in m for m in msgs)


def test_status_reports_sections(tmp_path, monkeypatch, capsys):
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    (tmp_path / "lm/x/y").mkdir(parents=True)
    (tmp_path / "lm/x/y/real.gguf").write_bytes(gguf_bytes(tmp_path))
    lines = []
    sync.status(c, ["lmstudio"], out=lines.append)
    text = "\n".join(lines)
    assert "owned: 1" in text and "lmstudio:x/y/real.gguf" in text
```

- [ ] **Step 2: Write the failing CLI tests**

`tests/test_cli.py`:
```python
from hfhub import cli


def test_import_export_delegate(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli._xfer, "main", lambda argv: seen.setdefault("argv", argv) or 0)
    assert cli.xfer_main(["import", "--local-dir", "/x"]) == 0
    assert seen["argv"] == ["import", "--local-dir", "/x"]


def test_sync_parses_flags(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sync, "run", lambda config, views, execute, offline, out: seen.update(
        views=views, execute=execute, offline=offline) or {})
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(path=None, views={}))
    assert cli.xfer_main(["sync", "--view", "ollama", "--execute", "--offline"]) == 0
    assert seen == {"views": ["ollama"], "execute": True, "offline": True}


def test_view_status_default_all_views(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sync, "status", lambda config, views, out: seen.update(views=views))
    monkeypatch.setattr(cli.cfg, "load", lambda: cli.cfg.Config(path=None, views={}))
    assert cli.xfer_main(["view", "status"]) == 0
    assert seen["views"] == ["lmstudio", "ollama"]


def test_hfu_no_sync_flag_is_own_option():
    from hfhub.hfu import _split_own_args
    own, extra = _split_own_args(["org/name", "--no-sync", "--revision", "main"])
    assert "--no-sync" in own and extra == ["--revision", "main"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sync.py tests/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: hfhub.sync` and attribute errors in test_cli.

- [ ] **Step 4: Implement sync.py**

`hfhub/sync.py`:
```python
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
    out(f"[{view.name}] {mode}: " + ", ".join(f"{k}={v}" for k, v in sorted(plan.summary().items())) or "nothing")
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
        registry_404 = view.root / "manifests" and getattr(view, "_registry_cache", None)
        if registry_404 and execute:
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
```

- [ ] **Step 5: Implement cli.py**

`hfhub/cli.py`:
```python
"""Console entry points for hfu and hf-xfer."""
from __future__ import annotations

import argparse
import sys

from hfhub import config as cfg, sync
from hfhub import hfu as _hfu
from hfhub import xfer as _xfer

ALL_VIEWS = list(cfg.VIEW_NAMES)


def hfu_main() -> None:
    _hfu.main()


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="hf-xfer", description="HF cache <-> local-dir transfer, and consumer views (sync/view/dupes).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def view_opt(p):
        p.add_argument("--view", action="append", choices=ALL_VIEWS, help="limit to one view (repeatable)")

    s = sub.add_parser("sync", help="reconcile LM Studio / Ollama views with the HF cache")
    view_opt(s)
    s.add_argument("--execute", action="store_true", help="write changes (default: dry run)")
    s.add_argument("--offline", action="store_true", help="never contact huggingface.co")

    v = sub.add_parser("view", help="inspect or edit view entries")
    vs = v.add_subparsers(dest="vcmd", required=True)
    p = vs.add_parser("status"); view_opt(p)
    for name in ("add", "remove"):
        p = vs.add_parser(name)
        p.add_argument("key", help="<org/name>:<relpath> or, with --foreign, a foreign key")
        view_opt(p)
        p.add_argument("--execute", action="store_true")
        if name == "remove":
            p.add_argument("--foreign", action="store_true", help="delete a foreign (non-owned) item")
    p = vs.add_parser("adopt")
    p.add_argument("key")
    p.add_argument("--repo-id", required=True)
    view_opt(p)
    p.add_argument("--move", action="store_true")
    p.add_argument("--execute", action="store_true")

    d = sub.add_parser("dupes", help="list foreign files that look like cache entries")
    view_opt(d)
    return ap


def xfer_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("import", "export"):
        return _xfer.main(argv)
    args = _parser().parse_args(argv)
    config = cfg.load()
    views = args.view or ALL_VIEWS
    if args.cmd == "sync":
        sync.run(config, views, execute=args.execute, offline=args.offline)
        return 0
    if args.cmd == "view":
        if args.vcmd == "status":
            sync.status(config, views)
        elif args.vcmd == "add":
            sync.view_add(config, args.key, views, execute=args.execute)
        elif args.vcmd == "remove":
            if args.foreign:
                from hfhub import adopt
                adopt.remove_foreign(config, args.key, views, execute=args.execute)
            else:
                sync.view_remove(config, args.key, views, execute=args.execute)
        elif args.vcmd == "adopt":
            from hfhub import adopt
            adopt.run(config, args.key, args.repo_id, views, move=args.move, execute=args.execute)
        return 0
    if args.cmd == "dupes":
        from hfhub import dupes
        dupes.run(config, views)
        return 0
    return 2
```

(`adopt` and `dupes` modules arrive in Tasks 11 and 12; until then those branches import lazily so the rest works.)

- [ ] **Step 6: Add the hfu hook**

In `hfhub/hfu.py`:
- Change `_OWN_FLAGS = {"-h", "--help"}` to `_OWN_FLAGS = {"-h", "--help", "--no-sync"}`.
- In `parse_args`, after the `--mmproj` argument add:
```python
    parser.add_argument("--no-sync", action="store_true",
                        help="don't refresh LM Studio / Ollama views after the download")
```
- In `main`, replace `print("\n✅ Successfully downloaded")` with:
```python
        print("\n✅ Successfully downloaded")
        if not ns.no_sync:
            try:
                from hfhub import config as _cfg, sync as _sync
                _sync.run(_cfg.load(), list(_cfg.VIEW_NAMES), execute=True, offline=False)
            except Exception as e:  # the download succeeded; a sync problem must not change the exit code
                print(f"⚠️  view sync failed: {type(e).__name__}: {e}", file=sys.stderr)
```

- [ ] **Step 7: Run all tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. If `test_status_reports_sections` fails on the `owned: 1` string, match the exact format written in `status()`.

- [ ] **Step 8: Smoke test against the real cache, dry run**

```bash
mkdir -p ~/.config/hfhub
cat > ~/.config/hfhub/config.toml <<'EOF'
[views.lmstudio]
root = "~/.lmstudio/models"

[views.ollama]
root = "/media/stephen/2f5f6aa6-3182-4426-b774-5829d6ba8260/huggingface/ollama"
EOF
.venv/bin/hf-xfer sync | head -40
.venv/bin/hf-xfer view status | head -60
```
Expected: LM Studio plan shows mostly `adopt` (existing links) plus some `create`; Ollama plan shows `create` for every weight entry (root does not exist yet); status lists the 29 real LM Studio files as foreign and the unlinked blobs. Nothing is written.

- [ ] **Step 9: Commit**

```bash
git add hfhub/sync.py hfhub/cli.py hfhub/hfu.py tests/test_sync.py tests/test_cli.py
git commit -m "Add sync driver, hf-xfer sync/view commands, and hfu post-download sync"
```

---

### Task 11: Adopt foreign files into the HF cache

**Files:**
- Create: `hfhub/adopt.py`, `tests/test_adopt.py`

**Interfaces:**
- Consumes: `xfer.map_from_hub`, `xfer.RepoMap`, `xfer._place_file`, `xfer._rel_symlink_target`, `cache.repo_folder`, `views.base.ForeignItem`, `sync.build_view`, `sync.sync_view`, `config.save`, `ollama.REGISTRY_CACHE_DIR`, `ollama_registry.small_layers`.
- Produces: `AdoptPlan(repo_id, relpath, commit, etag, hub_backed: bool, note: str)`; `plan_adopt(item: ForeignItem, repo_id: str, sha256: str, hub: Callable[[str], RepoMap | None]) -> AdoptPlan`; `file_sha256(path) -> str`; `execute_adopt(item, plan, hub_dir, move) -> Path` (returns snapshot file path); `find_foreign(config, view_names) -> list[tuple[str, ForeignItem]]`; `run(config, key, repo_id, view_names, move, execute, out=print)`; `remove_foreign(config, key, view_names, execute, out=print)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_adopt.py`:
```python
import hashlib
import json
from pathlib import Path

from hfhub import adopt, cache, config as cfg, sync, state as st
from hfhub.views.base import ForeignItem
from hfhub.xfer import RepoMap
from tests.hub_fixture import add_repo
from tests.test_ollama import gguf_bytes, FIX


def test_plan_hub_backed_when_hash_in_repo(tmp_path: Path):
    f = tmp_path / "x.gguf"; f.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    hub = lambda rid: RepoMap(commit_hash="c" * 40, etags={"real-name.gguf": sha}, source="hub")
    p = adopt.plan_adopt(ForeignItem("lmstudio:o/r/x.gguf", f), "o/r", sha, hub)
    assert p.hub_backed and p.relpath == "real-name.gguf" and p.commit == "c" * 40 and p.etag == sha


def test_plan_synthesises_when_repo_missing_or_hash_unknown(tmp_path: Path):
    f = tmp_path / "x.gguf"; f.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    p = adopt.plan_adopt(ForeignItem("lmstudio:o/r/x.gguf", f), "o/r", sha, lambda rid: None)
    assert not p.hub_backed and p.relpath == "x.gguf" and p.etag == sha and len(p.commit) == 40
    q = adopt.plan_adopt(ForeignItem("lmstudio:o/r/x.gguf", f), "o/r", sha,
                         lambda rid: RepoMap(commit_hash="c" * 40, etags={"other.gguf": "0" * 64}, source="hub"))
    assert not q.hub_backed and "not in" in q.note


def test_ollama_foreign_gets_name_tag_filename(tmp_path: Path):
    f = tmp_path / "sha256-" + "1" * 64
    p = adopt.plan_adopt(ForeignItem("ollama:registry.ollama.ai/library/llama3.1:8b", Path(f)), "ollama/llama3.1", "1" * 64, lambda rid: None)
    assert p.relpath == "llama3.1-8b.gguf"


def test_execute_adopt_move_creates_cache_entry_and_removes_source(tmp_path: Path):
    hub = tmp_path / "hub"; hub.mkdir()
    src = tmp_path / "src.gguf"; src.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    plan = adopt.AdoptPlan("o/r", "x.gguf", "c" * 40, sha, False, "")
    snap_file = adopt.execute_adopt(ForeignItem("k", src), plan, hub, move=True)
    assert not src.exists()
    assert (hub / "models--o--r/blobs" / sha).read_bytes() == b"abc"
    assert snap_file.is_symlink() and snap_file.read_bytes() == b"abc"
    assert (hub / "models--o--r/refs/main").read_text() == "c" * 40
    assert cache.scan(hub)[0].key == "o/r:x.gguf"


def test_run_lmstudio_end_to_end(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hub"; hub.mkdir()
    lm = tmp_path / "lm"; (lm / "o/r").mkdir(parents=True)
    real = lm / "o/r/x.gguf"; real.write_bytes(gguf_bytes(tmp_path))
    monkeypatch.setattr(sync, "hub_dir", lambda: hub)
    monkeypatch.setattr(adopt, "hub_lookup", lambda rid: None)
    c = cfg.Config(path=tmp_path / "c.toml", views={"lmstudio": cfg.ViewConfig(root=lm), "ollama": cfg.ViewConfig()})
    adopt.run(c, "lmstudio:o/r/x.gguf", "o/r", ["lmstudio"], move=True, execute=True, out=lambda *_: None)
    assert real.is_symlink() and real.resolve().parent.name == "blobs"
    assert "o/r:x.gguf" in st.load(lm).owned


def test_run_ollama_carries_layers_and_adds_alias(tmp_path: Path, monkeypatch):
    hub = tmp_path / "hub"; hub.mkdir()
    ol = tmp_path / "ol"
    data = gguf_bytes(tmp_path, arch="llama")
    sha = hashlib.sha256(data).hexdigest()
    (ol / "blobs").mkdir(parents=True); (ol / "blobs" / f"sha256-{sha}").write_bytes(data)
    tmpl = b"{{ .Prompt }}"; tsha = hashlib.sha256(tmpl).hexdigest()
    (ol / "blobs" / f"sha256-{tsha}").write_bytes(tmpl)
    mp = ol / "manifests/registry.ollama.ai/library/llama3.1/8b"; mp.parent.mkdir(parents=True)
    mp.write_text(json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "2" * 64, "size": 1},
                              "layers": [{"digest": f"sha256:{sha}", "mediaType": "application/vnd.ollama.image.model", "size": len(data)},
                                         {"digest": f"sha256:{tsha}", "mediaType": "application/vnd.ollama.image.template", "size": len(tmpl)}]}))
    monkeypatch.setattr(sync, "hub_dir", lambda: hub)
    monkeypatch.setattr(adopt, "hub_lookup", lambda rid: None)
    monkeypatch.setattr(sync.reg, "fetch_manifest", lambda *a, **k: None)
    monkeypatch.setattr(sync.reg, "fetch_blob", lambda *a, **k: b"{}")
    c = cfg.Config(path=tmp_path / "c.toml", views={"lmstudio": cfg.ViewConfig(), "ollama": cfg.ViewConfig(root=ol)})
    adopt.run(c, "ollama:registry.ollama.ai/library/llama3.1:8b", "ollama/llama3.1", ["ollama"], move=True, execute=True, out=lambda *_: None)
    assert cfg.load(c.path).views["ollama"].aliases["llama3.1:8b"] == "ollama/llama3.1:llama3.1-8b.gguf"
    assert (ol / "blobs" / f"sha256-{sha}").is_symlink()
    new = json.loads((ol / "manifests/hf.co/ollama/llama3.1/llama3.1-8b.gguf").read_text())
    kinds = {l["mediaType"].split(".")[-1] for l in new["layers"]}
    assert kinds == {"model", "template"}
    assert json.loads(mp.read_text()) == new  # alias rewritten in place
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_adopt.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`hfhub/adopt.py`:
```python
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


def plan_adopt(item: ForeignItem, repo_id: str, sha256: str, hub: Callable[[str], RepoMap | None] = None) -> AdoptPlan:
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


def find_foreign(config: cfg.Config, view_names: list[str]) -> list[tuple[str, ForeignItem]]:
    out = []
    for name in view_names:
        view = sync.build_view(name, config, offline=True)
        if view is None:
            continue
        for f in view.foreign(st.load(view.root)):
            out.append((name, f))
    return out


def _seed_registry_cache(view_root: Path, plan: AdoptPlan, item: ForeignItem) -> None:
    """Keep the foreign manifest's template/params/license layers for the new hf.co manifest."""
    layers = [l for l in item.extra.get("layers", []) if l["mediaType"] not in (MT_MODEL, MT_PROJECTOR)]
    if not layers:
        return
    fake = {"schemaVersion": 2, "config": {"digest": "sha256:" + "0" * 64, "size": 0},
            "layers": [{"digest": "sha256:" + plan.etag, "mediaType": MT_MODEL, "size": 0}] + layers}
    p = view_root / REGISTRY_CACHE_DIR / plan.repo_id / (plan.relpath + ".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fake))


def run(config: cfg.Config, key: str, repo_id: str, view_names: list[str], move: bool, execute: bool, out: Out = print) -> None:
    match = [(n, f) for n, f in find_foreign(config, view_names) if f.key == key]
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
        _seed_registry_cache(view.root, plan, item)
        name_tag = key.split("/")[-1]
        config.views["ollama"].aliases[name_tag] = f"{plan.repo_id}:{plan.relpath}"
        cfg.save(config)
        view = sync.build_view(view_name, config)
        old_manifest = view.root / item.extra["manifest"]
        if old_manifest.exists():
            old_manifest.unlink()
    execute_adopt(item, plan, hub, move)
    entries = cache.scan(hub)
    plan_, _ = sync.sync_view(view, entries, execute=True)
    out(f"[{view_name}] synced: " + ", ".join(f"{k}={v}" for k, v in sorted(plan_.summary().items())))


def remove_foreign(config: cfg.Config, key: str, view_names: list[str], execute: bool, out: Out = print) -> None:
    match = [(n, f) for n, f in find_foreign(config, view_names) if f.key == key]
    if not match:
        out(f"no foreign item with key {key}")
        return
    view_name, item = match[0]
    view = sync.build_view(view_name, config, offline=True)
    targets = [item.path]
    if view_name == "ollama":
        targets = [view.root / item.extra["manifest"]]
        others = [p for p in (view.root / "manifests").rglob("*") if p.is_file() and p != targets[0]]
        referenced = set()
        for p in others:
            try:
                referenced |= {l["digest"][7:] for l in json.loads(p.read_text())["layers"]}
            except (ValueError, KeyError, TypeError):
                pass
        for l in item.extra.get("layers", []):
            if l["digest"][7:] not in referenced:
                targets.append(view.root / blob_path(l["digest"][7:]))
    for t in targets:
        out(f"delete {t}")
    if execute:
        from hfhub.views.lmstudio import remove_empty_parents
        for t in targets:
            if t.exists() or t.is_symlink():
                t.unlink()
            remove_empty_parents(view.root, t)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_adopt.py -q`
Expected: 6 passed. The Ollama end-to-end test depends on the registry-cache seeding making the new manifest carry the template layer; if it carries only `model`, check that `_seed_registry_cache` wrote the file at `<root>/.hfhub-registry/ollama/llama3.1/llama3.1-8b.gguf.json` and that `OllamaView._registry_cache` reads the same path (`repo_id` + `filename + ".json"`).

- [ ] **Step 5: Commit**

```bash
git add hfhub/adopt.py tests/test_adopt.py
git commit -m "Add adoption of foreign files into the HF cache"
```

---

### Task 12: Duplicate detection

**Files:**
- Create: `hfhub/dupes.py`, `tests/test_dupes.py`

**Interfaces:**
- Consumes: `gguf_header.read_header`, `file_type_name`, `adopt.find_foreign`, `cache.scan`, `cache.weights`, `sync.hub_dir`.
- Produces: `Signature(architecture: str, file_type: int, params: int)`; `signature(path: Path) -> Signature`; `Dupe(foreign_key, cache_key, reason)`; `find_dupes(foreign: list[tuple[str, Signature]], entries: list[tuple[str, Signature]], tolerance: float = 0.02) -> list[Dupe]`; `run(config, view_names, out=print)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dupes.py`:
```python
from pathlib import Path

from hfhub import dupes
from hfhub.dupes import Signature
from tests.gguf_fixture import write_gguf


def test_signature_reads_header(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"general.architecture": "qwen35", "general.file_type": 15}, [("a", [1000])])
    assert dupes.signature(p) == Signature("qwen35", 15, 1000)


def test_find_dupes_matches_within_tolerance():
    f = [("ollama:x:8b", Signature("llama", 15, 8_030_000_000))]
    c = [("unsloth/L-GGUF:L-Q4_K_M.gguf", Signature("llama", 15, 8_000_000_000)),
         ("unsloth/L-GGUF:L-Q8_0.gguf", Signature("llama", 7, 8_000_000_000)),
         ("o/g:g.gguf", Signature("gemma4", 15, 8_000_000_000)),
         ("o/big:b.gguf", Signature("llama", 15, 9_000_000_000))]
    d = dupes.find_dupes(f, c)
    assert [(x.foreign_key, x.cache_key) for x in d] == [("ollama:x:8b", "unsloth/L-GGUF:L-Q4_K_M.gguf")]
    assert "Q4_K_M" in d[0].reason and "llama" in d[0].reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dupes.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`hfhub/dupes.py`:
```python
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
    for name, item in adopt.find_foreign(config, view_names):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hfhub/dupes.py tests/test_dupes.py
git commit -m "Add duplicate detection for foreign files"
```

---

### Task 13: Opt-in Ollama integration test

**Files:**
- Create: `tests/test_ollama_integration.py`

**Interfaces:**
- Consumes: `sync.run`, `config.Config`; the `ollama` binary.

- [ ] **Step 1: Write the test**

`tests/test_ollama_integration.py`:
```python
"""Opt-in: HFHUB_OLLAMA=1 starts a scratch `ollama serve` and checks synced models are listed."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hfhub import config as cfg, sync
from tests.hub_fixture import add_repo
from tests.test_ollama import FIX

pytestmark = pytest.mark.skipif(not os.environ.get("HFHUB_OLLAMA") or not shutil.which("ollama"),
                                reason="set HFHUB_OLLAMA=1 with ollama installed")


@pytest.fixture
def server(tmp_path: Path):
    root = tmp_path / "ol"
    root.mkdir()
    env = dict(os.environ, OLLAMA_MODELS=str(root), OLLAMA_HOST="127.0.0.1:11499")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    yield root, env
    proc.terminate()
    proc.wait(timeout=10)


def test_synced_model_and_alias_are_listed(tmp_path: Path, server, monkeypatch):
    root, env = server
    hub = tmp_path / "hub"
    src = Path(os.environ["HF_HOME"]) / "hub" / "models--ibm-granite--granite-docling-258M-GGUF"
    real = next(src.glob("snapshots/*/granite-docling-258M-BF16.gguf")).read_bytes()
    add_repo(hub, "ibm-granite/granite-docling-258M-GGUF", {"granite-docling-258M-BF16.gguf": real})
    monkeypatch.setattr(sync, "hub_dir", lambda: hub)
    c = cfg.Config(path=tmp_path / "c.toml", views={
        "lmstudio": cfg.ViewConfig(),
        "ollama": cfg.ViewConfig(root=root, aliases={"docling:bf16": "ibm-granite/granite-docling-258M-GGUF:granite-docling-258M-BF16.gguf"})})
    sync.run(c, ["ollama"], execute=True, offline=False, out=print)
    listed = subprocess.run(["ollama", "list"], env=env, capture_output=True, text=True).stdout
    assert "hf.co/ibm-granite/granite-docling-258M-GGUF:BF16" in listed
    assert "docling:bf16" in listed
    show = subprocess.run(["ollama", "show", "docling:bf16"], env=env, capture_output=True, text=True)
    assert show.returncode == 0
```

- [ ] **Step 2: Run it**

Run: `HFHUB_OLLAMA=1 .venv/bin/python -m pytest tests/test_ollama_integration.py -q -s`
Expected: passes; `ollama list` output shows both names. Note the tag derived for a single-quant repo is whatever `group_quants` names it; if the assertion on `:BF16` fails, print `listed` and adjust the expected tag to what `derive_tags` produced, then add that case to `test_derive_tags_uses_quant_name_else_filename` in `tests/test_ollama.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ollama_integration.py
git commit -m "Add opt-in Ollama integration test"
```

---

### Task 14: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Consumer views" section after the `hf-xfer` section**

Insert before `## Requirements`:

```markdown
---

## Consumer views — LM Studio and Ollama

The HF hub cache is the only place model bytes live. `hf-xfer sync` maintains two derived views of it:

| View | What it writes |
|------|----------------|
| `lmstudio` | `<root>/<org>/<name>/<file>.gguf` symlinks into the cache (mmproj beside the weights) |
| `ollama` | `<root>/blobs/sha256-<hash>` symlinks plus manifests under `manifests/hf.co/<org>/<name>/<tag>`, and alias manifests |

Configure roots in `~/.config/hfhub/config.toml` (or `$HFHUB_CONFIG`):

```toml
[views.lmstudio]
root = "~/.lmstudio/models"

[views.ollama]
root = "/path/to/huggingface/ollama"   # set OLLAMA_MODELS to the same path

[views.ollama.aliases]
"qwen3.6:27b" = "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf"
```

```bash
hf-xfer sync                 # dry run: what would change
hf-xfer sync --execute       # write links and manifests
hf-xfer view status          # owned / tombstoned / foreign per view, unlinked cache blobs
hf-xfer view remove <org/name>:<file> --execute   # remove from views and remember it (tombstone)
hf-xfer view add    <org/name>:<file> --execute   # bring it back
hf-xfer view adopt  <foreign key> --repo-id <org/name> --move --execute   # move a foreign file into the cache
hf-xfer dupes                # foreign files that look like a cache entry
hf-xfer view remove --foreign <foreign key> --execute
```

Rules:

- Dry run is the default. Nothing is written without `--execute`.
- Deleting a model inside LM Studio or with `ollama rm` sticks: sync records a tombstone and does not recreate it.
- Sync never deletes or overwrites anything it did not create. Files that appear in a view without HF backing are listed as *foreign*.
- Ollama manifests are built from `https://huggingface.co/v2/<org>/<name>/manifests/<file>` so templates and parameters match what `ollama pull hf.co/...` would produce, but the model layer always points at your local file, so cached files that are behind the Hub still work. Repos that do not exist on the Hub get a minimal manifest and rely on the chat template embedded in the GGUF.
- `hfu` runs `sync --execute` after every successful download; pass `--no-sync` to skip.

Ollama needs read access to the cache and write access to the view root. With Ollama running as its own user, put the view root on the same drive as the cache, `chgrp ollama` it, `chmod 2775`, and set `OLLAMA_MODELS` in a systemd override together with `RequiresMountsFor=<mount point>`.
```

- [ ] **Step 2: Update the top table**

Change the table under the title to:

```markdown
| Tool | Purpose |
|------|---------|
| `hfu` | Download any Hub repo from a URL, path, or `org/name` shorthand, then refresh the views |
| `hf-xfer` | Translate between a `--local-dir` flat copy and the shared hub cache; maintain LM Studio and Ollama views |
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document consumer views in README"
```

---

## Self-review

**Spec coverage.** Package layout: Task 1. gguf_header: Task 2. config: Task 3. state: Task 4. cache: Task 5. views/base reconcile table: Task 6 (all seven rows tested). LM Studio view incl. basename collision and foreign detection: Task 7. Registry client, patch and synth manifests: Task 8. Ollama view incl. tags, aliases, registry cache with 404 sentinel, offline skip, adopt-existing-real-blob, foreign manifests: Task 9. sync/view status/add/remove, disabled view, StateError and PermissionError handling, hfu hook with `--no-sync`: Task 10. Adopt (Hub-verified, synthesised, Ollama alias + layer carry-over), `view remove --foreign`: Task 11. Dupes: Task 12. Opt-in integration test: Task 13. README: Task 14. Migration steps are operational and stay in the spec.

**Placeholder scan.** No TBD/TODO. Every code step has full code.

**Type consistency.** `Desired(key, sha256, links, extra)`, `Presence`, `SkipEntry`, `ForeignItem(key, path, extra)` used identically in Tasks 6, 7, 9, 10, 11. `sync.build_view`, `sync.sync_view`, `sync.hub_dir` used in Tasks 11 and 12 as defined in Task 10. `OllamaView._registry_cache(d)` used by `sync.view_add` (Task 10) and mirrored by `adopt._seed_registry_cache` path `<root>/.hfhub-registry/<repo_id>/<filename>.json` (Task 11). `xfer.RepoMap(commit_hash, etags, source)` matches the existing dataclass fields.
