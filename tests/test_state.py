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


def test_owned_entry_missing_sha256_raises(tmp_path: Path):
    (tmp_path / st.STATE_FILE).write_text(json.dumps(
        {"version": 1, "owned": {"k1": {"paths": ["a"]}}, "tombstones": {}}))
    with pytest.raises(st.StateError):
        st.load(tmp_path)


def test_owned_entry_paths_as_string_raises(tmp_path: Path):
    (tmp_path / st.STATE_FILE).write_text(json.dumps(
        {"version": 1, "owned": {"k1": {"paths": "a/b/c.gguf", "sha256": "x" * 64}}, "tombstones": {}}))
    with pytest.raises(st.StateError):
        st.load(tmp_path)


def test_tombstones_as_list_raises(tmp_path: Path):
    (tmp_path / st.STATE_FILE).write_text(json.dumps(
        {"version": 1, "owned": {}, "tombstones": ["a/b:c.gguf"]}))
    with pytest.raises(st.StateError):
        st.load(tmp_path)
