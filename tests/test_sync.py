from pathlib import Path

from hfhub import config as cfg, state as st, sync
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


def test_status_skips_a_view_with_a_corrupt_state_file(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    (tmp_path / "lm").mkdir()
    (tmp_path / "lm" / st.STATE_FILE).write_text("{ not json")
    lines = []
    sync.status(c, ["lmstudio", "ollama"], out=lines.append)
    text = "\n".join(lines)
    assert "[lmstudio] aborted:" in text and "unreadable state file" in text
    assert "[ollama] root=" in text and "desired: 1" in text


def test_view_remove_on_corrupt_state_reports_and_continues(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    (tmp_path / "lm").mkdir()
    (tmp_path / "lm" / st.STATE_FILE).write_text("{ not json")
    lines = []
    sync.view_remove(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lines.append)
    assert any("[lmstudio] aborted:" in m for m in lines)


def test_view_add_on_corrupt_state_reports_and_continues(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    (tmp_path / "lm").mkdir()
    (tmp_path / "lm" / st.STATE_FILE).write_text("{ not json")
    lines = []
    sync.view_add(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lines.append)
    assert any("[lmstudio] aborted:" in m for m in lines)
