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
    (tmp_path / "lm").mkdir(exist_ok=True)
    (tmp_path / "ol").mkdir(exist_ok=True)
    c = cfg.Config(path=tmp_path / "c.toml", views={
        "lmstudio": cfg.ViewConfig(root=tmp_path / "lm"),
        "ollama": cfg.ViewConfig(root=tmp_path / "ol", aliases={"m:q4": "org/M-GGUF:M-Q4_K_M.gguf"})})
    return hub, c


def test_dry_run_plans_but_writes_nothing(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    plans = sync.run(c, ["lmstudio", "ollama"], execute=False, offline=False, out=lambda *_: None)
    assert plans["lmstudio"].summary() == {"create": 1}
    assert plans["ollama"].summary() == {"create": 1}
    assert list((tmp_path / "lm").iterdir()) == [] and list((tmp_path / "ol").iterdir()) == []


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
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None,
                     allow_mass_removal=True)
    assert plans["lmstudio"].summary() == {"tombstone": 1}
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    assert plans["lmstudio"].summary() == {"skip": 1}
    add_repo(hub, "org/N-GGUF", {"N-Q4_K_M.gguf": gguf_bytes(tmp_path)})
    lines = []
    sync.view_add(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lines.append)
    assert (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").is_symlink()
    assert "org/M-GGUF:M-Q4_K_M.gguf" not in st.load(tmp_path / "lm").tombstones
    # the whole view is synced, so the whole plan is reported: the requested key
    # is starred, the rest of the plan is listed plainly.
    starred = [m for m in lines if m.startswith("*")]
    assert starred == [m for m in starred if "org/M-GGUF:M-Q4_K_M.gguf" in m] and starred
    assert any("org/N-GGUF:N-Q4_K_M.gguf" in m and not m.startswith("*") for m in lines)
    assert (tmp_path / "lm/org/N-GGUF/N-Q4_K_M.gguf").is_symlink()


def test_absent_cache_dir_does_nothing(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    before = st.load(tmp_path / "lm")
    monkeypatch.setattr(sync, "hub_dir", lambda: tmp_path / "no-such-cache")
    lines = []
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lines.append)
    assert plans == {}
    assert lines == [f"[cache] not found: {tmp_path / 'no-such-cache'}; nothing done"]
    assert (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").is_symlink()
    assert st.load(tmp_path / "lm").owned == before.owned
    lines = []
    sync.view_add(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lines.append)
    assert lines == [f"[cache] not found: {tmp_path / 'no-such-cache'}; nothing done"]
    assert st.load(tmp_path / "lm").owned == before.owned


def test_all_entries_gone_from_cache_is_refused_without_the_flag(tmp_path, monkeypatch):
    import shutil
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    shutil.rmtree(hub / "models--org--M-GGUF")
    lines = []
    plans = sync.run(c, ["lmstudio"], execute=False, offline=False, out=lines.append)
    assert plans["lmstudio"].summary() == {"prune": 1}          # dry run still reports the plan
    assert not any("refused" in m for m in lines)
    lines = []
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lines.append)
    assert "lmstudio" not in plans
    assert lines == ["[lmstudio] refused: plan would remove every owned entry (1); "
                     "pass --allow-mass-removal if this is intended"]
    assert (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").is_symlink()
    assert st.load(tmp_path / "lm").owned
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None, allow_mass_removal=True)
    assert not (tmp_path / "lm/org").exists() and not st.load(tmp_path / "lm").owned


def test_all_entries_gone_from_disk_is_refused_without_the_flag(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None)
    (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").unlink()
    lines = []
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lines.append)
    assert any("refused: plan would remove every owned entry (1)" in m for m in lines)
    assert st.load(tmp_path / "lm").owned and not st.load(tmp_path / "lm").tombstones
    sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None, allow_mass_removal=True)
    assert st.load(tmp_path / "lm").tombstones and not st.load(tmp_path / "lm").owned


def test_missing_view_root_is_skipped_and_not_created(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    root = tmp_path / "nope"
    c.views["lmstudio"].root = root
    notice = f"[lmstudio] skipped: root does not exist: {root}"
    for call in (lambda out: sync.run(c, ["lmstudio"], execute=True, offline=False, out=out),
                 lambda out: sync.status(c, ["lmstudio"], out=out),
                 lambda out: sync.view_add(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=out),
                 lambda out: sync.view_remove(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=out)):
        lines = []
        call(lines.append)
        assert notice in lines, lines
        assert not root.exists()


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
    plans = sync.run(c, ["lmstudio"], execute=True, offline=False, out=lambda *_: None,
                     allow_mass_removal=True)
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
    (tmp_path / "lm" / st.STATE_FILE).write_text("{ not json")
    lines = []
    sync.status(c, ["lmstudio", "ollama"], out=lines.append)
    text = "\n".join(lines)
    assert "[lmstudio] aborted:" in text and "unreadable state file" in text
    assert "[ollama] root=" in text and "desired: 1" in text


def test_view_remove_on_corrupt_state_reports_and_continues(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    (tmp_path / "lm" / st.STATE_FILE).write_text("{ not json")
    lines = []
    sync.view_remove(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lines.append)
    assert any("[lmstudio] aborted:" in m for m in lines)


def test_view_add_on_corrupt_state_reports_and_continues(tmp_path, monkeypatch):
    hub, c = make(tmp_path, monkeypatch)
    (tmp_path / "lm" / st.STATE_FILE).write_text("{ not json")
    lines = []
    sync.view_add(c, "org/M-GGUF:M-Q4_K_M.gguf", ["lmstudio"], execute=True, out=lines.append)
    assert any("[lmstudio] aborted:" in m for m in lines)
