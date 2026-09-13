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


def test_missing_mmproj_link_is_partial_and_create_fills(tmp_path: Path):
    hub, entries, view = setup(tmp_path)
    (d,) = view.desired(entries).values()
    rel = "org/M-GGUF/M-Q4_K_M.gguf"
    p = view.root / rel
    p.parent.mkdir(parents=True)
    p.symlink_to(d.links[rel])
    assert view.present(d) is Presence.PARTIAL
    view.create(d)
    assert view.present(d) is Presence.CORRECT
