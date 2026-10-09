"""`hf-xfer view clean`: broken links into the HF cache that sync does not own."""
import shutil
from pathlib import Path

from hfhub import state as st, sync
from tests.hub_fixture import folder
from tests.test_ollama import gguf_bytes
from tests.test_shared_blobs import make_views, run

VIEWS = ["lmstudio", "ollama"]


def clean(c, execute, lines=None):
    return sync.clean(c, VIEWS, execute=execute, out=(lines.append if lines is not None else lambda *_: None))


def test_removes_only_unowned_broken_links_into_the_cache(tmp_path, monkeypatch):
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path)})
    run(c)
    lm = tmp_path / "lm"
    gone = hub / folder("org/M-GGUF") / "blobs" / ("f" * 64)
    stale = lm / "org/M-GGUF/MTP/old-MTP.gguf"            # left by an older layout, target since deleted
    stale.parent.mkdir(parents=True)
    stale.symlink_to(gone)
    outside = lm / "mine/broken.gguf"; outside.parent.mkdir()
    outside.symlink_to(tmp_path / "nowhere.gguf")         # broken, but not into the cache: the user's
    working = lm / "org/M-GGUF/extra.gguf"
    working.symlink_to(next((hub / folder("org/M-GGUF") / "blobs").iterdir()))

    lines = []
    assert clean(c, execute=False, lines=lines) == 1 and stale.is_symlink()   # dry run writes nothing
    assert any("would remove org/M-GGUF/MTP/old-MTP.gguf" in l for l in lines)
    assert clean(c, execute=True) == 1
    assert not stale.is_symlink() and not stale.parent.exists()               # empty MTP/ folder goes too
    assert outside.is_symlink() and working.is_symlink()
    assert clean(c, execute=True) == 0


def test_owned_broken_links_are_left_to_sync(tmp_path, monkeypatch):
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path)})
    run(c)
    shutil.rmtree(hub / folder("org/M-GGUF") / "blobs")   # every owned link now dangles
    lines = []
    assert clean(c, execute=True, lines=lines) == 0
    assert (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").is_symlink()
    assert any(l.startswith("  owned") and "sync --execute" in l for l in lines)
    assert st.load(tmp_path / "lm").tombstones == {}


def test_ollama_names_the_manifest_that_used_a_removed_blob(tmp_path, monkeypatch):
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path)})
    run(c)
    ol = tmp_path / "ol"
    digest = "e" * 64
    blob = ol / "blobs" / f"sha256-{digest}"
    blob.symlink_to(hub / folder("org/M-GGUF") / "blobs" / digest)          # broken, not owned
    m = ol / "manifests/registry.ollama.ai/library/someone/latest"
    m.parent.mkdir(parents=True)
    m.write_text('{"config": {"digest": "sha256:%s"}, "layers": [{"digest": "sha256:%s"}]}' % ("0" * 64, digest))
    lines = []
    assert clean(c, execute=True, lines=lines) == 1 and not blob.is_symlink()
    assert any("registry.ollama.ai/library/someone/latest" in l for l in lines)
    assert m.exists()                                                         # reported, not deleted
