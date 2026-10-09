"""The huggingface_hub 2.x shared blob store, sync relinking an updated cache file,
adopt's refs/main handling, and `hf-xfer verify`."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from hfhub import adopt, blobs, cache, cache_ops, config as cfg, state as st, sync, verify, xfer
from hfhub.views.base import ForeignItem
from hfhub.xfer import RepoMap
from tests.hub_fixture import add_repo, folder
from tests.test_ollama import FIX, gguf_bytes

shared_store = pytest.mark.skipif(blobs._hf_shared is None, reason="huggingface_hub without a shared blob store")


def share(hub: Path, repo_id: str, content: bytes) -> Path:
    """Turn a repo's regular blob into a link into the shared store, as a 2.x download does."""
    blob = hub / folder(repo_id) / "blobs" / hashlib.sha256(content).hexdigest()
    xet_hash = hashlib.sha256(b"xet:" + content).hexdigest()
    assert blobs._hf_shared.publish_blob_to_shared_store(
        blob_path=str(blob), xet_hash=xet_hash, cache_dir=hub, expected_size=len(content))
    assert blob.is_symlink()
    return blob


def store_files(hub: Path) -> list[Path]:
    return sorted(p for p in (hub / "blobs").rglob("*") if p.is_file() and len(p.name) == 64)


# ------------------------------------------------------------------ scan
@shared_store
def test_scan_reports_the_sha256_not_the_store_name(tmp_path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m-Q4_K_M.gguf": b"weights"})
    blob = share(hub, "o/r", b"weights")
    [e] = cache.scan(hub)
    assert e.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert e.blob == Path(os.path.realpath(blob.parent)) / blob.name   # the repo blob, not the store file
    assert e.size == len(b"weights")


def test_scan_of_a_regular_blob_is_unchanged(tmp_path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m-Q4_K_M.gguf": b"weights"})
    [e] = cache.scan(hub)
    snap_link = next((hub / folder("o/r") / "snapshots").rglob("*.gguf"))
    assert e.blob == Path(os.path.realpath(snap_link))   # what scan returned before the store existed


# ------------------------------------------------------------------ cache ops
@shared_store
def test_thin_collects_the_store_file(tmp_path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m-Q4_K_M.gguf": b"big-q4", "m-Q8_0.gguf": b"big-q8"})
    share(hub, "o/r", b"big-q4")
    res = cache_ops.thin(hub, execute=True, min_bits=8, out=lambda *_: None)
    assert res.actions == 1 and res.bytes == len(b"big-q4")
    assert store_files(hub) == []
    assert [e.relpath for e in cache.scan(hub)] == ["m-Q8_0.gguf"]


@shared_store
def test_thin_keeps_a_store_file_another_repo_uses(tmp_path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/a", {"m-Q4_K_M.gguf": b"same", "m-Q8_0.gguf": b"a-q8"})
    add_repo(hub, "o/b", {"m-Q4_K_M.gguf": b"same"})
    share(hub, "o/a", b"same")
    share(hub, "o/b", b"same")
    assert cache_ops.thin(hub, execute=False, min_bits=8, repos=["o/a"], out=lambda *_: None).bytes == 0
    res = cache_ops.thin(hub, execute=True, min_bits=8, repos=["o/a"], out=lambda *_: None)
    assert res.actions == 1 and res.bytes == 0          # nothing freed: o/b still uses the bytes
    assert len(store_files(hub)) == 1
    assert (hub / folder("o/b") / "blobs" / hashlib.sha256(b"same").hexdigest()).read_bytes() == b"same"


@shared_store
def test_dedupe_links_a_plain_copy_into_the_store(tmp_path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/a", {"m.gguf": b"same"})
    add_repo(hub, "o/b", {"m.gguf": b"same"})
    a = share(hub, "o/a", b"same")
    b = hub / folder("o/b") / "blobs" / a.name
    res = cache_ops.dedupe(hub, execute=True, out=lambda *_: None)
    assert res.actions == 1 and a.is_symlink() and b.is_symlink()
    assert os.path.realpath(a) == os.path.realpath(b) and b.read_bytes() == b"same"
    assert len(store_files(hub)) == 1
    assert cache_ops.dedupe(hub, execute=True, out=lambda *_: None).actions == 0
    # the store still knows both links: dropping one repo keeps the bytes for the other
    assert blobs.delete_blob(a) == 0 and b.read_bytes() == b"same"
    assert blobs.delete_blob(b) == len(b"same") and store_files(hub) == []


@shared_store
def test_dedupe_hardlinks_copies_the_store_would_not_take(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    for r in ("o/a", "o/b", "o/c"):
        add_repo(hub, r, {"m.gguf": b"same"})
    share(hub, "o/a", b"same")
    monkeypatch.setattr(blobs, "link_into_store", lambda blob, store: False)
    lines = []
    res = cache_ops.dedupe(hub, execute=True, out=lines.append)
    b, c = (hub / folder(r) / "blobs" / hashlib.sha256(b"same").hexdigest() for r in ("o/b", "o/c"))
    assert not any("linked into the shared store" in l for l in lines)
    assert any("could not link into the shared store" in n for n in res.notes)
    assert not b.is_symlink() and b.stat().st_ino == c.stat().st_ino   # still deduplicated, by hardlink
    assert res.actions == 1


@shared_store
def test_gc_collects_a_leaked_store_file(tmp_path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"leak"})
    blob = share(hub, "o/r", b"leak")
    blob.unlink()  # what a pre-fix thin/prune did: the repo link goes, the store file stays
    assert cache_ops.gc(hub, execute=False, out=lambda *_: None).actions == 1
    assert len(store_files(hub)) == 1
    assert cache_ops.gc(hub, execute=True, out=lambda *_: None).bytes == len(b"leak")
    assert store_files(hub) == []


# ------------------------------------------------------------------ export
@shared_store
def test_export_move_copies_a_store_file_and_uses_the_etag(tmp_path):
    hub = tmp_path / "hub"
    commit = add_repo(hub, "o/r", {"m.gguf": b"payload"})
    share(hub, "o/r", b"payload")
    repo_root = hub / folder("o/r")
    plan = xfer.plan_export(repo_root, commit)
    assert [c[1] for c in plan.can] == [hashlib.sha256(b"payload").hexdigest()]
    out = tmp_path / "out"
    xfer.execute_export(repo_root, commit, out, plan, "move")
    assert (out / "m.gguf").read_bytes() == b"payload"
    assert store_files(hub) == []                        # collected, not renamed out of the store


# ------------------------------------------------------------------ sync: relinking an updated file
def make_views(tmp_path: Path, monkeypatch, files: dict[str, bytes]):
    hub = tmp_path / "hub"
    add_repo(hub, "org/M-GGUF", files, commit="a" * 40)
    monkeypatch.setattr(sync, "hub_dir", lambda: hub)
    monkeypatch.setattr(sync.reg, "fetch_manifest", lambda *a, **k: FIX)
    monkeypatch.setattr(sync.reg, "fetch_blob", lambda *a, **k: b"{}")
    for d in ("lm", "ol"):
        (tmp_path / d).mkdir(exist_ok=True)
    c = cfg.Config(path=tmp_path / "c.toml", views={
        "lmstudio": cfg.ViewConfig(root=tmp_path / "lm"),
        "ollama": cfg.ViewConfig(root=tmp_path / "ol", aliases={"m:q4": "org/M-GGUF:M-Q4_K_M.gguf"})})
    return hub, c


def new_revision(hub: Path, files: dict[str, bytes]) -> None:
    """A newer download: a second snapshot that refs/main moves to."""
    add_repo(hub, "org/M-GGUF", files, commit="b" * 40)


def run(c, execute=True):
    return sync.run(c, ["lmstudio", "ollama"], execute=execute, offline=False, out=lambda *_: None)


def model_digest(manifest: Path) -> str:
    return next(l["digest"] for l in json.loads(manifest.read_text())["layers"] if l["mediaType"].endswith(".model"))


def test_updated_file_is_relinked_in_both_views(tmp_path, monkeypatch):
    old, new = gguf_bytes(tmp_path), gguf_bytes(tmp_path, ft=7)
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": old})
    run(c)
    new_revision(hub, {"M-Q4_K_M.gguf": new})
    plans = run(c, execute=False)
    assert plans["lmstudio"].summary() == {"update": 1} and plans["ollama"].summary() == {"update": 1}
    run(c)
    new_sha = hashlib.sha256(new).hexdigest()
    assert (tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf").read_bytes() == new
    ol = tmp_path / "ol"
    assert model_digest(ol / "manifests/hf.co/org/M-GGUF/Q4_K_M") == "sha256:" + new_sha
    assert model_digest(ol / "manifests/registry.ollama.ai/library/m/q4") == "sha256:" + new_sha  # alias too
    assert not (ol / "blobs" / f"sha256-{hashlib.sha256(old).hexdigest()}").exists()            # old link gone
    assert st.load(ol).owned["org/M-GGUF:M-Q4_K_M.gguf"].sha256 == new_sha
    plans = run(c)
    assert plans["lmstudio"].changes() == [] and plans["ollama"].changes() == []


def test_added_mmproj_updates_the_ollama_manifest(tmp_path, monkeypatch):
    w = gguf_bytes(tmp_path)
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": w})
    run(c)
    new_revision(hub, {"M-Q4_K_M.gguf": w, "mmproj-F16.gguf": b"projector"})
    plans = run(c)
    assert plans["ollama"].summary() == {"update": 1}
    layers = json.loads((tmp_path / "ol/manifests/hf.co/org/M-GGUF/Q4_K_M").read_text())["layers"]
    assert any(l["mediaType"].endswith(".projector") for l in layers)
    assert (tmp_path / "lm/org/M-GGUF/mmproj-F16.gguf").read_bytes() == b"projector"


def test_link_into_cache_sees_through_the_windows_extended_path_prefix(tmp_path, monkeypatch):
    from hfhub.views.base import link_into_cache
    target = tmp_path / "hub" / folder("o/r") / "blobs" / ("a" * 64)
    link = tmp_path / "view" / "m.gguf"; link.parent.mkdir()
    link.symlink_to(target)
    real_readlink = os.readlink
    monkeypatch.setattr(os, "readlink", lambda p: "\\\\?\\" + real_readlink(p))   # what Windows returns
    assert link_into_cache(link, target)


def test_a_user_file_in_place_of_a_link_is_still_foreign(tmp_path, monkeypatch):
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path)})
    run(c)
    link = tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf"
    link.unlink()
    elsewhere = tmp_path / "mine.gguf"; elsewhere.write_bytes(b"mine")
    link.symlink_to(elsewhere)
    plans = run(c)
    assert plans["lmstudio"].summary() == {"foreign": 1}
    assert os.readlink(link) == str(elsewhere)


def test_an_ollama_manifest_rewritten_by_someone_else_is_still_foreign(tmp_path, monkeypatch):
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path)})
    run(c)
    m = tmp_path / "ol/manifests/hf.co/org/M-GGUF/Q4_K_M"
    data = json.loads(m.read_text())
    for l in data["layers"]:
        if l["mediaType"].endswith(".model"):
            l["digest"] = "sha256:" + "9" * 64
    m.write_text(json.dumps(data))
    plans = run(c)
    assert plans["ollama"].summary() == {"foreign": 1}
    assert model_digest(m) == "sha256:" + "9" * 64


@shared_store
def test_links_written_before_the_store_fix_are_relinked(tmp_path, monkeypatch):
    """A view link to the store file (what a pre-fix scan produced) is updated to the repo blob."""
    w = gguf_bytes(tmp_path)
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": w})
    blob = share(hub, "org/M-GGUF", w)
    run(c)
    link = tmp_path / "lm/org/M-GGUF/M-Q4_K_M.gguf"
    link.unlink()
    link.symlink_to(os.path.realpath(blob))   # old behaviour: straight to hub/blobs/xx/<xet hash>
    assert run(c)["lmstudio"].summary() == {"update": 1}
    assert os.readlink(link) == str(Path(os.path.realpath(blob.parent)) / blob.name)


def test_a_file_renamed_in_the_cache_takes_its_alias_along_in_one_sync(tmp_path, monkeypatch):
    q8 = gguf_bytes(tmp_path, ft=8)   # a second model, so the rename is not a mass removal
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path), "M-Q8_0.gguf": q8})
    run(c)
    new_revision(hub, {"M-UD-Q4_K_M.gguf": gguf_bytes(tmp_path, ft=7), "M-Q8_0.gguf": q8})
    import shutil
    shutil.rmtree(hub / folder("org/M-GGUF") / "snapshots" / ("a" * 40))   # the old name is gone
    c.views["ollama"].aliases["m:q4"] = "org/M-GGUF:M-UD-Q4_K_M.gguf"
    plans = run(c)
    assert {a.kind for a in plans["ollama"].changes()} == {"create", "prune"}
    alias = tmp_path / "ol/manifests/registry.ollama.ai/library/m/q4"
    assert model_digest(alias) == "sha256:" + hashlib.sha256(gguf_bytes(tmp_path, ft=7)).hexdigest()
    assert run(c)["ollama"].changes() == []


# ------------------------------------------------------------------ adopt refs/main
def test_hub_backed_adopt_moves_refs_main_off_a_synthesised_revision(tmp_path):
    hub = tmp_path / "hub"; hub.mkdir()
    a = tmp_path / "a.gguf"; a.write_bytes(b"old build")
    sha_a = hashlib.sha256(b"old build").hexdigest()
    first = adopt.plan_adopt(ForeignItem("lmstudio:o/r/a.gguf", a), "o/r", sha_a, lambda rid: None)
    adopt.execute_adopt(ForeignItem("k", a), first, hub, move=False)
    root = hub / "models--o--r"
    assert adopt.is_synthesised(root, "o/r", first.commit)
    b = tmp_path / "b.gguf"; b.write_bytes(b"mmproj")
    sha_b = hashlib.sha256(b"mmproj").hexdigest()
    real = lambda rid: RepoMap(commit_hash="c" * 40, etags={"mmproj-F16.gguf": sha_b}, source="hub")
    second = adopt.plan_adopt(ForeignItem("lmstudio:o/r/b.gguf", b), "o/r", sha_b, real)
    adopt.execute_adopt(ForeignItem("k", b), second, hub, move=False)
    assert (root / "refs/main").read_text() == "c" * 40
    assert not adopt.is_synthesised(root, "o/r", "c" * 40)
    assert {e.key for e in cache.scan(hub)} == {"o/r:a.gguf", "o/r:mmproj-F16.gguf"}


def test_adopt_reports_which_kind_of_revision_refs_main_stayed_on(tmp_path, monkeypatch):
    hub = tmp_path / "hub"; hub.mkdir()
    add_repo(hub, "o/r", {"real.gguf": b"real"}, commit="c" * 40)
    lm = tmp_path / "lm"; (lm / "o/r").mkdir(parents=True)
    (lm / "o/r/x.gguf").write_bytes(gguf_bytes(tmp_path))
    monkeypatch.setattr(sync, "hub_dir", lambda: hub)
    monkeypatch.setattr(adopt, "hub_lookup", lambda rid: None)
    c = cfg.Config(path=tmp_path / "c.toml", views={"lmstudio": cfg.ViewConfig(root=lm), "ollama": cfg.ViewConfig()})
    lines = []
    adopt.run(c, "lmstudio:o/r/x.gguf", "o/r", ["lmstudio"], move=True, execute=True, out=lines.append)
    msg = next(l for l in lines if l.startswith("refs/main left at"))
    assert "an existing revision" in msg and "real Hub revision" not in msg


# ------------------------------------------------------------------ verify
def test_verify_clean_setup_has_no_problems(tmp_path, monkeypatch):
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path), "mmproj-F16.gguf": b"p"})
    run(c)
    assert verify.run(c, ["lmstudio", "ollama"], out=lambda *_: None) == 0


def test_verify_finds_broken_links_orphans_and_pending_updates(tmp_path, monkeypatch):
    old = gguf_bytes(tmp_path)
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": old})
    run(c)
    (tmp_path / "ol/blobs" / ("sha256-" + "e" * 64)).write_bytes(b"orphan")
    stray = tmp_path / "lm/org/Other/x.gguf"; stray.parent.mkdir(parents=True)
    stray.symlink_to(hub / folder("org/M-GGUF") / "blobs" / ("f" * 64))   # broken link into the cache
    new_revision(hub, {"M-Q4_K_M.gguf": gguf_bytes(tmp_path, ft=7)})
    lines = []
    problems = verify.run(c, ["lmstudio", "ollama"], out=lines.append)
    text = "\n".join(lines)
    assert "broken link" in text and "no manifest uses" in text and "links an older cache file" in text
    assert problems == 4   # broken link, orphan blob, and one pending update per view


def test_verify_hub_flags_outdated_and_renamed_files(tmp_path, monkeypatch):
    hub, c = make_views(tmp_path, monkeypatch, {"M-Q4_K_M.gguf": b"q4", "m-mxfp4.gguf": b"mx"})

    class Sib:
        def __init__(self, name, sha):
            self.rfilename, self.lfs = name, {"sha256": sha}

    info = type("Info", (), {"siblings": [Sib("M-Q4_K_M.gguf", "1" * 64), Sib("m-MXFP4.gguf", "2" * 64)]})
    lines = []
    r = verify.Report(lines.append)
    verify.check_hub(cache.scan(hub), r, info_fn=lambda rid: info)
    assert r.problems == 2
    assert any("outdated" in l for l in lines) and any("renamed on the Hub to m-MXFP4.gguf" in l for l in lines)
