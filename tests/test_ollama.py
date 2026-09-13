import json
from pathlib import Path

from hfhub import cache, ollama_registry as reg
from hfhub.state import Owned, State
from hfhub.views.base import Presence, SkipEntry
from hfhub.views.ollama import OllamaView, alias_manifest_path, blob_path, derive_tags
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
    add_repo(hub, "o/ud", {"m-UD-Q4_K_XL.gguf": b"4"})
    add_repo(hub, "o/dup", {"a-Q4_K_M.gguf": b"5", "sub/b-Q4_K_M.gguf": b"6"})
    tags = derive_tags(cache.scan(hub))
    assert tags["o/r:r-Q4_K_M.gguf"] == "Q4_K_M"
    assert tags["o/r:r-Q8_0.gguf"] == "Q8_0"
    assert tags["o/r:weird name.gguf"] == "weird_name.gguf"
    assert "o/r:mmproj-F16.gguf" not in tags
    assert tags["o/ud:m-UD-Q4_K_XL.gguf"] == "UD-Q4_K_XL"
    assert tags["o/dup:a-Q4_K_M.gguf"] == "a-Q4_K_M.gguf"
    assert tags["o/dup:sub/b-Q4_K_M.gguf"] == "sub_b-Q4_K_M.gguf"


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


def two_models(tmp_path: Path, **kw):
    """One repo, two weights sharing a basename (so one cached registry response)."""
    files = {"M-Q4_K_M.gguf": gguf_bytes(tmp_path), "sub/M-Q4_K_M.gguf": gguf_bytes(tmp_path, arch="gemma5"),
             "mmproj-F16.gguf": b"proj"}
    return make(tmp_path, files=files, **kw)


def test_offline_without_cached_response_skips(tmp_path: Path):
    hub, entries, view, calls = make(tmp_path, offline=True)
    (d,) = view.desired(entries).values()
    try:
        view.create(d)
        assert False, "expected SkipEntry"
    except SkipEntry:
        pass
    assert calls["manifest"] == []
    assert not (view.root / "blobs").exists()  # skipped before anything was written


def test_offline_with_full_cache_creates_without_network(tmp_path: Path):
    hub, entries, view, calls = two_models(tmp_path)
    desired = view.desired(entries)
    view.create(desired["org/M-GGUF:M-Q4_K_M.gguf"])
    seen = {"manifest": [], "blob": []}

    def fm(org, name, tag, token=None, timeout=30):
        seen["manifest"].append(tag)
        return FIX

    def fb(org, name, digest, token=None, timeout=60):
        seen["blob"].append(digest)
        return b"x"

    off = OllamaView(view.root, aliases={}, offline=True, fetch_manifest=fm, fetch_blob=fb)
    d2 = off.desired(entries)["org/M-GGUF:sub/M-Q4_K_M.gguf"]
    off.create(d2)
    assert seen == {"manifest": [], "blob": []}
    assert off.present(d2) is Presence.CORRECT


def test_alias_added_after_create_gives_partial_and_create_fills(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path)
    (d,) = view.desired(entries).values()
    view.create(d)
    assert view.present(d) is Presence.CORRECT
    view.aliases = {"m:latest": d.key}
    (d2,) = view.desired(entries).values()
    assert view.present(d2) is Presence.PARTIAL
    paths = view.create(d2)
    assert alias_manifest_path("m:latest") in paths
    assert view.present(d2) is Presence.CORRECT


def test_alias_of_another_model_is_kept_but_stale_alias_of_ours_is_refreshed(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path, aliases={"m:latest": "org/M-GGUF:M-Q4_K_M.gguf"})
    (d,) = view.desired(entries).values()
    p = view.root / alias_manifest_path("m:latest")
    p.parent.mkdir(parents=True)
    theirs = json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "2" * 64},
                         "layers": [{"digest": "sha256:" + "3" * 64, "mediaType": reg.MT_MODEL, "size": 1}]})
    p.write_text(theirs)
    paths = view.create(d)
    assert p.read_text() == theirs
    assert alias_manifest_path("m:latest") not in paths
    assert any("another model" in w for w in view.warnings)
    stale = json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "2" * 64},
                        "layers": [{"digest": "sha256:" + d.sha256, "mediaType": reg.MT_MODEL, "size": 1}]})
    p.write_text(stale)
    paths = view.create(d)
    assert alias_manifest_path("m:latest") in paths
    assert json.loads(p.read_text()) == json.loads((view.root / d.extra["manifest"]).read_text())


def test_remove_keeps_blob_referenced_by_foreign_manifest(tmp_path: Path):
    hub, entries, view, _ = make(tmp_path)
    (d,) = view.desired(entries).values()
    paths = view.create(d)
    mp = view.root / "manifests/registry.ollama.ai/library/other/latest"
    mp.parent.mkdir(parents=True)
    mp.write_text(json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "2" * 64},
                              "layers": [{"digest": "sha256:" + d.sha256, "mediaType": reg.MT_MODEL, "size": 1}]}))
    view.remove(paths)
    mm = next(e for e in entries if e.is_mmproj)
    assert (view.root / blob_path(d.sha256)).exists()          # still referenced by their manifest
    assert not (view.root / blob_path(mm.sha256)).exists()     # unreferenced: removed
    assert not (view.root / d.extra["manifest"]).exists()


def test_remove_keeps_small_blob_shared_with_our_other_manifest(tmp_path: Path):
    hub, entries, view, _ = two_models(tmp_path)
    desired = view.desired(entries)
    a, b = desired["org/M-GGUF:M-Q4_K_M.gguf"], desired["org/M-GGUF:sub/M-Q4_K_M.gguf"]
    paths_a = view.create(a)
    view.create(b)
    tmpl = blob_path(next(l for l in FIX["layers"] if l["mediaType"] == reg.MT_TEMPLATE)["digest"][7:])
    assert tmpl in paths_a
    view.remove(paths_a)
    assert (view.root / tmpl).is_file()                        # b's manifest still points at it
    assert not (view.root / blob_path(a.sha256)).exists()      # a's own weight blob goes
    assert view.present(b) is Presence.CORRECT


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
    assert not (view.root / "blobs").exists() or not any((view.root / "blobs").iterdir())


def test_shards_are_skipped_with_warning(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"r-BF16-00001-of-00002.gguf": b"1", "r-BF16-00002-of-00002.gguf": b"2"})
    view = OllamaView(tmp_path / "ol", aliases={}, fetch_manifest=lambda *a, **k: None, fetch_blob=lambda *a, **k: b"")
    assert view.desired(cache.scan(hub)) == {}
    assert any("shard" in w for w in view.warnings)
