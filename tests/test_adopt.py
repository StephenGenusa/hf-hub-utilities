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
    f = tmp_path / ("sha256-" + "1" * 64)
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


def test_remove_foreign_keeps_shared_blobs_and_needs_execute(tmp_path: Path):
    ol = tmp_path / "ol"
    (ol / "blobs").mkdir(parents=True)
    model, other, shared = ("a" * 64, "b" * 64, "c" * 64)
    for d in (model, other, shared):
        (ol / "blobs" / f"sha256-{d}").write_bytes(b"x")
    MT_MODEL = "application/vnd.ollama.image.model"
    MT_TEMPLATE = "application/vnd.ollama.image.template"

    def manifest(rel: str, digest: str) -> Path:
        p = ol / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "0" * 64, "size": 0},
                                 "layers": [{"digest": f"sha256:{digest}", "mediaType": MT_MODEL, "size": 1},
                                            {"digest": f"sha256:{shared}", "mediaType": MT_TEMPLATE, "size": 1}]}))
        return p

    mine = manifest("manifests/registry.ollama.ai/library/m/8b", model)
    theirs = manifest("manifests/registry.ollama.ai/library/n/8b", other)
    c = cfg.Config(path=tmp_path / "c.toml", views={"lmstudio": cfg.ViewConfig(), "ollama": cfg.ViewConfig(root=ol)})
    key = "ollama:registry.ollama.ai/library/m:8b"

    lines: list[str] = []
    adopt.remove_foreign(c, key, ["ollama"], execute=False, out=lines.append)
    assert any(str(mine) in l for l in lines) and any(f"sha256-{model}" in l for l in lines)
    assert not any(f"sha256-{shared}" in l for l in lines)
    assert mine.is_file() and (ol / "blobs" / f"sha256-{model}").is_file()  # dry run wrote nothing

    adopt.remove_foreign(c, key, ["ollama"], execute=True, out=lambda *_: None)
    assert not mine.exists()
    assert not (ol / "blobs" / f"sha256-{model}").exists()
    assert (ol / "blobs" / f"sha256-{shared}").is_file()  # still referenced by the other manifest
    assert theirs.is_file() and (ol / "blobs" / f"sha256-{other}").is_file()
