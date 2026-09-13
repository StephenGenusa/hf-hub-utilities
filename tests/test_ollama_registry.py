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


def test_http_400_is_not_found(monkeypatch):
    import urllib.error, urllib.request

    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert reg.fetch_manifest("o", "r", "sub%2Fx.gguf") is None
