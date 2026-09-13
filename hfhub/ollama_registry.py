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
        # huggingface.co's registry returns 401 (not 404) for a repo that
        # doesn't exist when the request is unauthenticated, to avoid
        # revealing whether a private repo exists. Treat both as "not found".
        if e.code in (404, 401):
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
