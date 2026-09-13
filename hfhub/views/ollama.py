"""Ollama view: blob symlinks + hand-written manifests under an OLLAMA_MODELS directory."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from hfhub import ollama_registry as reg
from hfhub.cache import GgufEntry, mmproj_for, weights
from hfhub.gguf_header import read_header
from hfhub.state import State
from hfhub.views.base import Desired, ForeignItem, Presence, SkipEntry
from hfhub.views.lmstudio import remove_empty_parents

REGISTRY_CACHE_DIR = ".hfhub-registry"
_TAG_OK = re.compile(r"^[A-Za-z0-9._-]+$")
_QUANT = re.compile(r"(?:^|[-._])((?:UD-)?(?:IQ|Q|TQ)\d[A-Za-z0-9_]*|BF16|F16|F32|MXFP4(?:_MOE)?)$")


def blob_path(sha256: str) -> str:
    return f"blobs/sha256-{sha256}"


def manifest_path(repo_id: str, tag: str) -> str:
    return f"manifests/hf.co/{repo_id}/{tag}"


def alias_manifest_path(alias: str) -> str:
    name, _, tag = alias.partition(":")
    tag = tag or "latest"
    ns, _, short = name.rpartition("/")
    return f"manifests/registry.ollama.ai/{ns or 'library'}/{short}/{tag}"


def _safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def derive_tags(entries: list[GgufEntry]) -> dict[str, str]:
    """key -> Ollama tag for every non-shard weight entry, per repo."""
    out: dict[str, str] = {}
    by_repo: dict[str, list[GgufEntry]] = {}
    for e in weights(entries):
        if not e.is_shard:
            by_repo.setdefault(e.repo_id, []).append(e)
    for es in by_repo.values():
        for e in es:
            stem = e.basename[: -len(".gguf")] if e.basename.lower().endswith(".gguf") else e.basename
            m = _QUANT.search(stem)
            out[e.key] = m.group(1) if m and _TAG_OK.match(m.group(1)) else _safe_tag(e.basename)
        used: dict[str, int] = {}
        for e in es:
            used[out[e.key]] = used.get(out[e.key], 0) + 1
        for e in es:
            if used[out[e.key]] > 1:
                out[e.key] = _safe_tag(e.relpath.replace("/", "_"))
    return out


class OllamaView:
    name = "ollama"

    def __init__(self, root: Path, aliases: dict[str, str], offline: bool = False, token: str | None = None,
                 fetch_manifest=reg.fetch_manifest, fetch_blob=reg.fetch_blob):
        self.root = root
        self.aliases = dict(aliases)
        self.offline = offline
        self.token = token
        self._fetch_manifest = fetch_manifest
        self._fetch_blob = fetch_blob
        self.warnings: list[str] = []

    # ---- desired -------------------------------------------------------
    def desired(self, entries: list[GgufEntry]) -> dict[str, Desired]:
        self.warnings = []
        tags = derive_tags(entries)
        out: dict[str, Desired] = {}
        for e in weights(entries):
            if e.is_shard:
                self.warnings.append(f"{e.key}: sharded GGUF not supported by Ollama; skipped")
                continue
            links = {blob_path(e.sha256): e.blob}
            mm = mmproj_for(e, entries)
            if mm is not None:
                links[blob_path(mm.sha256)] = mm.blob
            tag = tags[e.key]
            extra = {"tag": tag, "manifest": manifest_path(e.repo_id, tag), "repo_id": e.repo_id,
                     "filename": e.basename, "model": (e.sha256, e.size),
                     "projector": (mm.sha256, mm.size) if mm else None, "blob": str(e.blob),
                     "aliases": [a for a, t in self.aliases.items() if t == e.key]}
            out[e.key] = Desired(key=e.key, sha256=e.sha256, links=links, extra=extra)
        return out

    # ---- presence ------------------------------------------------------
    def _link_status(self, rel: str, target: Path) -> str:
        p = self.root / rel
        if p.is_symlink():
            return "ok" if os.readlink(p) == str(target) else "wrong"
        if p.is_file():
            return "ok"  # Ollama downloaded identical bytes; adopt
        return "missing"

    def _manifest_status(self, d: Desired) -> str:
        p = self.root / d.extra["manifest"]
        if not p.is_file():
            return "missing"
        try:
            m = json.loads(p.read_text())
            layers = {l["mediaType"]: l["digest"] for l in m["layers"]}
        except (ValueError, KeyError, TypeError):
            return "wrong"
        model_ok = layers.get(reg.MT_MODEL) == "sha256:" + d.extra["model"][0]
        proj = d.extra["projector"]
        proj_ok = layers.get(reg.MT_PROJECTOR) == ("sha256:" + proj[0] if proj else None)
        return "ok" if model_ok and proj_ok else "wrong"

    def present(self, d: Desired) -> Presence:
        statuses = {self._link_status(rel, t) for rel, t in d.links.items()}
        statuses.add(self._manifest_status(d))
        for a in d.extra.get("aliases", []):
            statuses.add("ok" if (self.root / alias_manifest_path(a)).is_file() else "missing")
        if "wrong" in statuses:
            return Presence.WRONG
        return Presence.CORRECT if statuses == {"ok"} else Presence.ABSENT

    # ---- create --------------------------------------------------------
    def _registry_cache(self, d: Desired) -> Path:
        return self.root / REGISTRY_CACHE_DIR / d.extra["repo_id"] / (d.extra["filename"] + ".json")

    def _registry_manifest(self, d: Desired) -> dict | None:
        cache_file = self._registry_cache(d)
        if cache_file.is_file():
            data = json.loads(cache_file.read_text())
            return None if data.get("missing") else data
        if self.offline:
            raise SkipEntry("offline and no cached registry response")
        org, name = d.extra["repo_id"].split("/", 1)
        m = self._fetch_manifest(org, name, d.extra["filename"], token=self.token)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(m if m is not None else {"missing": True}))
        return m

    def _write_blob(self, sha256: str, data: bytes) -> str:
        rel = blob_path(sha256)
        p = self.root / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return rel

    def create(self, d: Desired) -> list[str]:
        created: list[str] = []
        for rel, target in d.links.items():
            p = self.root / rel
            if not (p.is_symlink() or p.exists()):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.symlink_to(target)
            created.append(rel)
        registry = self._registry_manifest(d)
        if registry is not None:
            org, name = d.extra["repo_id"].split("/", 1)
            config = json.loads(self._fetch_blob(org, name, registry["config"]["digest"], token=self.token) or b"{}")
            for layer in reg.small_layers(registry):
                data = self._fetch_blob(org, name, layer["digest"], token=self.token)
                created.append(self._write_blob(layer["digest"][7:], data))
            manifest, cfg_bytes = reg.patch_manifest(registry, config, d.extra["model"], d.extra["projector"])
        else:
            header = read_header(Path(d.extra["blob"]))
            manifest, cfg_bytes = reg.synth_manifest(header, d.extra["model"], d.extra["projector"])
            self.warnings.append(f"{d.key}: not on the Hub registry; model relies on its embedded chat template")
        created.append(self._write_blob(manifest["config"]["digest"][7:], cfg_bytes))
        text = json.dumps(manifest, separators=(",", ":"))
        for rel in [d.extra["manifest"]] + [alias_manifest_path(a) for a in d.extra.get("aliases", [])]:
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            created.append(rel)
        return created

    # ---- remove / foreign ---------------------------------------------
    def remove(self, paths: list[str]) -> None:
        for rel in paths:
            p = self.root / rel
            if p.is_symlink() or p.exists():
                p.unlink()
            remove_empty_parents(self.root, p)

    def foreign(self, state: State) -> list[ForeignItem]:
        owned = {p for o in state.owned.values() for p in o.paths}
        out: list[ForeignItem] = []
        mroot = self.root / "manifests"
        if not mroot.is_dir():
            return out
        for p in sorted(x for x in mroot.rglob("*") if x.is_file()):
            rel = str(p.relative_to(self.root))
            if rel in owned:
                continue
            try:
                m = json.loads(p.read_text())
                model = next(l for l in m["layers"] if l["mediaType"] == reg.MT_MODEL)
            except (ValueError, KeyError, StopIteration, TypeError):
                continue
            blob = self.root / blob_path(model["digest"][7:])
            if blob.is_file() and not blob.is_symlink():
                parts = p.relative_to(mroot).parts  # registry, ns, name, tag
                key = "ollama:" + "/".join(parts[:-1]) + ":" + parts[-1]
                out.append(ForeignItem(key=key, path=blob, extra={"manifest": rel, "layers": m["layers"]}))
        return out
