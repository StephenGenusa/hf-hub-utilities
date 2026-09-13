"""Ollama view: blob symlinks + hand-written manifests under an OLLAMA_MODELS directory."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
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
                     "filename": e.basename, "relpath": e.relpath, "model": (e.sha256, e.size),
                     "projector": (mm.sha256, mm.size) if mm else None, "blob": str(e.blob),
                     "aliases": [a for a, t in self.aliases.items() if t == e.key]}
            out[e.key] = Desired(key=e.key, sha256=e.sha256, links=links, extra=extra)
        return out

    # ---- presence ------------------------------------------------------
    def _link_status(self, rel: str, target: Path) -> str:
        p = self.root / rel
        if p.is_symlink():
            current = os.readlink(p)
            # The same bytes can live in several HF repos (identical mmproj or
            # MTP files); the blob path is keyed by sha, so a link to any copy
            # of the same sha is correct.
            return "ok" if current == str(target) or os.path.basename(current) == target.name else "wrong"
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
        """Primary artefacts are the hf.co manifest and the model blob link.

        The projector link and the alias manifests are secondary: when they are
        the only thing missing the model still runs, so report PARTIAL (a repair)
        rather than ABSENT (which reconcile would read as "deleted outside sync").
        """
        links = {rel: self._link_status(rel, t) for rel, t in d.links.items()}
        statuses = set(links.values())
        manifest = self._manifest_status(d)
        statuses.add(manifest)
        for a in d.extra.get("aliases", []):
            statuses.add("ok" if (self.root / alias_manifest_path(a)).is_file() else "missing")
        if "wrong" in statuses:
            return Presence.WRONG
        if manifest != "ok" or links.get(blob_path(d.extra["model"][0])) != "ok":
            return Presence.ABSENT
        return Presence.CORRECT if statuses == {"ok"} else Presence.PARTIAL

    # ---- create --------------------------------------------------------
    def _registry_cache(self, d: Desired) -> Path:
        return self.root / REGISTRY_CACHE_DIR / d.extra["repo_id"] / (d.extra["filename"] + ".json")

    def _config_cache(self, d: Desired) -> Path:
        return self.root / REGISTRY_CACHE_DIR / d.extra["repo_id"] / (d.extra["filename"] + ".config.json")

    def _fetch(self, fn, *args):
        """Call a registry fetch, turning a network outage into a skip.

        A transient outage must not abort the whole sync run: the entry is left
        for the next run instead. A missing blob (FileNotFoundError) is an OSError
        too and is skipped the same way - we cannot build a manifest without it.
        """
        try:
            return fn(*args, token=self.token)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise SkipEntry(f"registry unreachable: {e}") from e

    def _registry_manifest(self, d: Desired) -> dict | None:
        cache_file = self._registry_cache(d)
        if cache_file.is_file():
            data = json.loads(cache_file.read_text())
            return None if data.get("missing") else data
        if self.offline:
            raise SkipEntry("offline and no cached registry response")
        org, name = d.extra["repo_id"].split("/", 1)
        # The registry accepts a file path as the tag only when '/' is percent-encoded.
        tag = urllib.parse.quote(d.extra.get("relpath", d.extra["filename"]), safe="")
        m = self._fetch(self._fetch_manifest, org, name, tag)
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

    def _registry_layers(self, d: Desired, registry: dict) -> tuple[dict, list[str]]:
        """Config dict + the view-relative paths of the manifest's small layers.

        Everything is cached under the view root so that a later offline run needs
        no network at all; offline with anything missing skips the entry before
        anything is written.
        """
        org, name = d.extra["repo_id"].split("/", 1)
        layers = reg.small_layers(registry)
        missing = [l for l in layers if not (self.root / blob_path(l["digest"][7:])).exists()]
        cache_file = self._config_cache(d)
        if self.offline and (not cache_file.is_file() or missing):
            raise SkipEntry("offline and registry layers not cached")
        if cache_file.is_file():
            raw = cache_file.read_bytes()
        else:
            raw = self._fetch(self._fetch_blob, org, name, registry["config"]["digest"]) or b"{}"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(raw)
        for layer in missing:
            self._write_blob(layer["digest"][7:], self._fetch(self._fetch_blob, org, name, layer["digest"]))
        return json.loads(raw or b"{}"), [blob_path(l["digest"][7:]) for l in layers]

    def _alias_action(self, d: Desired, alias: str, text: str) -> str:
        """write | skip (already ours and identical) | refuse (belongs to someone else)."""
        p = self.root / alias_manifest_path(alias)
        if not p.exists():
            return "write"
        try:
            existing = p.read_text()
        except OSError:
            return "refuse"
        if existing == text:
            return "skip"
        try:
            m = json.loads(existing)
            model = next(l for l in m["layers"] if l["mediaType"] == reg.MT_MODEL)
        except (ValueError, KeyError, TypeError, StopIteration):
            return "refuse"
        return "write" if model.get("digest") == "sha256:" + d.extra["model"][0] else "refuse"

    def create(self, d: Desired) -> list[str]:
        created: list[str] = []
        # Resolve the registry (and its small layers) first: anything that can
        # raise SkipEntry must do so before we put a single link on disk.
        registry = self._registry_manifest(d)
        if registry is not None:
            config, layer_paths = self._registry_layers(d, registry)
            created.extend(layer_paths)
            manifest, cfg_bytes = reg.patch_manifest(registry, config, d.extra["model"], d.extra["projector"])
        else:
            header = read_header(Path(d.extra["blob"]))
            manifest, cfg_bytes = reg.synth_manifest(header, d.extra["model"], d.extra["projector"])
            self.warnings.append(f"{d.key}: not on the Hub registry; model relies on its embedded chat template")
        for rel, target in d.links.items():
            p = self.root / rel
            if not (p.is_symlink() or p.exists()):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.symlink_to(target)
            created.append(rel)
        created.append(self._write_blob(manifest["config"]["digest"][7:], cfg_bytes))
        text = json.dumps(manifest, separators=(",", ":"))
        writes = [(d.extra["manifest"], "write")]
        for alias in d.extra.get("aliases", []):
            action = self._alias_action(d, alias, text)
            if action == "refuse":
                self.warnings.append(f"{d.key}: alias {alias!r} exists and belongs to another model; not overwritten")
                continue
            writes.append((alias_manifest_path(alias), action))
        for rel, action in writes:
            p = self.root / rel
            if action == "write":
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            created.append(rel)
        return created

    # ---- remove / foreign ---------------------------------------------
    def _unlink(self, rel: str) -> None:
        p = self.root / rel
        if p.is_symlink() or p.exists():
            p.unlink()
        remove_empty_parents(self.root, p)

    def _referenced_digests(self, exclude: Path | None = None) -> set[str]:
        """Every digest any manifest still on disk points at (config + layers).

        `exclude` skips one manifest file: the caller is about to delete it and
        wants to know what nothing *else* refers to any more.
        """
        out: set[str] = set()
        mroot = self.root / "manifests"
        if not mroot.is_dir():
            return out
        for p in mroot.rglob("*"):
            if not p.is_file() or p == exclude:
                continue
            try:
                m = json.loads(p.read_text())
                digests = [l["digest"] for l in m["layers"]]
            except (ValueError, KeyError, TypeError, OSError):
                continue
            config = m.get("config") if isinstance(m, dict) else None
            if isinstance(config, dict) and isinstance(config.get("digest"), str):
                digests.append(config["digest"])
            out.update(x.removeprefix("sha256:") for x in digests if isinstance(x, str))
        return out

    def remove(self, paths: list[str]) -> None:
        """Drop our manifests first, then any blob no remaining manifest refers to.

        Blobs are shared: Ollama's own models and our other models point at the
        same template/params/config layers, and a blob path we merely adopted may
        hold bytes Ollama downloaded itself. Reference-count against what is left
        on disk rather than trusting our own path list.
        """
        for rel in [p for p in paths if p.startswith("manifests/")]:
            self._unlink(rel)
        referenced = self._referenced_digests()
        for rel in [p for p in paths if not p.startswith("manifests/")]:
            if rel.startswith("blobs/") and os.path.basename(rel).removeprefix("sha256-") in referenced:
                continue
            self._unlink(rel)

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
            parts = p.relative_to(mroot).parts  # registry, ns, name, tag
            if len(parts) < 2:
                continue
            try:
                m = json.loads(p.read_text())
                model = next(l for l in m["layers"] if l["mediaType"] == reg.MT_MODEL)
                digest = model["digest"][7:]
            except (ValueError, KeyError, StopIteration, TypeError, OSError):
                continue
            blob = self.root / blob_path(digest)
            if blob.is_file() and not blob.is_symlink():
                key = "ollama:" + "/".join(parts[:-1]) + ":" + parts[-1]
                out.append(ForeignItem(key=key, path=blob, extra={"manifest": rel, "layers": m["layers"]}))
        return out
