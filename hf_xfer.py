#!/usr/bin/env python3
"""
hf-xfer.py - Bidirectional translator between a HuggingFace `--local-dir` flat
copy and the canonical content-addressed hub cache (`~/.cache/huggingface/hub`,
or wherever HF_HOME / HF_HUB_CACHE point).

  import  : local-dir  -> cache   (seed the shared cache from a flat copy)
  export  : cache      -> local-dir (materialize a flat copy out of the cache)

Dry-run is the DEFAULT. Pass --execute to actually write anything.

Why this exists: `hf` ships no first-party command to import a `--local-dir`
copy back into the shared cache. The two layouts are deliberately separate
(content-addressed blobs+snapshots+refs vs. flat path-addressed files). This
tool does the translation using huggingface_hub's own query + metadata APIs so
the result is byte-compatible with `scan_cache_dir()` / `hf download`.

Verified against huggingface_hub 1.20.1 source:
  - cache layout : <cache>/<repo_folder>/{blobs,snapshots/<commit>/<relpath>,refs/<rev>}
  - blob name    : etag -> sha256 (LFS) or git-blob-sha1 (regular)
  - snapshot link: os.path.relpath(blob, snapshot_file_dir)  (relative)
  - local meta   : <local_dir>/.cache/huggingface/download/<relpath>.metadata
                   3 plain-text lines: commit_hash / etag / timestamp
  - cache dir    : HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > HF_HOME/hub > default
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# huggingface_hub is used for QUERIES and for its own metadata/path helpers so
# we never diverge from the library's actual on-disk contract.
try:
    from huggingface_hub import HfApi, constants
    from huggingface_hub.file_download import repo_folder_name
    from huggingface_hub._local_folder import write_download_metadata
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
except Exception as e:  # pragma: no cover
    sys.stderr.write(
        f"huggingface_hub is required: {e}\n  pip install huggingface_hub\n"
    )
    raise


# --------------------------------------------------------------------------- #
# Resolution / small helpers
# --------------------------------------------------------------------------- #

def resolve_cache_dir(explicit: Optional[str]) -> tuple[Path, str]:
    """Return (cache_dir, how_it_was_resolved). No hard-coded ~/.cache guess:
    we defer to huggingface_hub.constants, which encodes the full precedence
    (HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > HF_HOME/hub > platform default)."""
    if explicit:
        return Path(explicit).expanduser().resolve(), "--cache-dir flag"
    if os.getenv("HF_HUB_CACHE"):
        how = "HF_HUB_CACHE env"
    elif os.getenv("HUGGINGFACE_HUB_CACHE"):
        how = "HUGGINGFACE_HUB_CACHE env"
    elif os.getenv("HF_HOME"):
        how = "HF_HOME/hub"
    elif os.getenv("XDG_CACHE_HOME"):
        how = "XDG_CACHE_HOME/huggingface/hub"
    else:
        how = "default ~/.cache/huggingface/hub"
    return Path(constants.HF_HUB_CACHE).resolve(), how


def infer_repo_id(local_dir: Path) -> Optional[str]:
    """Derive `org/name` from the last two components of the dir path.
    e.g. .../AIModels/google/diffusiongemma-26B-A4B-it -> google/diffusiongemma-26B-A4B-it"""
    parts = local_dir.resolve().parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return None


def etag_kind(etag: str) -> str:
    e = etag.lower()
    if len(e) == 64 and all(c in "0123456789abcdef" for c in e):
        return "sha256(LFS)"
    if len(e) == 40 and all(c in "0123456789abcdef" for c in e):
        return "git-sha1"
    return "unknown"


def _looks_like_commit(rev: str) -> bool:
    r = rev.lower()
    return len(r) == 40 and all(c in "0123456789abcdef" for c in r)


def _place_file(src: Path, dst: Path, mode: str) -> str:
    """Put src's content at dst. mode: 'copy' | 'move' | 'hardlink'.
    copy/move preserve the original mtime+atime (copy2 / rename). hardlinks
    share the inode so times are identical inherently. Returns the verb used.

    Note: on Linux the file *birth* time (crtime) is not settable from
    userspace; mtime and atime are preserved exactly (ns precision)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if mode == "move":
            try:
                src.unlink()          # content already at dst; drop the redundant source
            except OSError:
                pass
            return "exists (src removed)"
        return "exists"
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)    # cross-device fallback, preserves times
            return "copy (xdev)"
    if mode == "move":
        shutil.move(str(src), str(dst))   # rename when possible; copy2+unlink across fs
        return "move"
    shutil.copy2(src, dst)            # default: preserve mtime/atime + mode bits
    return "copy"


def prune_empty_dirs(root: Path, remove_root: bool = False) -> list[str]:
    """Bottom-up remove empty directories under root. A directory holding only
    a dangling symlink is NOT empty, so broken snapshot entries are preserved."""
    removed: list[str] = []
    if not root.is_dir():
        return removed
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        if d == root and not remove_root:
            continue
        try:
            if not any(d.iterdir()):
                d.rmdir()
                removed.append(str(d))
        except OSError:
            pass
    if remove_root and root.is_dir():
        try:
            if not any(root.iterdir()):
                root.rmdir()
                removed.append(str(root))
        except OSError:
            pass
    return removed


def _blob_refcounts(repo_root: Path) -> dict[str, int]:
    """Count how many snapshot symlinks (across all commits) resolve to each
    blob name, so export --move can refuse to move a shared blob."""
    counts: dict[str, int] = {}
    snaps = repo_root / "snapshots"
    if snaps.is_dir():
        for p in snaps.rglob("*"):
            if p.is_symlink():
                name = os.path.basename(os.path.realpath(p))
                counts[name] = counts.get(name, 0) + 1
    return counts


def _rel_symlink_target(blob_path: Path, snapshot_file: Path) -> str:
    """Exactly how huggingface_hub builds it: relpath(blob, dir(snapshot_file))."""
    return os.path.relpath(
        os.path.abspath(blob_path), os.path.dirname(os.path.abspath(snapshot_file))
    )


# --------------------------------------------------------------------------- #
# Repo file map (relpath -> etag) + commit hash
# --------------------------------------------------------------------------- #

@dataclass
class RepoMap:
    commit_hash: Optional[str]
    etags: dict[str, str]          # relpath -> etag
    source: str                    # "hub" | "local-metadata" | "mixed"
    multi_commit: bool = False     # True if local metadata disagreed on commit


def map_from_hub(repo_id: str, repo_type: str, revision: str) -> RepoMap:
    """Authoritative: ask the Hub for the commit + per-file etag.
    blob_id = git-blob-sha1 (regular files), lfs.sha256 = etag for LFS files."""
    api = HfApi()
    info = api.repo_info(
        repo_id=repo_id, repo_type=repo_type, revision=revision, files_metadata=True
    )
    etags: dict[str, str] = {}
    for sib in info.siblings or []:
        if sib.lfs is not None:
            etag = sib.lfs.get("sha256") if isinstance(sib.lfs, dict) else getattr(sib.lfs, "sha256", None)
        else:
            etag = sib.blob_id
        if etag:
            etags[sib.rfilename] = etag
    return RepoMap(commit_hash=info.sha, etags=etags, source="hub")


def _read_metadata_file(meta_path: Path) -> Optional[tuple[str, str]]:
    """Parse a local-dir download metadata file directly: 3 plain-text lines
    (commit_hash, etag, timestamp). We read it ourselves rather than via
    huggingface_hub.read_download_metadata, which discards metadata whose
    timestamp predates the file's mtime — a resumable-download safeguard that
    wrongly rejects files that were merely copied/moved. Returns (commit, etag)."""
    try:
        with meta_path.open() as f:
            commit = f.readline().strip()
            etag = f.readline().strip()
    except OSError:
        return None
    if not commit or not etag:
        return None
    return commit, etag


def map_from_local_metadata(local_dir: Path) -> RepoMap:
    """Offline fallback: read every <relpath>.metadata the original
    `--local-dir` download wrote."""
    dl_root = local_dir / ".cache" / "huggingface" / "download"
    etags: dict[str, str] = {}
    commits: dict[str, int] = {}
    if dl_root.is_dir():
        for meta in dl_root.rglob("*.metadata"):
            relpath = str(meta.relative_to(dl_root))[: -len(".metadata")]
            relpath = relpath.replace(os.sep, "/")
            parsed = _read_metadata_file(meta)
            if parsed is None:
                continue
            commit, etag = parsed
            etags[relpath] = etag
            commits[commit] = commits.get(commit, 0) + 1
    commit = max(commits, key=commits.get) if commits else None
    return RepoMap(
        commit_hash=commit,
        etags=etags,
        source="local-metadata",
        multi_commit=len(commits) > 1,
    )


def build_repo_map(local_dir: Path, repo_id: str, repo_type: str, revision: str,
                   offline: bool) -> RepoMap:
    """Prefer the Hub (covers every file authoritatively); fall back to local
    metadata when offline or the query fails."""
    if not offline:
        try:
            hub = map_from_hub(repo_id, repo_type, revision)
            if hub.commit_hash:
                return hub
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            sys.stderr.write(f"Hub query failed ({type(e).__name__}); using local metadata.\n")
        except Exception as e:
            sys.stderr.write(f"Hub query failed ({e}); using local metadata.\n")
    return map_from_local_metadata(local_dir)


# --------------------------------------------------------------------------- #
# IMPORT  (local-dir -> cache)
# --------------------------------------------------------------------------- #

def verify_after_import(repo_id: str, repo_type: str, revision: str,
                        cache_dir: Path) -> tuple[bool, Optional[str], Optional[Exception]]:
    """Run the equivalent of `hf download <repo_id>` (huggingface_hub.snapshot_download)
    against the resolved cache. This is the real completeness check: it queries the
    Hub for the full file manifest, skips files already placed, FETCHES anything
    missing, and confirms every snapshot symlink resolves to a real blob.
    Needs network. Returns (ok, snapshot_path, error)."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=repo_id, repo_type=repo_type, revision=revision,
            cache_dir=str(cache_dir),
        )
        return True, path, None
    except Exception as e:
        return False, None, e


def wiring_check(repo_root: Path, commit: str) -> tuple[int, list[str]]:
    """Offline best-effort: confirm every snapshot symlink for this commit resolves
    to an existing blob. CANNOT detect files missing from the snapshot entirely,
    since the repo manifest is only knowable from the Hub. Returns (ok_count, broken)."""
    snap_dir = repo_root / "snapshots" / commit
    ok, broken = 0, []
    if snap_dir.is_dir():
        for p in snap_dir.rglob("*"):
            if p.is_dir():
                continue
            if Path(os.path.realpath(p)).exists():
                ok += 1
            else:
                broken.append(p.relative_to(snap_dir).as_posix())
    return ok, broken


@dataclass
class Plan:
    can: list[tuple]      # (relpath, etag, kind, action)
    cannot: list[tuple]   # (relpath, reason)


def plan_import(local_dir: Path, cache_dir: Path, repo_id: str, repo_type: str,
                rmap: RepoMap) -> tuple[Plan, Path, Optional[str]]:
    repo_root = cache_dir / repo_folder_name(repo_id=repo_id, repo_type=repo_type)
    commit = rmap.commit_hash
    can, cannot = [], []

    # Files physically present in the local dir, excluding the .cache metadata area.
    present: set[str] = set()
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            if rel.startswith(".cache/"):
                continue
            present.add(rel)

    if commit is None:
        for rel in sorted(present):
            cannot.append((rel, "no commit hash (Hub unreachable and no local metadata)"))
        return Plan(can, cannot), repo_root, commit

    for rel in sorted(present):
        etag = rmap.etags.get(rel)
        if etag is None:
            cannot.append((rel, "etag unknown (not in repo listing / no metadata)"))
            continue
        kind = etag_kind(etag)
        if kind == "unknown":
            cannot.append((rel, f"unrecognized etag form: {etag!r}"))
            continue
        snap = repo_root / "snapshots" / commit / rel
        blob = repo_root / "blobs" / etag
        if snap.is_symlink() and blob.exists():
            action = "already present (skip)"
        else:
            action = "place blob + symlink"
        can.append((rel, etag, kind, action))

    # Local files the repo listing doesn't know about.
    for rel in sorted(present - set(rmap.etags)):
        if all(rel != c[0] for c in cannot):
            cannot.append((rel, "etag unknown (not in repo listing / no metadata)"))

    return Plan(can, cannot), repo_root, commit


def execute_import(local_dir: Path, repo_root: Path, commit: str, revision: str,
                   plan: Plan, mode: str) -> list[str]:
    (repo_root / "blobs").mkdir(parents=True, exist_ok=True)
    (repo_root / "snapshots" / commit).mkdir(parents=True, exist_ok=True)
    (repo_root / "refs").mkdir(parents=True, exist_ok=True)

    for rel, etag, _kind, action in plan.can:
        if action.startswith("already"):
            continue
        src = local_dir / rel
        blob = repo_root / "blobs" / etag
        _place_file(src, blob, mode)
        snap = repo_root / "snapshots" / commit / rel
        snap.parent.mkdir(parents=True, exist_ok=True)
        if snap.is_symlink() or snap.exists():
            snap.unlink()
        snap.symlink_to(_rel_symlink_target(blob, snap))

    # refs/<revision> -> commit. Skip writing a ref named like a bare commit.
    if not _looks_like_commit(revision):
        (repo_root / "refs" / revision).parent.mkdir(parents=True, exist_ok=True)
        (repo_root / "refs" / revision).write_text(commit)

    if mode == "move":
        return prune_empty_dirs(local_dir, remove_root=True)
    return []


# --------------------------------------------------------------------------- #
# EXPORT  (cache -> local-dir)
# --------------------------------------------------------------------------- #

def resolve_commit_in_cache(repo_root: Path, revision: str) -> Optional[str]:
    ref = repo_root / "refs" / revision
    if ref.is_file():
        return ref.read_text().strip()
    if (repo_root / "snapshots" / revision).is_dir():
        return revision  # revision was already a commit hash
    return None


def plan_export(repo_root: Path, commit: str) -> Plan:
    snap_dir = repo_root / "snapshots" / commit
    can, cannot = [], []
    for p in sorted(snap_dir.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(snap_dir).as_posix()
        target = Path(os.path.realpath(p))
        if target.exists():
            etag = target.name
            can.append((rel, etag, etag_kind(etag), "copy blob -> local-dir"))
        else:
            cannot.append((rel, "blob missing (broken snapshot symlink)"))
    return Plan(can, cannot)


def execute_export(repo_root: Path, commit: str, local_dir: Path, plan: Plan,
                   mode: str) -> tuple[list[str], list[str]]:
    snap_dir = repo_root / "snapshots" / commit
    refcounts = _blob_refcounts(repo_root) if mode == "move" else {}
    notes: list[str] = []
    for rel, etag, _kind, _action in plan.can:
        link = snap_dir / rel
        src = Path(os.path.realpath(link))
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        effective = mode
        if mode == "move" and refcounts.get(src.name, 0) > 1:
            effective = "copy"
            notes.append(f"{rel}: blob shared by {refcounts[src.name]} snapshots — copied, not moved")
        _place_file(src, dst, effective)
        # Regenerate metadata so a later `hf download --local-dir` is incremental.
        write_download_metadata(local_dir, rel, commit_hash=commit, etag=etag)
        if effective == "move":
            try:
                link.unlink()         # blob is gone; drop the now-dangling snapshot symlink
            except OSError:
                pass
    removed = prune_empty_dirs(repo_root, remove_root=True) if mode == "move" else []
    return notes, removed


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_report(direction: str, cache_dir: Path, how: str, repo_root: Path,
                 commit: Optional[str], src_label: str, plan: Plan,
                 execute: bool, mode: str, extra: str = "") -> None:
    bar = "=" * 72
    print(bar)
    print(f"{direction}   ({'EXECUTE' if execute else 'DRY RUN'})")
    print(bar)
    print(f"cache dir   : {cache_dir}   [resolved via {how}]")
    print(f"repo folder : {repo_root.name}")
    print(f"commit      : {commit or '<unknown>'}")
    print(f"source      : {src_label}")
    print(f"mode        : {mode}" + ("  (emptied dirs erased)" if mode == "move" else ""))
    if extra:
        print(extra)
    print()
    print(f"CAN  ({len(plan.can)}):")
    if plan.can:
        w = max((len(c[0]) for c in plan.can), default=4)
        for rel, etag, kind, action in plan.can:
            print(f"  {rel:<{w}}  {kind:<11}  {etag[:16]}…  {action}")
    else:
        print("  (none)")
    print()
    print(f"CANNOT ({len(plan.cannot)}):")
    if plan.cannot:
        w = max((len(c[0]) for c in plan.cannot), default=4)
        for rel, reason in plan.cannot:
            print(f"  {rel:<{w}}  {reason}")
    else:
        print("  (none)")
    print()
    if not execute:
        print("Dry run only — re-run with --execute to apply.")
    print(bar)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Translate between a HuggingFace --local-dir copy and the hub cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("import", "local-dir -> cache"), ("export", "cache -> local-dir")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--local-dir", required=True, type=Path)
        p.add_argument("--repo-id", default=None,
                       help="org/name. If omitted, inferred from the last two "
                            "path components of --local-dir.")
        p.add_argument("--repo-type", default="model",
                       choices=["model", "dataset", "space"])
        p.add_argument("--revision", default="main",
                       help="branch/tag name, or a commit hash")
        p.add_argument("--cache-dir", default=None,
                       help="override cache dir (else HF_HUB_CACHE/HF_HOME resolution)")
        p.add_argument("--execute", action="store_true",
                       help="actually write (default is dry run)")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--move", action="store_true",
                       help="move files instead of copying, then erase emptied directories")
        g.add_argument("--hardlink", action="store_true",
                       help="hardlink instead of copying (same filesystem, fastest, no extra space)")
        if name == "import":
            p.add_argument("--offline", action="store_true",
                           help="don't query the Hub; use local .metadata only")
            p.add_argument("--no-verify", action="store_true",
                           help="skip the post-import 'hf download' completeness check")

    args = ap.parse_args(argv)
    cache_dir, how = resolve_cache_dir(args.cache_dir)
    local_dir = args.local_dir.expanduser().resolve()

    repo_id = args.repo_id or infer_repo_id(local_dir)
    if not repo_id or "/" not in repo_id:
        sys.stderr.write(
            f"could not determine --repo-id (got {repo_id!r}); pass it explicitly.\n")
        return 2
    if not args.repo_id:
        print(f"[inferred repo-id from path: {repo_id}]")
    args.repo_id = repo_id

    mode = "move" if args.move else "hardlink" if args.hardlink else "copy"

    if args.cmd == "import":
        if not local_dir.is_dir():
            sys.stderr.write(f"local-dir not found: {local_dir}\n")
            return 2
        rmap = build_repo_map(local_dir, args.repo_id, args.repo_type,
                              args.revision, args.offline)
        plan, repo_root, commit = plan_import(
            local_dir, cache_dir, args.repo_id, args.repo_type, rmap)
        extra = ""
        if rmap.multi_commit:
            extra = ("WARNING: local metadata referenced multiple commits; "
                     f"using the most common ({commit}).")
        already = repo_root.exists()
        if already:
            extra = (extra + "\n" if extra else "") + \
                f"NOTE: repo folder already exists in cache; present files are skipped."
        print_report("IMPORT  local-dir -> cache", cache_dir, how, repo_root,
                     commit, f"{rmap.source}", plan, args.execute, mode, extra)
        if args.execute and commit and plan.can:
            removed = execute_import(local_dir, repo_root, commit, args.revision, plan, mode)
            print("Import applied.")
            if removed:
                print(f"Erased {len(removed)} empty director"
                      f"{'y' if len(removed) == 1 else 'ies'} under {local_dir}.")
            if not args.no_verify:
                if args.offline:
                    ok, broken = wiring_check(repo_root, commit)
                    print(f"Wiring check (offline): {ok} file(s) resolve to blobs.")
                    if broken:
                        print(f"  BROKEN ({len(broken)}): " + ", ".join(broken))
                    print("  Note: offline check confirms wiring only; it cannot detect "
                          "files missing from the snapshot. Re-run online (drop --offline) "
                          "for a true completeness check + fetch.")
                else:
                    print("Verifying cache completeness (hf download)…")
                    ok, path, err = verify_after_import(
                        args.repo_id, args.repo_type, args.revision, cache_dir)
                    if ok:
                        print(f"  OK — all repo files present in cache:\n  {path}")
                    else:
                        print(f"  VERIFY FAILED ({type(err).__name__}): {err}")
        return 0

    # export
    repo_root = cache_dir / repo_folder_name(repo_id=args.repo_id, repo_type=args.repo_type)
    if not repo_root.is_dir():
        sys.stderr.write(f"repo not found in cache: {repo_root}\n")
        return 2
    commit = resolve_commit_in_cache(repo_root, args.revision)
    if commit is None:
        sys.stderr.write(
            f"could not resolve revision {args.revision!r} in {repo_root}\n"
            f"  (no refs/{args.revision} and no snapshots/{args.revision})\n")
        return 2
    plan = plan_export(repo_root, commit)
    print_report("EXPORT  cache -> local-dir", cache_dir, how, repo_root,
                 commit, f"snapshots/{commit}", plan, args.execute, mode,
                 extra=f"dest local-dir : {local_dir}")
    if args.execute and plan.can:
        local_dir.mkdir(parents=True, exist_ok=True)
        notes, removed = execute_export(repo_root, commit, local_dir, plan, mode)
        print("Export applied.")
        for n in notes:
            print(f"  note: {n}")
        if removed:
            print(f"Erased {len(removed)} empty director"
                  f"{'y' if len(removed) == 1 else 'ies'} under {repo_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())