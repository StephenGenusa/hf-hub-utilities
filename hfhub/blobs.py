"""Repo blobs in both cache layouts.

Before huggingface_hub 2.x every ``models--*/blobs/<etag>`` was a regular file. Since
then a download may put the bytes in a cache-wide store at ``<hub>/blobs/<xx>/<xet hash>``
and make the repo blob a relative symlink to it (see huggingface_hub's
``utils/_shared_blobs.py``). The repo blob's own name is still the etag (the sha256 of an
LFS file); the store file is named by its Xet hash, which is *not* the sha256.

So: follow a snapshot link one hop to the repo blob, never all the way, when the name
matters; and when a repo blob that is a store link is deleted, let huggingface_hub
collect the store file once nothing else references it.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # huggingface_hub >= 2.x; absent before the shared store existed
    from huggingface_hub.utils import _shared_blobs as _hf_shared
except ImportError:  # pragma: no cover - depends on the installed huggingface_hub
    _hf_shared = None


def read_link(p: Path) -> str:
    r"""os.readlink without the \\?\ prefix Windows adds to absolute targets,
    so the result compares equal to the str(Path) the link was made from."""
    t = os.readlink(p)
    if t.startswith("\\\\?\\UNC\\"):
        return "\\\\" + t[8:]
    return t[4:] if t.startswith("\\\\?\\") else t


def repo_blob(link: Path) -> Path:
    """The ``models--*/blobs/<etag>`` entry a snapshot link points at.

    The directory part is resolved (so the result matches what ``realpath`` gave
    for a regular-file blob, which keeps existing view links valid); the final
    component is not, so a shared-store blob keeps its etag name.
    """
    target = Path(os.path.join(os.path.dirname(link), read_link(link)))
    return Path(os.path.realpath(target.parent)) / target.name


def is_shared(blob: Path) -> bool:
    """True when a repo blob is a link into the cache-wide shared store."""
    return blob.is_symlink()


def hub_of(blob: Path) -> Path:
    """``<hub>`` for ``<hub>/models--x/blobs/<etag>``."""
    return blob.parent.parent.parent


def delete_blob(blob: Path) -> int:
    """Remove a repo blob; collect its shared-store file when nothing else uses it.

    Returns the bytes actually freed: 0 when the store file is still referenced by
    another repo (or huggingface_hub cannot prove it is unreferenced).
    """
    if not (blob.is_symlink() or blob.exists()):
        return 0
    if not is_shared(blob):
        size = blob.stat().st_size if blob.stat().st_nlink == 1 else 0
        blob.unlink()
        return size
    store = Path(os.path.realpath(blob))
    blob.unlink()
    if _hf_shared is None or not store.exists():
        return 0
    return _hf_shared.sweep_shared_blob(store, cache_dir=hub_of(blob))


def link_into_store(blob: Path, store: Path) -> bool:
    """Replace a regular repo blob with a link to the identical shared-store file.

    Uses huggingface_hub's own publishing path, so the store's reference list
    records the new link and garbage collection keeps working.
    """
    if _hf_shared is None:
        return False
    return _hf_shared.publish_blob_to_shared_store(
        blob_path=str(blob), xet_hash=store.name, cache_dir=hub_of(blob),
        expected_size=store.stat().st_size)


def unreferenced_store_blobs(hub: Path) -> dict[Path, int]:
    """Shared-store files no repo blob links to any more (leaked space), with sizes."""
    if _hf_shared is None or not _hf_shared.is_shared_blobs_dir(_hf_shared.shared_blobs_dir(hub)):
        return {}
    return _hf_shared.unreferenced_shared_blobs(hub)
