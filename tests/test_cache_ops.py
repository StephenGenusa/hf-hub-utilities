import hashlib
import os
from pathlib import Path

import pytest

from hfhub import cache, cache_ops as ops
from tests.hub_fixture import add_repo


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def big(tag: bytes, n: int = 2_000_000) -> bytes:
    return tag * (n // len(tag))


# ---------------------------------------------------------------- parsing helpers
def test_quant_bits_and_parsers():
    assert ops.quant_bits("m-Q4_K_M.gguf") == 4
    assert ops.quant_bits("m-UD-IQ2_M.gguf") == 2
    assert ops.quant_bits("m-CD-Q2_K.gguf") == 2
    assert ops.quant_bits("m.q8_0.gguf") == 8
    assert ops.quant_bits("m-BF16.gguf") == 16
    assert ops.quant_bits("plain.gguf") is None
    assert ops.parse_min_quant("IQ4_XS") == 4 and ops.parse_min_quant("6") == 6
    assert ops.parse_size("23G") == 23_000_000_000 and ops.parse_size("500MB") == 500_000_000
    with pytest.raises(ValueError):
        ops.parse_size("lots")


# ---------------------------------------------------------------- repair
def test_repair_relinks_orphan_blobs_from_hub_tree(tmp_path: Path):
    hub = tmp_path / "hub"
    repo = hub / "models--o--r"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / "old").mkdir(parents=True)
    data = b"weights"
    (repo / "blobs" / sha(data)).write_bytes(data)
    stray = b"x" * 5
    (repo / "blobs" / sha(stray)).write_bytes(stray)   # not in the Hub tree
    tree = lambda rid: ("c" * 40, {sha(data): ("sub/m.gguf", len(data))})
    msgs = []
    res = ops.repair(hub, execute=False, out=msgs.append, tree_fn=tree)
    assert res.actions == 1 and not (repo / "snapshots" / ("c" * 40)).exists()
    res = ops.repair(hub, execute=True, out=msgs.append, tree_fn=tree)
    link = repo / "snapshots" / ("c" * 40) / "sub" / "m.gguf"
    assert link.is_symlink() and link.read_bytes() == data
    assert (repo / "refs" / "main").read_text() == "c" * 40
    assert not (repo / "snapshots" / "old").exists()
    assert any("not in the current revision" in m for m in msgs)
    assert cache.scan(hub)[0].key == "o/r:sub/m.gguf"


def test_repair_leaves_linked_repos_alone(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"1"})
    res = ops.repair(hub, execute=True, out=lambda *_: None, tree_fn=lambda rid: pytest.fail("should not be called"))
    assert res.actions == 0


# ---------------------------------------------------------------- dedupe
def test_dedupe_hardlinks_identical_blobs_and_links_plain_copies(tmp_path: Path):
    hub = tmp_path / "hub"
    data = big(b"same")
    add_repo(hub, "a/r", {"m.gguf": data})
    add_repo(hub, "b/r", {"m.gguf": data})
    # a plain copy sitting in b's snapshot instead of a link
    snap = next((hub / "models--b--r" / "snapshots").iterdir())
    (snap / "copy.gguf").write_bytes(data)
    res = ops.dedupe(hub, execute=True, out=lambda *_: None)
    a = hub / "models--a--r" / "blobs" / sha(data)
    b = hub / "models--b--r" / "blobs" / sha(data)
    assert a.stat().st_ino == b.stat().st_ino
    assert (snap / "copy.gguf").is_symlink() and (snap / "copy.gguf").read_bytes() == data
    assert res.actions == 2 and res.bytes == 2 * len(data)
    assert ops.dedupe(hub, execute=False, out=lambda *_: None).actions == 0


def test_dedupe_moves_a_copy_with_no_blob_into_blobs(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "a/r", {"m.gguf": b"1"})
    snap = next((hub / "models--a--r" / "snapshots").iterdir())
    data = big(b"lonely")
    (snap / "lonely.gguf").write_bytes(data)
    ops.dedupe(hub, execute=True, out=lambda *_: None)
    assert (hub / "models--a--r" / "blobs" / sha(data)).read_bytes() == data
    assert (snap / "lonely.gguf").is_symlink()


# ---------------------------------------------------------------- prune superseded
def test_prune_superseded_deletes_only_replaced_paths(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"old-version", "only-old.gguf": b"keep-me"}, commit="a" * 40, main=False)
    add_repo(hub, "o/r", {"m.gguf": b"new-version"}, commit="b" * 40, main=True)
    msgs = []
    res = ops.prune_superseded(hub, execute=False, out=msgs.append)
    assert res.actions == 1 and (hub / "models--o--r" / "blobs" / sha(b"old-version")).exists()
    res = ops.prune_superseded(hub, execute=True, out=msgs.append)
    blobs = hub / "models--o--r" / "blobs"
    assert not (blobs / sha(b"old-version")).exists()
    assert (blobs / sha(b"keep-me")).exists() and (blobs / sha(b"new-version")).exists()
    old_snap = hub / "models--o--r" / "snapshots" / ("a" * 40)
    assert (old_snap / "only-old.gguf").is_symlink() and not (old_snap / "m.gguf").exists()
    assert any("sync --execute" in m for m in msgs)


# ---------------------------------------------------------------- thin
def test_thin_applies_bits_and_size_rules_and_keeps_one_quant(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m-Q2_K.gguf": b"2", "m-Q4_K_M.gguf": b"4444", "m-Q8_0.gguf": b"8" * 10,
                          "mmproj-F16.gguf": b"p", "MTP/m-MTP-Q2_K.gguf": b"d"})
    add_repo(hub, "o/tiny", {"t-Q2_K.gguf": b"22"})
    msgs = []
    res = ops.thin(hub, execute=False, min_bits=4, max_size=9, out=msgs.append)
    assert res.actions == 2  # Q2_K (too small) and Q8_0 (too big) in o/r; o/tiny skipped
    assert any("skipped o/tiny" in m for m in msgs)
    ops.thin(hub, execute=True, min_bits=4, max_size=9, out=lambda *_: None)
    keys = {e.key for e in cache.scan(hub)}
    assert "o/r:m-Q4_K_M.gguf" in keys and "o/r:mmproj-F16.gguf" in keys and "o/r:MTP/m-MTP-Q2_K.gguf" in keys
    assert "o/r:m-Q2_K.gguf" not in keys and "o/r:m-Q8_0.gguf" not in keys
    assert "o/tiny:t-Q2_K.gguf" in keys
    res = ops.thin(hub, execute=True, min_bits=4, repos=["o/tiny"], allow_empty=True, out=lambda *_: None)
    assert res.actions == 1 and "o/tiny:t-Q2_K.gguf" not in {e.key for e in cache.scan(hub)}


# ---------------------------------------------------------------- report
def test_report_lists_categories(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {f"m-Q{i}_K.gguf": bytes([i]) * 3 for i in (2, 3, 4, 5)})
    msgs = []
    ops.report(hub, out=msgs.append)
    text = "\n".join(msgs)
    assert "superseded" in text and "identical" in text and "unlinked" in text and "4 quants  o/r" in text
