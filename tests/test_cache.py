from pathlib import Path

from hfhub import cache
from tests.hub_fixture import add_repo


def test_scan_finds_ggufs_with_sha_and_currency(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "org/Model-GGUF", {"m-Q4_K_M.gguf": b"aaa", "mmproj-F16.gguf": b"bbb", "README.md": b"x"})
    entries = cache.scan(hub)
    keys = {e.key: e for e in entries}
    assert set(keys) == {"org/Model-GGUF:m-Q4_K_M.gguf", "org/Model-GGUF:mmproj-F16.gguf"}
    w = keys["org/Model-GGUF:m-Q4_K_M.gguf"]
    assert w.sha256 == "9834876dcfb05cb167a5c24953eba58c4ac89b1adf57f28f2f9d09af107ee8f0"
    assert w.blob == hub / "models--org--Model-GGUF" / "blobs" / w.sha256
    assert w.size == 3 and w.is_current and not w.is_mmproj and not w.is_shard
    assert keys["org/Model-GGUF:mmproj-F16.gguf"].is_mmproj


def test_refs_main_snapshot_wins_over_older(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"old"}, commit="a" * 40, main=False)
    add_repo(hub, "o/r", {"m.gguf": b"new"}, commit="b" * 40, main=True)
    (e,) = cache.scan(hub)
    assert e.blob.read_bytes() == b"new" and e.is_current


def test_shards_are_flagged(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"BF16/m-BF16-00001-of-00002.gguf": b"1", "BF16/m-BF16-00002-of-00002.gguf": b"2"})
    assert all(e.is_shard for e in cache.scan(hub))
    assert {e.relpath for e in cache.scan(hub)} == {"BF16/m-BF16-00001-of-00002.gguf", "BF16/m-BF16-00002-of-00002.gguf"}


def test_unlinked_blobs_reported(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"linked"})
    orphan = hub / "models--o--r" / "blobs" / ("f" * 64)
    orphan.write_bytes(b"orphan")
    assert cache.unlinked(hub) == [("o/r", orphan)]


def test_mmproj_for_prefers_f16(tmp_path: Path):
    hub = tmp_path / "hub"
    add_repo(hub, "o/r", {"m.gguf": b"m", "mmproj-BF16.gguf": b"1", "mmproj-F16.gguf": b"2"})
    add_repo(hub, "o/other", {"mmproj-F16.gguf": b"3"})
    entries = cache.scan(hub)
    w = next(e for e in entries if e.relpath == "m.gguf")
    assert cache.mmproj_for(w, entries).relpath == "mmproj-F16.gguf"
    assert cache.mmproj_for(w, entries).repo_id == "o/r"


def test_repo_folder_roundtrip():
    assert cache.repo_folder("unsloth/Qwen3.6-27B-GGUF") == "models--unsloth--Qwen3.6-27B-GGUF"
    assert cache.repo_id_from_folder("models--unsloth--Qwen3.6-27B-GGUF") == "unsloth/Qwen3.6-27B-GGUF"
