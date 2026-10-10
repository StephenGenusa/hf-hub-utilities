"""`hf-xfer import`: every file is checked against its etag before it is placed."""
import hashlib
from pathlib import Path

from hfhub import xfer
from hfhub.xfer import RepoMap

COMMIT = "c" * 40
GIT_HELLO = "3b18e512dba79e4c8300dd08aeb37f8e728b8dad"   # `git hash-object` of b"hello world\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def local(tmp_path: Path, files: dict[str, bytes]) -> Path:
    d = tmp_path / "org" / "name"
    for rel, data in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(data)
    return d


def run(tmp_path, d, rmap, revision="main", mode="copy"):
    plan, root, commit = xfer.plan_import(d, tmp_path / "hub", "org/name", "model", rmap)
    removed, problems = xfer.execute_import(d, root, commit, revision, plan, mode, rmap.source)
    return root, problems


def test_matching_files_are_placed_and_ref_written(tmp_path):
    d = local(tmp_path, {"m.gguf": b"weights", "README.md": b"hello world\n"})
    rmap = RepoMap(commit_hash=COMMIT, etags={"m.gguf": sha(b"weights"), "README.md": GIT_HELLO}, source="hub")
    root, problems = run(tmp_path, d, rmap)
    assert problems == []
    assert (root / "snapshots" / COMMIT / "m.gguf").read_bytes() == b"weights"
    assert (root / "blobs" / GIT_HELLO).read_bytes() == b"hello world\n"
    assert (root / "refs" / "main").read_text() == COMMIT


def test_a_file_from_another_revision_is_refused_not_misfiled(tmp_path):
    d = local(tmp_path, {"m.gguf": b"old weights", "README.md": b"hello world\n"})
    rmap = RepoMap(commit_hash=COMMIT, etags={"m.gguf": sha(b"new weights"), "README.md": GIT_HELLO}, source="hub")
    root, problems = run(tmp_path, d, rmap, mode="move")
    assert len(problems) == 1 and "m.gguf" in problems[0]
    assert not (root / "blobs" / sha(b"new weights")).exists()
    assert not (root / "snapshots" / COMMIT / "m.gguf").exists()
    assert (d / "m.gguf").read_bytes() == b"old weights"           # left in place, even with --move
    assert not (root / "refs" / "main").exists()                    # incomplete snapshot gets no ref


def test_offline_xet_hash_metadata_is_filed_under_the_sha256(tmp_path):
    d = local(tmp_path, {"m.gguf": b"weights"})
    xet = sha(b"xet:" + b"weights")                                 # what 2.x metadata records
    rmap = RepoMap(commit_hash=COMMIT, etags={"m.gguf": xet}, source="local-metadata")
    root, problems = run(tmp_path, d, rmap)
    assert problems == []
    assert (root / "blobs" / sha(b"weights")).read_bytes() == b"weights"
    assert not (root / "blobs" / xet).exists()


def test_offline_git_file_with_wrong_hash_is_refused(tmp_path):
    d = local(tmp_path, {"README.md": b"edited\n"})
    rmap = RepoMap(commit_hash=COMMIT, etags={"README.md": GIT_HELLO}, source="local-metadata")
    _, problems = run(tmp_path, d, rmap)
    assert len(problems) == 1


def test_failed_hub_lookup_does_not_fall_back_to_metadata(tmp_path, monkeypatch, capsys):
    d = local(tmp_path, {"m.gguf": b"weights"})
    md = d / ".cache/huggingface/download"
    md.mkdir(parents=True)
    (md / "m.gguf.metadata").write_text(f"{COMMIT}\n{sha(b'weights')}\n0\n")

    def boom(*a, **k):
        raise RuntimeError("503")
    monkeypatch.setattr(xfer, "map_from_hub", boom)
    rmap = xfer.build_repo_map(d, "org/name", "model", "main", offline=False)
    assert rmap.commit_hash is None and rmap.etags == {}
    assert "--offline" in capsys.readouterr().err
    assert xfer.build_repo_map(d, "org/name", "model", "main", offline=True).commit_hash == COMMIT


# ------------------------------------------------------------------ metadata-first default
OLD, NEW = "a" * 40, "b" * 40


def with_metadata(tmp_path, files: dict[str, bytes], commit: str, etags: dict[str, str]) -> Path:
    d = local(tmp_path, files)
    md = d / ".cache/huggingface/download"
    for rel, etag in etags.items():
        (md / (rel + ".metadata")).parent.mkdir(parents=True, exist_ok=True)
        (md / (rel + ".metadata")).write_text(f"{commit}\n{etag}\n0\n")
    return d


def import_default(tmp_path, d, monkeypatch, tips):
    monkeypatch.setattr(xfer, "hub_ref_names", lambda *a: tips)
    root = None
    for rmap in xfer.build_repo_maps(d, "org/name", "model", None, offline=False):
        plan, root, commit = xfer.plan_import(d, tmp_path / "hub", "org/name", "model", rmap)
        _, problems = xfer.execute_import(d, root, commit, rmap.ref, plan, "copy", rmap.source, rmap.ref_if_absent)
        assert problems == []
    return root


def test_default_imports_each_file_at_its_download_commit(tmp_path, monkeypatch):
    xet = sha(b"xet:weights")                                        # 2.x metadata: not the sha256
    d = with_metadata(tmp_path, {"m.gguf": b"weights", "README.md": b"hello world\n"}, OLD,
                      {"m.gguf": xet, "README.md": GIT_HELLO})
    root = import_default(tmp_path, d, monkeypatch, tips={NEW: ["main"]})   # main has moved on
    snap = root / "snapshots" / OLD
    assert (snap / "m.gguf").read_bytes() == b"weights" and (snap / "README.md").read_bytes() == b"hello world\n"
    assert (root / "blobs" / sha(b"weights")).exists() and not (root / "blobs" / xet).exists()
    assert (root / "refs" / "main").read_text() == OLD               # no main in the cache yet


def test_an_older_commit_never_moves_an_existing_main_back(tmp_path, monkeypatch):
    d = with_metadata(tmp_path, {"README.md": b"hello world\n"}, OLD, {"README.md": GIT_HELLO})
    refs = tmp_path / "hub" / "models--org--name" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(NEW)
    import_default(tmp_path, d, monkeypatch, tips={NEW: ["main"]})
    assert (refs / "main").read_text() == NEW


def test_a_branch_tip_gets_its_branch_ref(tmp_path, monkeypatch):
    d = with_metadata(tmp_path, {"m.safetensors": b"exl3"}, OLD, {"m.safetensors": sha(b"x")})
    root = import_default(tmp_path, d, monkeypatch, tips={OLD: ["3.05bpw"], NEW: ["main"]})
    assert (root / "refs" / "3.05bpw").read_text() == OLD and not (root / "refs" / "main").exists()


def test_files_without_metadata_use_the_hub_listing(tmp_path, monkeypatch):
    d = with_metadata(tmp_path, {"README.md": b"hello world\n", "extra.bin": b"x"}, OLD, {"README.md": GIT_HELLO})
    monkeypatch.setattr(xfer, "map_from_hub", lambda *a: RepoMap(NEW, {"extra.bin": sha(b"x")}, "hub"))
    root = import_default(tmp_path, d, monkeypatch, tips={NEW: ["main"]})
    assert (root / "snapshots" / NEW / "extra.bin").read_bytes() == b"x"
    assert (root / "snapshots" / OLD / "README.md").exists()
    assert (root / "refs" / "main").read_text() == NEW
