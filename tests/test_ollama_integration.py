"""Opt-in: HFHUB_OLLAMA=1 starts a scratch `ollama serve` and checks synced models are listed."""
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hfhub import config as cfg, sync
from tests.hub_fixture import add_repo

pytestmark = pytest.mark.skipif(not os.environ.get("HFHUB_OLLAMA") or not shutil.which("ollama"),
                                reason="set HFHUB_OLLAMA=1 with ollama installed")


def _wait_ready(env, proc, timeout: float = 30.0) -> None:
    """Poll `ollama list` against the scratch server until it answers."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"scratch `ollama serve` on {env['OLLAMA_HOST']} exited with {proc.returncode}")
        if subprocess.run(["ollama", "list"], env=env, capture_output=True).returncode == 0:
            return
        time.sleep(0.5)
    pytest.fail(f"scratch `ollama serve` on {env['OLLAMA_HOST']} was not ready within {timeout:.0f}s")


@pytest.fixture
def server(tmp_path: Path):
    root = tmp_path / "ol"
    root.mkdir()
    env = dict(os.environ, OLLAMA_MODELS=str(root), OLLAMA_HOST="127.0.0.1:11499")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_ready(env, proc)
        yield root, env
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_synced_model_and_alias_are_listed(tmp_path: Path, server, monkeypatch):
    root, env = server
    hub = tmp_path / "hub"
    src = Path(os.environ["HF_HOME"]) / "hub" / "models--ibm-granite--granite-docling-258M-GGUF"
    real = next(src.glob("snapshots/*/granite-docling-258M-BF16.gguf")).read_bytes()
    add_repo(hub, "ibm-granite/granite-docling-258M-GGUF", {"granite-docling-258M-BF16.gguf": real})
    monkeypatch.setattr(sync, "hub_dir", lambda: hub)
    c = cfg.Config(path=tmp_path / "c.toml", views={
        "lmstudio": cfg.ViewConfig(),
        "ollama": cfg.ViewConfig(root=root, aliases={"docling:bf16": "ibm-granite/granite-docling-258M-GGUF:granite-docling-258M-BF16.gguf"})})
    sync.run(c, ["ollama"], execute=True, offline=False, out=print)
    listed = subprocess.run(["ollama", "list"], env=env, capture_output=True, text=True).stdout
    assert "hf.co/ibm-granite/granite-docling-258M-GGUF:BF16" in listed, listed
    assert "docling:bf16" in listed, listed
    show = subprocess.run(["ollama", "show", "docling:bf16"], env=env, capture_output=True, text=True)
    assert show.returncode == 0
