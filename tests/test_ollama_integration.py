"""Opt-in: HFHUB_OLLAMA=1 starts a scratch `ollama serve` and checks synced models are listed."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hfhub import config as cfg, sync
from tests.hub_fixture import add_repo
from tests.test_ollama import FIX

pytestmark = pytest.mark.skipif(not os.environ.get("HFHUB_OLLAMA") or not shutil.which("ollama"),
                                reason="set HFHUB_OLLAMA=1 with ollama installed")


@pytest.fixture
def server(tmp_path: Path):
    root = tmp_path / "ol"
    root.mkdir()
    env = dict(os.environ, OLLAMA_MODELS=str(root), OLLAMA_HOST="127.0.0.1:11499")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    yield root, env
    proc.terminate()
    proc.wait(timeout=10)


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
