from pathlib import Path

from hfhub import config as cfg


TOML = '''
[views.lmstudio]
root = "~/lm"

[views.ollama]
root = "/data/ollama"

[views.ollama.aliases]
"qwen3.6:27b" = "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf"
'''


def test_load_parses_roots_and_aliases(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(TOML)
    c = cfg.load(p)
    assert c.views["lmstudio"].root == Path("~/lm").expanduser()
    assert c.views["ollama"].root == Path("/data/ollama")
    assert c.views["ollama"].aliases["qwen3.6:27b"].startswith("unsloth/")
    assert c.views["lmstudio"].aliases == {}


def test_missing_file_gives_disabled_views(tmp_path: Path):
    c = cfg.load(tmp_path / "nope.toml")
    assert c.views["lmstudio"].root is None
    assert c.views["ollama"].root is None


def test_env_override_selects_path(tmp_path: Path, monkeypatch):
    p = tmp_path / "x.toml"
    p.write_text('[views.lmstudio]\nroot = "/a"\n')
    monkeypatch.setenv("HFHUB_CONFIG", str(p))
    assert cfg.load().views["lmstudio"].root == Path("/a")


def test_dump_roundtrips(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(TOML)
    c = cfg.load(p)
    c.views["ollama"].aliases["llama3.1:8b"] = "ollama/llama3.1:llama3.1-8b.gguf"
    cfg.save(c)
    again = cfg.load(p)
    assert again.views["ollama"].aliases["llama3.1:8b"] == "ollama/llama3.1:llama3.1-8b.gguf"
    assert again.views["lmstudio"].root == Path("~/lm").expanduser()
