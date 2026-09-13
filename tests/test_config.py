from pathlib import Path

import pytest

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


HAND_WRITTEN = '''# my hfhub config
[views.lmstudio]
root = "~/lm"      # trailing comment

[views.ollama]
root = "/data/ollama"
unknown_key = 3

[views.ollama.aliases]
"qwen3.6:27b" = "unsloth/Qwen3.6-27B-GGUF:Qwen3.6-27B-UD-Q4_K_XL.gguf"

[other]
keep = "me"
'''


def test_add_alias_preserves_the_rest_of_the_file(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(HAND_WRITTEN)
    c = cfg.load(p)
    cfg.add_alias(c, "ollama", "llama3.1:8b", "ollama/llama3.1:llama3.1-8b.gguf")
    text = p.read_text()
    assert "# my hfhub config" in text
    assert "[other]\nkeep = \"me\"" in text
    assert "unknown_key = 3" in text
    assert "root = \"~/lm\"      # trailing comment" in text
    assert '"llama3.1:8b" = "ollama/llama3.1:llama3.1-8b.gguf"' in text
    # inserted inside the aliases table, not after [other]
    lines = text.splitlines()
    assert lines.index('"llama3.1:8b" = "ollama/llama3.1:llama3.1-8b.gguf"') < lines.index("[other]")
    again = cfg.load(p)
    assert again.views["ollama"].aliases["llama3.1:8b"] == "ollama/llama3.1:llama3.1-8b.gguf"
    assert again.views["ollama"].aliases["qwen3.6:27b"].startswith("unsloth/")
    assert c.views["ollama"].aliases["llama3.1:8b"] == "ollama/llama3.1:llama3.1-8b.gguf"


def test_add_alias_appends_a_table_when_there_is_none(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[views.ollama]\nroot = "/data/ollama"\n')
    c = cfg.load(p)
    cfg.add_alias(c, "ollama", "m:q4", "org/M-GGUF:M-Q4_K_M.gguf")
    assert p.read_text() == ('[views.ollama]\nroot = "/data/ollama"\n\n'
                             '[views.ollama.aliases]\n"m:q4" = "org/M-GGUF:M-Q4_K_M.gguf"\n')
    assert cfg.load(p).views["ollama"].aliases == {"m:q4": "org/M-GGUF:M-Q4_K_M.gguf"}


def test_add_alias_replaces_an_existing_line(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(HAND_WRITTEN)
    c = cfg.load(p)
    cfg.add_alias(c, "ollama", "qwen3.6:27b", "other/Repo-GGUF:Repo-Q8_0.gguf")
    text = p.read_text()
    assert text.count('"qwen3.6:27b"') == 1
    assert '"qwen3.6:27b" = "other/Repo-GGUF:Repo-Q8_0.gguf"' in text
    assert "[other]" in text and "# my hfhub config" in text
    assert cfg.load(p).views["ollama"].aliases == {"qwen3.6:27b": "other/Repo-GGUF:Repo-Q8_0.gguf"}


def test_add_alias_matches_a_header_with_a_trailing_comment(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[views.ollama]\nroot = "/d"\n\n'
                 '[views.ollama.aliases]   # my aliases\n"a:b" = "o/r:a.gguf"\n\n[other]\nk = 1\n')
    c = cfg.load(p)
    cfg.add_alias(c, "ollama", "m:q4", "org/M-GGUF:M-Q4_K_M.gguf")
    text = p.read_text()
    assert "[views.ollama.aliases]   # my aliases" in text
    lines = text.splitlines()
    assert lines.index('"m:q4" = "org/M-GGUF:M-Q4_K_M.gguf"') < lines.index("[other]")
    assert cfg.load(p).views["ollama"].aliases == {"a:b": "o/r:a.gguf", "m:q4": "org/M-GGUF:M-Q4_K_M.gguf"}


def test_add_alias_replaces_a_single_quoted_key(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[views.ollama.aliases]\n'm:q4' = 'old/Repo:old.gguf'\n")
    c = cfg.load(p)
    cfg.add_alias(c, "ollama", "m:q4", "org/M-GGUF:M-Q4_K_M.gguf")
    text = p.read_text()
    assert "old/Repo:old.gguf" not in text
    assert text.count("m:q4") == 1
    assert cfg.load(p).views["ollama"].aliases == {"m:q4": "org/M-GGUF:M-Q4_K_M.gguf"}


def test_add_alias_refuses_an_inline_aliases_table(tmp_path: Path):
    p = tmp_path / "config.toml"
    before = '[views.ollama]\nroot = "/d"\naliases = { "a:b" = "o/r:a.gguf" }\n'
    p.write_text(before)
    c = cfg.load(p)
    with pytest.raises(ValueError, match="inline aliases table"):
        cfg.add_alias(c, "ollama", "m:q4", "org/M-GGUF:M-Q4_K_M.gguf")
    assert p.read_text() == before
    assert "m:q4" not in c.views["ollama"].aliases


def test_add_alias_never_writes_unparseable_toml(tmp_path: Path):
    p = tmp_path / "config.toml"
    before = '[views.ollama.aliases]\n"broken = "o/r:a.gguf\n'
    p.write_text(before)
    c = cfg.Config(path=p, views={"ollama": cfg.ViewConfig()})
    with pytest.raises(ValueError, match="unparseable TOML"):
        cfg.add_alias(c, "ollama", "m:q4", "org/M-GGUF:M-Q4_K_M.gguf")
    assert p.read_text() == before
    assert c.views["ollama"].aliases == {}
