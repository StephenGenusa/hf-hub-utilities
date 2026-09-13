from pathlib import Path

from hfhub import dupes
from hfhub.dupes import Signature
from tests.gguf_fixture import write_gguf


def test_signature_reads_header(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"general.architecture": "qwen35", "general.file_type": 15}, [("a", [1000])])
    assert dupes.signature(p) == Signature("qwen35", 15, 1000)


def test_find_dupes_matches_within_tolerance():
    f = [("ollama:x:8b", Signature("llama", 15, 8_030_000_000))]
    c = [("unsloth/L-GGUF:L-Q4_K_M.gguf", Signature("llama", 15, 8_000_000_000)),
         ("unsloth/L-GGUF:L-Q8_0.gguf", Signature("llama", 7, 8_000_000_000)),
         ("o/g:g.gguf", Signature("gemma4", 15, 8_000_000_000)),
         ("o/big:b.gguf", Signature("llama", 15, 9_000_000_000))]
    d = dupes.find_dupes(f, c)
    assert [(x.foreign_key, x.cache_key) for x in d] == [("ollama:x:8b", "unsloth/L-GGUF:L-Q4_K_M.gguf")]
    assert "Q4_K_M" in d[0].reason and "llama" in d[0].reason
