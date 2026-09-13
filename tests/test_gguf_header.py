from pathlib import Path

import pytest

from hfhub.gguf_header import file_type_name, read_header, size_label
from tests.gguf_fixture import write_gguf


def test_reads_scalar_and_string_kv(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"general.architecture": "qwen35", "general.file_type": 15}, [])
    h = read_header(p)
    assert h.version == 3
    assert h.kv["general.architecture"] == "qwen35"
    assert h.kv["general.file_type"] == 15


def test_param_count_sums_tensor_elements(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"general.architecture": "llama"}, [("a", [4, 8]), ("b", [10])])
    assert read_header(p).param_count == 42
    assert read_header(p).n_tensors == 2


def test_long_string_arrays_are_truncated(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {"tokenizer.ggml.tokens": [f"t{i}" for i in range(100)]}, [])
    h = read_header(p, max_array_items=8)
    assert h.kv["tokenizer.ggml.tokens"] == [f"t{i}" for i in range(8)]


def test_rejects_non_gguf(tmp_path: Path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(ValueError):
        read_header(p)


def test_file_type_names():
    assert file_type_name(15) == "Q4_K_M"
    assert file_type_name(1) == "F16"
    assert file_type_name(999) == "unknown"


def test_size_label():
    assert size_label(4_650_000_000) == "4.7B"
    assert size_label(137_000_000) == "137.0M"
