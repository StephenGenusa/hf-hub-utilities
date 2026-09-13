"""Minimal GGUF v3 writer for tests: header + KV + tensor infos, no tensor data."""
import struct
from pathlib import Path

_STR, _U32, _ARR, _U64, _F32 = 8, 4, 9, 10, 6


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", 7) + struct.pack("<?", v)
    if isinstance(v, int):
        return struct.pack("<I", _U32) + struct.pack("<I", v) if v < 2**32 else struct.pack("<I", _U64) + struct.pack("<Q", v)
    if isinstance(v, float):
        return struct.pack("<I", _F32) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", _STR) + _s(v)
    if isinstance(v, list):
        out = struct.pack("<I", _ARR) + struct.pack("<I", _STR) + struct.pack("<Q", len(v))
        return out + b"".join(_s(x) for x in v)
    raise TypeError(type(v))


def write_gguf(path: Path, kv: dict[str, object], tensors: list[tuple[str, list[int]]]) -> None:
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kv))
    for k, v in kv.items():
        out += _s(k) + _value(v)
    for name, dims in tensors:
        out += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<I", 0) + struct.pack("<Q", 0)  # dtype F32, offset 0
    path.write_bytes(out)
