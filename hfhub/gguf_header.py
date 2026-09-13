"""Read the GGUF header (KV metadata + tensor infos) without touching tensor data."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
           6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
_STRING, _ARRAY = 8, 9

_FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
               11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
               17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
               23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
               29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
               38: "MXFP4_MOE"}


@dataclass(frozen=True)
class GgufHeader:
    version: int
    n_tensors: int
    kv: dict[str, object]
    param_count: int


class _Reader:
    def __init__(self, f, max_array_items: int):
        self.f = f
        self.max_array_items = max_array_items

    def raw(self, n: int) -> bytes:
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t: int):
        if t == _STRING:
            return self.string()
        if t == _ARRAY:
            et, n = self.u32(), self.u64()
            keep = min(n, self.max_array_items)
            items = [self.value(et) for _ in range(keep)]
            for _ in range(n - keep):
                self.value(et)
            return items
        fmt, size = _SCALAR[t]
        return struct.unpack(fmt, self.raw(size))[0]


def read_header(path: Path, max_array_items: int = 64) -> GgufHeader:
    with open(path, "rb") as f:
        r = _Reader(f, max_array_items)
        if r.raw(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = r.u32()
        n_tensors, n_kv = r.u64(), r.u64()
        kv: dict[str, object] = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.u32())
        params = 0
        for _ in range(n_tensors):
            r.string()
            ndim = r.u32()
            count = 1
            for _ in range(ndim):
                count *= r.u64()
            r.u32()
            r.u64()
            params += count
    return GgufHeader(version=version, n_tensors=n_tensors, kv=kv, param_count=params)


def file_type_name(ft: int) -> str:
    return _FILE_TYPES.get(ft, "unknown")


def size_label(param_count: int) -> str:
    if param_count >= 1_000_000_000:
        return f"{param_count / 1e9:.1f}B"
    return f"{param_count / 1e6:.1f}M"
