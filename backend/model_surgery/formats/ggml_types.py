"""GGML tensor type table and llama.cpp file-type names.

Block sizes and per-block byte sizes mirror ggml's ``type_traits`` table
(ggml/src/ggml.c). Removed type ids (4, 5, 31-33, 36-38) are intentionally
absent so files using them are rejected rather than mis-sized.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GGMLType:
    id: int
    name: str
    block_size: int  # elements per block
    type_size: int  # bytes per block
    quantized: bool

    @property
    def bits_per_weight(self) -> float:
        return self.type_size * 8 / self.block_size


_TYPES = [
    GGMLType(0, "F32", 1, 4, False),
    GGMLType(1, "F16", 1, 2, False),
    GGMLType(2, "Q4_0", 32, 18, True),
    GGMLType(3, "Q4_1", 32, 20, True),
    GGMLType(6, "Q5_0", 32, 22, True),
    GGMLType(7, "Q5_1", 32, 24, True),
    GGMLType(8, "Q8_0", 32, 34, True),
    GGMLType(9, "Q8_1", 32, 36, True),
    GGMLType(10, "Q2_K", 256, 84, True),
    GGMLType(11, "Q3_K", 256, 110, True),
    GGMLType(12, "Q4_K", 256, 144, True),
    GGMLType(13, "Q5_K", 256, 176, True),
    GGMLType(14, "Q6_K", 256, 210, True),
    GGMLType(15, "Q8_K", 256, 292, True),
    GGMLType(16, "IQ2_XXS", 256, 66, True),
    GGMLType(17, "IQ2_XS", 256, 74, True),
    GGMLType(18, "IQ3_XXS", 256, 98, True),
    GGMLType(19, "IQ1_S", 256, 50, True),
    GGMLType(20, "IQ4_NL", 32, 18, True),
    GGMLType(21, "IQ3_S", 256, 110, True),
    GGMLType(22, "IQ2_S", 256, 82, True),
    GGMLType(23, "IQ4_XS", 256, 136, True),
    GGMLType(24, "I8", 1, 1, False),
    GGMLType(25, "I16", 1, 2, False),
    GGMLType(26, "I32", 1, 4, False),
    GGMLType(27, "I64", 1, 8, False),
    GGMLType(28, "F64", 1, 8, False),
    GGMLType(29, "IQ1_M", 256, 56, True),
    GGMLType(30, "BF16", 1, 2, False),
    GGMLType(34, "TQ1_0", 256, 54, True),
    GGMLType(35, "TQ2_0", 256, 66, True),
    GGMLType(39, "MXFP4", 32, 17, True),
]

GGML_TYPES: dict[int, GGMLType] = {t.id: t for t in _TYPES}
GGML_TYPES_BY_NAME: dict[str, GGMLType] = {t.name: t for t in _TYPES}

# llama.cpp `enum llama_ftype` -> human label (general.file_type in GGUF).
LLAMA_FTYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
}

# llama.cpp marks "guessed" file types by OR-ing this bit.
LLAMA_FTYPE_GUESSED = 1024

UNQUANTIZED_FTYPES = {"F32", "F16", "BF16"}


def ftype_label(value: int) -> str:
    guessed = bool(value & LLAMA_FTYPE_GUESSED)
    base = LLAMA_FTYPES.get(value & ~LLAMA_FTYPE_GUESSED, f"unknown({value & ~LLAMA_FTYPE_GUESSED})")
    return f"{base} (guessed)" if guessed else base


def tensor_nbytes(ggml_type: GGMLType, shape: list[int]) -> int:
    """Bytes occupied by a tensor. Caller must ensure shape[0] % block_size == 0."""
    n = 1
    for d in shape:
        n *= d
    return n // ggml_type.block_size * ggml_type.type_size
