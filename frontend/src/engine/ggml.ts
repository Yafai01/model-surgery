// GGML tensor type table and llama.cpp file types. Mirrors backend/model_surgery/formats/ggml_types.py.

export interface GGMLType {
  id: number;
  name: string;
  blockSize: number;
  typeSize: number;
  quantized: boolean;
}

const T = (id: number, name: string, blockSize: number, typeSize: number, quantized: boolean): GGMLType => ({ id, name, blockSize, typeSize, quantized });

const TYPES: GGMLType[] = [
  T(0, "F32", 1, 4, false),
  T(1, "F16", 1, 2, false),
  T(2, "Q4_0", 32, 18, true),
  T(3, "Q4_1", 32, 20, true),
  T(6, "Q5_0", 32, 22, true),
  T(7, "Q5_1", 32, 24, true),
  T(8, "Q8_0", 32, 34, true),
  T(9, "Q8_1", 32, 36, true),
  T(10, "Q2_K", 256, 84, true),
  T(11, "Q3_K", 256, 110, true),
  T(12, "Q4_K", 256, 144, true),
  T(13, "Q5_K", 256, 176, true),
  T(14, "Q6_K", 256, 210, true),
  T(15, "Q8_K", 256, 292, true),
  T(16, "IQ2_XXS", 256, 66, true),
  T(17, "IQ2_XS", 256, 74, true),
  T(18, "IQ3_XXS", 256, 98, true),
  T(19, "IQ1_S", 256, 50, true),
  T(20, "IQ4_NL", 32, 18, true),
  T(21, "IQ3_S", 256, 110, true),
  T(22, "IQ2_S", 256, 82, true),
  T(23, "IQ4_XS", 256, 136, true),
  T(24, "I8", 1, 1, false),
  T(25, "I16", 1, 2, false),
  T(26, "I32", 1, 4, false),
  T(27, "I64", 1, 8, false),
  T(28, "F64", 1, 8, false),
  T(29, "IQ1_M", 256, 56, true),
  T(30, "BF16", 1, 2, false),
  T(34, "TQ1_0", 256, 54, true),
  T(35, "TQ2_0", 256, 66, true),
  T(39, "MXFP4", 32, 17, true),
];

export const GGML_BY_ID = new Map(TYPES.map((t) => [t.id, t]));
export const GGML_BY_NAME = new Map(TYPES.map((t) => [t.name, t]));

export const LLAMA_FTYPES: Record<number, string> = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
  13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
  21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
  29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
};
export const LLAMA_FTYPE_GUESSED = 1024;
export const UNQUANTIZED_FTYPES = new Set(["F32", "F16", "BF16"]);

export function ftypeLabel(v: number): string {
  const guessed = (v & LLAMA_FTYPE_GUESSED) !== 0;
  const base = v & ~LLAMA_FTYPE_GUESSED;
  const name = LLAMA_FTYPES[base] ?? `unknown(${base})`;
  return guessed ? `${name} (guessed)` : name;
}

export function tensorNBytes(t: GGMLType, shape: number[]): number {
  const n = shape.reduce((a, b) => a * b, 1);
  return Math.floor(n / t.blockSize) * t.typeSize;
}
