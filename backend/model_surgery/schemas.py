"""API / report data model.

Provenance vocabulary used everywhere:
  verified  - read directly from the file (header field, metadata key, tensor descriptor)
  derived   - computed deterministically from verified values (e.g. parameter count)
  inferred  - heuristic (e.g. component role guessed from a tensor name)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Provenance = Literal["verified", "derived", "inferred"]
ModelKind = Literal["gguf", "hf_safetensors", "safetensors"]


class Fact(BaseModel):
    key: str
    label: str
    value: int | float | str | bool | None
    provenance: Provenance
    source: str


class SourceFile(BaseModel):
    path: str
    size_bytes: int
    role: str
    parsed: bool
    note: str | None = None


class MetadataEntry(BaseModel):
    key: str
    type: str
    value: Any
    length: int | None = None
    truncated: bool = False


class TensorRecord(BaseModel):
    name: str
    shape: list[int]
    shape_order: Literal["ggml", "row_major"]
    row_length: int  # elements along the contiguous axis (GGML ne0)
    dtype: str
    n_elements: int
    n_bytes: int | None
    offset: int | None
    file_index: int
    stack: str | None = None
    layer: int | None = None
    component: str  # embedding | attention | ffn | experts | router | norm | output | ssm | other
    role: str | None = None


class ComponentSummary(BaseModel):
    component: str
    roles: list[str]
    tensors: list[str]
    n_elements: int
    n_bytes: int


class LayerSummary(BaseModel):
    index: int
    components: list[ComponentSummary]
    n_elements: int
    n_bytes: int
    dtypes: dict[str, int]


class StackSummary(BaseModel):
    prefix: str
    n_layers: int
    layers: list[LayerSummary]
    n_bytes: int
    primary: bool


class ArchitectureSummary(BaseModel):
    family: str | None
    family_provenance: Provenance | None
    stacks: list[StackSummary]
    globals: list[ComponentSummary]  # embedding / output / final norm / other non-layer tensors
    uniform_layers: bool
    notes: list[str]


class DTypeRow(BaseModel):
    dtype: str
    n_tensors: int
    n_elements: int
    n_bytes: int


class QuantizationSummary(BaseModel):
    label: str | None
    label_provenance: Provenance | None
    is_quantized: bool
    effective_bits_per_weight: float | None
    packed_checkpoint: bool  # HF GPTQ/AWQ/bnb: tensor numel != logical parameter count
    config: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)


class Integrity(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]


class FormatInfo(BaseModel):
    name: str
    version: str | None
    details: dict[str, Any]


class ScanResult(BaseModel):
    id: str
    scanned_at: str
    app_version: str
    kind: ModelKind
    name: str
    path: str
    files: list[SourceFile]
    total_size_bytes: int
    format: FormatInfo
    integrity: Integrity
    hyperparameters: list[Fact]
    quantization: QuantizationSummary
    dtype_breakdown: list[DTypeRow]
    architecture: ArchitectureSummary
    metadata: list[MetadataEntry]
    tensors: list[TensorRecord]
    # Repository facts reported by the Hugging Face Hub API, for scans of hf:// models (see frontend engine/hfhub.ts).
    hub: dict[str, Any] | None = None

    @property
    def is_remote(self) -> bool:
        return self.path.startswith("hf://")

    def fact(self, key: str) -> Fact | None:
        return next((f for f in self.hyperparameters if f.key == key), None)

    def fact_value(self, key: str) -> Any:
        f = self.fact(key)
        return None if f is None else f.value


class ScanSummary(BaseModel):
    id: str
    scanned_at: str
    path: str
    kind: ModelKind
    name: str
    total_size_bytes: int
    n_params: int | None
    quant_label: str | None
    valid: bool
