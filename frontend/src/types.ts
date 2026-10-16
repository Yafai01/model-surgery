// Mirrors backend/model_surgery/schemas.py, memory.py, jobs.py, tooling.py.

export type Provenance = "verified" | "derived" | "inferred";
export type ModelKind = "gguf" | "hf_safetensors" | "safetensors";

export interface Fact {
  key: string;
  label: string;
  value: number | string | boolean | null;
  provenance: Provenance;
  source: string;
}

export interface SourceFile {
  path: string;
  size_bytes: number;
  role: string;
  parsed: boolean;
  note: string | null;
}

export interface MetadataEntry {
  key: string;
  type: string;
  value: unknown;
  length: number | null;
  truncated: boolean;
}

export interface TensorRecord {
  name: string;
  shape: number[];
  shape_order: "ggml" | "row_major";
  row_length: number;
  dtype: string;
  n_elements: number;
  n_bytes: number | null;
  offset: number | null;
  file_index: number;
  stack: string | null;
  layer: number | null;
  component: string;
  role: string | null;
}

export interface ComponentSummary {
  component: string;
  roles: string[];
  tensors: string[];
  n_elements: number;
  n_bytes: number;
}

export interface LayerSummary {
  index: number;
  components: ComponentSummary[];
  n_elements: number;
  n_bytes: number;
  dtypes: Record<string, number>;
}

export interface StackSummary {
  prefix: string;
  n_layers: number;
  layers: LayerSummary[];
  n_bytes: number;
  primary: boolean;
}

export interface ArchitectureSummary {
  family: string | null;
  family_provenance: Provenance | null;
  stacks: StackSummary[];
  globals: ComponentSummary[];
  uniform_layers: boolean;
  notes: string[];
}

export interface DTypeRow {
  dtype: string;
  n_tensors: number;
  n_elements: number;
  n_bytes: number;
}

export interface QuantizationSummary {
  label: string | null;
  label_provenance: Provenance | null;
  is_quantized: boolean;
  effective_bits_per_weight: number | null;
  packed_checkpoint: boolean;
  config: Record<string, unknown> | null;
  notes: string[];
}

export interface FormatInfo {
  name: string;
  version: string | null;
  details: Record<string, unknown>;
}

export interface ScanResult {
  id: string;
  scanned_at: string;
  app_version: string;
  kind: ModelKind;
  name: string;
  path: string;
  files: SourceFile[];
  total_size_bytes: number;
  format: FormatInfo;
  integrity: { valid: boolean; errors: string[]; warnings: string[] };
  hyperparameters: Fact[];
  quantization: QuantizationSummary;
  dtype_breakdown: DTypeRow[];
  architecture: ArchitectureSummary;
  metadata: MetadataEntry[];
  tensors: TensorRecord[];
  hub?: HubInfo | null;
}

/** Repository facts reported by the Hugging Face Hub API (not read from the weights). */
export interface HubInfo {
  repo: string;
  url: string;
  revision: string;
  sha: string;
  scanned_target: string; // "repository root" or a file path inside the repo
  author: string | null;
  pipeline_tag: string | null;
  library_name: string | null;
  license: string | null;
  downloads: number | null;
  likes: number | null;
  created_at: string | null;
  last_modified: string | null;
  gated: string | null; // null = not gated, otherwise "auto" / "manual"
  private: boolean;
  tags: string[];
  base_models: string[];
  datasets: string[];
  languages: string[];
  api_parameters: { total: number | null; by_dtype: Record<string, number> } | null;
  api_gguf: { total: number | null; architecture: string | null; context_length: number | null } | null;
  used_storage_bytes: number | null;
  file_count: number;
  total_file_bytes: number;
  files: { path: string; size: number | null }[];
  gguf_variants: { path: string; size: number | null; label: string | null; shards: number }[];
  readme_excerpt: string | null;
  bytes_fetched: number;
  requests: number;
}

export interface ScanSummary {
  id: string;
  scanned_at: string;
  path: string;
  kind: ModelKind;
  name: string;
  total_size_bytes: number;
  n_params: number | null;
  quant_label: string | null;
  valid: boolean;
}

export interface FsEntry {
  name: string;
  path: string;
  type: "dir" | "file";
  kind: string | null;
  size_bytes: number | null;
  is_model_dir: boolean;
}

export interface FsListing {
  path: string;
  parent: string | null;
  entries: FsEntry[];
  truncated: boolean;
}

export interface FsRoot {
  label: string;
  path: string;
}

// ---- memory -----------------------------------------------------------------

export interface DeviceProfile {
  id: string;
  name: string;
  memory_kind: "system_ram" | "discrete_gpu" | "unified";
  capacity_bytes: number;
  reserved_bytes: number;
  runtime_overhead_bytes: number;
  host_ram_bytes: number | null;
  bandwidth_gbps: number | null;
  assumptions: string[];
}

export type KvType = "f32" | "f16" | "bf16" | "q8_0" | "q4_0";

export interface MemoryRequest {
  weight_format: string;
  context_length: number;
  batch_size: number;
  kv_cache_type: KvType;
  flash_attention: boolean;
  n_ubatch: number;
  device_id: string | null;
  custom_device: DeviceProfile | null;
}

export interface WeightEstimate {
  id: string;
  label: string;
  family: string;
  bytes: number;
  bits_per_weight: number;
  provenance: Provenance;
  method: string;
  type_mix: Record<string, number>;
}

export interface MemoryComponent {
  key: string;
  label: string;
  bytes: number;
  provenance: Provenance;
  formula: string;
}

export interface FitResult {
  device_id: string;
  device_name: string;
  memory_kind: string;
  capacity_bytes: number;
  usable_bytes: number;
  required_bytes: number;
  headroom_bytes: number;
  verdict: "fits" | "tight" | "partial_offload" | "does_not_fit";
  gpu_layers: number | null;
  total_layers: number | null;
  decode_ceiling_tokens_per_s: number | null;
  notes: string[];
  assumptions: string[];
}

export interface MemoryEstimate {
  is_measurement: false;
  disclaimer: string;
  weights: WeightEstimate[];
  selected_weights: WeightEstimate;
  inference: MemoryComponent[];
  total_bytes: number;
  assumptions: string[];
  warnings: string[];
  fit: FitResult | null;
}

// ---- tools & jobs -------------------------------------------------------------

export interface QuantType {
  name: string;
  ftype_id: number;
  description: string;
}

export interface ServerTool {
  available: boolean;
  path: string | null;
  version: string | null;
  detail: string | null;
}

export interface ToolsReport {
  server?: ServerTool;
  quantize: {
    available: boolean;
    path: string | null;
    version: string | null;
    types: QuantType[];
    supports_dry_run: boolean;
    detail: string | null;
  };
  convert: {
    available: boolean;
    script: string | null;
    python: string | null;
    outtypes: string[];
    supported_architectures: string[] | null;
    detail: string | null;
  };
  checked_at: number;
  configuration_help: string[];
}

export interface PlanRequest {
  scan_id: string;
  operation: "quantize" | "convert";
  target: string;
  output_dir: string | null;
  allow_requantize: boolean;
  threads: number | null;
}

export interface Plan {
  plan_id: string;
  created_at: string;
  expires_at: number;
  operation: "quantize" | "convert";
  scan_id: string;
  source_path: string;
  source_label: string | null;
  target: string;
  output_dir: string | null;
  output_path: string | null;
  partial_path: string | null;
  argv: string[];
  tool: string | null;
  tool_version: string | null;
  estimated_output_bytes: number | null;
  estimate_note: string;
  free_disk_bytes: number | null;
  warnings: string[];
  blockers: string[];
  can_execute: boolean;
  confirm_token: string | null;
  supports_tool_dry_run: boolean;
}

export interface ToolDryRun {
  model_size_mib: number | null;
  quant_size_mib: number | null;
  quant_bpw: number | null;
  fallback_tensors: number;
  errors: string[];
  exit_code: number | null;
  argv: string[];
  tail: string[];
}

export interface ValidationCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export type JobStatus = "running" | "validating" | "succeeded" | "failed" | "cancelled" | "interrupted";

export interface Job {
  id: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  status: JobStatus;
  plan: Plan;
  progress: number | null;
  progress_label: string | null;
  exit_code: number | null;
  log_tail: string[];
  output_path: string | null;
  output_size_bytes: number | null;
  validation: { passed: boolean; checks: ValidationCheck[] } | null;
  output_scan_id: string | null;
  error: string | null;
}

export interface CompareRow {
  section: string;
  label: string;
  a: string;
  b: string;
  differs: boolean;
}

export interface Comparison {
  a_id: string;
  b_id: string;
  a_name: string;
  b_name: string;
  rows: CompareRow[];
  tensors_only_in_a: number;
  tensors_only_in_b: number;
  tensors_shape_changed: number;
  tensors_dtype_changed: number;
  dtype_changes: { from: string; to: string; count: number }[];
  note: string;
}

// ---- local inference (llama-server) -------------------------------------------

export interface RuntimeAllocation {
  kind: string; // model | KV | output | compute | RS
  device: string;
  bytes: number;
  line: string;
}

export interface RuntimeStatus {
  state: "unavailable" | "stopped" | "starting" | "ready" | "failed";
  tool: ServerTool;
  scan_id: string | null;
  model_name: string | null;
  context_length: number | null;
  n_ctx_runtime: number | null;
  n_gpu_layers_requested: number | null;
  gpu_layers_offloaded: number | null;
  total_layers: number | null;
  pid: number | null;
  started_at: string | null;
  load_seconds: number | null;
  allocations: RuntimeAllocation[];
  process_rss_bytes: number | null;
  generating: boolean;
  error: string | null;
  log_tail: string[];
  capabilities: Record<string, boolean>;
  trace_note: string;
}

export interface RuntimeStartRequest {
  scan_id: string;
  context_length: number;
  n_gpu_layers: number | null;
  threads: number | null;
  allow_memory_risk: boolean;
}

export interface GenerateRequest {
  prompt: string;
  chat: boolean;
  system: string | null;
  n_predict: number;
  temperature: number;
  top_k: number;
  top_p: number;
  seed: number | null;
  n_probs: number;
}

export interface TokenProb {
  id: number;
  token: string;
  logprob: number;
  top: { id: number; token: string; logprob: number }[];
}

export type GenerationEvent =
  | { type: "notice"; message: string }
  | { type: "prompt"; formatted: string; chat_template_applied: boolean; n_tokens: number; tokens: { id: number; piece: string | null }[] }
  | { type: "token"; content: string; ids: number[]; probs: TokenProb[]; t_ms: number }
  | {
      type: "done";
      stop_type: string | null;
      truncated: boolean;
      tokens_predicted: number | null;
      tokens_evaluated: number | null;
      tokens_cached: number | null;
      timings: Record<string, number | null>;
      wall: { first_token_ms: number | null; total_ms: number };
      sampling: Record<string, unknown>;
    }
  | { type: "cancelled"; wall: { first_token_ms: number | null; total_ms: number } }
  | { type: "error"; message: string };

export interface Measurement {
  at: string;
  scan_id: string | null;
  model_name: string | null;
  n_ctx: number | null;
  gpu_layers_offloaded: number | null;
  prompt_tokens: number | null;
  prompt_ms: number | null;
  prompt_tokens_per_s: number | null;
  generated_tokens: number | null;
  decode_ms: number | null;
  decode_tokens_per_s: number | null;
  wall_first_token_ms: number | null;
  wall_total_ms: number | null;
  process_rss_bytes: number | null;
  stop_type: string | null;
}

export interface SystemInfo {
  version: string;
  platform: string;
  cpu_count: number | null;
  memory: { total_bytes: number | null; available_bytes: number | null; source: string };
  data_dir: string;
  outputs_dir: string;
  allowed_roots: string[];
  llama_cpp_bin_dir: string | null;
  network: string;
}
