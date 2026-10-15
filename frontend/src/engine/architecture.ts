// Port of backend/model_surgery/architecture.py: tensor names -> layers and components (inferred).

import type { ArchitectureSummary, ComponentSummary, LayerSummary, Provenance, StackSummary, TensorRecord } from "../types";

const LAYER_RE = /^(.*?(?:^|\.)(?:blk|layers|layer|h|blocks|block))\.(\d+)\.(.+)$/;

const COMPONENT_RULES: [string, RegExp][] = [
  ["norm", /(norm|layernorm|(^|[._])ln(_?\d+|_f|_attn|_mlp)?([._]|$))/],
  ["router", /(ffn_gate_inp|router|block_sparse_moe\.gate\.|mlp\.gate\.(weight|bias)$|(^|\.)gate\.(weight|bias)$|exp_probs)/],
  ["experts", /(_exps\b|_exps\.|experts\.|\.experts$)/],
  ["ssm", /(ssm_|mixer\.|time_mix|channel_mix|conv1d|\bx_proj|dt_proj|A_log)/],
  ["attention", /(attn|attention|self_attn|q_proj|k_proj|v_proj|o_proj|\bquery\b|\bkey\b|\bvalue\b|qkv|\bwq\b|\bwk\b|\bwv\b|\bwo\b)/],
  ["ffn", /(mlp|ffn|feed_forward|gate_proj|up_proj|down_proj|\bfc\d\b|c_fc|\bw[123]\b|dense_h_to_4h|dense_4h_to_h|shexp|shared_expert)/],
];

const ROLE_RULES: [string, RegExp][] = [
  ["qkv", /(attn_qkv|qkv_proj|c_attn|query_key_value|wqkv|\.qkv\.)/],
  ["q_norm", /(attn_q_norm|q_norm|q_layernorm)/],
  ["k_norm", /(attn_k_norm|k_norm|k_layernorm)/],
  ["q", /(attn_q\b|attn_q\.|q_proj|\.query\.|\bwq\b)/],
  ["k", /(attn_k\b|attn_k\.|k_proj|\.key\.|\bwk\b)/],
  ["v", /(attn_v\b|attn_v\.|v_proj|\.value\.|\bwv\b)/],
  ["o", /(attn_output|attn_out\b|o_proj|out_proj|attn\.c_proj|attention\.dense|self_attention\.dense|\bwo\b)/],
  ["gate_up", /(gate_up_proj|ffn_gate_up)/],
  ["gate", /(ffn_gate|gate_proj|\bw1\b)/],
  ["up", /(ffn_up|up_proj|\bw3\b|c_fc|\bfc1\b|dense_h_to_4h)/],
  ["down", /(ffn_down|down_proj|\bw2\b|mlp\.c_proj|\bfc2\b|dense_4h_to_h)/],
  ["post_attn_norm", /post_attention_norm/],
  ["post_ffn_norm", /(post_ffw_norm|post_feedforward_layernorm)/],
  ["attn_norm", /(attn_norm|input_layernorm|ln_1|ln_attn|attention_norm)/],
  ["ffn_norm", /(ffn_norm|pre_feedforward_layernorm|post_attention_layernorm|ln_2|ln_mlp)/],
];

const GLOBAL_RULES: [string, string, RegExp][] = [
  ["embedding", "position", /(wpe|position_embd|embed_positions|position_embeddings)/],
  ["embedding", "token", /(token_embd|embed_tokens|\bwte\b|tok_embeddings|word_embeddings|embed_in)\b/],
  ["norm", "final", /(output_norm|ln_f|final_layernorm|final_layer_norm|(^|\.)norm\.(weight|bias)$)/],
  ["output", "lm_head", /(^output\.|lm_head|embed_out)/],
  ["other", "rope", /rope/],
];

export interface Classification {
  stack: string | null;
  layer: number | null;
  component: string;
  role: string | null;
}

export function classify(name: string): Classification {
  const m = LAYER_RE.exec(name);
  if (m) {
    const low = m[3].toLowerCase();
    const component = COMPONENT_RULES.find(([, rx]) => rx.test(low))?.[0] ?? "other";
    let role: string | null = component === "router" ? "router" : ROLE_RULES.find(([, rx]) => rx.test(low))?.[0] ?? null;
    if (component === "experts" && role === null) role = "experts";
    if (role === null && low.endsWith("bias")) role = "bias";
    return { stack: m[1], layer: Number(m[2]), component, role };
  }
  const low = name.toLowerCase();
  for (const [component, role, rx] of GLOBAL_RULES) if (rx.test(low)) return { stack: null, layer: null, component, role };
  return { stack: null, layer: null, component: "other", role: null };
}

const ORDER = ["embedding", "norm", "attention", "router", "experts", "ffn", "ssm", "output", "other"];

function componentSummaries(tensors: TensorRecord[]): ComponentSummary[] {
  const groups = new Map<string, TensorRecord[]>();
  for (const t of tensors) {
    if (!groups.has(t.component)) groups.set(t.component, []);
    groups.get(t.component)!.push(t);
  }
  const idx = (c: string) => (ORDER.includes(c) ? ORDER.indexOf(c) : 99);
  return [...groups.keys()]
    .sort((a, b) => idx(a) - idx(b))
    .map((comp) => {
      const ts = groups.get(comp)!;
      return {
        component: comp,
        roles: [...new Set(ts.map((t) => t.role).filter((r): r is string => !!r))].sort(),
        tensors: ts.map((t) => t.name),
        n_elements: ts.reduce((a, t) => a + t.n_elements, 0),
        n_bytes: ts.reduce((a, t) => a + (t.n_bytes ?? 0), 0),
      };
    });
}

export function buildArchitecture(tensors: TensorRecord[], family: string | null, familyProv: Provenance | null): ArchitectureSummary {
  const byStack = new Map<string, Map<number, TensorRecord[]>>();
  const globals: TensorRecord[] = [];
  for (const t of tensors) {
    if (t.stack !== null && t.layer !== null) {
      if (!byStack.has(t.stack)) byStack.set(t.stack, new Map());
      const m = byStack.get(t.stack)!;
      if (!m.has(t.layer)) m.set(t.layer, []);
      m.get(t.layer)!.push(t);
    } else globals.push(t);
  }
  const stacks: StackSummary[] = [];
  for (const [prefix, layers] of byStack) {
    const summaries: LayerSummary[] = [...layers.keys()]
      .sort((a, b) => a - b)
      .map((i) => {
        const ts = layers.get(i)!;
        const dtypes: Record<string, number> = {};
        for (const t of ts) dtypes[t.dtype] = (dtypes[t.dtype] ?? 0) + 1;
        return {
          index: i,
          components: componentSummaries(ts),
          n_elements: ts.reduce((a, t) => a + t.n_elements, 0),
          n_bytes: ts.reduce((a, t) => a + (t.n_bytes ?? 0), 0),
          dtypes,
        };
      });
    stacks.push({ prefix, n_layers: summaries.length, layers: summaries, n_bytes: summaries.reduce((a, s) => a + s.n_bytes, 0), primary: false });
  }
  stacks.sort((a, b) => b.n_bytes - a.n_bytes || b.n_layers - a.n_layers);
  const notes: string[] = [];
  let uniform = true;
  if (stacks.length) {
    stacks[0].primary = true;
    const p = stacks[0];
    const idxs = p.layers.map((l) => l.index);
    if (idxs.some((v, i) => v !== i)) notes.push(`layer indices in '${p.prefix}' are not contiguous from 0 (pruned or partial model?)`);
    const sigs = new Set(p.layers.map((l) => JSON.stringify(l.components.map((c) => [c.component, c.roles]).sort())));
    uniform = sigs.size === 1;
    if (!uniform) notes.push(`layers in '${p.prefix}' do not all share the same component set (interleaved attention types, MoE/dense mixing, or hybrid architecture)`);
    for (const s of stacks.slice(1)) notes.push(`additional layer stack '${s.prefix}' with ${s.n_layers} layers (e.g. vision encoder or MTP head)`);
  } else notes.push("no repeating layer pattern recognised in tensor names; architecture view is limited to a flat tensor list");
  const otherLayer = tensors.filter((t) => t.layer !== null && t.component === "other").length;
  if (otherLayer) notes.push(`${otherLayer} in-layer tensor(s) did not match a known component pattern and are shown as 'other'`);
  return { family, family_provenance: familyProv, stacks, globals: componentSummaries(globals), uniform_layers: uniform, notes };
}
