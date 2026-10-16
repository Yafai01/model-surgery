"""Reconstruct a layer/component view from tensor names.

Tensor names, shapes and dtypes are verified data; the *grouping* into layers
and the attention/FFN/norm roles are inferred from naming conventions used by
llama.cpp (``blk.N.attn_q``) and Hugging Face (``model.layers.N.self_attn.q_proj``).
Anything that does not match a known pattern is reported as ``other`` rather
than guessed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .schemas import ArchitectureSummary, ComponentSummary, LayerSummary, StackSummary, TensorRecord

# Layer index patterns. Group 1 = stack prefix, group 2 = index, group 3 = remainder.
_LAYER_RE = re.compile(r"^(.*?(?:^|\.)(?:blk|layers|layer|h|blocks|block))\.(\d+)\.(.+)$")

_COMPONENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    # Order matters: the first match wins.
    ("norm", re.compile(r"(norm|layernorm|(^|[._])ln(_?\d+|_f|_attn|_mlp)?([._]|$))")),
    ("router", re.compile(r"(ffn_gate_inp|router|block_sparse_moe\.gate\.|mlp\.gate\.(weight|bias)$|(^|\.)gate\.(weight|bias)$|exp_probs)")),
    ("experts", re.compile(r"(_exps\b|_exps\.|(?<!shared_)experts\.|\.experts$)")),
    # Attention/FFN projections inside a generic "mixer." block (NemotronH) are not SSM tensors.
    ("ssm", re.compile(r"(ssm_|mamba\.|mixer\.(?!(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)\b)|time_mix|channel_mix|conv1d|\bx_proj|dt_proj|a_log)")),
    ("attention", re.compile(r"(attn|attention|self_attn|q_proj|k_proj|v_proj|o_proj|\bquery\b|\bkey\b|\bvalue\b|qkv|\bwq\b|\bwk\b|\bwv\b|\bwo\b)")),
    ("ffn", re.compile(r"(mlp|ffn|feed_forward|gate_proj|up_proj|down_proj|\bfc\d\b|c_fc|\bw[123]\b|dense_h_to_4h|dense_4h_to_h|shexp|shared_expert)")),
]

_ROLE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("qkv", re.compile(r"(attn_qkv|qkv_proj|c_attn|query_key_value|wqkv|\.qkv\.)")),
    ("q_norm", re.compile(r"(attn_q_norm|q_norm|q_layernorm)")),
    ("k_norm", re.compile(r"(attn_k_norm|k_norm|k_layernorm)")),
    ("q", re.compile(r"(attn_q\b|attn_q\.|q_proj|\.query\.|\bwq\b)")),
    ("k", re.compile(r"(attn_k\b|attn_k\.|k_proj|\.key\.|\bwk\b)")),
    ("v", re.compile(r"(attn_v\b|attn_v\.|v_proj|\.value\.|\bwv\b)")),
    ("o", re.compile(r"(attn_output|attn_out\b|o_proj|out_proj|attn\.c_proj|attention\.dense|self_attention\.dense|\bwo\b)")),
    ("gate_up", re.compile(r"(gate_up_proj|ffn_gate_up)")),
    ("gate", re.compile(r"(ffn_gate|gate_proj|\bw1\b)")),
    ("up", re.compile(r"(ffn_up|up_proj|\bw3\b|c_fc|\bfc1\b|dense_h_to_4h)")),
    ("down", re.compile(r"(ffn_down|down_proj|\bw2\b|mlp\.c_proj|\bfc2\b|dense_4h_to_h)")),
    ("post_attn_norm", re.compile(r"post_attention_norm")),
    ("post_ffn_norm", re.compile(r"(post_ffw_norm|post_feedforward_layernorm)")),
    ("attn_norm", re.compile(r"(attn_norm|input_layernorm|ln_1|ln_attn|attention_norm)")),
    ("ffn_norm", re.compile(r"(ffn_norm|pre_feedforward_layernorm|post_attention_layernorm|ln_2|ln_mlp)")),
]

_GLOBAL_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    ("embedding", "position", re.compile(r"(wpe|position_embd|embed_positions|position_embeddings)")),
    ("embedding", "token", re.compile(r"(token_embd|embed_tokens|\bwte\b|tok_embeddings|word_embeddings|embed_in)\b")),
    ("norm", "final", re.compile(r"(output_norm|ln_f|final_layernorm|final_layer_norm|(^|\.)norm\.(weight|bias)$)")),
    ("output", "lm_head", re.compile(r"(^output\.|lm_head|embed_out)")),
    ("other", "rope", re.compile(r"rope")),
]


@dataclass
class Classification:
    stack: str | None
    layer: int | None
    component: str
    role: str | None


def classify(name: str) -> Classification:
    m = _LAYER_RE.match(name)
    if m:
        stack, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        low = rest.lower()
        component = next((c for c, rx in _COMPONENT_RULES if rx.search(low)), "other")
        role = "router" if component == "router" else next((r for r, rx in _ROLE_RULES if rx.search(low)), None)
        if component == "experts" and role is None:
            role = "experts"
        if role is None and low.endswith("bias"):
            role = "bias"
        return Classification(stack, idx, component, role)
    low = name.lower()
    for component, role, rx in _GLOBAL_RULES:
        if rx.search(low):
            return Classification(None, None, component, role)
    return Classification(None, None, "other", None)


def _component_summaries(tensors: list[TensorRecord]) -> list[ComponentSummary]:
    groups: dict[str, list[TensorRecord]] = defaultdict(list)
    for t in tensors:
        groups[t.component].append(t)
    order = ["embedding", "norm", "attention", "router", "experts", "ffn", "ssm", "output", "other"]
    out = []
    for comp in sorted(groups, key=lambda c: order.index(c) if c in order else 99):
        ts = groups[comp]
        out.append(
            ComponentSummary(
                component=comp,
                roles=sorted({t.role for t in ts if t.role}),
                tensors=[t.name for t in ts],
                n_elements=sum(t.n_elements for t in ts),
                n_bytes=sum(t.n_bytes or 0 for t in ts),
            )
        )
    return out


def build_architecture(tensors: list[TensorRecord], family: str | None, family_provenance) -> ArchitectureSummary:
    by_stack: dict[str, dict[int, list[TensorRecord]]] = defaultdict(lambda: defaultdict(list))
    globals_: list[TensorRecord] = []
    for t in tensors:
        if t.stack is not None and t.layer is not None:
            by_stack[t.stack][t.layer].append(t)
        else:
            globals_.append(t)

    stacks: list[StackSummary] = []
    for prefix, layers in by_stack.items():
        summaries = []
        for idx in sorted(layers):
            ts = layers[idx]
            summaries.append(
                LayerSummary(
                    index=idx,
                    components=_component_summaries(ts),
                    n_elements=sum(t.n_elements for t in ts),
                    n_bytes=sum(t.n_bytes or 0 for t in ts),
                    dtypes=dict(Counter(t.dtype for t in ts)),
                )
            )
        stacks.append(
            StackSummary(prefix=prefix, n_layers=len(summaries), layers=summaries, n_bytes=sum(s.n_bytes for s in summaries), primary=False)
        )
    stacks.sort(key=lambda s: (s.n_bytes, s.n_layers), reverse=True)
    notes: list[str] = []
    uniform = True
    if stacks:
        stacks[0].primary = True
        primary = stacks[0]
        idxs = [l.index for l in primary.layers]
        if idxs != list(range(len(idxs))):
            notes.append(f"layer indices in '{primary.prefix}' are not contiguous from 0 (pruned or partial model?)")
        sigs = {tuple(sorted((c.component, tuple(c.roles)) for c in l.components)) for l in primary.layers}
        uniform = len(sigs) == 1
        if not uniform:
            notes.append(
                f"layers in '{primary.prefix}' do not all share the same component set "
                "(interleaved attention types, MoE/dense mixing, or hybrid architecture)"
            )
        for s in stacks[1:]:
            notes.append(f"additional layer stack '{s.prefix}' with {s.n_layers} layers (e.g. vision encoder or MTP head)")
    else:
        notes.append("no repeating layer pattern recognised in tensor names; architecture view is limited to a flat tensor list")

    other_layer = sum(1 for t in tensors if t.layer is not None and t.component == "other")
    if other_layer:
        notes.append(f"{other_layer} in-layer tensor(s) did not match a known component pattern and are shown as 'other'")

    return ArchitectureSummary(
        family=family,
        family_provenance=family_provenance,
        stacks=stacks,
        globals=_component_summaries(globals_),
        uniform_layers=uniform,
        notes=notes,
    )
