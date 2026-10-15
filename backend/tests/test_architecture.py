import pytest

from model_surgery.architecture import classify


@pytest.mark.parametrize(
    "name,layer,component,role",
    [
        ("blk.0.attn_q.weight", 0, "attention", "q"),
        ("blk.3.attn_k.bias", 3, "attention", "k"),
        ("blk.3.attn_qkv.weight", 3, "attention", "qkv"),
        ("blk.1.attn_output.weight", 1, "attention", "o"),
        ("blk.1.attn_norm.weight", 1, "norm", "attn_norm"),
        ("blk.1.attn_q_norm.weight", 1, "norm", "q_norm"),
        ("blk.1.ffn_norm.weight", 1, "norm", "ffn_norm"),
        ("blk.1.post_attention_norm.weight", 1, "norm", "post_attn_norm"),
        ("blk.2.ffn_gate.weight", 2, "ffn", "gate"),
        ("blk.2.ffn_down.weight", 2, "ffn", "down"),
        ("blk.2.ffn_gate_inp.weight", 2, "router", "router"),
        ("blk.2.ffn_up_exps.weight", 2, "experts", "up"),
        ("blk.2.ffn_down_shexp.weight", 2, "ffn", "down"),
        ("blk.5.ssm_in.weight", 5, "ssm", None),
        ("model.layers.10.self_attn.q_proj.weight", 10, "attention", "q"),
        ("model.layers.10.self_attn.o_proj.weight", 10, "attention", "o"),
        ("model.layers.10.mlp.gate_proj.weight", 10, "ffn", "gate"),
        ("model.layers.10.input_layernorm.weight", 10, "norm", "attn_norm"),
        ("model.layers.10.post_attention_layernorm.weight", 10, "norm", "ffn_norm"),
        ("model.layers.0.mlp.gate.weight", 0, "router", "router"),
        ("model.layers.0.mlp.experts.7.down_proj.weight", 0, "experts", "down"),
        ("model.layers.0.block_sparse_moe.gate.weight", 0, "router", "router"),
        ("transformer.h.4.attn.c_attn.weight", 4, "attention", "qkv"),
        ("transformer.h.4.attn.c_proj.weight", 4, "attention", "o"),
        ("transformer.h.4.mlp.c_fc.weight", 4, "ffn", "up"),
        ("transformer.h.4.mlp.c_proj.weight", 4, "ffn", "down"),
        ("transformer.h.4.ln_1.weight", 4, "norm", "attn_norm"),
        ("model.layers.2.weird_thing.weight", 2, "other", None),
    ],
)
def test_layer_tensors(name, layer, component, role):
    c = classify(name)
    assert (c.layer, c.component, c.role) == (layer, component, role)


@pytest.mark.parametrize(
    "name,component,role",
    [
        ("token_embd.weight", "embedding", "token"),
        ("model.embed_tokens.weight", "embedding", "token"),
        ("transformer.wte.weight", "embedding", "token"),
        ("transformer.wpe.weight", "embedding", "position"),
        ("output.weight", "output", "lm_head"),
        ("lm_head.weight", "output", "lm_head"),
        ("output_norm.weight", "norm", "final"),
        ("model.norm.weight", "norm", "final"),
        ("transformer.ln_f.weight", "norm", "final"),
        ("rope_freqs.weight", "other", "rope"),
        ("mystery", "other", None),
    ],
)
def test_global_tensors(name, component, role):
    c = classify(name)
    assert c.layer is None and (c.component, c.role) == (component, role)


def test_multimodal_stacks_are_separate():
    a = classify("language_model.model.layers.3.self_attn.q_proj.weight")
    v = classify("vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.weight")
    assert a.stack != v.stack and a.layer == v.layer == 3
