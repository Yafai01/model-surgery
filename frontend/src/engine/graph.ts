// Normalized architecture graph built from a scan.
//
// Every node says where it comes from:
//   - nodes backed by tensors list them; tensor names/shapes are verified, the role is inferred from names
//   - residual adds, the input and the logits have no tensors in the file: they are `illustrative`
//   - dimensions carry the provenance of the hyperparameter they were read from
// Anything that cannot be placed in the dataflow (unclassified tensors, unknown layouts) is listed in
// `unrepresented` instead of being forced into a standard transformer block.

import type { ComponentSummary, LayerSummary, Provenance, ScanResult, StackSummary, TensorRecord } from "../types";

export type NodeKind =
  | "input"
  | "embedding"
  | "norm"
  | "attention"
  | "ssm"
  | "router"
  | "experts"
  | "ffn"
  | "residual_add"
  | "output"
  | "logits"
  | "other";

export interface Dim {
  label: string;
  value: string;
  provenance: Provenance;
}

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  layer: number | null;
  tensors: string[];
  n_elements: number;
  n_bytes: number;
  dims: Dim[];
  provenance: Provenance;
  illustrative: boolean;
}

export interface GraphEdge {
  from: string;
  to: string;
  kind: "data" | "residual";
}

export type LayerLayout = "sequential_prenorm" | "parallel" | "ssm" | "unknown";

export interface LayerGraph {
  index: number;
  layout: LayerLayout;
  nodes: GraphNode[];
  edges: GraphEdge[];
  notes: string[];
}

export interface ArchitectureGraph {
  schema: "model-surgery/architecture-graph@1";
  model: string;
  family: string | null;
  stack: string | null;
  n_layers: number;
  head: GraphNode[];
  layers: LayerGraph[];
  tail: GraphNode[];
  unrepresented: string[];
  notes: string[];
}

type Facts = ScanResult["hyperparameters"];

function fact(facts: Facts, key: string) {
  return facts.find((f) => f.key === key) ?? null;
}

function dim(facts: Facts, key: string, label: string): Dim | null {
  const f = fact(facts, key);
  if (!f || f.value === null || f.value === undefined) return null;
  return { label, value: typeof f.value === "number" ? f.value.toLocaleString("en-US") : String(f.value), provenance: f.provenance };
}

const dims = (...d: (Dim | null)[]) => d.filter((x): x is Dim => x !== null);

function tensorNode(id: string, kind: NodeKind, label: string, layer: number | null, tensors: TensorRecord[], d: Dim[] = []): GraphNode {
  const shapes: Dim[] = tensors.slice(0, 6).map((t) => ({ label: t.role ?? t.name.split(".").slice(-2).join("."), value: `[${t.shape.join(", ")}] ${t.dtype}`, provenance: "verified" }));
  return {
    id,
    kind,
    label,
    layer,
    tensors: tensors.map((t) => t.name),
    n_elements: tensors.reduce((a, t) => a + t.n_elements, 0),
    n_bytes: tensors.reduce((a, t) => a + (t.n_bytes ?? 0), 0),
    dims: [...d, ...shapes],
    provenance: "inferred", // role from naming conventions; the tensors themselves are verified
    illustrative: false,
  };
}

function virtualNode(id: string, kind: NodeKind, label: string, layer: number | null, d: Dim[] = []): GraphNode {
  return { id, kind, label, layer, tensors: [], n_elements: 0, n_bytes: 0, dims: d, provenance: "inferred", illustrative: true };
}

function layerGraph(layer: LayerSummary, facts: Facts, byName: Map<string, TensorRecord>): { graph: LayerGraph; unplaced: string[] } {
  const L = layer.index;
  const comp = new Map<string, ComponentSummary>(layer.components.map((c) => [c.component, c]));
  const ts = (c: ComponentSummary | undefined, pred: (t: TensorRecord) => boolean = () => true) =>
    (c?.tensors ?? []).map((n) => byName.get(n)).filter((t): t is TensorRecord => !!t && pred(t));
  const normsBy = (role: string) => ts(comp.get("norm"), (t) => t.role === role);
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const notes: string[] = [];
  const unplaced: string[] = [];
  const id = (s: string) => `L${L}.${s}`;
  const add = (n: GraphNode) => (nodes.push(n), n.id);
  const link = (from: string, to: string, kind: GraphEdge["kind"] = "data") => edges.push({ from, to, kind });

  const attnNorm = normsBy("attn_norm");
  const ffnNorm = normsBy("ffn_norm");
  const postAttn = normsBy("post_attn_norm");
  const postFfn = normsBy("post_ffn_norm");
  const qkNorm = ts(comp.get("norm"), (t) => t.role === "q_norm" || t.role === "k_norm");
  const placedNorms = new Set([...attnNorm, ...ffnNorm, ...postAttn, ...postFfn, ...qkNorm].map((t) => t.name));
  const otherNorms = ts(comp.get("norm"), (t) => !placedNorms.has(t.name));

  const attnT = [...ts(comp.get("attention")), ...qkNorm];
  const ssmT = ts(comp.get("ssm"));
  const ffnT = ts(comp.get("ffn"));
  const routerT = ts(comp.get("router"));
  const expertsT = ts(comp.get("experts"));
  const hasMixer = attnT.length > 0 || ssmT.length > 0;
  const hasFfn = ffnT.length > 0 || expertsT.length > 0;

  const attnDims = dims(
    dim(facts, "n_heads", "query heads"),
    dim(facts, "n_kv_heads", "KV heads"),
    dim(facts, "head_dim", "head dim (K)"),
    dim(facts, "head_dim_v", "head dim (V)"),
    dim(facts, "sliding_window", "sliding window"),
  );
  const ffnDims = dims(dim(facts, "ffn_size", "FFN width"));
  const moeDims = dims(dim(facts, "n_experts", "experts"), dim(facts, "n_experts_used", "experts per token"));

  let layout: LayerLayout;
  if (attnT.length && hasFfn && attnNorm.length && !ffnNorm.length && !postAttn.length) layout = "parallel";
  else if (!attnT.length && ssmT.length) layout = "ssm";
  else if (hasMixer && (attnNorm.length || ffnNorm.length || !hasFfn)) layout = "sequential_prenorm";
  else layout = "unknown";

  const input = add(virtualNode(id("in"), "input", "hidden state in", L, dims(dim(facts, "hidden_size", "hidden size"))));

  const mixer = (): string | null => {
    if (attnT.length) return add(tensorNode(id("attn"), "attention", ssmT.length ? "Attention (+SSM)" : "Self-attention", L, [...attnT, ...ssmT], attnDims));
    if (ssmT.length) return add(tensorNode(id("ssm"), "ssm", "SSM / recurrent mixer", L, ssmT, dims(dim(facts, "ssm_state_size", "state size"))));
    return null;
  };
  const feedForward = (from: string): string => {
    if (expertsT.length) {
      const r = routerT.length ? add(tensorNode(id("router"), "router", "Router", L, routerT, moeDims)) : null;
      const ex = add(tensorNode(id("experts"), "experts", "Routed experts", L, expertsT, [...moeDims, ...ffnDims]));
      if (r) {
        link(from, r);
        link(r, ex);
      }
      link(from, ex);
      if (ffnT.length) {
        const sh = add(tensorNode(id("shared_ffn"), "ffn", "Shared expert FFN", L, ffnT, ffnDims));
        link(from, sh);
        const merge = add(virtualNode(id("moe_sum"), "residual_add", "sum", L));
        link(ex, merge);
        link(sh, merge);
        return merge;
      }
      return ex;
    }
    const f = add(tensorNode(id("ffn"), "ffn", "Feed-forward (MLP)", L, ffnT, ffnDims));
    link(from, f);
    return f;
  };

  if (layout === "parallel") {
    const n = add(tensorNode(id("norm"), "norm", "Norm", L, attnNorm));
    link(input, n);
    const a = mixer()!;
    link(n, a);
    const f = feedForward(n);
    const sum = add(virtualNode(id("add"), "residual_add", "+ residual", L));
    link(a, sum);
    link(f, sum);
    link(input, sum, "residual");
    link(sum, add(virtualNode(id("out"), "input", "hidden state out", L)));
    notes.push("parallel layout inferred: one norm feeds both attention and the MLP (GPT-J / Falcon style)");
  } else if (layout === "sequential_prenorm" || layout === "ssm") {
    let cur = input;
    if (hasMixer) {
      let x = cur;
      if (attnNorm.length) {
        x = add(tensorNode(id("attn_norm"), "norm", "Pre-norm", L, attnNorm));
        link(cur, x);
      }
      const m = mixer()!;
      link(x, m);
      let y = m;
      if (postAttn.length) {
        const pn = add(tensorNode(id("post_attn_norm"), "norm", "Post-norm", L, postAttn));
        link(y, pn);
        y = pn;
      }
      const add1 = add(virtualNode(id("add1"), "residual_add", "+ residual", L));
      link(y, add1);
      link(cur, add1, "residual");
      cur = add1;
    }
    if (hasFfn) {
      let x = cur;
      if (ffnNorm.length) {
        const fnode = add(tensorNode(id("ffn_norm"), "norm", "Pre-norm", L, ffnNorm));
        link(cur, fnode);
        x = fnode;
      }
      const f = feedForward(x);
      let y = f;
      if (postFfn.length) {
        const pn = add(tensorNode(id("post_ffn_norm"), "norm", "Post-norm", L, postFfn));
        link(y, pn);
        y = pn;
      }
      const add2 = add(virtualNode(id("add2"), "residual_add", "+ residual", L));
      link(y, add2);
      link(cur, add2, "residual");
      cur = add2;
    }
    const out = add(virtualNode(id("out"), "input", "hidden state out", L));
    link(cur, out);
  } else {
    notes.push("layer layout not recognised: components are listed without connections");
    for (const c of layer.components) {
      if (c.component === "other") continue;
      add(tensorNode(id(c.component), c.component as NodeKind, c.component, L, ts(c)));
    }
  }
  if (otherNorms.length) unplaced.push(...otherNorms.map((t) => t.name));
  for (const t of ts(comp.get("other"))) unplaced.push(t.name);
  if (unplaced.length) notes.push(`${unplaced.length} tensor(s) in this layer could not be placed in the dataflow and are listed separately`);
  return { graph: { index: L, layout, nodes, edges, notes }, unplaced };
}

export function buildArchitectureGraph(scan: ScanResult, stack?: StackSummary): ArchitectureGraph {
  const facts = scan.hyperparameters;
  const byName = new Map(scan.tensors.map((t) => [t.name, t]));
  const st = stack ?? scan.architecture.stacks.find((s) => s.primary) ?? null;
  const globalsBy = (component: string, pred: (t: TensorRecord) => boolean = () => true) =>
    (scan.architecture.globals.find((g) => g.component === component)?.tensors ?? []).map((n) => byName.get(n)).filter((t): t is TensorRecord => !!t && pred(t));

  const unrepresented: string[] = [];
  const notes: string[] = [];
  const head: GraphNode[] = [virtualNode("tokens", "input", "token ids", null, dims(dim(facts, "context_length", "training context")))];
  const tokEmb = globalsBy("embedding");
  if (tokEmb.length) head.push(tensorNode("embedding", "embedding", "Token embedding", null, tokEmb, dims(dim(facts, "vocab_size", "vocabulary"), dim(facts, "hidden_size", "hidden size"))));
  else notes.push("no embedding tensor recognised");

  const layers: LayerGraph[] = [];
  for (const l of st?.layers ?? []) {
    const { graph, unplaced } = layerGraph(l, facts, byName);
    layers.push(graph);
    unrepresented.push(...unplaced);
  }
  for (const s of scan.architecture.stacks) if (s !== st) notes.push(`stack '${s.prefix}' (${s.n_layers} layers) is not part of this graph; select it to view it`);

  const tail: GraphNode[] = [];
  const finalNorm = globalsBy("norm");
  if (finalNorm.length) tail.push(tensorNode("final_norm", "norm", "Final norm", null, finalNorm));
  const lmHead = globalsBy("output");
  const tied = fact(facts, "tied_embeddings");
  if (lmHead.length) tail.push(tensorNode("lm_head", "output", "Output head (lm_head)", null, lmHead, dims(dim(facts, "vocab_size", "vocabulary"))));
  else if (tokEmb.length)
    tail.push({ ...virtualNode("lm_head", "output", "Output head (tied to embedding)", null, dims(dim(facts, "vocab_size", "vocabulary"))), provenance: tied?.provenance ?? "inferred" });
  tail.push(virtualNode("logits", "logits", "logits", null));
  unrepresented.push(...globalsBy("other").map((t) => t.name));

  return {
    schema: "model-surgery/architecture-graph@1",
    model: scan.name,
    family: scan.architecture.family,
    stack: st?.prefix ?? null,
    n_layers: layers.length,
    head,
    layers,
    tail,
    unrepresented,
    notes,
  };
}
