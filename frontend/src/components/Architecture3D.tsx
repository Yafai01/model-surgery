import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { componentColor, fmtBytes } from "../format";
import { type RenderQuality, cssVar } from "../prefs";
import type { ComponentSummary, StackSummary } from "../types";
import type { Selection } from "./ArchitecturePage";

interface Props {
  stack: StackSummary;
  globals: ComponentSummary[];
  selection: Selection;
  onSelect: (s: Selection) => void;
  /** Layers matching the current search; null = no search active. */
  matches: Set<number> | null;
  /** Components to show; others are drawn as faint ghosts. */
  visible: Set<string>;
  quality: RenderQuality;
  showLabels: boolean;
}

interface Block {
  label: string;
  component: string;
  bytes: number;
  sel: Exclude<Selection, null>;
  layer: number | null;
  color: THREE.Color;
}

interface Row {
  y: number;
  h: number;
  w: number;
  label: string;
  sel: Exclude<Selection, null>;
}

const ORDER = ["norm", "attention", "ssm", "router", "experts", "ffn", "other"];
const SLAB_H = 0.42;
const GAP = 0.16;
const DEPTH = 2.2;
const MAX_W = 7;
const MIN_W = 0.12;
const QUALITY: Record<RenderQuality, { aa: boolean; dpr: number; edges: boolean }> = {
  high: { aa: true, dpr: 2, edges: true },
  balanced: { aa: true, dpr: 1.5, edges: true },
  low: { aa: false, dpr: 1, edges: false },
};

/** Schematic 3D stack: one slab per layer, block width ∝ bytes. Rendered on demand with a single instanced mesh. */
export default function Architecture3D({ stack, globals, selection, onSelect, matches, visible, quality, showLabels }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;
  const api = useRef<{ fit: (sel?: Selection) => void; reset: () => void; restyle: () => void; step: (d: number) => void } | null>(null);
  const styleRef = useRef({ selection, matches, visible, showLabels });
  styleRef.current = { selection, matches, visible, showLabels };
  const [lost, setLost] = useState(false);
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const el = host.current!;
    const q = QUALITY[quality];
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: q.aa, alpha: true, powerPreference: "low-power" });
    } catch {
      setLost(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, q.dpr));
    renderer.setSize(el.clientWidth || 1, el.clientHeight || 1);
    el.prepend(renderer.domElement);
    const canvas = renderer.domElement;
    canvas.setAttribute("aria-hidden", "true");

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(6, 10, 8);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-8, 4, -6);
    scene.add(fill);

    // ---- layout ----------------------------------------------------------------
    const maxBytes = Math.max(1, ...stack.layers.map((l) => l.n_bytes), ...globals.map((g) => g.n_bytes));
    const blocks: Block[] = [];
    const rows: Row[] = [];
    const matrices: THREE.Matrix4[] = [];
    const m4 = new THREE.Matrix4();
    const addRow = (comps: ComponentSummary[], y: number, sel: Exclude<Selection, null>, label: string, h = SLAB_H) => {
      const rowBytes = comps.reduce((a, c) => a + c.n_bytes, 0);
      const rowW = Math.max((rowBytes / maxBytes) * MAX_W, 0.6);
      const widths = comps.map((c) => Math.max((c.n_bytes / Math.max(rowBytes, 1)) * rowW, MIN_W));
      const total = widths.reduce((a, b) => a + b, 0);
      let x = -total / 2;
      comps.forEach((c, i) => {
        const w = widths[i];
        m4.makeScale(w * 0.97, h, DEPTH).setPosition(x + w / 2, y, 0);
        matrices.push(m4.clone());
        blocks.push({ label: `${label} · ${c.component} (${c.tensors.length} tensors)`, component: c.component, bytes: c.n_bytes, sel, layer: sel.kind === "layer" ? sel.index : null, color: new THREE.Color(componentColor(c.component)) });
        x += w;
      });
      rows.push({ y, h, w: total, label, sel });
    };
    const sorted = (comps: ComponentSummary[]) => [...comps].sort((a, b) => ORDER.indexOf(a.component) - ORDER.indexOf(b.component));
    let y = 0;
    const emb = globals.filter((g) => g.component === "embedding");
    if (emb.length) {
      addRow(emb, y, { kind: "global", component: "embedding" }, "embedding", SLAB_H * 1.3);
      y += SLAB_H * 1.3 + GAP * 2;
    }
    for (const l of stack.layers) {
      addRow(sorted(l.components), y, { kind: "layer", index: l.index }, `L${l.index}`);
      y += SLAB_H + GAP;
    }
    y += GAP;
    for (const comp of ["norm", "output", "other"]) {
      const g = globals.filter((x) => x.component === comp);
      if (!g.length) continue;
      const h = comp === "output" ? SLAB_H * 1.3 : SLAB_H * 0.6;
      addRow(g, y, { kind: "global", component: comp }, comp, h);
      y += h + GAP * 2;
    }
    const totalH = y;

    // ---- geometry: one instanced mesh for every block, one merged edge geometry -----
    const unit = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.05 });
    const mesh = new THREE.InstancedMesh(unit, material, Math.max(blocks.length, 1));
    matrices.forEach((m, i) => mesh.setMatrixAt(i, m));
    mesh.count = blocks.length;
    blocks.forEach((b, i) => mesh.setColorAt(i, b.color));
    mesh.instanceMatrix.needsUpdate = true;
    scene.add(mesh);

    const disposables: { dispose(): void }[] = [unit, material];
    if (q.edges && blocks.length) {
      const unitEdges = new THREE.EdgesGeometry(unit).getAttribute("position").array as Float32Array;
      const pos = new Float32Array(unitEdges.length * blocks.length);
      const v = new THREE.Vector3();
      matrices.forEach((m, i) => {
        for (let j = 0; j < unitEdges.length; j += 3) {
          v.set(unitEdges[j], unitEdges[j + 1], unitEdges[j + 2]).applyMatrix4(m);
          pos.set([v.x, v.y, v.z], i * unitEdges.length + j);
        }
      });
      const eg = new THREE.BufferGeometry();
      eg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      const em = new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar("--scene-edge") || "#ffffff"), transparent: true, opacity: 0.55 });
      scene.add(new THREE.LineSegments(eg, em));
      disposables.push(eg, em);
    }
    const frameGeo = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    const frameMat = new THREE.LineBasicMaterial({ color: new THREE.Color(cssVar("--accent") || "#0071e3") });
    const frame = new THREE.LineSegments(frameGeo, frameMat);
    frame.visible = false;
    scene.add(frame);
    const grid = new THREE.GridHelper(Math.max(MAX_W * 2, 12), 12, new THREE.Color(cssVar("--scene-grid") || "#e3e3e8"), new THREE.Color(cssVar("--scene-grid") || "#e3e3e8"));
    grid.position.y = -0.6;
    scene.add(grid);
    disposables.push(frameGeo, frameMat, grid.geometry, grid.material as THREE.Material);

    // ---- camera ----------------------------------------------------------------
    const camera = new THREE.PerspectiveCamera(40, (el.clientWidth || 1) / (el.clientHeight || 1), 0.1, 5000);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    controls.screenSpacePanning = true;
    const dir = new THREE.Vector3(0.55, 0.25, 0.8).normalize();

    const frameBox = (minY: number, maxY: number, halfW: number) => {
      // Fit a box of the given extent using both the vertical and the horizontal field of view.
      const center = new THREE.Vector3(0, (minY + maxY) / 2, 0);
      const radius = new THREE.Vector3(halfW, (maxY - minY) / 2, DEPTH / 2).length();
      const vfov = THREE.MathUtils.degToRad(camera.fov);
      const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
      const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.05;
      camera.position.copy(center).addScaledVector(dir, dist);
      camera.near = Math.max(0.01, dist / 500);
      camera.far = dist * 20;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
      requestRender();
    };
    const fitAll = () => frameBox(-SLAB_H, totalH, MAX_W / 2);
    const rowFor = (s: Selection) => rows.find((r) => s && (s.kind === "layer" ? r.sel.kind === "layer" && r.sel.index === s.index : r.sel.kind === "global" && r.sel.component === s.component));

    // ---- styling (selection, search matches, isolation) ---------------------------
    const tmp = new THREE.Color();
    const ghost = new THREE.Color(cssVar("--scene-grid") || "#e3e3e8");
    const restyle = () => {
      const { selection: s, matches: mt, visible: vis } = styleRef.current;
      blocks.forEach((b, i) => {
        const hidden = !vis.has(b.component);
        const unmatched = mt !== null && (b.layer === null || !mt.has(b.layer));
        tmp.copy(b.color);
        if (hidden) tmp.copy(ghost);
        else if (unmatched) tmp.lerp(ghost, 0.75);
        mesh.setColorAt(i, tmp);
      });
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      const r = rowFor(s);
      frame.visible = !!r;
      if (r) {
        frame.scale.set(r.w + 0.12, r.h + 0.1, DEPTH + 0.12);
        frame.position.set(0, r.y, 0);
      }
      requestRender();
    };

    // ---- labels & tooltip (imperative: no React re-render per frame) --------------
    const labelHost = labelsRef.current!;
    labelHost.replaceChildren();
    const every = Math.max(1, Math.ceil(stack.layers.length / 14));
    const labelled = rows.filter((r) => r.sel.kind === "global" || (r.sel.kind === "layer" && (r.sel.index % every === 0 || r.sel.index === stack.layers[stack.layers.length - 1]?.index)));
    const labelEls = labelled.map((r) => {
      const d = document.createElement("div");
      d.className = "label-3d";
      d.textContent = r.label;
      labelHost.appendChild(d);
      return { r, d };
    });
    const proj = new THREE.Vector3();
    const placeLabels = () => {
      const on = styleRef.current.showLabels;
      labelHost.style.display = on ? "" : "none";
      if (!on) return;
      const w = el.clientWidth;
      const h = el.clientHeight;
      for (const { r, d } of labelEls) {
        proj.set(-r.w / 2 - 0.15, r.y, DEPTH / 2).project(camera);
        const visibleNow = proj.z < 1 && Math.abs(proj.x) <= 1.05 && Math.abs(proj.y) <= 1.05;
        d.style.display = visibleNow ? "" : "none";
        if (visibleNow) d.style.transform = `translate(${((proj.x + 1) / 2) * w}px, ${((1 - proj.y) / 2) * h}px) translate(-100%, -50%)`;
      }
    };

    // ---- render on demand ----------------------------------------------------------
    let raf = 0;
    function requestRender() {
      if (!raf) raf = requestAnimationFrame(frameTick);
    }
    function frameTick() {
      raf = 0;
      const moving = controls.update(); // true while damping is still settling
      renderer.render(scene, camera);
      placeLabels();
      if (moving) requestRender();
    }
    controls.addEventListener("change", requestRender);

    // ---- picking ----------------------------------------------------------------
    const ray = new THREE.Raycaster();
    const ptr = new THREE.Vector2();
    const pick = (ev: PointerEvent): number | null => {
      const r = canvas.getBoundingClientRect();
      ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ptr, camera);
      const hit = ray.intersectObject(mesh, false)[0];
      return hit?.instanceId ?? null;
    };
    const tip = tipRef.current!;
    const onMove = (ev: PointerEvent) => {
      const i = pick(ev);
      const r = canvas.getBoundingClientRect();
      if (i !== null) {
        tip.textContent = `${blocks[i].label} — ${fmtBytes(blocks[i].bytes)}`;
        tip.style.display = "block";
        tip.style.left = `${ev.clientX - r.left}px`;
        tip.style.top = `${ev.clientY - r.top}px`;
        canvas.style.cursor = "pointer";
      } else {
        tip.style.display = "none";
        canvas.style.cursor = "grab";
      }
    };
    let downAt = 0;
    let downX = 0;
    let downY = 0;
    const onDown = (ev: PointerEvent) => {
      downAt = performance.now();
      downX = ev.clientX;
      downY = ev.clientY;
    };
    const onUp = (ev: PointerEvent) => {
      if (performance.now() - downAt > 300 || Math.hypot(ev.clientX - downX, ev.clientY - downY) > 5) return; // a drag, not a click
      const i = pick(ev);
      if (i !== null) selectRef.current(blocks[i].sel);
    };
    const onLeave = () => (tip.style.display = "none");
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointerleave", onLeave);

    const onLost = (e: Event) => {
      e.preventDefault();
      cancelAnimationFrame(raf);
      setLost(true);
    };
    canvas.addEventListener("webglcontextlost", onLost);

    const ro = new ResizeObserver(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      requestRender();
    });
    ro.observe(el);

    const layerIdx = stack.layers.map((l) => l.index);
    api.current = {
      fit: (s) => {
        const r = s ? rowFor(s) : undefined;
        if (r) frameBox(r.y - 4 * (SLAB_H + GAP), r.y + 4 * (SLAB_H + GAP), Math.max(r.w / 2, 2));
        else fitAll();
      },
      reset: fitAll,
      restyle,
      step: (d) => {
        const s = styleRef.current.selection;
        const cur = s?.kind === "layer" ? layerIdx.indexOf(s.index) : -1;
        const next = layerIdx[Math.min(layerIdx.length - 1, Math.max(0, cur + d))];
        if (next !== undefined) selectRef.current({ kind: "layer", index: next });
      },
    };

    fitAll();
    restyle();

    return () => {
      api.current = null;
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("webglcontextlost", onLost);
      controls.removeEventListener("change", requestRender);
      controls.dispose();
      mesh.dispose();
      disposables.forEach((d) => d.dispose());
      labelHost.replaceChildren();
      renderer.dispose();
      renderer.forceContextLoss(); // dispose() alone keeps the GL context alive until GC; browsers cap live contexts
      canvas.remove();
    };
  }, [stack, globals, quality, generation]);

  useEffect(() => {
    api.current?.restyle();
  }, [selection, matches, visible, showLabels]);

  const onKey = (e: KeyboardEvent) => {
    const a = api.current;
    if (!a) return;
    const k = e.key.toLowerCase();
    if (k === "f") a.fit(selection);
    else if (k === "r" || k === "home") a.reset();
    else if (k === "arrowdown" || k === "j") a.step(1);
    else if (k === "arrowup" || k === "k") a.step(-1);
    else return;
    e.preventDefault();
  };

  if (lost)
    return (
      <div className="canvas-3d lost" role="alert">
        <p>The 3D view lost its graphics context (GPU reset, driver issue or too many open 3D views).</p>
        <button
          className="btn sm"
          onClick={() => {
            setLost(false);
            setGeneration((g) => g + 1);
          }}
        >
          Restart 3D view
        </button>
      </div>
    );

  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row toolbar-3d">
        <button className="btn sm" onClick={() => api.current?.fit(selection)} title="Frame the selected layer (F)">
          Fit selection
        </button>
        <button className="btn sm" onClick={() => api.current?.reset()} title="Frame the whole model (R)">
          Fit all / reset
        </button>
        <span className="small faint">Drag to orbit · right-drag or Shift-drag to pan · scroll to zoom · ↑/↓ to step through layers</span>
      </div>
      <div
        className="canvas-3d"
        ref={host}
        tabIndex={0}
        onKeyDown={onKey}
        role="application"
        aria-label={`3D view of ${stack.prefix}, ${stack.n_layers} layers. Arrow keys change the selected layer, F fits the selection, R resets the camera.`}
      >
        <div className="labels-3d" ref={labelsRef} aria-hidden />
        <div className="tooltip-3d" ref={tipRef} style={{ display: "none" }} />
      </div>
    </div>
  );
}
