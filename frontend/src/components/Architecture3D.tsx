import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { componentColor, fmtBytes } from "../format";
import type { ComponentSummary, StackSummary } from "../types";
import type { Selection } from "./ArchitecturePage";

interface Props {
  stack: StackSummary;
  globals: ComponentSummary[];
  selection: Selection;
  onSelect: (s: Selection) => void;
}

interface BlockInfo {
  label: string;
  bytes: number;
  sel: Selection;
}

const ORDER = ["norm", "attention", "ssm", "router", "experts", "ffn", "other"];
const SLAB_H = 0.42;
const GAP = 0.16;
const DEPTH = 2.2;
const MAX_W = 7;
const MIN_W = 0.12;

export default function Architecture3D({ stack, globals, selection, onSelect }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;
  const highlightRef = useRef<(s: Selection) => void>(() => {});

  useEffect(() => {
    const el = host.current!;
    const width = el.clientWidth;
    const height = el.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(6, 10, 8);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x88ccff, 0.4);
    rim.position.set(-8, 4, -6);
    scene.add(rim);

    const maxBytes = Math.max(1, ...stack.layers.map((l) => l.n_bytes), ...globals.map((g) => g.n_bytes));
    const pickables: THREE.Mesh[] = [];
    const info = new Map<THREE.Object3D, BlockInfo>();
    const geoms: THREE.BufferGeometry[] = [];
    const mats: THREE.Material[] = [];

    const addRow = (comps: ComponentSummary[], y: number, sel: Selection, label: string, height = SLAB_H) => {
      const rowBytes = comps.reduce((a, c) => a + c.n_bytes, 0);
      const rowW = Math.max((rowBytes / maxBytes) * MAX_W, 0.6);
      const widths = comps.map((c) => Math.max((c.n_bytes / Math.max(rowBytes, 1)) * rowW, MIN_W));
      const total = widths.reduce((a, b) => a + b, 0);
      let x = -total / 2;
      comps.forEach((c, i) => {
        const w = widths[i];
        const g = new THREE.BoxGeometry(w * 0.97, height, DEPTH);
        const m = new THREE.MeshStandardMaterial({ color: componentColor(c.component), roughness: 0.55, metalness: 0.1, emissive: 0x000000 });
        const mesh = new THREE.Mesh(g, m);
        mesh.position.set(x + w / 2, y, 0);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: 0x0b1220, transparent: true, opacity: 0.5 }));
        mesh.add(edges);
        scene.add(mesh);
        pickables.push(mesh);
        info.set(mesh, { label: `${label} · ${c.component} (${c.tensors.length} tensors)`, bytes: c.n_bytes, sel });
        geoms.push(g, edges.geometry);
        mats.push(m, edges.material as THREE.Material);
        x += w;
      });
    };

    const sorted = (comps: ComponentSummary[]) => [...comps].sort((a, b) => ORDER.indexOf(a.component) - ORDER.indexOf(b.component));

    let y = 0;
    const emb = globals.filter((g) => g.component === "embedding");
    if (emb.length) {
      addRow(emb, y, { kind: "global", component: "embedding" }, "embedding", SLAB_H * 1.3);
      y += SLAB_H * 1.3 + GAP * 2;
    }
    for (const l of stack.layers) {
      addRow(sorted(l.components), y, { kind: "layer", index: l.index }, `${stack.prefix}.${l.index}`);
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

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 2000);
    const center = new THREE.Vector3(0, totalH / 2, 0);
    const dist = Math.max(totalH, MAX_W) * 1.25 + 6;
    camera.position.set(dist * 0.55, totalH / 2 + dist * 0.25, dist * 0.8);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.update();

    const grid = new THREE.GridHelper(Math.max(MAX_W * 2, 12), 12, 0x2a3b5c, 0x1a2740);
    grid.position.y = -0.6;
    scene.add(grid);

    // Highlight selection
    highlightRef.current = (s: Selection) => {
      for (const mesh of pickables) {
        const bi = info.get(mesh)!;
        const on =
          s !== null &&
          ((s.kind === "layer" && bi.sel?.kind === "layer" && bi.sel.index === s.index) ||
            (s.kind === "global" && bi.sel?.kind === "global" && bi.sel.component === s.component));
        (mesh.material as THREE.MeshStandardMaterial).emissive.setHex(on ? 0x335566 : 0x000000);
        mesh.scale.z = on ? 1.12 : 1;
      }
    };

    const ray = new THREE.Raycaster();
    const ptr = new THREE.Vector2();
    const pick = (ev: PointerEvent): THREE.Mesh | null => {
      const r = renderer.domElement.getBoundingClientRect();
      ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ptr, camera);
      const hit = ray.intersectObjects(pickables, false)[0];
      return (hit?.object as THREE.Mesh) ?? null;
    };
    const onMove = (ev: PointerEvent) => {
      const m = pick(ev);
      const r = renderer.domElement.getBoundingClientRect();
      if (m) {
        const bi = info.get(m)!;
        setTip({ x: ev.clientX - r.left, y: ev.clientY - r.top, text: `${bi.label} — ${fmtBytes(bi.bytes)}` });
        renderer.domElement.style.cursor = "pointer";
      } else {
        setTip(null);
        renderer.domElement.style.cursor = "grab";
      }
    };
    let downAt = 0;
    const onDown = () => (downAt = performance.now());
    const onUp = (ev: PointerEvent) => {
      if (performance.now() - downAt > 250) return; // treat as orbit drag, not click
      const m = pick(ev);
      if (m) selectRef.current(info.get(m)!.sel);
    };
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    renderer.domElement.addEventListener("pointerleave", () => setTip(null));

    let raf = 0;
    const loop = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    const ro = new ResizeObserver(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      geoms.forEach((g) => g.dispose());
      mats.forEach((m) => m.dispose());
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [stack, globals]);

  useEffect(() => {
    highlightRef.current(selection);
  }, [selection, stack, globals]);

  return (
    <div className="canvas-3d" ref={host}>
      {tip && (
        <div className="tooltip-3d" style={{ left: tip.x, top: tip.y }}>
          {tip.text}
        </div>
      )}
    </div>
  );
}
