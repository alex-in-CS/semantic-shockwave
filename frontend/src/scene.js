// The 3D view, in plain three.js.
//
// Built for ~1,500 pinned nodes whose style changes every frame while a search
// replays, so everything is batched:
//   nodes   one InstancedMesh (per-instance color and scale)
//   labels  one instanced quad mesh reading a single canvas texture atlas,
//           billboarded in the vertex shader and placed under each node
//   links   one LineSegments with per-vertex colors
// Paths, comets and the blast flash are a handful of extra meshes.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BG = 0x05060a;
const NODE_RADIUS = 3;
const MAX_EXTRA_NODES = 16;
const MAX_EXTRA_EDGES = 128;
const LABEL_FONT_PX = 26;
const LABEL_CELL_H = 36;
const ATLAS_W = 2048;
const EXTRA_CELL_W = 512;

export const easeOutCubic = (t) => 1 - (1 - t) ** 3;
export const easeInCubic = (t) => t ** 3;
export const easeInOutCubic = (t) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2);

/** Run `onFrame(progress)` every animation frame for `ms`; resolves when done. */
export function animate(ms, onFrame) {
  if (ms <= 0) { onFrame(1); return Promise.resolve(); }
  const start = performance.now();
  return new Promise((resolve) => {
    function frame(now) {
      const t = Math.min(1, (now - start) / ms);
      onFrame(t);
      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    }
    requestAnimationFrame(frame);
  });
}

// ---------------------------------------------------------------------------
// Label atlas: every label drawn once into one canvas, white on transparent.

function labelFont() {
  return `600 ${LABEL_FONT_PX}px Inter, system-ui, -apple-system, "Segoe UI", sans-serif`;
}

function buildAtlas(texts) {
  const measure = document.createElement('canvas').getContext('2d');
  measure.font = labelFont();
  const widths = texts.map((t) => Math.min(ATLAS_W, Math.ceil(measure.measureText(t).width) + 12));
  // Shelf packing, then a final shelf of fixed-width cells for free-text labels.
  const cells = [];
  let x = 0;
  let y = 0;
  widths.forEach((w) => {
    if (x + w > ATLAS_W) { x = 0; y += LABEL_CELL_H; }
    cells.push({ x, y, w });
    x += w;
  });
  y += LABEL_CELL_H;
  const extraCells = [];
  for (let i = 0; i < MAX_EXTRA_NODES; i += 1) {
    const col = i % (ATLAS_W / EXTRA_CELL_W);
    if (i > 0 && col === 0) y += LABEL_CELL_H;
    extraCells.push({ x: col * EXTRA_CELL_W, y, w: EXTRA_CELL_W });
  }
  const height = y + LABEL_CELL_H;
  const canvas = document.createElement('canvas');
  canvas.width = ATLAS_W;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const draw = (text, cell) => {
    ctx.clearRect(cell.x, cell.y, cell.w, LABEL_CELL_H);
    ctx.font = labelFont();
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 5;
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
    const cx = cell.x + cell.w / 2;
    const cy = cell.y + LABEL_CELL_H / 2;
    ctx.strokeText(text, cx, cy, cell.w - 8);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, cx, cy, cell.w - 8);
  };
  texts.forEach((t, i) => draw(t, cells[i]));
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  const rect = (cell) => [cell.x / ATLAS_W, 1 - (cell.y + LABEL_CELL_H) / height,
    (cell.x + cell.w) / ATLAS_W, 1 - cell.y / height];
  return { texture, cells, extraCells, rect, draw, measure };
}

const LABEL_VERTEX = /* glsl */ `
  attribute vec3 iPos;
  attribute vec4 iRect;
  attribute float iAspect;
  attribute vec4 iColor;
  attribute float iScale;
  uniform float uPx;      // world units per screen pixel, at distance 1
  uniform float uLabelPx; // label height on screen, in pixels
  uniform float uGap;
  varying vec2 vUv;
  varying vec4 vColor;
  void main() {
    vec4 mv = modelViewMatrix * vec4(iPos, 1.0);
    float dist = max(-mv.z, 1.0);
    // Constant size on screen (so close labels aren't huge), hanging just below the node.
    float h = uLabelPx * uPx * dist * iScale;
    mv.xy += vec2(position.x * h * iAspect, (position.y - 0.5) * h - uGap * iScale);
    // Fade with distance so the overview isn't a wall of text; highlighted labels stay.
    float fade = clamp(1.0 - (dist - 180.0) / 420.0, 0.0, 1.0);
    vColor = vec4(iColor.rgb, iColor.a * max(fade, step(1.5, iScale)));
    vUv = mix(iRect.xy, iRect.zw, position.xy + 0.5);
    gl_Position = projectionMatrix * mv;
  }
`;

const LABEL_FRAGMENT = /* glsl */ `
  uniform sampler2D uAtlas;
  varying vec2 vUv;
  varying vec4 vColor;
  void main() {
    vec4 tex = texture2D(uAtlas, vUv);
    float a = tex.a * vColor.a;
    if (a < 0.02) discard;
    gl_FragColor = vec4(tex.rgb * vColor.rgb, a);
  }
`;

function textSprite(text, { size = 28, color = '#ffffff', opacity = 0.3, weight = 800 } = {}) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const font = `${weight} 64px Inter, system-ui, sans-serif`;
  ctx.font = font;
  canvas.width = Math.ceil(ctx.measureText(text).width) + 24;
  canvas.height = 84;
  ctx.font = font;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.fillText(text, 12, 44);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set((size * canvas.width) / canvas.height, size, 1);
  return sprite;
}

// ---------------------------------------------------------------------------

export function createScene(container, { onPick = () => {}, onHover = () => {} } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(BG);
  container.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 12000);
  camera.position.set(0, 0, 1000);
  scene.add(camera);
  scene.add(new THREE.AmbientLight(0xffffff, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(0.5, 1, 1);
  camera.add(key);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotateSpeed = 0.35;
  controls.minDistance = 20;
  controls.maxDistance = 4000;

  // ---- state ----
  let capacity = 0;
  let count = 0; // nodes in use, vocabulary + extras
  let baseCount = 0;
  let pos; let home; let baseColor; let color; let brightness; let scale; let labelAlpha; let labelScale;
  let nodeMesh = null;
  let labelMesh = null;
  let atlas = null;
  let edges = []; // [a, b]
  let baseEdgeCount = 0;
  const edgeIndexMap = new Map();
  let edgeBaseColor; let linkMesh = null;
  let regionGroup = new THREE.Group();
  scene.add(regionGroup);
  let nodesDirty = false;
  let edgesDirty = false;
  let positionsDirty = false;
  const pathGroup = new THREE.Group();
  scene.add(pathGroup);
  const flows = []; // particles along shown paths

  const edgeKey = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
  const pixelScale = () => (2 * Math.tan((camera.fov * Math.PI) / 360)) / window.innerHeight;
  const tmpMatrix = new THREE.Matrix4();
  const tmpColor = new THREE.Color();

  function setData({ nodes, links, regions = [] }) {
    baseCount = count = nodes.length;
    capacity = nodes.length + MAX_EXTRA_NODES;
    pos = new Float32Array(capacity * 3);
    home = new Float32Array(capacity * 3);
    baseColor = new Float32Array(capacity * 3);
    color = new Float32Array(capacity * 3);
    brightness = new Float32Array(capacity).fill(1);
    scale = new Float32Array(capacity).fill(1);
    labelAlpha = new Float32Array(capacity).fill(0.85);
    labelScale = new Float32Array(capacity).fill(1);
    nodes.forEach((n, i) => {
      pos.set([n.x, n.y, n.z], i * 3);
      home.set([n.x, n.y, n.z], i * 3);
      baseColor.set([n.color.r, n.color.g, n.color.b], i * 3);
    });

    // Nodes
    const geometry = new THREE.SphereGeometry(NODE_RADIUS, 14, 10);
    const material = new THREE.MeshLambertMaterial({ color: 0xffffff });
    nodeMesh = new THREE.InstancedMesh(geometry, material, capacity);
    nodeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    nodeMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    nodeMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    nodeMesh.count = count;
    nodeMesh.frustumCulled = false;
    scene.add(nodeMesh);

    // Labels
    atlas = buildAtlas(nodes.map((n) => n.label));
    const quad = new THREE.PlaneGeometry(1, 1);
    const lg = new THREE.InstancedBufferGeometry();
    lg.index = quad.index;
    lg.setAttribute('position', quad.getAttribute('position'));
    const iPos = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const iRect = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
    const iAspect = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    const iColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const iScale = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1).setUsage(THREE.DynamicDrawUsage);
    nodes.forEach((n, i) => {
      iRect.setXYZW(i, ...atlas.rect(atlas.cells[i]));
      iAspect.setX(i, atlas.cells[i].w / LABEL_CELL_H);
    });
    lg.setAttribute('iPos', iPos);
    lg.setAttribute('iRect', iRect);
    lg.setAttribute('iAspect', iAspect);
    lg.setAttribute('iColor', iColor);
    lg.setAttribute('iScale', iScale);
    lg.instanceCount = count;
    labelMesh = new THREE.Mesh(lg, new THREE.ShaderMaterial({
      uniforms: { uAtlas: { value: atlas.texture }, uPx: { value: pixelScale() }, uLabelPx: { value: 12 }, uGap: { value: NODE_RADIUS + 1 } },
      vertexShader: LABEL_VERTEX,
      fragmentShader: LABEL_FRAGMENT,
      transparent: true,
      depthWrite: false,
    }));
    labelMesh.frustumCulled = false;
    labelMesh.renderOrder = 2;
    scene.add(labelMesh);

    // Links
    edges = [];
    edgeIndexMap.clear();
    links.forEach(({ a, b }) => {
      const k = edgeKey(a, b);
      if (a === b || edgeIndexMap.has(k)) return;
      edgeIndexMap.set(k, edges.length);
      edges.push([a, b]);
    });
    baseEdgeCount = edges.length;
    const edgeCapacity = edges.length + MAX_EXTRA_EDGES;
    edgeBaseColor = new Float32Array(edgeCapacity * 3);
    const lgeo = new THREE.BufferGeometry();
    lgeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edgeCapacity * 6), 3).setUsage(THREE.DynamicDrawUsage));
    lgeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(edgeCapacity * 6), 3).setUsage(THREE.DynamicDrawUsage));
    linkMesh = new THREE.LineSegments(lgeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    linkMesh.frustumCulled = false;
    scene.add(linkMesh);
    edges.forEach((_, e) => setEdgeBase(e));

    // Regions
    scene.remove(regionGroup);
    regionGroup = new THREE.Group();
    regions.forEach((r) => {
      const sprite = textSprite(r.label.toUpperCase(), { color: r.color, size: 26, opacity: 0.5 });
      sprite.position.set(r.x, r.y, r.z);
      sprite.renderOrder = 1;
      regionGroup.add(sprite);
    });
    scene.add(regionGroup);

    resetStyle();
    positionsDirty = true;
  }

  function setEdgeBase(e) {
    const [a, b] = edges[e];
    for (let c = 0; c < 3; c += 1) {
      edgeBaseColor[e * 3 + c] = (baseColor[a * 3 + c] + baseColor[b * 3 + c]) * 0.5 * 0.045;
    }
  }

  // ---- styling ----
  function resetStyle() {
    for (let i = 0; i < count; i += 1) {
      color[i * 3] = baseColor[i * 3];
      color[i * 3 + 1] = baseColor[i * 3 + 1];
      color[i * 3 + 2] = baseColor[i * 3 + 2];
      brightness[i] = 1;
      scale[i] = 1;
      labelAlpha[i] = 0.85;
      labelScale[i] = 1;
    }
    const colors = linkMesh.geometry.getAttribute('color').array;
    for (let e = 0; e < edges.length; e += 1) {
      for (let v = 0; v < 2; v += 1) colors.set(edgeBaseColor.subarray(e * 3, e * 3 + 3), e * 6 + v * 3);
    }
    nodesDirty = edgesDirty = true;
  }

  /** Partial update: { color: '#hex' | THREE.Color | null (back to group color), brightness, scale, labelAlpha, labelScale }. */
  function setNode(i, style) {
    if (i < 0 || i >= count) return;
    if ('color' in style) {
      if (style.color === null) {
        color.set(baseColor.subarray(i * 3, i * 3 + 3), i * 3);
      } else {
        tmpColor.set(style.color);
        color.set([tmpColor.r, tmpColor.g, tmpColor.b], i * 3);
      }
    }
    if (style.brightness !== undefined) brightness[i] = style.brightness;
    if (style.scale !== undefined) scale[i] = style.scale;
    if (style.labelAlpha !== undefined) labelAlpha[i] = style.labelAlpha;
    if (style.labelScale !== undefined) labelScale[i] = style.labelScale;
    nodesDirty = true;
  }

  function setAllNodes(style) {
    for (let i = 0; i < count; i += 1) setNode(i, style);
  }

  /** Light an edge with a color at an intensity (0 = back to its dim base color). */
  function setEdge(e, edgeColor, intensity = 1) {
    if (e === undefined || e < 0) return;
    const colors = linkMesh.geometry.getAttribute('color').array;
    if (!edgeColor || intensity <= 0) {
      for (let v = 0; v < 2; v += 1) colors.set(edgeBaseColor.subarray(e * 3, e * 3 + 3), e * 6 + v * 3);
    } else {
      tmpColor.set(edgeColor);
      for (let v = 0; v < 2; v += 1) {
        colors[e * 6 + v * 3] = tmpColor.r * intensity;
        colors[e * 6 + v * 3 + 1] = tmpColor.g * intensity;
        colors[e * 6 + v * 3 + 2] = tmpColor.b * intensity;
      }
    }
    edgesDirty = true;
  }

  function dimAllEdges(factor) {
    const colors = linkMesh.geometry.getAttribute('color').array;
    for (let e = 0; e < edges.length; e += 1) {
      for (let v = 0; v < 2; v += 1) {
        for (let c = 0; c < 3; c += 1) colors[e * 6 + v * 3 + c] = edgeBaseColor[e * 3 + c] * factor;
      }
    }
    edgesDirty = true;
  }

  const edgeIndex = (a, b) => edgeIndexMap.get(edgeKey(a, b));

  // ---- free-text extras ----
  function addExtras(extraNodes, extraLinks) {
    const first = count;
    extraNodes.slice(0, MAX_EXTRA_NODES).forEach((n, k) => {
      const i = first + k;
      pos.set([n.x, n.y, n.z], i * 3);
      home.set([n.x, n.y, n.z], i * 3);
      baseColor.set([n.color.r, n.color.g, n.color.b], i * 3);
      color.set([n.color.r, n.color.g, n.color.b], i * 3);
      brightness[i] = 1;
      scale[i] = 1;
      labelAlpha[i] = 1;
      labelScale[i] = 1;
      const cell = atlas.extraCells[k];
      atlas.draw(n.label, cell);
      const geometry = labelMesh.geometry;
      const width = Math.min(EXTRA_CELL_W, Math.ceil(atlas.measure.measureText(n.label).width) + 12);
      // Use only the drawn middle of the fixed-width cell.
      const trimmed = { x: cell.x + (EXTRA_CELL_W - width) / 2, y: cell.y, w: width };
      geometry.getAttribute('iRect').setXYZW(i, ...atlas.rect(trimmed));
      geometry.getAttribute('iAspect').setX(i, width / LABEL_CELL_H);
      geometry.getAttribute('iRect').needsUpdate = true;
      geometry.getAttribute('iAspect').needsUpdate = true;
    });
    atlas.texture.needsUpdate = true;
    count = first + Math.min(extraNodes.length, MAX_EXTRA_NODES);
    extraLinks.forEach(({ a, b }) => {
      const k = edgeKey(a, b);
      if (a === b || edgeIndexMap.has(k) || edges.length >= baseEdgeCount + MAX_EXTRA_EDGES) return;
      edgeIndexMap.set(k, edges.length);
      edges.push([a, b]);
      setEdgeBase(edges.length - 1);
      setEdge(edges.length - 1, null);
    });
    nodeMesh.count = count;
    labelMesh.geometry.instanceCount = count;
    positionsDirty = nodesDirty = edgesDirty = true;
    return first;
  }

  function clearExtras() {
    for (let e = baseEdgeCount; e < edges.length; e += 1) edgeIndexMap.delete(edgeKey(...edges[e]));
    edges.length = baseEdgeCount;
    count = baseCount;
    nodeMesh.count = count;
    labelMesh.geometry.instanceCount = count;
    positionsDirty = nodesDirty = edgesDirty = true;
  }

  // ---- geometry updates ----
  function writeNodes() {
    const iColor = labelMesh.geometry.getAttribute('iColor');
    const iScale = labelMesh.geometry.getAttribute('iScale');
    for (let i = 0; i < count; i += 1) {
      const b = brightness[i];
      nodeMesh.instanceColor.setXYZ(i, color[i * 3] * b, color[i * 3 + 1] * b, color[i * 3 + 2] * b);
      const lb = Math.max(0.35, Math.min(1, b + 0.4));
      iColor.setXYZW(i, lb, lb, lb, labelAlpha[i]);
      iScale.setX(i, labelScale[i]);
    }
    nodeMesh.instanceColor.needsUpdate = true;
    iColor.needsUpdate = true;
    iScale.needsUpdate = true;
  }

  function writePositions() {
    const iPos = labelMesh.geometry.getAttribute('iPos');
    for (let i = 0; i < count; i += 1) {
      const s = scale[i];
      tmpMatrix.makeScale(s, s, s).setPosition(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      nodeMesh.setMatrixAt(i, tmpMatrix);
      iPos.setXYZ(i, pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
    }
    nodeMesh.instanceMatrix.needsUpdate = true;
    iPos.needsUpdate = true;
    const lp = linkMesh.geometry.getAttribute('position');
    for (let e = 0; e < edges.length; e += 1) {
      const [a, b] = edges[e];
      lp.array.set(pos.subarray(a * 3, a * 3 + 3), e * 6);
      lp.array.set(pos.subarray(b * 3, b * 3 + 3), e * 6 + 3);
    }
    lp.needsUpdate = true;
    linkMesh.geometry.setDrawRange(0, edges.length * 2);
  }

  const position = (i) => ({ x: pos[i * 3], y: pos[i * 3 + 1], z: pos[i * 3 + 2] });
  const homeOf = (i) => ({ x: home[i * 3], y: home[i * 3 + 1], z: home[i * 3 + 2] });

  /** Tween every node from where it is to `destination(i)` (a point, or null for home). */
  function tweenPositions(destination, ms, ease = easeOutCubic) {
    const from = pos.slice(0, count * 3);
    const to = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      const d = destination(i) ?? homeOf(i);
      to.set([d.x, d.y, d.z], i * 3);
    }
    return animate(ms, (t) => {
      const e = ease(t);
      for (let k = 0; k < count * 3; k += 1) pos[k] = from[k] + (to[k] - from[k]) * e;
      positionsDirty = true;
    });
  }

  // ---- paths ----
  /** Draw glowing tubes along paths: [{ nodes: [i...], color, radius?, flow? }]. */
  function showPaths(paths) {
    clearPaths();
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    paths.forEach(({ nodes, color: c, radius = 1.1, flow = true, opacity = 0.95 }) => {
      const material = new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity, depthWrite: false });
      for (let k = 0; k < nodes.length - 1; k += 1) {
        a.set(...Object.values(homeOf(nodes[k])));
        b.set(...Object.values(homeOf(nodes[k + 1])));
        const length = a.distanceTo(b);
        if (length < 1e-6) continue;
        const tube = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 8, 1, true), material);
        tube.position.copy(a).lerp(b, 0.5);
        tube.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
        pathGroup.add(tube);
        if (flow) flows.push({ a: a.clone(), b: b.clone(), color: new THREE.Color(c) });
      }
    });
    if (flows.length) {
      const per = 3;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(flows.length * per * 3), 3));
      const colors = new Float32Array(flows.length * per * 3);
      flows.forEach((f, i) => { for (let p = 0; p < per; p += 1) f.color.toArray(colors, (i * per + p) * 3); });
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        size: 4.5, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      }));
      points.userData.per = per;
      points.frustumCulled = false;
      pathGroup.add(points);
      pathGroup.userData.flowPoints = points;
    }
  }

  function clearPaths() {
    pathGroup.children.forEach((o) => { o.geometry.dispose(); o.material.dispose(); });
    pathGroup.clear();
    pathGroup.userData.flowPoints = null;
    flows.length = 0;
  }

  function updateFlows(now) {
    const points = pathGroup.userData.flowPoints;
    if (!points) return;
    const per = points.userData.per;
    const attr = points.geometry.getAttribute('position');
    const p = new THREE.Vector3();
    flows.forEach((f, i) => {
      for (let k = 0; k < per; k += 1) {
        const t = ((now / 1400) + k / per + i * 0.13) % 1;
        p.lerpVectors(f.a, f.b, t);
        attr.setXYZ(i * per + k, p.x, p.y, p.z);
      }
    });
    attr.needsUpdate = true;
  }

  // ---- effects ----
  function makeComet(cometColor, n = 120) {
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const base = new THREE.Color(cometColor);
    for (let i = 0; i < n; i += 1) base.clone().multiplyScalar((1 - i / n) ** 1.6).toArray(colors, i * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 11, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    points.frustumCulled = false;
    return points;
  }

  const dispose = (objects) => objects.forEach((o) => { scene.remove(o); o.geometry.dispose(); o.material.dispose(); });

  /** Two particle streams race from a and b and collide at mid. */
  async function comets(a, b, mid, colors, ms = 1100) {
    const streams = [[a, makeComet(colors[0])], [b, makeComet(colors[1])]];
    streams.forEach(([, c]) => scene.add(c));
    const p = new THREE.Vector3();
    await animate(ms, (t) => {
      streams.forEach(([start, comet]) => {
        const attr = comet.geometry.attributes.position;
        for (let i = 0; i < attr.count; i += 1) {
          const lag = Math.max(0, t - i * 0.004);
          const jitter = (i / attr.count) * 10;
          p.lerpVectors(start, mid, easeInCubic(lag));
          attr.setXYZ(i, p.x + (Math.random() - 0.5) * jitter, p.y + (Math.random() - 0.5) * jitter, p.z + (Math.random() - 0.5) * jitter);
        }
        attr.needsUpdate = true;
      });
    });
    dispose(streams.map(([, c]) => c));
  }

  function flash(mid, ringColor = '#e0f2fe', ms = 900) {
    const additive = { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending };
    const core = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 24), new THREE.MeshBasicMaterial({ color: '#ffffff', ...additive }));
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 96), new THREE.MeshBasicMaterial({ color: ringColor, side: THREE.DoubleSide, ...additive }));
    core.position.copy(mid);
    ring.position.copy(mid);
    ring.lookAt(camera.position);
    scene.add(core, ring);
    return animate(ms, (t) => {
      const e = easeOutCubic(t);
      core.scale.setScalar(4 + 40 * e);
      core.material.opacity = (1 - e) ** 2;
      ring.scale.setScalar(10 + 700 * e);
      ring.material.opacity = 0.6 * (1 - t) ** 1.5;
    }).then(() => dispose([core, ring]));
  }

  // ---- camera ----
  function flyTo(position, target, ms) {
    const fromPos = camera.position.clone();
    const fromTarget = controls.target.clone();
    const toPos = new THREE.Vector3(position.x, position.y, position.z);
    const toTarget = new THREE.Vector3(target.x, target.y, target.z);
    return animate(ms, (t) => {
      const e = easeInOutCubic(t);
      camera.position.lerpVectors(fromPos, toPos, e);
      controls.target.lerpVectors(fromTarget, toTarget, e);
    });
  }

  let insets = () => ({ left: 0, bottom: 0 });
  /** Fit a sphere (center, radius) into the part of the screen the panel doesn't cover. */
  function fit(center, radius, ms = 1200) {
    const halfV = (camera.fov * Math.PI) / 360;
    const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
    const { left, bottom } = insets();
    const coveredX = left / window.innerWidth;
    const coveredY = bottom / window.innerHeight;
    const usableH = Math.atan(Math.tan(halfH) * (1 - coveredX));
    const usableV = Math.atan(Math.tan(halfV) * (1 - coveredY));
    const distance = (radius / Math.sin(Math.min(usableV, usableH))) * 1.05;
    // Keep the current viewing direction; shift the target so the subject centres in what's left.
    const dir = camera.position.clone().sub(controls.target).normalize();
    if (!Number.isFinite(dir.x) || dir.lengthSq() < 0.5) dir.set(0, 0, 1);
    const right = new THREE.Vector3().crossVectors(camera.up, dir).normalize();
    const up = new THREE.Vector3().crossVectors(dir, right).normalize();
    const shift = right.multiplyScalar(-distance * Math.tan(halfH) * coveredX)
      .add(up.multiplyScalar(-distance * Math.tan(halfV) * coveredY));
    const target = new THREE.Vector3(center.x, center.y, center.z).add(shift);
    return flyTo(target.clone().add(dir.multiplyScalar(distance)), target, ms);
  }

  function boundsOf(indices) {
    const c = { x: 0, y: 0, z: 0 };
    indices.forEach((i) => { const p = homeOf(i); c.x += p.x; c.y += p.y; c.z += p.z; });
    c.x /= indices.length; c.y /= indices.length; c.z /= indices.length;
    let r = 0;
    indices.forEach((i) => { const p = homeOf(i); r = Math.max(r, Math.hypot(p.x - c.x, p.y - c.y, p.z - c.z)); });
    return { center: c, radius: r };
  }

  function overview(ms = 1200) {
    const all = Array.from({ length: baseCount }, (_, i) => i);
    const { radius } = boundsOf(all);
    return fit({ x: 0, y: 0, z: 0 }, radius * 0.92, ms);
  }

  // ---- picking ----
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  function pick(clientX, clientY) {
    if (!nodeMesh) return null;
    pointer.set((clientX / window.innerWidth) * 2 - 1, -(clientY / window.innerHeight) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(nodeMesh, false)[0];
    return hit ? hit.instanceId : null;
  }

  let down = null;
  let hoverQueued = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; controls.autoRotate = false; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6) {
      const i = pick(e.clientX, e.clientY);
      if (i !== null) onPick(i);
    }
    down = null;
  });
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'touch') return;
    hoverQueued = { x: e.clientX, y: e.clientY };
  });
  renderer.domElement.addEventListener('pointerleave', () => { hoverQueued = null; onHover(null); });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    if (labelMesh) labelMesh.material.uniforms.uPx.value = pixelScale();
  });

  // ---- loop ----
  function frame(now) {
    if (hoverQueued) {
      const { x, y } = hoverQueued;
      hoverQueued = null;
      onHover(pick(x, y), x, y);
    }
    if (nodeMesh) {
      if (positionsDirty || nodesDirty) writePositions();
      if (nodesDirty) writeNodes();
      if (edgesDirty) linkMesh.geometry.getAttribute('color').needsUpdate = true;
      positionsDirty = nodesDirty = edgesDirty = false;
      updateFlows(now);
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return {
    camera,
    setData,
    get count() { return count; },
    get baseCount() { return baseCount; },
    position,
    home: homeOf,
    setNode,
    setAllNodes,
    resetStyle,
    setEdge,
    dimAllEdges,
    edgeIndex,
    addExtras,
    clearExtras,
    tweenPositions,
    showPaths,
    clearPaths,
    comets,
    flash,
    fit,
    boundsOf,
    overview,
    setInsets(fn) { insets = fn; },
    setRegionsVisible(visible) { regionGroup.visible = visible; },
    setAutoRotate(on) { controls.autoRotate = on; },
  };
}
