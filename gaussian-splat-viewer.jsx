import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

const SH_C0 = 0.28209479177387814;

function findHeaderEnd(uint8) {
  const enc = new TextEncoder();
  for (const ending of ["end_header\n", "end_header\r\n"]) {
    const marker = enc.encode(ending);
    outer: for (let i = 0; i <= uint8.length - marker.length; i++) {
      for (let j = 0; j < marker.length; j++) {
        if (uint8[i + j] !== marker[j]) continue outer;
      }
      return i + marker.length;
    }
  }
  return -1;
}

async function parsePLY(buffer, onProgress) {
  const uint8 = new Uint8Array(buffer);
  const headerEnd = findHeaderEnd(uint8);
  if (headerEnd < 0) throw new Error("Invalid PLY: end_header not found");

  const header = new TextDecoder().decode(uint8.slice(0, headerEnd));
  const numVertices = parseInt(header.match(/element vertex (\d+)/)[1]);
  const props = [...header.matchAll(/property float (\w+)/g)].map((m) => m[1]);
  const stride = props.length;

  const get = (name) => {
    const i = props.indexOf(name);
    if (i < 0) throw new Error(`Missing property: ${name}`);
    return i;
  };

  const xI = get("x"), yI = get("y"), zI = get("z");
  const r0I = get("f_dc_0"), g0I = get("f_dc_1"), b0I = get("f_dc_2");
  const opI = get("opacity");
  const s0I = get("scale_0"), s1I = get("scale_1"), s2I = get("scale_2");

  const dataSlice = buffer.slice(headerEnd);
  const f32 = new Float32Array(dataSlice);

  const positions = new Float32Array(numVertices * 3);
  const colors = new Float32Array(numVertices * 3);
  const opacities = new Float32Array(numVertices);
  const scales = new Float32Array(numVertices);

  const CHUNK = 80000;
  for (let i = 0; i < numVertices; i++) {
    const b = i * stride;
    positions[i * 3] = f32[b + xI];
    positions[i * 3 + 1] = f32[b + yI];
    positions[i * 3 + 2] = f32[b + zI];

    colors[i * 3] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f32[b + r0I]));
    colors[i * 3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f32[b + g0I]));
    colors[i * 3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f32[b + b0I]));

    opacities[i] = 1 / (1 + Math.exp(-f32[b + opI]));

    const s0 = Math.exp(f32[b + s0I]);
    const s1 = Math.exp(f32[b + s1I]);
    const s2 = Math.exp(f32[b + s2I]);
    scales[i] = Math.max(s0, Math.max(s1, s2));

    if (i % CHUNK === 0) {
      onProgress(i / numVertices);
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  return { positions, colors, opacities, scales, numVertices };
}

const VERT = `
  attribute float aOpacity;
  attribute float aScale;
  varying vec3 vColor;
  varying float vOpacity;
  varying float vScale;

  void main() {
    vColor = color;
    vOpacity = aOpacity;
    vScale = aScale;
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    float dist = -mvPos.z;
    float sz = exp(aScale * 0.5) * 600.0 / max(dist, 0.01);
    gl_PointSize = clamp(sz, 0.5, 40.0);
    gl_Position = projectionMatrix * mvPos;
  }
`;

const FRAG = `
  varying vec3 vColor;
  varying float vOpacity;
  uniform float uMode; // 0=splat, 1=point

  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    float alpha;
    if (uMode < 0.5) {
      if (r2 > 1.0) discard;
      alpha = exp(-r2 * 2.5) * vOpacity;
    } else {
      if (r2 > 1.0) discard;
      alpha = vOpacity * 0.9;
    }
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(vColor, alpha);
  }
`;

export default function GaussianSplatViewer() {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const frameRef = useRef(null);
  const mouse = useRef({ down: false, x: 0, y: 0, btn: 0 });
  const orbit = useRef({ theta: 0.3, phi: 1.2, radius: 5, tx: 0, ty: 0, tz: 0 });
  const pointsRef = useRef(null);
  const matRef = useRef(null);

  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [renderMode, setRenderMode] = useState(0); // 0=splat, 1=disc
  const [fps, setFps] = useState(0);
  const fpsRef = useRef({ last: 0, frames: 0 });

  useEffect(() => {
    const el = mountRef.current;
    if (!el) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x04060f);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(55, el.clientWidth / el.clientHeight, 0.001, 2000);
    cameraRef.current = camera;

    let last = performance.now();
    let frames = 0;
    const animate = (now) => {
      frameRef.current = requestAnimationFrame(animate);
      frames++;
      if (now - last > 500) {
        setFps(Math.round((frames * 1000) / (now - last)));
        frames = 0;
        last = now;
      }

      const { theta, phi, radius, tx, ty, tz } = orbit.current;
      camera.position.set(
        tx + radius * Math.sin(phi) * Math.sin(theta),
        ty + radius * Math.cos(phi),
        tz + radius * Math.sin(phi) * Math.cos(theta)
      );
      camera.lookAt(tx, ty, tz);
      renderer.render(scene, camera);
    };
    animate(performance.now());

    const onResize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, []);

  const onMouseDown = useCallback((e) => {
    mouse.current = { down: true, x: e.clientX, y: e.clientY, btn: e.button };
  }, []);

  const onMouseMove = useCallback((e) => {
    if (!mouse.current.down) return;
    const dx = e.clientX - mouse.current.x;
    const dy = e.clientY - mouse.current.y;
    mouse.current.x = e.clientX;
    mouse.current.y = e.clientY;
    if (mouse.current.btn === 0) {
      orbit.current.theta -= dx * 0.006;
      orbit.current.phi = Math.max(0.02, Math.min(Math.PI - 0.02, orbit.current.phi + dy * 0.006));
    } else if (mouse.current.btn === 2) {
      const cam = cameraRef.current;
      if (!cam) return;
      const right = new THREE.Vector3().crossVectors(cam.getWorldDirection(new THREE.Vector3()), cam.up).normalize();
      const up = cam.up.clone().normalize();
      const scale = orbit.current.radius * 0.002;
      orbit.current.tx -= right.x * dx * scale;
      orbit.current.ty += up.y * dy * scale;
      orbit.current.tz -= right.z * dx * scale;
    }
  }, []);

  const onMouseUp = useCallback(() => { mouse.current.down = false; }, []);

  const onWheel = useCallback((e) => {
    e.preventDefault();
    orbit.current.radius = Math.max(0.05, orbit.current.radius * (1 + e.deltaY * 0.001));
  }, []);

  const onContextMenu = useCallback((e) => e.preventDefault(), []);

  const toggleMode = useCallback(() => {
    setRenderMode((m) => {
      const next = 1 - m;
      if (matRef.current) matRef.current.uniforms.uMode.value = next;
      return next;
    });
  }, []);

  const loadFile = useCallback(async (file) => {
    setStatus("loading");
    setProgress(0);
    try {
      const buffer = await file.arrayBuffer();
      const data = await parsePLY(buffer, setProgress);

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(data.colors, 3));
      geo.setAttribute("aOpacity", new THREE.BufferAttribute(data.opacities, 1));
      geo.setAttribute("aScale", new THREE.BufferAttribute(data.scales, 1));
      geo.computeBoundingBox();

      const box = geo.boundingBox;
      const center = new THREE.Vector3();
      box.getCenter(center);
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);

      geo.translate(-center.x, -center.y, -center.z);
      orbit.current = { theta: 0.3, phi: 1.2, radius: maxDim * 1.6, tx: 0, ty: 0, tz: 0 };

      const mat = new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        vertexColors: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
        uniforms: { uMode: { value: 0 } },
      });
      matRef.current = mat;

      if (pointsRef.current) {
        sceneRef.current.remove(pointsRef.current);
        pointsRef.current.geometry.dispose();
        pointsRef.current.material.dispose();
      }

      const pts = new THREE.Points(geo, mat);
      sceneRef.current.add(pts);
      pointsRef.current = pts;

      setStats({
        n: data.numVertices.toLocaleString(),
        mb: (file.size / 1048576).toFixed(1),
        sx: size.x.toFixed(2),
        sy: size.y.toFixed(2),
        sz: size.z.toFixed(2),
        name: file.name,
      });
      setStatus("ready");
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  }, [loadFile]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#04060f", position: "relative", overflow: "hidden", fontFamily: "'Courier New', monospace", userSelect: "none" }}>
      {/* Scanline overlay */}
      <div style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex: 10,
        backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.08) 3px, rgba(0,0,0,0.08) 4px)",
      }} />

      {/* Canvas */}
      <div ref={mountRef}
        style={{ position: "absolute", inset: 0 }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onWheel={onWheel}
        onContextMenu={onContextMenu}
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
      />

      {/* Drop highlight */}
      {dragging && (
        <div style={{ position: "absolute", inset: 8, border: "2px solid #00e5ff", borderRadius: 2, pointerEvents: "none", zIndex: 20,
          boxShadow: "0 0 40px rgba(0,229,255,0.15) inset, 0 0 40px rgba(0,229,255,0.05)",
        }} />
      )}

      {/* IDLE */}
      {status === "idle" && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", zIndex: 30, pointerEvents: "none" }}>
          <div style={{ textAlign: "center", pointerEvents: "all" }}>
            <div style={{ fontSize: 80, lineHeight: 1, marginBottom: 20, opacity: 0.15, color: "#00e5ff" }}>◈</div>
            <div style={{ fontSize: 11, letterSpacing: 8, color: "#00e5ff", marginBottom: 6 }}>GAUSSIAN SPLAT VIEWER</div>
            <div style={{ fontSize: 10, letterSpacing: 3, color: "#1a4a5a", marginBottom: 32 }}>3D RADIANCE FIELD RENDERER</div>
            <div style={{ marginBottom: 12, fontSize: 10, color: "#2a5a6a", letterSpacing: 2 }}>PLY / SPLAT ファイルをドロップ</div>
            <label style={{ cursor: "pointer", display: "inline-block" }}>
              <input type="file" accept=".ply,.splat" style={{ display: "none" }} onChange={(e) => e.target.files[0] && loadFile(e.target.files[0])} />
              <span style={{ display: "inline-block", padding: "10px 28px", border: "1px solid #00e5ff", color: "#00e5ff",
                fontSize: 10, letterSpacing: 4, background: "rgba(0,229,255,0.04)",
                boxShadow: "0 0 20px rgba(0,229,255,0.1)",
              }}>
                ファイルを選択
              </span>
            </label>
          </div>
        </div>
      )}

      {/* LOADING */}
      {status === "loading" && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", zIndex: 30, background: "rgba(4,6,15,0.85)" }}>
          <div style={{ textAlign: "center", width: 320 }}>
            <div style={{ fontSize: 10, letterSpacing: 6, color: "#00e5ff", marginBottom: 24 }}>PARSING GAUSSIAN DATA</div>
            <div style={{ width: "100%", height: 1, background: "#0a1525", marginBottom: 4, position: "relative" }}>
              <div style={{ position: "absolute", top: 0, left: 0, width: `${progress * 100}%`, height: "100%", background: "#00e5ff",
                transition: "width 0.15s linear", boxShadow: "0 0 8px #00e5ff" }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#1a4a5a", letterSpacing: 2 }}>
              <span>0</span>
              <span style={{ color: "#00e5ff" }}>{Math.round(progress * 100)}%</span>
              <span>100</span>
            </div>
            <div style={{ marginTop: 20, fontSize: 9, color: "#0a3040", letterSpacing: 1 }}>
              {Math.round(progress * 3848892).toLocaleString()} / 3,848,892 vertices
            </div>
          </div>
        </div>
      )}

      {/* ERROR */}
      {status === "error" && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", zIndex: 30 }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#ff4455", fontSize: 10, letterSpacing: 4, marginBottom: 16 }}>PARSE ERROR</div>
            <label style={{ cursor: "pointer" }}>
              <input type="file" accept=".ply" style={{ display: "none" }} onChange={(e) => e.target.files[0] && loadFile(e.target.files[0])} />
              <span style={{ color: "#555", fontSize: 10, letterSpacing: 2 }}>別のファイルを試す</span>
            </label>
          </div>
        </div>
      )}

      {/* READY: HUD */}
      {status === "ready" && stats && (
        <>
          {/* Top-left: stats */}
          <div style={{ position: "absolute", top: 20, left: 20, zIndex: 20, lineHeight: 1.9 }}>
            <div style={{ fontSize: 9, letterSpacing: 5, color: "#00e5ff", marginBottom: 10, opacity: 0.7 }}>◈ GSS VIEWER</div>
            <div style={{ background: "rgba(4,6,15,0.7)", padding: "12px 16px", backdropFilter: "blur(4px)" }}>
              {[
                ["FILE", stats.name],
                ["GAUSSIANS", stats.n],
                ["FILE SIZE", `${stats.mb} MB`],
                ["BOUNDS X", `${stats.sx} m`],
                ["BOUNDS Y", `${stats.sy} m`],
                ["BOUNDS Z", `${stats.sz} m`],
                ["FPS", fps],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", gap: 16, fontSize: 9, letterSpacing: 2 }}>
                  <span style={{ color: "#1a4a5a", width: 80 }}>{k}</span>
                  <span style={{ color: "#7af" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom-left: controls */}
          <div style={{ position: "absolute", bottom: 20, left: 20, zIndex: 20, fontSize: 9, color: "#1a3a4a", letterSpacing: 2, lineHeight: 2 }}>
            <div>LEFT DRAG  ·  回転</div>
            <div>RIGHT DRAG ·  パン</div>
            <div>SCROLL     ·  ズーム</div>
          </div>

          {/* Top-right: controls */}
          <div style={{ position: "absolute", top: 20, right: 20, zIndex: 20, display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-end" }}>
            <button onClick={toggleMode}
              style={{ padding: "7px 16px", border: "1px solid #1a3a4a", background: "rgba(4,6,15,0.7)",
                color: "#7af", fontSize: 9, letterSpacing: 3, cursor: "pointer", backdropFilter: "blur(4px)" }}>
              MODE: {renderMode === 0 ? "SPLAT" : "DISC"}
            </button>
            <label style={{ cursor: "pointer" }}>
              <input type="file" accept=".ply,.splat" style={{ display: "none" }} onChange={(e) => e.target.files[0] && loadFile(e.target.files[0])} />
              <span style={{ display: "inline-block", padding: "7px 16px", border: "1px solid #1a3a4a", background: "rgba(4,6,15,0.7)",
                color: "#555", fontSize: 9, letterSpacing: 3, backdropFilter: "blur(4px)" }}>
                OPEN FILE
              </span>
            </label>
          </div>
        </>
      )}
    </div>
  );
}
