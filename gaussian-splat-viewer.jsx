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
  if (headerEnd < 0) throw new Error("Invalid PLY");
  const header = new TextDecoder().decode(uint8.slice(0, headerEnd));
  const numVertices = parseInt(header.match(/element vertex (\d+)/)[1]);
  const props = [...header.matchAll(/property float (\w+)/g)].map((m) => m[1]);
  const stride = props.length;
  const get = (n) => { const i = props.indexOf(n); if (i < 0) throw new Error(`Missing: ${n}`); return i; };
  const xI=get("x"),yI=get("y"),zI=get("z");
  const r0I=get("f_dc_0"),g0I=get("f_dc_1"),b0I=get("f_dc_2");
  const opI=get("opacity"),s0I=get("scale_0"),s1I=get("scale_1"),s2I=get("scale_2");
  const f32 = new Float32Array(buffer.slice(headerEnd));
  const positions = new Float32Array(numVertices * 3);
  const colors    = new Float32Array(numVertices * 3);
  const opacities = new Float32Array(numVertices);
  const scales    = new Float32Array(numVertices);
  const CHUNK = 80000;
  for (let i = 0; i < numVertices; i++) {
    const b = i * stride;
    positions[i*3]   = f32[b+xI]; positions[i*3+1] = f32[b+yI]; positions[i*3+2] = f32[b+zI];
    colors[i*3]   = Math.max(0, Math.min(1, 0.5 + SH_C0 * f32[b+r0I]));
    colors[i*3+1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f32[b+g0I]));
    colors[i*3+2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * f32[b+b0I]));
    opacities[i] = 1 / (1 + Math.exp(-f32[b+opI]));
    scales[i] = Math.max(Math.exp(f32[b+s0I]), Math.exp(f32[b+s1I]), Math.exp(f32[b+s2I]));
    if (i % CHUNK === 0) { onProgress(i / numVertices); await new Promise(r => setTimeout(r, 0)); }
  }
  return { positions, colors, opacities, scales, numVertices };
}

const VERT = `
  attribute float aOpacity; attribute float aScale;
  varying vec3 vColor; varying float vOpacity;
  void main() {
    vColor = color; vOpacity = aOpacity;
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    float sz = exp(aScale * 0.5) * 600.0 / max(-mvPos.z, 0.01);
    gl_PointSize = clamp(sz, 0.5, 40.0);
    gl_Position = projectionMatrix * mvPos;
  }
`;
const FRAG = `
  varying vec3 vColor; varying float vOpacity;
  uniform float uMode;
  uniform float uThreshold;
  void main() {
    if (vOpacity < uThreshold) discard;
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float alpha = (uMode < 0.5 ? exp(-r2 * 2.5) : 1.0) * vOpacity;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(vColor, alpha);
  }
`;

// ── Quaternion orbit: zero singularities ─────────────────────────────────────
function rotateOrbit(orbitRef, dx, dy) {
  const q = orbitRef.current.quat;
  const yaw   = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0), -dx * 0.005);
  const right = new THREE.Vector3(1,0,0).applyQuaternion(q);
  const pitch = new THREE.Quaternion().setFromAxisAngle(right, -dy * 0.005);
  q.premultiply(yaw).premultiply(pitch).normalize();
}

function panOrbit(orbitRef, dx, dy) {
  const { quat, radius } = orbitRef.current;
  const r = new THREE.Vector3(1,0,0).applyQuaternion(quat);
  const u = new THREE.Vector3(0,1,0).applyQuaternion(quat);
  const s = radius * 0.0012;
  orbitRef.current.target.addScaledVector(r, -dx * s).addScaledVector(u, dy * s);
}

function syncCamera(cam, orbitRef) {
  const { quat, target, radius } = orbitRef.current;
  cam.position.copy(target).addScaledVector(new THREE.Vector3(0,0,1).applyQuaternion(quat), radius);
  cam.quaternion.copy(quat);
}
// ─────────────────────────────────────────────────────────────────────────────

function Btn({ onClick, active, children, as: Tag = "button" }) {
  const style = {
    padding: "8px 0", width: "100%", border: `1px solid ${active ? "#00e5ff" : "#1e3a4a"}`,
    background: "rgba(4,6,15,0.8)", color: active ? "#00e5ff" : "#5a9ab5",
    fontSize: 9, letterSpacing: 3, cursor: "pointer", backdropFilter: "blur(6px)",
    display: "block", textAlign: "center", boxSizing: "border-box",
    boxShadow: active ? "0 0 10px rgba(0,229,255,0.15)" : "none",
  };
  return Tag === "label"
    ? <label style={style}>{children}</label>
    : <button onClick={onClick} style={style}>{children}</button>;
}

export default function GaussianSplatViewer() {
  const mountRef  = useRef(null);
  const rendRef   = useRef(null);
  const sceneRef  = useRef(null);
  const camRef    = useRef(null);
  const frameRef  = useRef(null);
  const ptsRef    = useRef(null);
  const matRef    = useRef(null);
  const mouse     = useRef({ down: false, x: 0, y: 0, btn: 0 });
  const touchR    = useRef({ touches: [], dist: 0 });
  const orbit     = useRef({ quat: new THREE.Quaternion(), target: new THREE.Vector3(), radius: 5 });

  const [status,   setStatus]   = useState("idle");
  const [progress, setProgress] = useState(0);
  const [stats,    setStats]    = useState(null);
  const [dragging, setDragging] = useState(false);
  const [mode,     setMode]     = useState(0);
  const [flipped,  setFlipped]  = useState(false);
  const [threshold, setThreshold] = useState(0);
  const [fps,      setFps]      = useState(0);

  useEffect(() => {
    const el = mountRef.current; if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);
    rendRef.current = renderer;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x04060f);
    sceneRef.current = scene;
    const cam = new THREE.PerspectiveCamera(55, el.clientWidth / el.clientHeight, 0.001, 2000);
    camRef.current = cam;

    let last = performance.now(), frames = 0;
    const loop = (now) => {
      frameRef.current = requestAnimationFrame(loop);
      frames++;
      if (now - last > 500) { setFps(Math.round(frames*1000/(now-last))); frames=0; last=now; }
      syncCamera(cam, orbit);
      renderer.render(scene, cam);
    };
    loop(performance.now());

    const onResize = () => {
      const w=el.clientWidth,h=el.clientHeight;
      renderer.setSize(w,h); cam.aspect=w/h; cam.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(frameRef.current); window.removeEventListener("resize",onResize); renderer.dispose(); el.removeChild(renderer.domElement); };
  }, []);

  const onMouseDown = useCallback((e) => { mouse.current={down:true,x:e.clientX,y:e.clientY,btn:e.button}; }, []);
  const onMouseMove = useCallback((e) => {
    if (!mouse.current.down) return;
    const dx=e.clientX-mouse.current.x, dy=e.clientY-mouse.current.y;
    mouse.current.x=e.clientX; mouse.current.y=e.clientY;
    mouse.current.btn===0 ? rotateOrbit(orbit,dx,dy) : panOrbit(orbit,dx,dy);
  }, []);
  const onMouseUp = useCallback(() => { mouse.current.down=false; }, []);
  const onWheel = useCallback((e) => {
    e.preventDefault();
    orbit.current.radius = Math.max(0.005, Math.min(2000, orbit.current.radius * (1 + e.deltaY * (e.deltaMode===1?0.05:0.001))));
  }, []);

  const onTouchStart = useCallback((e) => {
    touchR.current.touches=[...e.touches].map(t=>({x:t.clientX,y:t.clientY}));
    if (e.touches.length===2) { const dx=e.touches[0].clientX-e.touches[1].clientX,dy=e.touches[0].clientY-e.touches[1].clientY; touchR.current.dist=Math.sqrt(dx*dx+dy*dy); }
  }, []);
  const onTouchMove = useCallback((e) => {
    e.preventDefault();
    const prev=touchR.current.touches, cur=[...e.touches].map(t=>({x:t.clientX,y:t.clientY}));
    if (e.touches.length===1 && prev.length>=1) {
      rotateOrbit(orbit, cur[0].x-prev[0].x, cur[0].y-prev[0].y);
    } else if (e.touches.length===2 && prev.length>=2) {
      const dx=e.touches[0].clientX-e.touches[1].clientX, dy=e.touches[0].clientY-e.touches[1].clientY;
      const dist=Math.sqrt(dx*dx+dy*dy);
      orbit.current.radius=Math.max(0.005,Math.min(2000,orbit.current.radius*touchR.current.dist/dist));
      touchR.current.dist=dist;
      const cx=(cur[0].x+cur[1].x)/2,cy=(cur[0].y+cur[1].y)/2;
      const px=(prev[0].x+prev[1].x)/2,py=(prev[0].y+prev[1].y)/2;
      panOrbit(orbit, cx-px, cy-py);
    }
    touchR.current.touches=cur;
  }, []);

  const toggleMode = useCallback(() => {
    setMode(m => { const n=1-m; if(matRef.current) matRef.current.uniforms.uMode.value=n; return n; });
  }, []);

  const onThreshold = useCallback((val) => {
    setThreshold(val);
    if (matRef.current) matRef.current.uniforms.uThreshold.value = val;
  }, []);

  const toggleFlip = useCallback(() => {
    setFlipped(f => { if(ptsRef.current) ptsRef.current.scale.y = f ? 1 : -1; return !f; });
  }, []);

  const loadFile = useCallback(async (file) => {
    setStatus("loading"); setProgress(0); setFlipped(false);
    try {
      const data = await parsePLY(await file.arrayBuffer(), setProgress);
      setProgress(1);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(data.positions,3));
      geo.setAttribute("color",    new THREE.BufferAttribute(data.colors,3));
      geo.setAttribute("aOpacity", new THREE.BufferAttribute(data.opacities,1));
      geo.setAttribute("aScale",   new THREE.BufferAttribute(data.scales,1));
      geo.computeBoundingBox();
      const box=geo.boundingBox, center=new THREE.Vector3(), size=new THREE.Vector3();
      box.getCenter(center); box.getSize(size);
      geo.translate(-center.x,-center.y,-center.z);
      orbit.current = { quat: new THREE.Quaternion(), target: new THREE.Vector3(), radius: Math.max(size.x,size.y,size.z)*1.6 };
      rotateOrbit(orbit, 60, 100); // nice default angle
      const mat = new THREE.ShaderMaterial({
        vertexShader:VERT, fragmentShader:FRAG,
        transparent:true, vertexColors:true, depthWrite:false,
        blending:THREE.NormalBlending, uniforms:{uMode:{value:0}, uThreshold:{value:0}},
      });
      matRef.current=mat; setMode(0); setThreshold(0);
      if (ptsRef.current) { sceneRef.current.remove(ptsRef.current); ptsRef.current.geometry.dispose(); ptsRef.current.material.dispose(); }
      const pts = new THREE.Points(geo, mat);
      sceneRef.current.add(pts); ptsRef.current=pts;
      setStats({ n:data.numVertices.toLocaleString(), mb:(file.size/1048576).toFixed(1), name:file.name });
      setStatus("ready");
    } catch(err) { console.error(err); setStatus("error"); }
  }, []);

  const onDrop = useCallback((e) => { e.preventDefault(); setDragging(false); const f=e.dataTransfer.files[0]; if(f) loadFile(f); }, [loadFile]);

  return (
    <div style={{ width:"100vw",height:"100vh",background:"#04060f",position:"relative",overflow:"hidden",fontFamily:"'Courier New',monospace",userSelect:"none" }}>
      {/* Scanlines */}
      <div style={{ position:"absolute",inset:0,pointerEvents:"none",zIndex:10,
        backgroundImage:"repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.07) 3px,rgba(0,0,0,0.07) 4px)" }} />

      {/* Canvas */}
      <div ref={mountRef} style={{ position:"absolute",inset:0 }}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
        onWheel={onWheel} onContextMenu={e=>e.preventDefault()}
        onDrop={onDrop} onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)}
        onTouchStart={onTouchStart} onTouchMove={onTouchMove} onTouchEnd={()=>{touchR.current.touches=[];}}
      />

      {dragging && <div style={{ position:"absolute",inset:8,border:"2px solid #00e5ff",pointerEvents:"none",zIndex:20,boxShadow:"0 0 40px rgba(0,229,255,0.12) inset" }} />}

      {/* ══ TOP-RIGHT: always visible controls ══ */}
      <div style={{ position:"absolute",top:20,right:20,zIndex:30,display:"flex",flexDirection:"column",gap:6,width:148 }}>
        <Btn onClick={toggleFlip} active={flipped}>
          {flipped ? "↑ FLIP: ON" : "↓ FLIP: OFF"}
        </Btn>
        <Btn onClick={toggleMode}>{mode===0 ? "SPLAT MODE" : "DISC MODE"}</Btn>
        <label style={{ display:"block" }}>
          <input type="file" accept=".ply,.splat" style={{ display:"none" }} onChange={e=>e.target.files[0]&&loadFile(e.target.files[0])} />
          <Btn as="label">OPEN FILE</Btn>
        </label>

        {/* Opacity threshold meter */}
        <div style={{ marginTop:6, background:"rgba(4,6,15,0.8)", border:"1px solid #1e3a4a", padding:"10px 12px", backdropFilter:"blur(6px)" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:8 }}>
            <span style={{ fontSize:8, letterSpacing:3, color:"#1a4a5a" }}>OPACITY CUT</span>
            <span style={{ fontSize:11, letterSpacing:1, color: threshold > 0.5 ? "#ff6a6a" : threshold > 0.2 ? "#ffb347" : "#00e5ff", fontWeight:"bold" }}>
              {Math.round(threshold * 100)}%
            </span>
          </div>

          {/* Visual bar */}
          <div style={{ position:"relative", height:6, background:"#0a1828", marginBottom:10, borderRadius:1, overflow:"visible" }}>
            {/* Gradient fill */}
            <div style={{
              position:"absolute", top:0, left:0, right:0, bottom:0,
              background:"linear-gradient(to right, #00e5ff, #ffb347, #ff6a6a)",
              opacity:0.25,
            }}/>
            {/* Active fill */}
            <div style={{
              position:"absolute", top:0, left:0, bottom:0,
              width:`${threshold*100}%`,
              background: threshold > 0.5 ? "#ff6a6a" : threshold > 0.2 ? "#ffb347" : "#00e5ff",
              boxShadow:`0 0 6px ${threshold > 0.5 ? "#ff6a6a" : threshold > 0.2 ? "#ffb347" : "#00e5ff"}`,
              transition:"background 0.2s",
            }}/>
            {/* Thumb indicator */}
            <div style={{
              position:"absolute", top:"50%", left:`${threshold*100}%`,
              transform:"translate(-50%, -50%)",
              width:10, height:10,
              background: threshold > 0.5 ? "#ff6a6a" : threshold > 0.2 ? "#ffb347" : "#00e5ff",
              borderRadius:"50%",
              boxShadow:`0 0 8px ${threshold > 0.5 ? "#ff6a6a" : threshold > 0.2 ? "#ffb347" : "#00e5ff"}`,
              pointerEvents:"none",
            }}/>
            <input type="range" min="0" max="1" step="0.005" value={threshold}
              onChange={e => onThreshold(parseFloat(e.target.value))}
              style={{
                position:"absolute", inset:0, opacity:0, cursor:"pointer",
                width:"100%", margin:0, height:"100%",
              }}
            />
          </div>

          {/* Tick marks */}
          <div style={{ display:"flex", justifyContent:"space-between", fontSize:7, color:"#1a3a4a", letterSpacing:1 }}>
            <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
          </div>
        </div>
      </div>

      {/* IDLE */}
      {status==="idle" && (
        <div style={{ position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",zIndex:20,pointerEvents:"none" }}>
          <div style={{ textAlign:"center",pointerEvents:"all" }}>
            <div style={{ fontSize:80,lineHeight:1,marginBottom:20,opacity:0.1,color:"#00e5ff" }}>◈</div>
            <div style={{ fontSize:11,letterSpacing:8,color:"#00e5ff",marginBottom:6 }}>GAUSSIAN SPLAT VIEWER</div>
            <div style={{ fontSize:10,letterSpacing:3,color:"#1a4a5a",marginBottom:32 }}>3D RADIANCE FIELD RENDERER</div>
            <div style={{ marginBottom:16,fontSize:10,color:"#2a5a6a",letterSpacing:2 }}>PLY / SPLAT ファイルをドロップ</div>
            <label style={{ cursor:"pointer" }}>
              <input type="file" accept=".ply,.splat" style={{ display:"none" }} onChange={e=>e.target.files[0]&&loadFile(e.target.files[0])} />
              <span style={{ display:"inline-block",padding:"10px 28px",border:"1px solid #00e5ff",color:"#00e5ff",fontSize:10,letterSpacing:4,background:"rgba(0,229,255,0.04)",boxShadow:"0 0 20px rgba(0,229,255,0.1)" }}>
                ファイルを選択
              </span>
            </label>
          </div>
        </div>
      )}

      {/* LOADING */}
      {status==="loading" && (
        <div style={{ position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",zIndex:20,background:"rgba(4,6,15,0.88)" }}>
          <div style={{ textAlign:"center",width:320 }}>
            <div style={{ fontSize:10,letterSpacing:6,color:"#00e5ff",marginBottom:24 }}>PARSING GAUSSIAN DATA</div>
            <div style={{ width:"100%",height:1,background:"#0a1525",position:"relative",marginBottom:4 }}>
              <div style={{ position:"absolute",top:0,left:0,width:`${progress*100}%`,height:"100%",background:"#00e5ff",transition:"width 0.15s",boxShadow:"0 0 8px #00e5ff" }} />
            </div>
            <div style={{ display:"flex",justifyContent:"space-between",fontSize:9,color:"#1a4a5a",letterSpacing:2 }}>
              <span>0</span><span style={{ color:"#00e5ff" }}>{Math.round(progress*100)}%</span><span>100</span>
            </div>
          </div>
        </div>
      )}

      {status==="error" && (
        <div style={{ position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",zIndex:20 }}>
          <div style={{ color:"#ff4455",fontSize:10,letterSpacing:4 }}>PARSE ERROR — 右上からファイルを再選択してください</div>
        </div>
      )}

      {/* READY: stats + hint */}
      {status==="ready" && stats && (
        <>
          <div style={{ position:"absolute",top:20,left:20,zIndex:20 }}>
            <div style={{ fontSize:9,letterSpacing:5,color:"#00e5ff",marginBottom:8,opacity:0.6 }}>◈ GSS VIEWER</div>
            <div style={{ background:"rgba(4,6,15,0.7)",padding:"10px 14px",backdropFilter:"blur(4px)",lineHeight:1.9 }}>
              {[["FILE",stats.name],["GAUSSIANS",stats.n],["SIZE",`${stats.mb} MB`],["FPS",fps]].map(([k,v])=>(
                <div key={k} style={{ display:"flex",gap:14,fontSize:9,letterSpacing:2 }}>
                  <span style={{ color:"#1a4a5a",width:80 }}>{k}</span>
                  <span style={{ color:"#7af",maxWidth:180,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
          <div style={{ position:"absolute",bottom:20,left:20,zIndex:20,fontSize:9,color:"#1a3a4a",letterSpacing:2,lineHeight:2.2 }}>
            <div>LEFT DRAG  · 自由回転（特異点なし）</div>
            <div>RIGHT/MID  · パン</div>
            <div>SCROLL     · ズーム</div>
            <div>PINCH      · ズーム（タッチ）</div>
          </div>
        </>
      )}
    </div>
  );
}
