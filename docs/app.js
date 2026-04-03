// ============================================================
// Infusion — Loss Landscape Background (WebGL)
// ============================================================

const GRID_N = 100;
const HALF_N = GRID_N / 2;
const CAM_TILT = 0.55;
const COS_TILT = Math.cos(CAM_TILT);
const SIN_TILT = Math.sin(CAM_TILT);
const MAX_IMPULSES = 20;

/** Touch phones: don't tie window touchmove to the mesh warp (scroll would distort the landscape). */
function isCoarseTouchDevice() {
    return window.matchMedia('(hover: none) and (pointer: coarse)').matches;
}

const BASINS = [
    { wx: -6, wy: -3 },
    { wx:  8, wy:  5 },
    { wx: -2, wy: 10 },
    { wx: -15, wy: -8 },
    { wx: 12, wy: -6 },
    { wx:  0, wy:  0 },
];

// ---- Shaders ----

function vertSrc() {
    return `
precision mediump float;
attribute vec2 a_grid;
uniform float u_time;
uniform vec2 u_mouse;
uniform float u_mouseActive;
uniform vec4 u_impulses[${MAX_IMPULSES}];
uniform int u_impulseCount;
uniform vec2 u_scale;
uniform vec2 u_minPos;
uniform vec2 u_prevMinPos;
uniform float u_minGlow;
uniform float u_minTransition;
uniform float u_lockFlash;

varying float v_height;
varying float v_depth;
varying float v_minDist;
varying float v_prevMinDist;

float baseHeight(vec2 w, float t) {
    float s = 0.35;
    float d = t * 0.15;
    float h = 0.0;
    h += sin(w.x*s + d*0.7) * cos(w.y*s*0.8 + d*0.5) * 1.2;
    h += sin(w.x*s*2.3 + w.y*s*1.7 + d*0.3) * 0.6;
    h += cos(w.x*s*0.6 - w.y*s*1.2 + d*0.9) * 0.8;
    h += sin((w.x+w.y)*s*1.5 + d*0.4) * 0.4;
    h += 1.5 * exp(-((w.x-8.0)*(w.x-8.0)+(w.y-5.0)*(w.y-5.0))/40.0);
    h -= 1.2 * exp(-((w.x+6.0)*(w.x+6.0)+(w.y+3.0)*(w.y+3.0))/30.0);
    h += 1.0 * exp(-((w.x+2.0)*(w.x+2.0)+(w.y-10.0)*(w.y-10.0))/50.0);
    return h;
}

void main() {
    float half_N = ${HALF_N.toFixed(1)};
    vec2 w = a_grid - half_N;

    float h = baseHeight(w, u_time);

    // Mouse warp
    if (u_mouseActive > 0.5) {
        float md = length(w - u_mouse);
        if (md < 8.0) {
            float mt = 1.0 - md / 8.0;
            float mf = mt * mt * (3.0 - 2.0 * mt);
            h += -mf * 2.0;
        }
    }

    // Impulse warps
    for (int k = 0; k < ${MAX_IMPULSES}; k++) {
        if (k >= u_impulseCount) break;
        vec4 imp = u_impulses[k];
        float age = u_time - imp.z;
        float str = abs(imp.w);
        float sgn = sign(imp.w);
        float dur = 2.0 + str;
        if (age < 0.0 || age > dur) continue;
        float idist = length(w - imp.xy);
        float rr = age * (3.0 + str * 2.0);
        float rw = 2.5 + age * 1.2;
        float rd = abs(idist - rr);
        if (rd < rw) {
            float fade = 1.0 - age / dur;
            fade = fade * fade;
            float rf = 1.0 - rd / rw;
            h += sin(rd * 1.5 - age * 6.0) * rf * str * fade * sgn;
        }
    }

    // Minimum bowl
    float minD = length(w - u_minPos);
    h += -0.8 * u_minGlow * exp(-minD*minD / 50.0);
    float prevD = length(w - u_prevMinPos);
    h += -0.4 * (1.0 - u_minTransition) * exp(-prevD*prevD / 50.0);

    // Lock flash ring
    if (u_lockFlash > 0.01) {
        float fr = (1.0 - u_lockFlash) * 25.0;
        float fd = abs(minD - fr);
        if (fd < 3.0) {
            float rf = 1.0 - fd / 3.0;
            h += sin(fd * 2.0) * rf * u_lockFlash * 1.5;
        }
    }

    v_height = h;
    v_minDist = minD;
    v_prevMinDist = prevD;

    // 3D tilt + perspective
    float wz = h * 2.5;
    float y3d = w.y * ${COS_TILT.toFixed(6)} - wz * ${SIN_TILT.toFixed(6)};
    float z3d = w.y * ${SIN_TILT.toFixed(6)} + wz * ${COS_TILT.toFixed(6)};
    v_depth = z3d;
    float pF = 1.0 / (1.0 + z3d * 0.006);
    gl_Position = vec4(w.x * u_scale.x * pF, y3d * u_scale.y * pF, 0.0, 1.0);
}
`;
}

function fragSrc() {
    return `
precision mediump float;
varying float v_height;
varying float v_depth;
varying float v_minDist;
varying float v_prevMinDist;

uniform float u_minGlow;
uniform float u_minTransition;
uniform float u_lockFlash;
uniform float u_time;
uniform float u_lightMode;

void main() {
    float t = clamp((v_height + 3.0) / 6.0, 0.0, 1.0);

    // Dark mode: indigo -> purple -> cyan
    vec3 dLow  = vec3(0.30, 0.20, 0.70);
    vec3 dMid  = vec3(0.55, 0.30, 0.85);
    vec3 dHigh = vec3(0.40, 0.80, 1.00);

    // Light mode: soft indigo -> violet -> blue
    vec3 lLow  = vec3(0.35, 0.30, 0.75);
    vec3 lMid  = vec3(0.50, 0.30, 0.80);
    vec3 lHigh = vec3(0.30, 0.45, 0.85);

    vec3 low  = mix(dLow, lLow, u_lightMode);
    vec3 mid  = mix(dMid, lMid, u_lightMode);
    vec3 high = mix(dHigh, lHigh, u_lightMode);

    vec3 color;
    if (t < 0.5) {
        color = mix(low, mid, t * 2.0);
    } else {
        color = mix(mid, high, (t - 0.5) * 2.0);
    }

    // Gold glow near minimum
    vec3 goldDark = mix(vec3(0.7, 0.55, 0.1), vec3(1.0, 0.84, 0.0), 0.5 + 0.5 * sin(u_time * 2.0));
    vec3 goldLight = mix(vec3(0.7, 0.5, 0.0), vec3(0.85, 0.65, 0.0), 0.5 + 0.5 * sin(u_time * 2.0));
    vec3 gold = mix(goldDark, goldLight, u_lightMode);

    float minInf = exp(-v_minDist * v_minDist / 60.0) * u_minGlow;
    float prevInf = exp(-v_prevMinDist * v_prevMinDist / 60.0) * (1.0 - u_minTransition) * 0.5;

    float flashInf = 0.0;
    if (u_lockFlash > 0.01) {
        float fr = (1.0 - u_lockFlash) * 25.0;
        float fd = abs(v_minDist - fr);
        if (fd < 4.0) {
            flashInf = (1.0 - fd / 4.0) * u_lockFlash * 1.5;
        }
    }

    float totalGold = clamp(minInf + prevInf + flashInf, 0.0, 1.0);
    color = mix(color, gold, totalGold);

    float depthFade = clamp(1.0 - v_depth * 0.008, 0.5, 1.0);
    float baseAlpha = mix(0.7, 0.45, u_lightMode);
    float alpha = (baseAlpha + totalGold * 0.3) * depthFade;
    gl_FragColor = vec4(color * alpha, alpha);
}
`;
}

// ---- Compile helper with visible errors ----

function compileShader(gl, type, src) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const err = gl.getShaderInfoLog(shader);
        const name = type === gl.VERTEX_SHADER ? 'Vertex' : 'Fragment';
        showError(name + ' shader error: ' + err);
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function showError(msg) {
    console.error(msg);
    let el = document.getElementById('glsl-error');
    if (!el) {
        el = document.createElement('div');
        el.id = 'glsl-error';
        el.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:red;color:white;padding:12px;font:12px monospace;white-space:pre-wrap;';
        document.body.appendChild(el);
    }
    el.textContent += msg + '\n';
}

// ---- Landscape ----

class LossLandscape {
    constructor(canvas) {
        this.canvas = canvas;
        const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: true, antialias: true });
        if (!gl) { showError('WebGL not supported'); return; }
        this.gl = gl;
        this.mouse = { x: -9999, y: -9999 };
        this.mouseActive = false;
        this.editImpulses = [];
        this.time = 0;

        this.currentBasin = 0;
        this.minPos = { x: BASINS[0].wx, y: BASINS[0].wy };
        this.prevMinPos = { x: BASINS[0].wx, y: BASINS[0].wy };
        this.minGlow = 1.0;
        this.minTransition = 1.0;
        this.lockFlash = 0.0;
        this.editEnergy = 0;
        this.editsToTransition = 8 + Math.floor(Math.random() * 6);

        if (!this.initShaders()) return;
        this.initBuffers();
        this._cw = undefined;
        this._ch = undefined;
        this.resize();
        this.bindEvents();
        this.animate();
    }

    screenToWorld(sx, sy) {
        const clipX = (sx / this.canvas.width) * 2 - 1;
        const clipY = -((sy / this.canvas.height) * 2 - 1);
        return { wx: clipX / this.scaleX, wy: clipY / (this.scaleY * COS_TILT) };
    }

    computeScale() {
        const aspect = this.canvas.width / this.canvas.height;
        const overscan = 1.4;
        const yExtent = HALF_N * COS_TILT;
        let sy = overscan / yExtent;
        let sx = sy / aspect;
        if (HALF_N * sx < overscan) {
            const needed = overscan / HALF_N * aspect;
            sx = needed / aspect;
            sy = needed;
        }
        this.scaleX = sx;
        this.scaleY = sy;
    }

    initShaders() {
        const gl = this.gl;
        const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc());
        const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc());
        if (!vs || !fs) return false;

        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            showError('Program link error: ' + gl.getProgramInfoLog(this.program));
            return false;
        }

        gl.useProgram(this.program);
        this.loc = {
            grid: gl.getAttribLocation(this.program, 'a_grid'),
            time: gl.getUniformLocation(this.program, 'u_time'),
            mouse: gl.getUniformLocation(this.program, 'u_mouse'),
            mouseActive: gl.getUniformLocation(this.program, 'u_mouseActive'),
            impulseCount: gl.getUniformLocation(this.program, 'u_impulseCount'),
            scale: gl.getUniformLocation(this.program, 'u_scale'),
            minPos: gl.getUniformLocation(this.program, 'u_minPos'),
            prevMinPos: gl.getUniformLocation(this.program, 'u_prevMinPos'),
            minGlow: gl.getUniformLocation(this.program, 'u_minGlow'),
            minTransition: gl.getUniformLocation(this.program, 'u_minTransition'),
            lockFlash: gl.getUniformLocation(this.program, 'u_lockFlash'),
            lightMode: gl.getUniformLocation(this.program, 'u_lightMode'),
        };
        this.impulseLocs = [];
        for (let i = 0; i < MAX_IMPULSES; i++)
            this.impulseLocs.push(gl.getUniformLocation(this.program, 'u_impulses[' + i + ']'));
        return true;
    }

    initBuffers() {
        const gl = this.gl;
        const N = GRID_N;
        const indices = [];
        for (let j = 0; j <= N; j++)
            for (let i = 0; i < N; i++) {
                const a = j * (N + 1) + i;
                indices.push(a, a + 1);
            }
        for (let i = 0; i <= N; i++)
            for (let j = 0; j < N; j++) {
                const a = j * (N + 1) + i;
                indices.push(a, a + (N + 1));
            }
        this.indexCount = indices.length;

        const verts = new Float32Array((N + 1) * (N + 1) * 2);
        let vi = 0;
        for (let j = 0; j <= N; j++)
            for (let i = 0; i <= N; i++) {
                verts[vi++] = i;
                verts[vi++] = j;
            }

        this.vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

        const ext = gl.getExtension('OES_element_index_uint');
        this.ibo = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
        if (ext) {
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(indices), gl.STATIC_DRAW);
            this.indexType = gl.UNSIGNED_INT;
        } else {
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
            this.indexType = gl.UNSIGNED_SHORT;
        }
    }

    resize() {
        const vv = window.visualViewport;
        const w = Math.max(1, Math.round(vv ? vv.width : window.innerWidth));
        const h = Math.max(1, Math.round(vv ? vv.height : window.innerHeight));
        if (this._cw === w && this._ch === h) return;
        this._cw = w;
        this._ch = h;
        this.canvas.width = w;
        this.canvas.height = h;
        if (this.gl) this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.computeScale();
    }

    bindEvents() {
        window.addEventListener('resize', () => this.resize());
        if (window.visualViewport) {
            window.visualViewport.addEventListener('resize', () => this.resize());
        }
        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
            this.mouseActive = true;
        });
        window.addEventListener('mouseleave', () => { this.mouseActive = false; });
        /* Finger-drag scroll on phones was driving u_mouseActive and warping the whole mesh */
        if (!isCoarseTouchDevice()) {
            window.addEventListener('touchmove', (e) => {
                this.mouse.x = e.touches[0].clientX;
                this.mouse.y = e.touches[0].clientY;
                this.mouseActive = true;
            }, { passive: true });
            window.addEventListener('touchend', () => { this.mouseActive = false; });
        }
    }

    addImpulse(sx, sy, strength) {
        const { wx, wy } = this.screenToWorld(sx, sy);
        this.editImpulses.push({ wx, wy, birth: this.time, strength });
        if (this.editImpulses.length > MAX_IMPULSES) this.editImpulses.shift();
    }

    addWorldImpulse(wx, wy, strength) {
        this.editImpulses.push({ wx, wy, birth: this.time, strength });
        if (this.editImpulses.length > MAX_IMPULSES) this.editImpulses.shift();
    }

    onEdit() {
        this.editEnergy++;
        if (this.editEnergy >= this.editsToTransition) {
            this.transitionToNextBasin();
            this.editEnergy = 0;
            this.editsToTransition = 8 + Math.floor(Math.random() * 6);
        }
    }

    transitionToNextBasin() {
        let next;
        do { next = Math.floor(Math.random() * BASINS.length); } while (next === this.currentBasin);
        this.prevMinPos = { ...this.minPos };
        this.currentBasin = next;
        this.minTransition = 0;
        this.lockFlash = 1.0;
        this.addWorldImpulse(BASINS[next].wx, BASINS[next].wy, 2.5);
    }

    updateMinimum(dt) {
        const b = BASINS[this.currentBasin];
        this.minPos.x += (b.wx - this.minPos.x) * 2.0 * dt;
        this.minPos.y += (b.wy - this.minPos.y) * 2.0 * dt;
        this.minTransition = Math.min(1.0, this.minTransition + dt * 1.5);
        this.lockFlash = Math.max(0, this.lockFlash - dt * 0.8);
        this.minGlow = 0.3 + this.minTransition * 0.7;
    }

    draw(t) {
        const gl = this.gl;
        if (!gl) return;

        this.editImpulses = this.editImpulses.filter(imp =>
            (t - imp.birth) <= 2.0 + Math.abs(imp.strength));

        const isLight = document.documentElement.getAttribute('data-theme') === 'light';
        if (isLight) {
            gl.clearColor(0.973, 0.969, 1.0, 1.0); // #f8f7ff
        } else {
            gl.clearColor(0.039, 0.039, 0.071, 1.0); // #0a0a12
        }
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(this.program);
        gl.uniform1f(this.loc.time, t);
        gl.uniform2f(this.loc.scale, this.scaleX, this.scaleY);

        if (this.mouseActive) {
            const { wx, wy } = this.screenToWorld(this.mouse.x, this.mouse.y);
            gl.uniform2f(this.loc.mouse, wx, wy);
            gl.uniform1f(this.loc.mouseActive, 1.0);
        } else {
            gl.uniform1f(this.loc.mouseActive, 0.0);
        }

        gl.uniform2f(this.loc.minPos, this.minPos.x, this.minPos.y);
        gl.uniform2f(this.loc.prevMinPos, this.prevMinPos.x, this.prevMinPos.y);
        gl.uniform1f(this.loc.minGlow, this.minGlow);
        gl.uniform1f(this.loc.minTransition, this.minTransition);
        gl.uniform1f(this.loc.lockFlash, this.lockFlash);
        gl.uniform1f(this.loc.lightMode, isLight ? 1.0 : 0.0);

        gl.uniform1i(this.loc.impulseCount, this.editImpulses.length);
        for (let i = 0; i < MAX_IMPULSES; i++) {
            if (i < this.editImpulses.length) {
                const imp = this.editImpulses[i];
                gl.uniform4f(this.impulseLocs[i], imp.wx, imp.wy, imp.birth, imp.strength);
            } else {
                gl.uniform4f(this.impulseLocs[i], 0.0, 0.0, -999.0, 0.0);
            }
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.enableVertexAttribArray(this.loc.grid);
        gl.vertexAttribPointer(this.loc.grid, 2, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
        gl.drawElements(gl.LINES, this.indexCount, this.indexType, 0);
    }

    animate() {
        const now = performance.now() / 1000;
        const dt = Math.min(now - this.time, 0.1);
        this.time = now;
        this.updateMinimum(dt);
        this.draw(now);
        requestAnimationFrame(() => this.animate());
    }
}

// ---- Event watchers ----

class RevealWatcher {
    constructor(landscape) {
        const narrow = window.matchMedia('(max-width: 768px)').matches;
        const skipImpulse = isCoarseTouchDevice() || narrow;
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !entry.target.classList.contains('visible')) {
                    entry.target.classList.add('visible');
                    if (!skipImpulse) {
                        const r = entry.target.getBoundingClientRect();
                        landscape.addImpulse(r.left + r.width/2, r.top + r.height/2, 1.5);
                    }
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
        document.querySelectorAll('section:not(#hero)').forEach(s => observer.observe(s));
    }
}

class HoverWatcher {
    constructor(landscape) {
        document.addEventListener('mouseenter', (e) => {
            const el = e.target.closest('a, .btn, .card');
            if (!el || el._hp) return;
            el._hp = true;
            const r = el.getBoundingClientRect();
            landscape.addImpulse(r.left + r.width/2, r.top + r.height/2, 0.5);
            setTimeout(() => { el._hp = false; }, 600);
        }, true);
    }
}

class ClickWatcher {
    constructor(landscape) {
        document.addEventListener('mousedown', (e) => landscape.addImpulse(e.clientX, e.clientY, 2.0));
        document.addEventListener('touchstart', (e) => {
            landscape.addImpulse(e.touches[0].clientX, e.touches[0].clientY, 2.0);
        }, { passive: true });
    }
}

class EditWatcher {
    constructor(landscape) {
        document.addEventListener('input', (e) => {
            if (e.target.getAttribute('contenteditable') !== 'true') return;
            const r = e.target.getBoundingClientRect();
            const sel = window.getSelection();
            let x = r.left + r.width/2, y = r.top + r.height/2;
            if (sel && sel.rangeCount > 0) {
                const cr = sel.getRangeAt(0).getBoundingClientRect();
                if (cr.width > 0 || cr.height > 0) { x = cr.left + cr.width/2; y = cr.top + cr.height/2; }
            }
            landscape.addImpulse(x, y, 1.2);
            landscape.onEdit();
        });
    }
}

class ScrollProgress {
    constructor() {
        this.bar = document.getElementById('scroll-progress');
        if (this.bar)
            window.addEventListener('scroll', () => {
                const s = window.scrollY;
                const d = document.documentElement.scrollHeight - window.innerHeight;
                this.bar.style.width = (d > 0 ? (s/d)*100 : 0) + '%';
            }, { passive: true });
    }
}

// ---- Theme toggle ----
function initThemeToggle() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    btn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    });
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    new ScrollProgress();
    initThemeToggle();

    const canvas = document.getElementById('landscape-canvas');
    if (canvas && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const landscape = new LossLandscape(canvas);
        if (landscape.gl) {
            new RevealWatcher(landscape);
            new HoverWatcher(landscape);
            new ClickWatcher(landscape);
            new EditWatcher(landscape);
        }
    }
});
