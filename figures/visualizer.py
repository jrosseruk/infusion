import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import json


def generate_data(seed, n_per_class, spread, sep):
    """Generate initial dataset."""
    rng = np.random.default_rng(int(seed))
    mu0 = np.array([-sep / 2.0, 0.0])
    mu1 = np.array([+sep / 2.0, 0.0])
    X0 = rng.normal(loc=mu0, scale=spread, size=(n_per_class, 2))
    X1 = rng.normal(loc=mu1, scale=spread, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return X.tolist(), y.tolist()


def create_full_app(points_x, points_y, labels, l2, eps_up, wlim, grid):
    """Create the full interactive app in pure JavaScript."""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            height: 100%;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0e1117;
            color: #fafafa;
        }}
        .container {{
            display: flex;
            gap: 20px;
            padding: 10px;
            height: calc(100% - 20px);
            min-height: 740px;
        }}
        .panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        .panel h3 {{
            font-size: 14px;
            margin-bottom: 10px;
            color: #fafafa;
            flex-shrink: 0;
        }}
        #canvas-container {{
            position: relative;
            border: 1px solid #333;
            background: #1a1a2e;
            border-radius: 4px;
            flex: 1;
            min-height: 680px;
        }}
        #pointCanvas {{
            cursor: crosshair;
            width: 100%;
            height: 100%;
        }}
        .legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        #plot3d {{
            flex: 1;
            border-radius: 4px;
            min-height: 680px;
        }}
        .info {{
            font-size: 11px;
            color: #888;
            margin-top: 8px;
            padding: 4px;
            flex-shrink: 0;
        }}
        .info strong {{ color: #4a9eff; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="panel">
            <h3>Training data + decision boundary (drag points!)</h3>
            <div id="canvas-container">
                <canvas id="pointCanvas"></canvas>
                <div class="legend">
                    <div class="legend-item"><div class="legend-dot" style="background: #4a9eff;"></div>Class 0</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #ff6b6b;"></div>Class 1</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #50fa7b;"></div>Selected</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #ffd93d; width: 20px; height: 3px; border-radius: 0;"></div>Boundary</div>
                </div>
            </div>
            <div class="info" id="info">Drag any point to see the manifold change in real-time</div>
        </div>
        <div class="panel">
            <h3>Parameter manifold: loss surface over (w1, w2)</h3>
            <div id="plot3d"></div>
            <div class="info" id="info3d">Current optimum shown as cyan dot</div>
        </div>
    </div>

    <script>
        // ============== DATA ==============
        let points = [];
        const pointsX = {json.dumps(points_x)};
        const pointsY = {json.dumps(points_y)};
        const labels = {json.dumps(labels)};
        const l2 = {l2};
        const epsUp = {eps_up};
        const wlim = {wlim};
        const gridSize = {grid};

        for (let i = 0; i < pointsX.length; i++) {{
            points.push({{ idx: i, x: pointsX[i], y: pointsY[i], label: labels[i] }});
        }}

        let selectedIdx = -1;
        let dragging = false;
        let dragIdx = -1;

        // ============== MATH HELPERS ==============
        function sigmoid(t) {{
            return 1.0 / (1.0 + Math.exp(-t));
        }}

        function dot(a, b) {{
            let sum = 0;
            for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
            return sum;
        }}

        function matVec(M, v) {{
            const result = [];
            for (let i = 0; i < M.length; i++) {{
                result.push(dot(M[i], v));
            }}
            return result;
        }}

        function solve3x3(A, b) {{
            // Simple 3x3 solver using Cramer's rule
            const det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                      - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                      + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

            if (Math.abs(det) < 1e-10) {{
                // Add regularization
                A[0][0] += 1e-6; A[1][1] += 1e-6; A[2][2] += 1e-6;
                return solve3x3(A, b);
            }}

            const x = [];
            for (let col = 0; col < 3; col++) {{
                const Ac = A.map(row => [...row]);
                for (let row = 0; row < 3; row++) Ac[row][col] = b[row];
                const detC = Ac[0][0]*(Ac[1][1]*Ac[2][2] - Ac[1][2]*Ac[2][1])
                           - Ac[0][1]*(Ac[1][0]*Ac[2][2] - Ac[1][2]*Ac[2][0])
                           + Ac[0][2]*(Ac[1][0]*Ac[2][1] - Ac[1][1]*Ac[2][0]);
                x.push(detC / det);
            }}
            return x;
        }}

        function fitLogReg(pts) {{
            const n = pts.length;
            let theta = [0, 0, 0]; // [w1, w2, b]

            for (let iter = 0; iter < 30; iter++) {{
                const w = [theta[0], theta[1]];
                const b = theta[2];

                // Compute predictions
                const p = pts.map(pt => sigmoid(pt.x * w[0] + pt.y * w[1] + b));

                // Gradient
                let grad = [0, 0, 0];
                for (let i = 0; i < n; i++) {{
                    const r = p[i] - pts[i].label;
                    grad[0] += r * pts[i].x / n;
                    grad[1] += r * pts[i].y / n;
                    grad[2] += r / n;
                }}
                grad[0] += l2 * w[0];
                grad[1] += l2 * w[1];

                // Hessian
                const H = [[l2, 0, 0], [0, l2, 0], [0, 0, 0]];
                for (let i = 0; i < n; i++) {{
                    const s = p[i] * (1 - p[i]);
                    H[0][0] += s * pts[i].x * pts[i].x / n;
                    H[0][1] += s * pts[i].x * pts[i].y / n;
                    H[0][2] += s * pts[i].x / n;
                    H[1][0] += s * pts[i].y * pts[i].x / n;
                    H[1][1] += s * pts[i].y * pts[i].y / n;
                    H[1][2] += s * pts[i].y / n;
                    H[2][0] += s * pts[i].x / n;
                    H[2][1] += s * pts[i].y / n;
                    H[2][2] += s / n;
                }}

                // Newton step
                const step = solve3x3(H, grad);
                theta = [theta[0] - step[0], theta[1] - step[1], theta[2] - step[2]];

                if (Math.sqrt(step[0]**2 + step[1]**2 + step[2]**2) < 1e-8) break;
            }}

            return theta;
        }}

        function computeLoss(theta, pts) {{
            const n = pts.length;
            const w = [theta[0], theta[1]];
            const b = theta[2];
            let ce = 0;
            for (let i = 0; i < n; i++) {{
                const z = pts[i].x * w[0] + pts[i].y * w[1] + b;
                const p = sigmoid(z);
                ce += -pts[i].label * Math.log(p + 1e-12) - (1 - pts[i].label) * Math.log(1 - p + 1e-12);
            }}
            ce /= n;
            const reg = 0.5 * l2 * (w[0]**2 + w[1]**2);
            return ce + reg;
        }}

        function computeLossSurface(pts, bFixed) {{
            const w1s = [], w2s = [], zs = [];
            for (let i = 0; i < gridSize; i++) {{
                const w1 = -wlim + (2 * wlim * i / (gridSize - 1));
                w1s.push(w1);
            }}
            for (let j = 0; j < gridSize; j++) {{
                const w2 = -wlim + (2 * wlim * j / (gridSize - 1));
                w2s.push(w2);
            }}

            for (let j = 0; j < gridSize; j++) {{
                const row = [];
                for (let i = 0; i < gridSize; i++) {{
                    const theta = [w1s[i], w2s[j], bFixed];
                    row.push(computeLoss(theta, pts));
                }}
                zs.push(row);
            }}

            return {{ w1s, w2s, zs }};
        }}

        function computeInfluence(theta, pts, idx) {{
            const n = pts.length;
            const w = [theta[0], theta[1]];
            const b = theta[2];

            // Per-example gradient
            const xi = pts[idx];
            const pi = sigmoid(xi.x * w[0] + xi.y * w[1] + b);
            const g = pi - xi.label;
            const gradI = [g * xi.x, g * xi.y, g];

            // Hessian
            const H = [[l2, 0, 0], [0, l2, 0], [0, 0, 0]];
            for (let i = 0; i < n; i++) {{
                const z = pts[i].x * w[0] + pts[i].y * w[1] + b;
                const p = sigmoid(z);
                const s = p * (1 - p);
                H[0][0] += s * pts[i].x * pts[i].x / n;
                H[0][1] += s * pts[i].x * pts[i].y / n;
                H[0][2] += s * pts[i].x / n;
                H[1][0] += s * pts[i].y * pts[i].x / n;
                H[1][1] += s * pts[i].y * pts[i].y / n;
                H[1][2] += s * pts[i].y / n;
                H[2][0] += s * pts[i].x / n;
                H[2][1] += s * pts[i].y / n;
                H[2][2] += s / n;
            }}

            const step = solve3x3(H, gradI);
            return [-epsUp * step[0], -epsUp * step[1], -epsUp * step[2]];
        }}

        // ============== 2D CANVAS ==============
        const canvas = document.getElementById('pointCanvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('canvas-container');

        function resizeCanvas() {{
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }}
        resizeCanvas();
        window.addEventListener('resize', () => {{ resizeCanvas(); draw2D(); }});

        function getDataBounds() {{
            let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
            for (const p of points) {{
                xMin = Math.min(xMin, p.x); xMax = Math.max(xMax, p.x);
                yMin = Math.min(yMin, p.y); yMax = Math.max(yMax, p.y);
            }}
            const padX = (xMax - xMin) * 0.15 + 0.5;
            const padY = (yMax - yMin) * 0.15 + 0.5;
            return {{ xMin: xMin - padX, xMax: xMax + padX, yMin: yMin - padY, yMax: yMax + padY }};
        }}

        function dataToCanvas(dx, dy, bounds) {{
            const padding = 40;
            const cx = padding + (dx - bounds.xMin) / (bounds.xMax - bounds.xMin) * (canvas.width - 2 * padding);
            const cy = canvas.height - padding - (dy - bounds.yMin) / (bounds.yMax - bounds.yMin) * (canvas.height - 2 * padding);
            return [cx, cy];
        }}

        function canvasToData(cx, cy, bounds) {{
            const padding = 40;
            const dx = bounds.xMin + (cx - padding) / (canvas.width - 2 * padding) * (bounds.xMax - bounds.xMin);
            const dy = bounds.yMin + (canvas.height - padding - cy) / (canvas.height - 2 * padding) * (bounds.yMax - bounds.yMin);
            return [dx, dy];
        }}

        let currentTheta = [0, 0, 0];

        function draw2D() {{
            const bounds = getDataBounds();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Fit model
            currentTheta = fitLogReg(points);
            const w = [currentTheta[0], currentTheta[1]];
            const b = currentTheta[2];
            const trainLoss = computeLoss(currentTheta, points);

            // Draw grid
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 0.5;
            const padding = 40;

            // Draw axes
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 1;
            const [ax0, ay0] = dataToCanvas(bounds.xMin, 0, bounds);
            const [ax1, ay1] = dataToCanvas(bounds.xMax, 0, bounds);
            if (ay0 > padding && ay0 < canvas.height - padding) {{
                ctx.beginPath(); ctx.moveTo(padding, ay0); ctx.lineTo(canvas.width - padding, ay0); ctx.stroke();
            }}
            const [bx0, by0] = dataToCanvas(0, bounds.yMin, bounds);
            const [bx1, by1] = dataToCanvas(0, bounds.yMax, bounds);
            if (bx0 > padding && bx0 < canvas.width - padding) {{
                ctx.beginPath(); ctx.moveTo(bx0, padding); ctx.lineTo(bx0, canvas.height - padding); ctx.stroke();
            }}

            // Draw decision boundary
            ctx.strokeStyle = '#ffd93d';
            ctx.lineWidth = 2;
            let dbX1, dbY1, dbX2, dbY2;
            if (Math.abs(w[1]) > 1e-8) {{
                dbX1 = bounds.xMin; dbX2 = bounds.xMax;
                dbY1 = -(w[0] / w[1]) * dbX1 - b / w[1];
                dbY2 = -(w[0] / w[1]) * dbX2 - b / w[1];
            }} else if (Math.abs(w[0]) > 1e-8) {{
                dbX1 = dbX2 = -b / w[0];
                dbY1 = bounds.yMin; dbY2 = bounds.yMax;
            }} else {{
                dbX1 = dbX2 = 0; dbY1 = dbY2 = 0;
            }}
            const [lx1, ly1] = dataToCanvas(dbX1, dbY1, bounds);
            const [lx2, ly2] = dataToCanvas(dbX2, dbY2, bounds);
            ctx.beginPath(); ctx.moveTo(lx1, ly1); ctx.lineTo(lx2, ly2); ctx.stroke();

            // Draw points
            for (let i = 0; i < points.length; i++) {{
                const p = points[i];
                const [cx, cy] = dataToCanvas(p.x, p.y, bounds);

                ctx.beginPath();
                ctx.arc(cx, cy, i === selectedIdx ? 14 : 9, 0, Math.PI * 2);

                if (i === selectedIdx) {{
                    ctx.fillStyle = '#50fa7b';
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                }} else {{
                    ctx.fillStyle = p.label === 0 ? '#4a9eff' : '#ff6b6b';
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }}
            }}

            // Axis labels
            ctx.fillStyle = '#888';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('x₁', canvas.width / 2, canvas.height - 8);
            ctx.save();
            ctx.translate(12, canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('x₂', 0, 0);
            ctx.restore();

            // Update info
            document.getElementById('info').innerHTML =
                `<strong>w</strong> = (${{w[0].toFixed(3)}}, ${{w[1].toFixed(3)}}), <strong>b</strong> = ${{b.toFixed(3)}} | <strong>loss</strong> = ${{trainLoss.toFixed(4)}}`;

            // Update 3D plot
            update3D();
        }}

        function findPoint(cx, cy) {{
            const bounds = getDataBounds();
            for (let i = 0; i < points.length; i++) {{
                const [px, py] = dataToCanvas(points[i].x, points[i].y, bounds);
                const dist = Math.sqrt((cx - px) ** 2 + (cy - py) ** 2);
                if (dist < 18) return i;
            }}
            return -1;
        }}

        canvas.addEventListener('mousedown', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;

            const idx = findPoint(cx, cy);
            if (idx >= 0) {{
                selectedIdx = idx;
                dragIdx = idx;
                dragging = true;
                draw2D();
            }}
        }});

        canvas.addEventListener('mousemove', (e) => {{
            if (!dragging || dragIdx < 0) return;

            const rect = canvas.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;

            const bounds = getDataBounds();
            const [dx, dy] = canvasToData(cx, cy, bounds);
            points[dragIdx].x = dx;
            points[dragIdx].y = dy;
            draw2D();
        }});

        canvas.addEventListener('mouseup', () => {{
            dragging = false;
            dragIdx = -1;
        }});

        canvas.addEventListener('mouseleave', () => {{
            dragging = false;
            dragIdx = -1;
        }});

        // ============== 3D PLOTLY ==============
        let plot3dInitialized = false;

        function update3D() {{
            const b = currentTheta[2];
            const surface = computeLossSurface(points, b);

            const zHat = computeLoss(currentTheta, points);

            const traces = [
                {{
                    type: 'surface',
                    x: surface.w1s,
                    y: surface.w2s,
                    z: surface.zs,
                    showscale: false,
                    opacity: 0.9,
                    colorscale: 'Viridis',
                    name: 'Loss surface'
                }},
                {{
                    type: 'scatter3d',
                    x: [currentTheta[0]],
                    y: [currentTheta[1]],
                    z: [zHat],
                    mode: 'markers',
                    marker: {{ size: 6, color: 'cyan' }},
                    name: 'Current optimum'
                }}
            ];

            // Add influence arrow if point selected
            if (selectedIdx >= 0) {{
                const delta = computeInfluence(currentTheta, points, selectedIdx);
                const w2New = [currentTheta[0] + delta[0], currentTheta[1] + delta[1]];
                const zNew = computeLoss([w2New[0], w2New[1], b], points);

                // Arrow line
                traces.push({{
                    type: 'scatter3d',
                    x: [currentTheta[0], w2New[0]],
                    y: [currentTheta[1], w2New[1]],
                    z: [zHat, zNew],
                    mode: 'lines',
                    line: {{ color: 'lime', width: 6 }},
                    name: 'Influence direction'
                }});

                // Arrowhead using cone
                const dx = w2New[0] - currentTheta[0];
                const dy = w2New[1] - currentTheta[1];
                const dz = zNew - zHat;
                traces.push({{
                    type: 'cone',
                    x: [w2New[0]],
                    y: [w2New[1]],
                    z: [zNew],
                    u: [dx],
                    v: [dy],
                    w: [dz],
                    sizemode: 'absolute',
                    sizeref: 0.5,
                    anchor: 'tip',
                    colorscale: [[0, 'lime'], [1, 'lime']],
                    showscale: false,
                    name: 'Influence arrow'
                }});

                document.getElementById('info3d').innerHTML =
                    `Selected point ${{selectedIdx}} - <span style="color: #50fa7b;">green arrow</span> shows influence direction`;
            }} else {{
                document.getElementById('info3d').innerHTML = 'Click a point to see its influence direction';
            }}

            const layout = {{
                margin: {{ l: 0, r: 0, t: 0, b: 0 }},
                paper_bgcolor: '#0e1117',
                scene: {{
                    xaxis: {{ title: 'w₁', color: '#888', gridcolor: '#333' }},
                    yaxis: {{ title: 'w₂', color: '#888', gridcolor: '#333' }},
                    zaxis: {{ title: 'Loss', color: '#888', gridcolor: '#333' }},
                    bgcolor: '#0e1117'
                }},
                showlegend: false
            }};

            if (!plot3dInitialized) {{
                Plotly.newPlot('plot3d', traces, layout, {{ responsive: true }});
                plot3dInitialized = true;
            }} else {{
                Plotly.react('plot3d', traces, layout);
            }}
        }}

        // Initial draw
        draw2D();
    </script>
</body>
</html>
"""
    return html


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Influence manifold demo", layout="wide")

# Hide streamlit padding and minimize header
st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    header { visibility: hidden; }
    .stMainBlockContainer { padding-top: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data")
    seed = st.number_input("Random seed", value=0, step=1)
    n_per_class = st.slider("Points per class", 10, 100, 40)
    spread = st.slider("Cluster spread (std)", 0.2, 3.0, 1.0)
    sep = st.slider("Cluster separation", 0.0, 8.0, 3.0)

    st.header("Model")
    l2 = st.slider("L2 regularization (λ)", 1e-4, 1.0, 1e-2, format="%.4f")
    eps_up = st.slider("Influence step size (ε)", 0.01, 1.0, 0.2)

    st.header("Surface")
    grid = st.slider("Loss surface grid", 15, 60, 30)
    wlim = st.slider("w-range (±)", 2, 15, 6)

    if st.button("🔄 Regenerate dataset"):
        st.session_state.pop("data_params", None)

# Generate data
data_params = (int(seed), n_per_class, spread, sep)
if st.session_state.get("data_params") != data_params:
    X, y = generate_data(seed, n_per_class, spread, sep)
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.data_params = data_params

X = st.session_state.X
y = st.session_state.y

# Extract coordinates
points_x = [p[0] for p in X]
points_y = [p[1] for p in X]

# Create the full interactive app
app_html = create_full_app(points_x, points_y, y, l2, eps_up, wlim, grid)
components.html(app_html, height=780, scrolling=False)
