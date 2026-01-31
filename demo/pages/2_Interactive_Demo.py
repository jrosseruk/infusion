"""Page 3: Interactive 2D Influence Demo.

Refactored from the original figures/visualizer.py.
Replaces the 3D loss surface with an intuitive 2D probability bar chart.
Adds decision region shading, probe point star, and influence glow.
"""

import json

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Interactive Demo", page_icon=":joystick:", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .stMainBlockContainer { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


def generate_data(seed, n_per_class, spread, sep):
    """Generate a 2-class Gaussian dataset."""
    rng = np.random.default_rng(int(seed))
    mu0 = np.array([-sep / 2.0, 0.0])
    mu1 = np.array([+sep / 2.0, 0.0])
    X0 = rng.normal(loc=mu0, scale=spread, size=(n_per_class, 2))
    X1 = rng.normal(loc=mu1, scale=spread, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return X.tolist(), y.tolist()


def create_interactive_app(points_x, points_y, labels, l2):
    """Create the 2D interactive influence visualization."""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            height: 100%;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0e1117;
            color: #fafafa;
            overflow: hidden;
        }}
        .container {{
            display: flex;
            gap: 20px;
            padding: 10px;
            height: 100%;
        }}
        .panel {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        .panel-left {{ flex: 1.2; }}
        .panel-right {{ flex: 0.8; }}
        .panel h3 {{
            font-size: 14px;
            margin-bottom: 8px;
            color: #fafafa;
            flex-shrink: 0;
        }}
        #canvas-container {{
            position: relative;
            border: 1px solid #333;
            background: #1a1a2e;
            border-radius: 4px;
            flex: 1;
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
            flex-shrink: 0;
        }}
        #bar-container {{
            border: 1px solid #333;
            background: #1a1a2e;
            border-radius: 4px;
            flex: 1;
            position: relative;
        }}
        #barCanvas {{
            width: 100%;
            height: 100%;
        }}
        .info {{
            font-size: 11px;
            color: #888;
            margin-top: 8px;
            padding: 4px;
            flex-shrink: 0;
        }}
        .info strong {{ color: #4a9eff; }}

        /* Guided intro overlay */
        .intro-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.75);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .intro-card {{
            background: #1a1a2e;
            border: 1px solid #4a9eff;
            border-radius: 12px;
            padding: 2rem;
            max-width: 500px;
            text-align: center;
        }}
        .intro-card h2 {{ margin-bottom: 1rem; color: #4a9eff; font-size: 1.3rem; }}
        .intro-card p {{ margin-bottom: 1rem; color: #ccc; font-size: 0.95rem; line-height: 1.5; }}
        .intro-card .step {{ margin: 0.8rem 0; text-align: left; }}
        .intro-card .step-num {{
            display: inline-block;
            width: 24px; height: 24px;
            background: #4a9eff;
            color: #fff;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-size: 13px;
            font-weight: 700;
            margin-right: 8px;
        }}
        .intro-btn {{
            background: #4a9eff;
            color: #fff;
            border: none;
            padding: 10px 28px;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 0.5rem;
        }}
        .intro-btn:hover {{ background: #3a8eef; }}
    </style>
</head>
<body>
    <!-- Guided intro -->
    <div class="intro-overlay" id="introOverlay">
        <div class="intro-card">
            <h2>Interactive Influence Demo</h2>
            <p>See how individual training points influence a model's decision.</p>
            <div class="step">
                <span class="step-num">1</span>
                <strong>Blue</strong> and <strong style="color:#ff6b6b;">red</strong> dots are training data from two classes. The yellow line is the decision boundary.
            </div>
            <div class="step">
                <span class="step-num">2</span>
                Click a training point to select it. Its <strong style="color:#50fa7b;">influence</strong> on the probe point (star) is shown as a glow.
            </div>
            <div class="step">
                <span class="step-num">3</span>
                Drag any point to see the decision boundary and class probabilities update in real-time.
            </div>
            <button class="intro-btn" onclick="dismissIntro()">Got it</button>
        </div>
    </div>

    <div class="container">
        <div class="panel panel-left">
            <h3>Training data + decision boundary (drag points!)</h3>
            <div id="canvas-container">
                <canvas id="pointCanvas"></canvas>
                <div class="legend">
                    <div class="legend-item"><div class="legend-dot" style="background: #4a9eff;"></div>Class 0</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #ff6b6b;"></div>Class 1</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #50fa7b;"></div>Selected</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #ffd93d; width: 20px; height: 3px; border-radius: 0;"></div>Boundary</div>
                    <div class="legend-item"><div class="legend-dot" style="background: #fff; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);"></div>Probe</div>
                </div>
            </div>
            <div class="info" id="info">Drag any point to see changes in real-time</div>
        </div>
        <div class="panel panel-right">
            <h3>Probe point: P(class) prediction</h3>
            <div id="bar-container">
                <canvas id="barCanvas"></canvas>
            </div>
            <div class="info" id="infoBar">Shows how the model classifies the probe star</div>
        </div>
    </div>

    <script>
        // ============== INTRO ==============
        function dismissIntro() {{
            document.getElementById('introOverlay').style.display = 'none';
            try {{ sessionStorage.setItem('infusion_intro_seen', '1'); }} catch(e) {{}}
        }}
        try {{
            if (sessionStorage.getItem('infusion_intro_seen') === '1') {{
                document.getElementById('introOverlay').style.display = 'none';
            }}
        }} catch(e) {{}}

        // ============== DATA ==============
        let points = [];
        const pointsX = {json.dumps(points_x)};
        const pointsY = {json.dumps(points_y)};
        const labels = {json.dumps(labels)};
        const l2 = {l2};

        for (let i = 0; i < pointsX.length; i++) {{
            points.push({{ idx: i, x: pointsX[i], y: pointsY[i], label: labels[i] }});
        }}

        // Probe point: placed at origin initially
        let probeX = 0.0, probeY = 0.0;

        let selectedIdx = -1;
        let dragging = false;
        let dragIdx = -1;
        let draggingProbe = false;

        // ============== MATH ==============
        function sigmoid(t) {{
            return 1.0 / (1.0 + Math.exp(-t));
        }}

        function solve3x3(A, b) {{
            const det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                      - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                      + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);
            if (Math.abs(det) < 1e-10) {{
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
            let theta = [0, 0, 0];
            for (let iter = 0; iter < 30; iter++) {{
                const w = [theta[0], theta[1]];
                const b = theta[2];
                const p = pts.map(pt => sigmoid(pt.x * w[0] + pt.y * w[1] + b));
                let grad = [0, 0, 0];
                for (let i = 0; i < n; i++) {{
                    const r = p[i] - pts[i].label;
                    grad[0] += r * pts[i].x / n;
                    grad[1] += r * pts[i].y / n;
                    grad[2] += r / n;
                }}
                grad[0] += l2 * w[0];
                grad[1] += l2 * w[1];
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
                const step = solve3x3(H, grad);
                theta = [theta[0] - step[0], theta[1] - step[1], theta[2] - step[2]];
                if (Math.sqrt(step[0]**2 + step[1]**2 + step[2]**2) < 1e-8) break;
            }}
            return theta;
        }}

        function computeInfluence(theta, pts, idx) {{
            const n = pts.length;
            const w = [theta[0], theta[1]];
            const b = theta[2];
            const xi = pts[idx];
            const pi = sigmoid(xi.x * w[0] + xi.y * w[1] + b);
            const g = pi - xi.label;
            const gradI = [g * xi.x, g * xi.y, g];
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
            return [-step[0], -step[1], -step[2]];
        }}

        // ============== 2D CANVAS ==============
        const canvas = document.getElementById('pointCanvas');
        const ctx = canvas.getContext('2d');
        const container = document.getElementById('canvas-container');

        const barCanvas = document.getElementById('barCanvas');
        const barCtx = barCanvas.getContext('2d');
        const barContainer = document.getElementById('bar-container');

        function resizeCanvas() {{
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            barCanvas.width = barContainer.clientWidth;
            barCanvas.height = barContainer.clientHeight;
        }}
        resizeCanvas();
        window.addEventListener('resize', () => {{ resizeCanvas(); drawAll(); }});

        function getDataBounds() {{
            let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
            for (const p of points) {{
                xMin = Math.min(xMin, p.x); xMax = Math.max(xMax, p.x);
                yMin = Math.min(yMin, p.y); yMax = Math.max(yMax, p.y);
            }}
            xMin = Math.min(xMin, probeX); xMax = Math.max(xMax, probeX);
            yMin = Math.min(yMin, probeY); yMax = Math.max(yMax, probeY);
            const padX = (xMax - xMin) * 0.15 + 0.5;
            const padY = (yMax - yMin) * 0.15 + 0.5;
            return {{ xMin: xMin - padX, xMax: xMax + padX, yMin: yMin - padY, yMax: yMax + padY }};
        }}

        const PADDING = 40;

        function dataToCanvas(dx, dy, bounds) {{
            const cx = PADDING + (dx - bounds.xMin) / (bounds.xMax - bounds.xMin) * (canvas.width - 2 * PADDING);
            const cy = canvas.height - PADDING - (dy - bounds.yMin) / (bounds.yMax - bounds.yMin) * (canvas.height - 2 * PADDING);
            return [cx, cy];
        }}

        function canvasToData(cx, cy, bounds) {{
            const dx = bounds.xMin + (cx - PADDING) / (canvas.width - 2 * PADDING) * (bounds.xMax - bounds.xMin);
            const dy = bounds.yMin + (canvas.height - PADDING - cy) / (canvas.height - 2 * PADDING) * (bounds.yMax - bounds.yMin);
            return [dx, dy];
        }}

        let currentTheta = [0, 0, 0];

        function drawDecisionRegions(bounds) {{
            const w = canvas.width;
            const h = canvas.height;
            const imgData = ctx.getImageData(0, 0, w, h);
            const data = imgData.data;
            const theta = currentTheta;

            // Sample every 4 pixels for performance
            for (let py = PADDING; py < h - PADDING; py += 3) {{
                for (let px = PADDING; px < w - PADDING; px += 3) {{
                    const [dx, dy] = canvasToData(px, py, bounds);
                    const z = theta[0] * dx + theta[1] * dy + theta[2];
                    const p1 = sigmoid(z);

                    // Fill a 3x3 block
                    for (let oy = 0; oy < 3 && py + oy < h; oy++) {{
                        for (let ox = 0; ox < 3 && px + ox < w; ox++) {{
                            const idx = ((py + oy) * w + (px + ox)) * 4;
                            if (p1 > 0.5) {{
                                // Class 1 region: light red
                                data[idx] = 40;
                                data[idx + 1] = 20;
                                data[idx + 2] = 25;
                                data[idx + 3] = 255;
                            }} else {{
                                // Class 0 region: light blue
                                data[idx] = 20;
                                data[idx + 1] = 25;
                                data[idx + 2] = 45;
                                data[idx + 3] = 255;
                            }}
                        }}
                    }}
                }}
            }}
            ctx.putImageData(imgData, 0, 0);
        }}

        function drawStar(cx, cy, r, spikes, color) {{
            ctx.beginPath();
            const rot = -Math.PI / 2;
            for (let i = 0; i < spikes * 2; i++) {{
                const radius = i % 2 === 0 ? r : r * 0.45;
                const angle = rot + (i * Math.PI) / spikes;
                const x = cx + Math.cos(angle) * radius;
                const y = cy + Math.sin(angle) * radius;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }}

        function draw2D() {{
            const bounds = getDataBounds();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Fit model
            currentTheta = fitLogReg(points);
            const w = [currentTheta[0], currentTheta[1]];
            const b = currentTheta[2];

            // Decision regions
            drawDecisionRegions(bounds);

            // Decision boundary
            ctx.strokeStyle = '#ffd93d';
            ctx.lineWidth = 2.5;
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

            // Influence glow for selected point
            if (selectedIdx >= 0) {{
                const infDelta = computeInfluence(currentTheta, points, selectedIdx);
                const infMag = Math.sqrt(infDelta[0]**2 + infDelta[1]**2);
                const glowRadius = Math.min(60, Math.max(15, infMag * 200));

                const sp = points[selectedIdx];
                const [scx, scy] = dataToCanvas(sp.x, sp.y, bounds);

                // Glow
                const gradient = ctx.createRadialGradient(scx, scy, 0, scx, scy, glowRadius);
                gradient.addColorStop(0, 'rgba(80, 250, 123, 0.4)');
                gradient.addColorStop(1, 'rgba(80, 250, 123, 0)');
                ctx.beginPath();
                ctx.arc(scx, scy, glowRadius, 0, Math.PI * 2);
                ctx.fillStyle = gradient;
                ctx.fill();

                // Influence direction arrow from probe
                const [pcx, pcy] = dataToCanvas(probeX, probeY, bounds);
                const arrowScale = 150;
                const ax = pcx + infDelta[0] * arrowScale;
                const ay = pcy - infDelta[1] * arrowScale;

                ctx.strokeStyle = '#50fa7b';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.moveTo(pcx, pcy);
                ctx.lineTo(ax, ay);
                ctx.stroke();

                // Arrowhead
                const angle = Math.atan2(ay - pcy, ax - pcx);
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(ax - 10 * Math.cos(angle - 0.4), ay - 10 * Math.sin(angle - 0.4));
                ctx.lineTo(ax - 10 * Math.cos(angle + 0.4), ay - 10 * Math.sin(angle + 0.4));
                ctx.closePath();
                ctx.fillStyle = '#50fa7b';
                ctx.fill();
            }}

            // Draw training points
            for (let i = 0; i < points.length; i++) {{
                const p = points[i];
                const [cx, cy] = dataToCanvas(p.x, p.y, bounds);

                ctx.beginPath();
                ctx.arc(cx, cy, i === selectedIdx ? 12 : 8, 0, Math.PI * 2);

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

            // Probe point (star)
            const [pcx, pcy] = dataToCanvas(probeX, probeY, bounds);
            drawStar(pcx, pcy, 16, 5, '#ffffff');

            // Axis labels
            ctx.fillStyle = '#888';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('x\\u2081', canvas.width / 2, canvas.height - 8);
            ctx.save();
            ctx.translate(12, canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('x\\u2082', 0, 0);
            ctx.restore();

            // Info
            document.getElementById('info').innerHTML =
                `<strong>w</strong> = (${{w[0].toFixed(3)}}, ${{w[1].toFixed(3)}}), <strong>b</strong> = ${{b.toFixed(3)}}` +
                (selectedIdx >= 0 ? ` | Selected: point ${{selectedIdx}}` : ' | Click a point to see its influence');
        }}

        function drawBarChart() {{
            const bw = barCanvas.width;
            const bh = barCanvas.height;
            barCtx.clearRect(0, 0, bw, bh);

            // Background
            barCtx.fillStyle = '#1a1a2e';
            barCtx.fillRect(0, 0, bw, bh);

            const theta = currentTheta;
            const z = theta[0] * probeX + theta[1] * probeY + theta[2];
            const p1 = sigmoid(z);
            const p0 = 1 - p1;

            const probs = [p0, p1];
            const classNames = ['Class 0', 'Class 1'];
            const colors = ['#4a9eff', '#ff6b6b'];

            const marginLeft = 80;
            const marginRight = 60;
            const marginTop = 40;
            const barHeight = 40;
            const gap = 30;
            const maxBarWidth = bw - marginLeft - marginRight;

            barCtx.font = '13px sans-serif';

            for (let i = 0; i < 2; i++) {{
                const y = marginTop + i * (barHeight + gap);
                const barWidth = probs[i] * maxBarWidth;

                // Bar background
                barCtx.fillStyle = '#333';
                barCtx.fillRect(marginLeft, y, maxBarWidth, barHeight);

                // Bar fill
                barCtx.fillStyle = colors[i];
                barCtx.fillRect(marginLeft, y, barWidth, barHeight);

                // Bar border
                barCtx.strokeStyle = 'rgba(255,255,255,0.2)';
                barCtx.lineWidth = 1;
                barCtx.strokeRect(marginLeft, y, maxBarWidth, barHeight);

                // Label
                barCtx.fillStyle = '#fafafa';
                barCtx.textAlign = 'right';
                barCtx.fillText(classNames[i], marginLeft - 10, y + barHeight / 2 + 5);

                // Value
                barCtx.textAlign = 'left';
                barCtx.fillStyle = '#fafafa';
                barCtx.fillText((probs[i] * 100).toFixed(1) + '%', marginLeft + barWidth + 8, y + barHeight / 2 + 5);
            }}

            // Prediction text
            const pred = p1 > 0.5 ? 1 : 0;
            const conf = Math.max(p0, p1);
            barCtx.font = '15px sans-serif';
            barCtx.textAlign = 'center';
            barCtx.fillStyle = colors[pred];
            barCtx.fillText(
                `Prediction: ${{classNames[pred]}} (${{(conf * 100).toFixed(1)}}%)`,
                bw / 2,
                marginTop + 2 * (barHeight + gap) + 20
            );

            // Influence info
            if (selectedIdx >= 0) {{
                const infDelta = computeInfluence(currentTheta, points, selectedIdx);
                const infMag = Math.sqrt(infDelta[0]**2 + infDelta[1]**2);

                barCtx.font = '12px sans-serif';
                barCtx.fillStyle = '#50fa7b';
                barCtx.textAlign = 'center';
                barCtx.fillText(
                    `Influence magnitude: ${{infMag.toFixed(4)}}`,
                    bw / 2,
                    marginTop + 2 * (barHeight + gap) + 50
                );

                // Show what would happen if this point were removed
                const thetaNew = [
                    currentTheta[0] + infDelta[0],
                    currentTheta[1] + infDelta[1],
                    currentTheta[2] + infDelta[2]
                ];
                const zNew = thetaNew[0] * probeX + thetaNew[1] * probeY + thetaNew[2];
                const p1New = sigmoid(zNew);
                const deltaP = p1New - p1;

                barCtx.fillStyle = '#aaa';
                barCtx.fillText(
                    `Upweighting shifts P(class 1) by ${{deltaP >= 0 ? '+' : ''}}${{(deltaP * 100).toFixed(2)}}%`,
                    bw / 2,
                    marginTop + 2 * (barHeight + gap) + 72
                );
            }}

            document.getElementById('infoBar').innerHTML =
                `Probe at (${{probeX.toFixed(2)}}, ${{probeY.toFixed(2)}}) | P(class 0) = ${{(p0*100).toFixed(1)}}%, P(class 1) = ${{(p1*100).toFixed(1)}}%`;
        }}

        function drawAll() {{
            draw2D();
            drawBarChart();
        }}

        // ============== MOUSE EVENTS ==============
        function findPoint(cx, cy) {{
            const bounds = getDataBounds();
            for (let i = 0; i < points.length; i++) {{
                const [px, py] = dataToCanvas(points[i].x, points[i].y, bounds);
                if (Math.sqrt((cx - px)**2 + (cy - py)**2) < 18) return i;
            }}
            return -1;
        }}

        function isNearProbe(cx, cy) {{
            const bounds = getDataBounds();
            const [px, py] = dataToCanvas(probeX, probeY, bounds);
            return Math.sqrt((cx - px)**2 + (cy - py)**2) < 20;
        }}

        canvas.addEventListener('mousedown', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
            const cy = (e.clientY - rect.top) * (canvas.height / rect.height);

            if (isNearProbe(cx, cy)) {{
                draggingProbe = true;
                return;
            }}

            const idx = findPoint(cx, cy);
            if (idx >= 0) {{
                selectedIdx = idx;
                dragIdx = idx;
                dragging = true;
                drawAll();
            }}
        }});

        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
            const cy = (e.clientY - rect.top) * (canvas.height / rect.height);
            const bounds = getDataBounds();

            if (draggingProbe) {{
                const [dx, dy] = canvasToData(cx, cy, bounds);
                probeX = dx;
                probeY = dy;
                drawAll();
                return;
            }}

            if (dragging && dragIdx >= 0) {{
                const [dx, dy] = canvasToData(cx, cy, bounds);
                points[dragIdx].x = dx;
                points[dragIdx].y = dy;
                drawAll();
            }}
        }});

        canvas.addEventListener('mouseup', () => {{
            dragging = false;
            dragIdx = -1;
            draggingProbe = false;
        }});

        canvas.addEventListener('mouseleave', () => {{
            dragging = false;
            dragIdx = -1;
            draggingProbe = false;
        }});

        // Initial draw
        drawAll();
    </script>
</body>
</html>
"""
    return html


# ----------------------------
# Streamlit UI
# ----------------------------
with st.sidebar:
    st.header("Data")
    seed = st.number_input("Random seed", value=0, step=1)
    n_per_class = st.slider("Points per class", 10, 100, 40)
    spread = st.slider("Cluster spread (std)", 0.2, 3.0, 1.0)
    sep = st.slider("Cluster separation", 0.0, 8.0, 3.0)

    st.header("Model")
    l2 = st.slider("L2 regularization", 1e-4, 1.0, 1e-2, format="%.4f")

    if st.button("Regenerate dataset"):
        st.session_state.pop("demo_data_params", None)

# Generate data
data_params = (int(seed), n_per_class, spread, sep)
if st.session_state.get("demo_data_params") != data_params:
    X, y = generate_data(seed, n_per_class, spread, sep)
    st.session_state.demo_X = X
    st.session_state.demo_y = y
    st.session_state.demo_data_params = data_params

X = st.session_state.demo_X
y = st.session_state.demo_y

points_x = [p[0] for p in X]
points_y = [p[1] for p in X]

app_html = create_interactive_app(points_x, points_y, y, l2)
components.html(app_html, height=700, scrolling=False)
