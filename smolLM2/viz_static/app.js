var P = [];
var sel = -1;
var curRun = "";
var filterTerms = [];

document.getElementById("filterOut").addEventListener("input", function(){
  filterTerms = this.value.toLowerCase().split(",").map(function(s){ return s.trim(); }).filter(function(s){ return s.length > 0; });
  draw();
  var visible = 0;
  for (var i = 0; i < P.length; i++) if (!isFiltered(i)) visible++;
  document.getElementById("status").textContent = visible + "/" + P.length + " visible";
});

function isFiltered(i){
  if (filterTerms.length === 0) return false;
  var kws = P[i].kw.join(" ").toLowerCase();
  for (var j = 0; j < filterTerms.length; j++){
    if (kws.indexOf(filterTerms[j]) >= 0) return true;
  }
  return false;
}
var W, H, sc, ox, oy;

var cv = document.getElementById("c");
var ctx = cv.getContext("2d");
var tt = document.getElementById("tooltip");
var sb = document.getElementById("sidebar");
var infoEl = document.getElementById("info");
var docsEl = document.getElementById("docs");
var runSel = document.getElementById("runSelect");
var statusEl = document.getElementById("status");

// Load runs index
fetch("/api/runs").then(function(r){ return r.json(); }).then(function(runs){
  runs.forEach(function(r){
    var o = document.createElement("option");
    o.value = r.name;
    o.text = r.name + " (" + r.n + ")";
    runSel.appendChild(o);
  });
  if (runs.length > 0) loadRun(runs[0].name);
});

runSel.addEventListener("change", function(){ loadRun(runSel.value); });

function loadRun(name){
  statusEl.textContent = "Loading...";
  curRun = name;
  sel = -1;
  infoEl.innerHTML = '<h2 style="color:#666">Loading run...</h2>'; docsEl.innerHTML = "";
  fetch("/api/points/" + name).then(function(r){ return r.json(); }).then(function(data){
    P = data;
    statusEl.textContent = P.length + " features";
    resize();
  });
}

function resize(){
  var mapEl = cv.parentElement;
  W = mapEl.clientWidth;
  H = mapEl.clientHeight;
  cv.width = W * devicePixelRatio;
  cv.height = H * devicePixelRatio;
  cv.style.width = W + "px";
  cv.style.height = H + "px";
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  if (!P.length) return;

  var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (var i = 0; i < P.length; i++){
    if (P[i].x < x0) x0 = P[i].x;
    if (P[i].x > x1) x1 = P[i].x;
    if (P[i].y < y0) y0 = P[i].y;
    if (P[i].y > y1) y1 = P[i].y;
  }
  var pad = 50;
  sc = Math.min((W - 2*pad) / (x1 - x0 || 1), (H - 2*pad) / (y1 - y0 || 1));
  ox = pad + (W - 2*pad - (x1 - x0)*sc) / 2 - x0*sc;
  oy = pad + (H - 2*pad - (y1 - y0)*sc) / 2 - y0*sc;
  draw();
}

function tx(x){ return x * sc + ox; }
function ty(y){ return y * sc + oy; }

function draw(){
  ctx.clearRect(0, 0, W, H);
  for (var i = 0; i < P.length; i++){
    if (isFiltered(i) && i !== sel) continue;
    var p = P[i];
    var cx = tx(p.x), cy = ty(p.y);
    var r = Math.max(2, Math.min(10, Math.log(p.n + 1) * 0.8));
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = (i === sel) ? "#fff" : "rgb(" + p.r + "," + p.g + "," + p.b + ")";
    ctx.globalAlpha = (i === sel) ? 1 : 0.7;
    ctx.fill();
    ctx.globalAlpha = 1;
    // Gold outline based on distinctiveness
    if (p.dist > 0.1 || i === sel) {
      ctx.beginPath();
      ctx.arc(cx, cy, r + 1, 0, Math.PI * 2);
      if (i === sel) {
        ctx.strokeStyle = "#8bf";
        ctx.lineWidth = 2;
      } else {
        var gold_alpha = Math.min(0.9, p.dist);
        ctx.strokeStyle = "rgba(255,215,0," + gold_alpha + ")";
        ctx.lineWidth = Math.max(1, p.dist * 3);
      }
      ctx.stroke();
    }
  }
}

function findNearest(mx, my){
  var best = -1, bestD = 20;
  for (var i = 0; i < P.length; i++){
    if (isFiltered(i)) continue;
    var dx = tx(P[i].x) - mx;
    var dy = ty(P[i].y) - my;
    var d = Math.sqrt(dx*dx + dy*dy);
    if (d < bestD){ bestD = d; best = i; }
  }
  return best;
}

cv.addEventListener("mousemove", function(e){
  var rect = cv.getBoundingClientRect();
  var mx = e.clientX - rect.left;
  var my = e.clientY - rect.top;
  var i = findNearest(mx, my);
  if (i >= 0){
    var p = P[i];
    tt.style.display = "block";
    tt.style.left = (mx + 15) + "px";
    tt.style.top = (my - 10) + "px";
    var lbl = p.label ? "<br><i>" + p.label + "</i>" : "";
    tt.innerHTML = "<b>Feature " + p.feat + "</b>" + lbl +
      "<br>Coh: " + p.coh + " | Active: " + p.n.toLocaleString() +
      "<br>" + p.kw.join(", ");
  } else {
    tt.style.display = "none";
  }
});

cv.addEventListener("click", function(e){
  var rect = cv.getBoundingClientRect();
  var mx = e.clientX - rect.left;
  var my = e.clientY - rect.top;
  var i = findNearest(mx, my);
  statusEl.textContent = "Click at " + Math.round(mx) + "," + Math.round(my) + " -> idx=" + i;
  if (i >= 0) showFeat(i);
});

function showFeat(i){
  sel = i;
  var p = P[i];
  // sidebar always visible
  var labelHtml = p.label ? '<div style="color:#fd6;font-size:14px;margin-bottom:4px">' + p.label + '</div>' : '';
  var dimHtml = p.dim ? '<div style="color:#aaa;font-size:11px;margin-bottom:4px">Dimension: ' + p.dim + ' | Category: ' + (p.cat || '') + '</div>' : '';
  infoEl.innerHTML = "<h2>Feature " + p.feat + "</h2>" + labelHtml + dimHtml +
    '<div class="meta">Coherence: ' + p.coh + " | Active: " + p.n.toLocaleString() + "</div>" +
    '<div class="keywords">' + p.kw.join(", ") + "</div>";
  docsEl.innerHTML = '<div style="color:#888;padding:20px">Loading docs...</div>';
  draw();

  var url = "/api/docs/" + curRun + "/" + p.feat;
  statusEl.textContent = "Fetching " + url + "...";
  fetch(url)
    .then(function(r){
      statusEl.textContent = "Got response " + r.status;
      return r.json();
    })
    .then(function(data){
      statusEl.textContent = "Loaded " + (data.docs||[]).length + " docs";
      // Show TF-IDF distinctive terms
      var tfidf = data.tfidf || [];
      if (tfidf.length > 0) {
        var termsHtml = '<div style="margin-bottom:12px"><div class="role u" style="margin-bottom:4px">Distinctive terms (vs corpus)</div>';
        for (var ti = 0; ti < tfidf.length; ti++) {
          var t = tfidf[ti];
          var barW = Math.min(100, t.pct);
          termsHtml += '<div style="font-size:11px;margin:2px 0;display:flex;align-items:center;gap:6px">' +
            '<span style="width:80px;color:#eee">' + t.w + '</span>' +
            '<div style="flex:1;background:#333;border-radius:2px;height:10px">' +
            '<div style="width:' + barW + '%;background:rgba(255,215,0,0.6);height:100%;border-radius:2px"></div></div>' +
            '<span style="width:40px;text-align:right;color:#999;font-size:10px">' + t.pct + '%</span>' +
            '<span style="width:30px;text-align:right;color:#666;font-size:10px">' + t.ratio + 'x</span></div>';
        }
        termsHtml += '</div>';
        docsEl.insertAdjacentHTML("beforebegin", "");
        infoEl.innerHTML += termsHtml;
      }
      var freq = data.freq || {};
      docsEl.innerHTML = "";
      var docsList = data.docs || [];
      for (var di = 0; di < docsList.length; di++){
        var d = docsList[di];
        var div = document.createElement("div");
        div.className = "doc";

        var userHtml = highlightText(d.u || "", freq);
        var asstHtml = highlightText(d.a || "", freq);

        var inner = '<div class="role u">User (Doc ' + d.id + ')</div>';
        inner += '<div class="txt' + (di > 0 ? " c" : "") + '" data-idx="u' + di + '">' + userHtml + '</div>';
        if ((d.u || "").length > 300) inner += '<div class="expand">Show more</div>';
        inner += '<div class="role a">Assistant</div>';
        inner += '<div class="txt' + (di > 0 ? " c" : "") + '" data-idx="a' + di + '">' + asstHtml + '</div>';
        if ((d.a || "").length > 400) inner += '<div class="expand">Show more</div>';

        div.innerHTML = inner;

        // Attach expand handlers
        var expands = div.querySelectorAll(".expand");
        for (var j = 0; j < expands.length; j++){
          (function(btn){
            btn.addEventListener("click", function(e){
              e.stopPropagation();
              var txtEl = btn.previousElementSibling;
              if (txtEl && txtEl.classList.contains("txt")){
                txtEl.classList.toggle("open");
                btn.textContent = txtEl.classList.contains("open") ? "Show less" : "Show more";
              }
            });
          })(expands[j]);
        }

        docsEl.appendChild(div);
      }
    })
    .catch(function(err){
      docsEl.innerHTML = '<div style="color:#f66;padding:20px">Error: ' + err.message + '</div>';
    });

  // Don't resize here — it can cause the sidebar to flash away
}

function escapeHtml(s){
  if (!s) return "";
  var div = document.createElement("div");
  div.appendChild(document.createTextNode(s));
  var escaped = div.innerHTML;
  // Convert newlines to <br>
  return escaped.replace(/\n/g, "<br>");
}

function highlightText(s, freq){
  var escaped = escapeHtml(s);
  // freq scores are TF-IDF style: only words overrepresented in this feature vs corpus
  return escaped.replace(/([a-zA-Z]{3,})/g, function(word){
    var f = freq[word.toLowerCase()] || 0;
    if (f <= 0) return word;
    var alpha = Math.min(0.55, 0.15 + f * 0.5);
    var green = Math.round(200 * (1 - f * 0.5));
    return '<span class="hw" style="background:rgba(255,' + green + ',50,' + alpha + ')">' + word + '</span>';
  });
}

window.addEventListener("resize", resize);
