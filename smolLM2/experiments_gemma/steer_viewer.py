"""Interactive steering viewer — compare baseline vs SAE vs random for any feature.

Usage:
    python smolLM2/experiments_gemma/steer_viewer.py
"""
import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "/home/mac/infusion/infusion_hf/gemma3_4b/lora_smoltalk"
SAE_DIR = "/home/mac/infusion/infusion_hf/gemma3_4b/gradient_atoms_50k_topk50/sae_sae_32k"
PROJ_DIR = "/home/mac/infusion/infusion_hf/gemma3_4b/gradient_atoms_50k_topk50/projected_gradients"
FACTORS_DIR = "/home/mac/infusion/infusion_hf/gemma3_4b/ekfac_factors/gemma3_4b_lora/factors_gemma3_lora_factors"
CHAR_PATH = "/home/mac/infusion/infusion_hf/gemma3_4b/gradient_atoms_50k_topk50/sae_sae_32k"  # characterisations
device = "cuda:0"
PORT = 7862

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
    trust_remote_code=True, attn_implementation="flash_attention_2").to(device)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
model.eval()

print("Loading SAE...", flush=True)
sae = torch.load(os.path.join(SAE_DIR, "sae_results.pt"), weights_only=True, map_location="cpu")
W_dec = sae["dictionary"]; d_input = sae["d_input"]
get_col = lambda fi: W_dec[:, fi] if W_dec.shape[0] == d_input else W_dec[fi]

meta = torch.load(os.path.join(PROJ_DIR, "metadata.pt"), weights_only=True, map_location="cpu")
module_info = meta["module_info"]
act_evecs = load_file(os.path.join(FACTORS_DIR, "activation_eigenvectors.safetensors"))
grad_evecs = load_file(os.path.join(FACTORS_DIR, "gradient_eigenvectors.safetensors"))

# Load training docs for feature context
training_docs = {}
for char_dir in [CHAR_PATH, SAE_DIR]:
    docs_path = os.path.join(char_dir, "training_docs_compact.json")
    if os.path.exists(docs_path):
        with open(docs_path) as f:
            training_docs = json.load(f)
        break

# Load characterisations
atoms = []
for char_dir in [SAE_DIR, CHAR_PATH]:
    char_path = os.path.join(char_dir, "atom_characterisations.json")
    if os.path.exists(char_path):
        with open(char_path) as f:
            atoms = json.load(f)
        break
atom_map = {a["atom_idx"]: a for a in atoms}

def unproject(decoder_col):
    deltas = {}; po = 0
    for mi in module_info:
        k = mi["k"]; comp = decoder_col[po:po+k].clone()
        if k == 0: po += k; continue
        g_flat = torch.zeros(mi["d_out"]*mi["d_in"]); g_flat[mi["topk_idx"]] = comp
        g_eigen = g_flat.reshape(mi["d_out"], mi["d_in"])
        ek = mi["name"]
        if ek in grad_evecs:
            deltas[ek+".weight"] = grad_evecs[ek].float() @ g_eigen @ act_evecs[ek].float().T
        po += k
    return deltas

def steer_gen(deltas, prompt, alpha):
    originals = {}
    if alpha != 0:
        for name, param in model.named_parameters():
            ek = name.replace(".weight","").replace(".original_module","")+".weight"
            if ek in deltas:
                originals[name] = param.data.clone()
                param.data -= alpha * deltas[ek].to(param.device, param.dtype)
    inputs = tokenizer.apply_chat_template([{"role":"user","content":prompt}],
        tokenize=True, return_tensors="pt", add_generation_prompt=True).to(device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    resp = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    for name in originals:
        for n, p in model.named_parameters():
            if n == name: p.data = originals[name]
    return resp

# Cache unprojected vectors
cache = {}

print("Ready.", flush=True)

HTML = open(os.path.join(os.path.dirname(__file__), "steer_viewer.html")).read()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._respond(200, "text/html", HTML.encode())
        elif self.path == "/api/features":
            # Return feature list sorted by coherence (or n_active if no coherence)
            feat_list = []
            for a in sorted(atoms, key=lambda x: -(x.get("coherence", 0) or x.get("n_active", 0))):
                if a.get("n_active", 0) < 10: continue
                feat_list.append({
                    "idx": a["atom_idx"],
                    "coh": a.get("coherence", 0),
                    "active": a.get("n_active", 0),
                    "label": a.get("label", ""),
                    "dim": a.get("dimension", ""),
                    "kw": a.get("keywords", [])[:8],
                })
                if len(feat_list) >= 500: break
            self._respond(200, "application/json", json.dumps(feat_list).encode())
        elif self.path.startswith("/api/steer"):
            import urllib.parse
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            feat = int(params.get("feat", [0])[0])
            alpha = float(params.get("alpha", [5])[0])
            prompt = params.get("prompt", ["Tell me about something interesting."])[0]

            # Get or compute vectors
            if feat not in cache:
                dc = get_col(feat)
                torch.manual_seed(feat + 5000)
                rv = torch.randn(d_input); rv = rv / rv.norm() * dc.norm()
                cache[feat] = {"sae": unproject(dc), "random": unproject(rv)}

            # Generate all three
            baseline = steer_gen({}, prompt, 0)
            sae_resp = steer_gen(cache[feat]["sae"], prompt, alpha)
            rand_resp = steer_gen(cache[feat]["random"], prompt, alpha)

            # Feature info
            info = atom_map.get(feat, {})

            # Training docs for this feature
            feat_docs = []
            for idx in info.get("top_doc_indices", [])[:5]:
                key = str(idx)
                if key in training_docs:
                    d = training_docs[key]
                    feat_docs.append({"user": d.get("user","")[:500], "assistant": d.get("assistant","")[:800]})

            result = {
                "feat": feat, "alpha": alpha, "prompt": prompt,
                "baseline": baseline, "sae": sae_resp, "random": rand_resp,
                "info": {
                    "coherence": info.get("coherence", 0),
                    "n_active": info.get("n_active", 0),
                    "keywords": info.get("keywords", [])[:10],
                    "label": info.get("label", ""),
                },
                "docs": feat_docs,
            }
            self._respond(200, "application/json", json.dumps(result).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a): pass

print(f"Steering viewer at http://0.0.0.0:{PORT}")
HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
