#!/usr/bin/env python3
"""
active_label.py

Interactive browser-based active labeling loop (one image at a time).

Place this file in the project root alongside:
  - embeddings.npy  (N x D numpy array)
  - paths.npy       (N length numpy array of file paths to the images)
  - optionally labels.csv (existing mappings)

Run:
  pip install flask numpy scikit-learn pillow tqdm faiss-cpu  # optional faiss
  python active_label.py

Open in browser: http://127.0.0.1:5000/

UI actions:
 - Type a label and Confirm (autocomplete suggestions from previous labels)
 - Click one of the top-3 suggestions
 - Trash (label becomes "trash")
 - Skip (leave unlabeled, move on)
"""

import os
import csv
import json
import time
import random
from pathlib import Path
from typing import Dict, List

from flask import Flask, send_file, request, jsonify, render_template_string

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# Optional faiss fallback
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# --------------- Config ---------------
EMBEDDINGS_FILE = "embeddings.npy"
PATHS_FILE = "paths.npy"
LABELS_FILE = "labels.csv"    # will be created/appended
PORT = 5000
HOST = "127.0.0.1"

# KNN settings
K_SUGGEST = 5     # used for suggestion ranking
K_CONF = 5        # used for confidence computation (neighbors considered)
RANDOM_SEED = 42

# UI settings
THUMBNAIL_SIZE = 360  # not used server-side, just recommendation for client

# --------------- Helpers & load data ---------------

def load_embeddings_and_paths(emb_file=EMBEDDINGS_FILE, paths_file=PATHS_FILE):
    if not os.path.exists(emb_file) or not os.path.exists(paths_file):
        raise FileNotFoundError(f"Missing {emb_file} or {paths_file} in working directory.")
    embeddings = np.load(emb_file)
    paths = np.load(paths_file, allow_pickle=True)
    if len(paths.shape) > 1:
        paths = paths.flatten()
    if embeddings.shape[0] != len(paths):
        raise ValueError("Number of embeddings and paths mismatch.")
    # Normalize embeddings for cosine via inner product
    embeddings = embeddings.astype(np.float32)
    embeddings = normalize(embeddings, norm='l2', axis=1)
    return embeddings, list(paths)

def load_existing_labels(labels_file=LABELS_FILE):
    mapping = {}  # path -> label
    if os.path.exists(labels_file):
        try:
            with open(labels_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        p, lab = row[0], row[1]
                        mapping[p] = lab
        except Exception:
            # fallback: try parse simple two-column format
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]
    return mapping

def append_label_to_csv(path, label, labels_file=LABELS_FILE):
    header_needed = not os.path.exists(labels_file)
    with open(labels_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["path","label"])
        writer.writerow([path, label])

# --------------- KNN index wrapper ---------------
class KNNIndex:
    def __init__(self, embeddings: np.ndarray, use_faiss=HAVE_FAISS):
        self.embeddings = embeddings
        self.n = embeddings.shape[0]
        self.dim = embeddings.shape[1]
        self.use_faiss = use_faiss and HAVE_FAISS
        self._build_index()

    def _build_index(self):
        if self.use_faiss:
            # Faiss inner product index (embeddings normalized)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings.astype('float32'))
        else:
            # sklearn NearestNeighbors with cosine metric
            self.nn = NearestNeighbors(n_neighbors=min(50, max(1, self.n)), metric='cosine')
            self.nn.fit(self.embeddings)

    def query(self, vectors: np.ndarray, k: int = 5):
        # returns distances, indices
        if self.use_faiss:
            vecs = vectors.astype('float32')
            D, I = self.index.search(vecs, k)
            # faiss returns inner-products; convert to cosine distance-like for compatibility
            # We'll return similarity (inner product) for suggestions
            return D, I
        else:
            # sklearn returns distances (cosine), indices
            D, I = self.nn.kneighbors(vectors, n_neighbors=k, return_distance=True)
            # convert cosine distance to similarity score = 1 - distance
            sim = 1.0 - D
            return sim, I

# --------------- Core active-learning logic ---------------

class ActiveLabeler:
    def __init__(self, embeddings: np.ndarray, paths: List[str], label_map: Dict[str,str]):
        self.embeddings = embeddings
        self.paths = paths
        self.N = len(paths)
        self.label_map = dict(label_map)   # path -> label
        self.index = KNNIndex(self.embeddings)
        self._rebuild_labeled_index()

        # bookkeeping for skipped items to avoid immediate re-showing
        self.skip_counts = {p:0 for p in self.paths}
        random.seed(RANDOM_SEED)

    def _rebuild_labeled_index(self):
        # Build small index of labeled items (for suggestions and k-NN propagation)
        labeled_paths = [p for p,lab in self.label_map.items() if lab is not None]
        self.labeled_idx = [self.paths.index(p) for p in labeled_paths if p in self.paths]
        self.labels = {self.paths.index(p): lab for p,lab in self.label_map.items() if p in self.paths}
        if len(self.labeled_idx) > 0:
            coords = self.embeddings[self.labeled_idx]
            # small NN over labeled set
            self.labeled_nn = NearestNeighbors(n_neighbors=min(len(self.labeled_idx), max(1, K_SUGGEST)), metric='cosine')
            self.labeled_nn.fit(coords)
        else:
            self.labeled_nn = None

    def suggest_for_index(self, idx, top_k=3):
        """
        Return list of (label, score) suggestions for sample idx based on nearest labeled neighbors.
        Score is average similarity (1 - cosine distance).
        """
        if self.labeled_nn is None:
            return []
        # find nearest labeled neighbors (in vector space of labeled coords)
        qvec = self.embeddings[idx].reshape(1, -1)
        dist, nbr_idx = self.labeled_nn.kneighbors(qvec, n_neighbors=min(len(self.labeled_idx), K_SUGGEST), return_distance=True)
        # nbr_idx are indices into self.labeled_idx list
        label_scores = {}
        for drow, nrow in zip(dist[0], nbr_idx[0]):
            # convert cosine distance -> similarity
            sim = 1.0 - float(drow)
            real_idx = self.labeled_idx[nrow]
            lab = self.labels.get(real_idx, None)
            if lab is None:
                continue
            label_scores.setdefault(lab, []).append(sim)
        # average per label and sort
        results = []
        for lab, sims in label_scores.items():
            avg = sum(sims)/len(sims)
            results.append((lab, avg))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compute_confidences(self, k=K_CONF):
        """
        For all unlabeled samples compute confidence based on neighbors among labeled set.
        confidence = fraction of k nearest labeled neighbors that agree on majority label
        Lower confidence => more uncertain (we will show them first)
        """
        unlabeled_indices = [i for i in range(self.N) if i not in self.labels]
        confidences = {}
        if self.labeled_nn is None:
            # no labeled examples -> uniform confidence (0.0) so random selection
            for idx in unlabeled_indices:
                confidences[idx] = 0.0
            return confidences

        # For each unlabeled, get k nearest labeled neighbors
        labeled_coords = self.embeddings[self.labeled_idx]
        qvecs = self.embeddings[unlabeled_indices]
        # reuse labeled_nn.kneighbors: distances (cosine) and indices into labeled list
        dist, nbr_idx = self.labeled_nn.kneighbors(qvecs, n_neighbors=min(len(self.labeled_idx), k), return_distance=True)
        for q_i, (drow, nrow) in enumerate(zip(dist, nbr_idx)):
            votes = {}
            for dval, nval in zip(drow, nrow):
                real_idx = self.labeled_idx[nval]
                lab = self.labels.get(real_idx, None)
                if lab is None:
                    continue
                votes[lab] = votes.get(lab, 0) + 1
            if len(votes) == 0:
                confidences[unlabeled_indices[q_i]] = 0.0
            else:
                majority = max(votes.values())
                conf = majority / sum(votes.values())
                confidences[unlabeled_indices[q_i]] = conf
        return confidences

    def pick_next_index(self, prefer_uncertain=True):
        """
        Pick the next unlabeled index:
         - If there are no labeled items -> random unlabeled
         - Else compute confidences and pick the lowest-confidence item
         - Respect skip counts to avoid immediately re-showing skipped items too often
        """
        unlabeled = [i for i in range(self.N) if i not in self.labels]
        if not unlabeled:
            return None
        if self.labeled_nn is None or not prefer_uncertain:
            # random pick (but prefer low skip_count)
            unlabeled.sort(key=lambda x: self.skip_counts.get(self.paths[x], 0))
            # pick from low-skip pool randomly
            candidates = [u for u in unlabeled if self.skip_counts.get(self.paths[u],0) <= 1]
            if not candidates:
                candidates = unlabeled
            return random.choice(candidates)
        # compute confidences
        confs = self.compute_confidences(k=K_CONF)
        # choose indices with minimal conf
        sorted_unl = sorted(confs.items(), key=lambda kv: (kv[1], self.skip_counts.get(self.paths[kv[0]], 0)))
        # pick first that hasn't been skipped too much
        for idx, conf in sorted_unl:
            if self.skip_counts.get(self.paths[idx], 0) <= 3:
                return idx
        # fallback
        return sorted_unl[0][0]

    def add_label(self, idx, label):
        path = self.paths[idx]
        self.label_map[path] = label
        self.labels[idx] = label
        append_label_to_csv(path, label)
        # rebuild labeled structures
        self._rebuild_labeled_index()

    def mark_skip(self, idx):
        path = self.paths[idx]
        self.skip_counts[path] = self.skip_counts.get(path, 0) + 1

    def mark_trash(self, idx):
        self.add_label(idx, "trash")

# --------------- Flask app ---------------
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable caching for development

# Global state filled at startup
EMBEDDINGS, PATHS = load_embeddings_and_paths()
LABEL_MAP = load_existing_labels()
LABELER = ActiveLabeler(EMBEDDINGS, PATHS, LABEL_MAP)

# small helper to return list of known labels for autocomplete
def known_labels():
    labs = set(LABELER.label_map.values())
    labs.discard(None)
    return sorted([l for l in labs if l != "trash"])

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Active Labeling</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background:#f6f8fa; }
    .container { display:flex; gap: 24px; }
    .left { width: 420px; }
    .right { flex:1; }
    img { max-width: 100%; height: auto; border: 1px solid #ccc; background: #fff; padding: 6px; }
    .buttons { margin-top: 12px; display:flex; gap:8px; flex-wrap:wrap; }
    button { padding:8px 12px; font-size:14px; }
    input[type=text] { width: 330px; padding:8px; font-size:14px; }
    .meta { margin-top:8px; color:#444; font-size:13px; }
    .suggest { margin-top:10px; }
    .small { font-size:13px; color:#666; }
  </style>
</head>
<body>
  <h2>Active Labeling — one image at a time</h2>
  <div class="container">
    <div class="left">
      <div id="imgbox">
        <img id="current_img" src="" alt="image to label">
      </div>
      <div class="meta" id="meta">Loading...</div>
      <div class="buttons">
        <button id="trash_btn">🗑️ Trash</button>
        <button id="skip_btn">❓ Skip</button>
      </div>

      <div style="margin-top:14px;">
        <form id="label_form" onsubmit="return false;">
          <input list="labels_list" id="label_input" placeholder="Type label... (autocomplete)">
          <datalist id="labels_list"></datalist>
          <button id="confirm_btn" style="margin-left:6px;">✅ Confirm</button>
        </form>
      </div>

      <div class="suggest" id="suggestions_area">
        <div class="small">Top suggestions:</div>
        <div id="suggest_buttons" style="margin-top:6px;"></div>
      </div>

      <div style="margin-top:12px;">
        <button id="done_btn">Finish / Show summary</button>
      </div>
    </div>

    <div class="right">
      <h4>Instructions</h4>
      <ul>
        <li>Label the image with the champion name (autocomplete available).</li>
        <li>Use suggestion buttons to accept a quick guess.</li>
        <li>Click <b>Trash</b> for artifacts (they become labeled "trash").</li>
        <li>Click <b>Skip</b> to postpone a difficult case (it will reappear later).</li>
      </ul>
      <h4>Session stats</h4>
      <div id="stats"></div>
      <h4>Recent labels</h4>
      <div id="recent"></div>
    </div>
  </div>

<script>
let current_idx = null;

async function loadNext() {
  const resp = await fetch('/next');
  const j = await resp.json();
  if (j['done']) {
    document.getElementById('meta').innerText = 'All images labeled (or nothing to do).';
    document.getElementById('current_img').style.display = 'none';
    document.getElementById('suggest_buttons').innerHTML = '';
    return;
  }
  current_idx = j['index'];
  const imgUrl = '/image?idx=' + current_idx + '&t=' + Date.now();
  document.getElementById('current_img').src = imgUrl;
  document.getElementById('current_img').style.display = 'block';
  document.getElementById('meta').innerText = `Index: ${current_idx} — path: ${j['path']}`;
  // fill suggestions
  const sugDiv = document.getElementById('suggest_buttons');
  sugDiv.innerHTML = '';
  if (j['suggestions'] && j['suggestions'].length>0) {
    for (const s of j['suggestions']) {
      const lab = s[0];
      const score = (100 * s[1]).toFixed(0);
      const btn = document.createElement('button');
      btn.innerText = lab + ' (' + score + '%)';
      btn.onclick = () => submitLabel(lab);
      sugDiv.appendChild(btn);
    }
  } else {
    sugDiv.innerHTML = '<span class="small">No suggestions (not enough labeled samples)</span>';
  }
  // fill autocomplete list
  const list = document.getElementById('labels_list');
  list.innerHTML = '';
  for (const lab of j['known_labels']) {
    const opt = document.createElement('option');
    opt.value = lab;
    list.appendChild(opt);
  }
  // stats
  document.getElementById('stats').innerText = `Labeled: ${j['n_labeled']} / ${j['n_total']}`;
  // recent
  let recentHtml = '';
  for (const r of j['recent']) {
    recentHtml += `<div style="margin-bottom:6px;"><img src="${r[0]}" width="64" style="vertical-align:middle;margin-right:8px;">${r[1]}</div>`;
  }
  document.getElementById('recent').innerHTML = recentHtml;
}

async function submitLabel(label) {
  if (label === undefined || label === null) {
    label = document.getElementById('label_input').value.trim();
  }
  if (!label) {
    alert("Please type a label or click a suggestion / Trash.");
    return;
  }
  await fetch('/label', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({index: current_idx, label: label})
  });
  document.getElementById('label_input').value = '';
  await loadNext();
}

document.getElementById('confirm_btn').addEventListener('click', () => submitLabel());
document.getElementById('trash_btn').addEventListener('click', () => {
  if (!confirm("Mark this image as TRASH?")) return;
  submitLabel('trash');
});
document.getElementById('skip_btn').addEventListener('click', async () => {
  await fetch('/skip', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({index: current_idx})});
  await loadNext();
});
document.getElementById('done_btn').addEventListener('click', async () => {
  const resp = await fetch('/summary');
  const j = await resp.json();
  alert(`Labeled ${j.n_labeled} / ${j.n_total}. Unique labels: ${j.unique_labels.join(', ')}`);
});

window.onload = () => {
  loadNext();
};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/image")
def serve_image():
    """
    Serve image by index. Query param: idx
    """
    idx = request.args.get('idx', None)
    if idx is None:
        return "Missing idx", 400
    try:
        i = int(idx)
        p = LABELER.paths[i]
        # security: make sure file exists
        if not os.path.exists(p):
            return "File not found", 404
        return send_file(p, conditional=True)
    except Exception as e:
        return str(e), 400

@app.route("/next")
def next_item():
    """
    Return JSON with next index to label, suggestions, known labels for autocomplete.
    """
    next_idx = LABELER.pick_next_index(prefer_uncertain=True)
    n_labeled = len(LABELER.labels)
    n_total = LABELER.N

    # gather recent labeled thumbnails (last 6)
    recent = []
    # read last lines from labels.csv if exists
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                rows = f.read().strip().splitlines()
                tail = rows[-12:] if len(rows)>12 else rows
                # parse and pick last labeled entries
                for line in tail[::-1][:6]:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        path = parts[0]
                        lab = parts[1]
                        # build thumbnail url (serve the image directly)
                        if os.path.exists(path):
                            recent.append( ("/image?idx=" + str(LABELER.paths.index(path)) + "&t=1", lab) )
        except Exception:
            pass

    if next_idx is None:
        return jsonify({"done": True, "n_labeled": n_labeled, "n_total": n_total, "known_labels": known_labels(), "recent": recent})

    suggestions = LABELER.suggest_for_index(next_idx, top_k=3)
    # suggestions is list of (label, score)
    return jsonify({
        "done": False,
        "index": int(next_idx),
        "path": LABELER.paths[next_idx],
        "suggestions": suggestions,
        "known_labels": known_labels(),
        "n_labeled": n_labeled,
        "n_total": n_total,
        "recent": recent
    })

@app.route("/label", methods=['POST'])
def label_item():
    """
    Receive JSON: {index: int, label: str}
    """
    data = request.get_json(force=True)
    idx = data.get('index', None)
    label = data.get('label', None)
    if idx is None or label is None:
        return jsonify({"ok": False, "error": "missing index/label"}), 400
    try:
        idx = int(idx)
        # sanitize label
        label = str(label).strip()
        LABELER.add_label(idx, label)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/skip", methods=['POST'])
def skip_item():
    data = request.get_json(force=True)
    idx = data.get('index', None)
    if idx is None:
        return jsonify({"ok": False, "error": "missing index"}), 400
    try:
        idx = int(idx)
        LABELER.mark_skip(idx)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/summary")
def summary():
    unique = sorted(list(set(LABELER.label_map.values())))
    return jsonify({"n_labeled": len(LABELER.labels), "n_total": LABELER.N, "unique_labels": unique})

# --------------- Main ---------------
if __name__ == "__main__":
    print("Starting active labeling server...")
    print("Embeddings:", EMBEDDINGS_FILE, "Paths:", PATHS_FILE)
    print("Labels file:", LABELS_FILE)
    print("Open your browser at http://%s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT)
