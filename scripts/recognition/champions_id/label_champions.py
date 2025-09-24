#!/usr/bin/env python3
"""
cluster_and_label.py

Workflow:
1. Load all images from output/board and output/bench
2. Extract image embeddings with pretrained ResNet50 (torch)
3. Dimensionality reduction (PCA) and clustering (HDBSCAN if available, else DBSCAN)
4. Create cluster folders and an HTML gallery to inspect clusters
5. Optionally run a tiny labeling server (Flask) so you can type one label per cluster
6. Save labels.csv and build a FAISS index (if faiss available) for fast lookup of new images

Requirements:
  pip install torch torchvision pillow numpy scikit-learn faiss-cpu flask hdbscan tqdm

If you don't have hdbscan/faiss you can still run the script; it will fall back to DBSCAN and skip index building.

Usage:
  python cluster_and_label.py             # runs the pipeline and writes clusters + gallery
  python cluster_and_label.py --serve     # also starts a web server (http://127.0.0.1:5000) for labeling
  python cluster_and_label.py --index-only path/to/new_image.png  # compute embedding and query index

Author: assistant
"""
import os
import sys
import argparse
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# ---- ML libs ----
try:
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
except Exception as e:
    print("ERROR: torch/torchvision required. Install with `pip install torch torchvision`.")
    raise

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# optional libs
try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# optional Flask for labeling server
try:
    from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory
    HAVE_FLASK = True
except Exception:
    HAVE_FLASK = False

# ------------------------
# Configurable paths / params
# ------------------------
INPUT_DIRS = ["output/board", "output/bench"]   # where your crops are
CLUSTERS_DIR = "clusters"                       # clusters output
GALLERY_HTML = "clusters_gallery.html"
LABELS_CSV = "cluster_labels.csv"
EMBEDDINGS_NPY = "embeddings.npy"
PATHS_NPY = "paths.npy"
FAISS_INDEX_FILE = "faiss.index"

BATCH_SIZE = 32
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Clustering params
PCA_DIM = 64           # reduce to this before clustering
USE_HDBSCAN = HAVE_HDBSCAN
DBSCAN_EPS = 0.6       # fallback param
MIN_SAMPLES = 2

# gallery thumbnail size
THUMBNAIL_MAX = 220

# ------------------------
# Utilities
# ------------------------
def gather_image_paths(input_dirs):
    exts = (".png", ".jpg", ".jpeg")
    paths = []
    for d in input_dirs:
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            if entry.lower().endswith(exts):
                paths.append(os.path.join(d, entry))
    return sorted(paths)

# image loader + transform for resnet
def make_transform(image_size=IMAGE_SIZE):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
    ])
    return transform

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

# ResNet feature extractor (remove final fc)
class FeatureExtractor:
    def __init__(self, device=DEVICE):
        self.device = device
        model = models.resnet50(pretrained=True)
        # remove final fc
        modules = list(model.children())[:-1]  # output: (batch, 2048, 1, 1)
        self.model = torch.nn.Sequential(*modules).to(self.device).eval()
        self.transform = make_transform()

    @torch.no_grad()
    def embed_batch(self, pil_images):
        # pil_images: list of PIL images
        tensors = [self.transform(img) for img in pil_images]
        x = torch.stack(tensors).to(self.device)
        feats = self.model(x)  # (B, 2048, 1, 1)
        feats = feats.view(feats.size(0), -1).cpu().numpy()
        # normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats = feats / norms
        return feats

# ------------------------
# Main pipeline functions
# ------------------------
def compute_embeddings(paths, extractor, batch_size=BATCH_SIZE, cache_path=None):
    n = len(paths)
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        arr = np.load(cache_path, allow_pickle=False)
        return arr
    embeddings = []
    for i in tqdm(range(0, n, batch_size), desc="Embedding batches"):
        batch_paths = paths[i:i+batch_size]
        imgs = [load_image(p) for p in batch_paths]
        feats = extractor.embed_batch(imgs)
        embeddings.append(feats)
    embeddings = np.vstack(embeddings)
    if cache_path:
        np.save(cache_path, embeddings)
    return embeddings

def reduce_dim(embeddings, pca_dim=PCA_DIM):
    print(f"Reducing dimensionality to {pca_dim} with PCA...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=min(pca_dim, emb_scaled.shape[1]), random_state=42)
    reduced = pca.fit_transform(emb_scaled)
    return reduced, pca, scaler

def cluster_embeddings(reduced, use_hdbscan=USE_HDBSCAN, db_eps=DBSCAN_EPS, min_samples=MIN_SAMPLES, metric="euclidean"):
    print("Clustering embeddings...")
    if use_hdbscan:
        print(f"Using HDBSCAN (preferred). Metric={metric}")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric=metric, prediction_data=True)
        labels = clusterer.fit_predict(reduced)
        return labels, ("hdbscan", None)
    else:
        print(f"HDBSCAN not available → using DBSCAN fallback. Metric={metric}")
        clusterer = DBSCAN(eps=db_eps, min_samples=min_samples, metric=metric).fit(reduced)
        labels = clusterer.labels_
        return labels, ("dbscan", db_eps)


def make_cluster_folders(paths, labels, out_dir=CLUSTERS_DIR):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    clusters = {}
    for p, lab in zip(paths, labels):
        lab_str = f"cluster_{int(lab)}" if lab != -1 else "cluster_noise"
        cluster_path = os.path.join(out_dir, lab_str)
        os.makedirs(cluster_path, exist_ok=True)
        # copy or link thumbnail to cluster folder
        dst = os.path.join(cluster_path, os.path.basename(p))
        try:
            shutil.copy2(p, dst)
        except Exception:
            # fallback to image write
            img = Image.open(p).convert("RGB")
            img.save(dst)
        clusters.setdefault(lab, []).append(dst)
    return clusters

def make_thumbnails(cluster_dir, thumb_size=THUMBNAIL_MAX):
    # create small versions in each cluster folder (overwrite)
    for root, _, files in os.walk(cluster_dir):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                img = Image.open(path)
                img.thumbnail((thumb_size, thumb_size))
                thumb_path = os.path.join(root, "thumb__" + fname)
                img.save(thumb_path)
            except Exception:
                pass

def generate_html_gallery(clusters, out_html=GALLERY_HTML, clusters_dir=CLUSTERS_DIR):
    # clusters: dict label -> list(filepaths)
    html_parts = []
    html_parts.append("<html><head><meta charset='utf-8'><title>Clusters gallery</title>"
                      "<style>body{font-family:Arial}.cluster{margin:12px;padding:8px;border:1px solid #ddd}"
                      ".thumb{margin:3px;border:1px solid #666;}</style></head><body>")
    html_parts.append("<h1>Cluster gallery</h1>")
    html_parts.append("<p>Each cluster row shows thumbnails; rename cluster by editing the input and clicking Save.</p>")
    html_parts.append("<form id='labelForm' method='post' action='save_labels'>")
    html_parts.append("<table>")
    # sort cluster keys for deterministic order, noise last
    keys = sorted([k for k in clusters.keys() if k != -1]) + ([-1] if -1 in clusters else [])
    for lab in keys:
        files = clusters[lab]
        html_parts.append("<tr class='cluster'><td valign='top'>")
        html_parts.append(f"<b>Cluster {lab}</b><br><small>{len(files)} images</small></td><td>")
        for f in files[:50]:  # show up to 50 thumbs
            fn = os.path.basename(f)
            thumb = os.path.join(os.path.basename(os.path.dirname(f)), "thumb__" + fn)
            thumb_path = os.path.join(CLUSTERS_DIR, os.path.basename(os.path.dirname(f)), "thumb__" + fn)
            # Use relative path
            rel = thumb_path.replace("\\", "/")
            html_parts.append(f"<img src='{rel}' class='thumb' width=90>")
        html_parts.append("</td><td valign='top'>")
        html_parts.append(f"<input type='text' name='label_{lab}' placeholder='type label for cluster {lab}' />")
        html_parts.append("</td></tr>")
    html_parts.append("</table>")
    html_parts.append("<input type='submit' value='Save labels' />")
    html_parts.append("</form>")
    html_parts.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Wrote gallery HTML to {out_html}")

def save_labels_csv(mapping, out_csv=LABELS_CSV):
    # mapping: cluster_label -> name
    lines = ["cluster,label"]
    for k, v in mapping.items():
        lines.append(f"{k},{v}")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved cluster->label mapping to {out_csv}")

def build_faiss_index(embeddings, labels_map=None, index_file=FAISS_INDEX_FILE):
    if not HAVE_FAISS:
        print("Faiss not available; skipping index build.")
        return None
    d = embeddings.shape[1]
    print(f"Building FAISS index of dim {d} ...")
    index = faiss.IndexFlatIP(d)  # inner product if embeddings are normalized
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_file)
    print(f"Wrote faiss index to {index_file}")
    return index

# ------------------------
# Flask app for labeling (optional)
# ------------------------
FLASK_APP = None
if HAVE_FLASK:
    app = Flask(__name__)
else:
    app = None

def start_label_server(clusters_dir=CLUSTERS_DIR, gallery_html=GALLERY_HTML):
    if not HAVE_FLASK:
        print("Flask not installed. Install with `pip install flask` to enable labeling server.")
        return
    # Serve static gallery file and handle form submit
    @app.route("/")
    def index():
        return send_from_directory(".", gallery_html)

    @app.route("/save_labels", methods=["POST"])
    def save_labels():
        mapping = {}
        for key, val in request.form.items():
            if key.startswith("label_") and val.strip():
                lab = int(key.replace("label_", ""))
                mapping[lab] = val.strip()
        if mapping:
            save_labels_csv(mapping)
            # Apply mapping: create folders named by label and move files
            for lab, name in mapping.items():
                from_dir = os.path.join(CLUSTERS_DIR, f"cluster_{lab}" if lab != -1 else "cluster_noise")
                if os.path.isdir(from_dir):
                    to_dir = os.path.join("labeled", name)
                    os.makedirs(to_dir, exist_ok=True)
                    for fn in os.listdir(from_dir):
                        if fn.startswith("thumb__"):
                            continue
                        src = os.path.join(from_dir, fn)
                        dst = os.path.join(to_dir, fn)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
            return redirect(url_for("index"))
        return redirect(url_for("index"))

    @app.route("/clusters/<path:filename>")
    def clusters_static(filename):
        # serve from clusters directory for thumbs
        return send_from_directory(CLUSTERS_DIR, filename)

    print("Starting labeling server at http://127.0.0.1:5000 (open in browser).")
    app.run(debug=False, port=5000)

# ------------------------
# End-to-end run
# ------------------------
def run_pipeline(args):
    # 0. collect images
    paths = gather_image_paths(INPUT_DIRS)
    if len(paths) == 0:
        print("No images found in input dirs:", INPUT_DIRS)
        return

    print(f"Found {len(paths)} images. Device: {DEVICE}")

    # 1. feature extractor
    extractor = FeatureExtractor(device=DEVICE)

    # 2. compute embeddings (cache to file)
    if os.path.exists(EMBEDDINGS_NPY) and os.path.exists(PATHS_NPY) and not args.force:
        print("Loading embeddings cache.")
        embeddings = np.load(EMBEDDINGS_NPY)
        saved_paths = np.load(PATHS_NPY)
        if list(saved_paths) != paths:
            print("Warning: paths changed since cache; recomputing embeddings.")
            embeddings = compute_embeddings(paths, extractor, cache_path=EMBEDDINGS_NPY)
            np.save(PATHS_NPY, np.array(paths))
    else:
        embeddings = compute_embeddings(paths, extractor, cache_path=EMBEDDINGS_NPY)
        np.save(PATHS_NPY, np.array(paths))

    # 3. reduce dim
    reduced, pca, scaler = reduce_dim(embeddings, pca_dim=args.pca_dim)

    # 4. clustering
    labels, info = cluster_embeddings(
        reduced,
        use_hdbscan=args.hdbscan and HAVE_HDBSCAN,
        db_eps=args.db_eps,
        min_samples=args.min_samples,
        metric=args.metric
    )
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clustering done. Unique labels: {len(set(labels))}. (noise = -1). Found {n_clusters} clusters (excluding noise).")

    # 5. write cluster folders
    clusters = make_cluster_folders(paths, labels, out_dir=CLUSTERS_DIR)

    # 6. make thumbnails
    make_thumbnails(CLUSTERS_DIR)

    # 7. write gallery HTML
    generate_html_gallery(clusters, out_html=GALLERY_HTML, clusters_dir=CLUSTERS_DIR)

    # 8. build faiss index
    if HAVE_FAISS:
        build_faiss_index(embeddings)

    print("Pipeline finished. Open", GALLERY_HTML, "in a browser to inspect clusters.")
    if args.serve:
        start_label_server()

# ------------------------
# CLI
# ------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Cluster champion crops and label clusters quickly.")
    p.add_argument("--serve", action="store_true", help="Start labeling server after creating gallery (requires Flask).")
    p.add_argument("--pca-dim", type=int, default=PCA_DIM, help="PCA dim for reduction.")
    p.add_argument("--hdbscan", action="store_true", help="Prefer HDBSCAN if installed.")
    p.add_argument("--db-eps", type=float, default=DBSCAN_EPS, help="DBSCAN eps fallback.")
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES, help="min_samples for DBSCAN/HDBSCAN.")
    p.add_argument("--metric", type=str, default="euclidean", help="Distance metric: euclidean, cosine, manhattan, etc.")
    p.add_argument("--force", action="store_true", help="Force recomputing embeddings even if cache exists.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
