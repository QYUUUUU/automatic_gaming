#!/usr/bin/env python3
"""
recognition/query_db.py

Usage:
    python recognition/query_db.py \
        --units_dir ../output/temp/units \
        --top_k 5 \
        --threshold 0.65 \
        --out_csv ../output/predictions.csv

Notes:
 - The script expects these files in --db_dir:
    embeddings_db.npy, labels_db.npy, paths_db.npy
   and optionally faiss_db.index
 - If faiss is installed and faiss_db.index exists it will use it for fast search.
 - Otherwise it will compute cosine similarity with NumPy (slower but accurate).
"""
import os
import argparse
import numpy as np
from PIL import Image
import pandas as pd
import sys

# Torch + transforms + model
import torch
import torchvision.transforms as T
import torchvision.models as models

# Try to import faiss if available
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

IMAGE_SIZE = 224
DB_DIR = "C:\\Users\\Alex\\Desktop\\automatic\\tft_recognition\\data\\active_learning_ressources"
def make_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # load ResNet50 up to the avgpool
        try:
            model = models.resnet50(pretrained=True)
        except TypeError:
            # fallback for newer torchvision API (weights enum)
            model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(model.children())[:-1]  # remove final fc
        self.model = torch.nn.Sequential(*modules).to(self.device).eval()
        self.transform = make_transform()
    
    @torch.no_grad()
    def embed(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(x).view(1,-1).cpu().numpy().astype('float32')
        # normalize to unit length
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        feat = feat / norm
        return feat  # shape (1, D)

def load_db(db_dir):
    emb_path = os.path.join(db_dir, "embeddings_db.npy")
    labels_path = os.path.join(db_dir, "labels_db.npy")
    paths_path = os.path.join(db_dir, "paths_db.npy")
    faiss_path = os.path.join(db_dir, "faiss_db.index")
    if not os.path.exists(emb_path) or not os.path.exists(labels_path) or not os.path.exists(paths_path):
        raise FileNotFoundError(f"Database files missing in {db_dir}. Expecting embeddings_db.npy, labels_db.npy, paths_db.npy")
    all_embeddings = np.load(emb_path).astype('float32')
    all_labels = np.load(labels_path, allow_pickle=True)
    all_paths = np.load(paths_path, allow_pickle=True)

    # Ensure embeddings are normalized (safety)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8
    all_embeddings = all_embeddings / norms

    index = None
    if _HAS_FAISS and os.path.exists(faiss_path):
        try:
            index = faiss.read_index(faiss_path)
            # If the index is on CPU and you have GPU-enabled faiss, you could move it to GPU here.
        except Exception as e:
            print("⚠️ Failed to load FAISS index, will build a fresh IndexFlatIP. Error:", e, file=sys.stderr)
            index = None

    if index is None:
        # build a CPU IndexFlatIP (fast enough for moderate DB sizes)
        d = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(all_embeddings)  # index order matches labels/paths arrays
    return all_embeddings, all_labels, all_paths, index

def query_with_faiss(index, q_emb, top_k=5):
    # q_emb: (1, d) float32
    q = np.ascontiguousarray(q_emb.astype('float32'))
    D, I = index.search(q, top_k)
    # D is similarity (inner prod) if IndexFlatIP
    return D[0], I[0]

def query_bruteforce(all_embeddings, q_emb, top_k=5):
    # q_emb (1, d), all_embeddings (N, d) - both L2-normalized
    sims = (all_embeddings @ q_emb.T).squeeze()  # shape (N,)
    # get top_k indices
    idx = np.argsort(-sims)[:top_k]
    return sims[idx], idx

def predict_all(units_dir, db_data, top_k=5, threshold=0.65, out_csv=None, device=None, verbose=False):
    extractor = FeatureExtractor(device=device)
    all_embeddings, all_labels, all_paths, index = db_data

    use_faiss = _HAS_FAISS and index is not None

    results = []
    # Look inside both "bench" and "board" subfolders (recursively)
    unit_files = []
    for root, _, files in os.walk(units_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                unit_files.append(os.path.join(root, f))

    unit_files = sorted(unit_files)
    if not unit_files:
        print("No unit images found in", units_dir)
        return None
    if verbose:
        print(f"Found {len(unit_files)} unit images. Using {'FAISS' if use_faiss else 'NumPy brute-force'} search. top_k={top_k}, threshold={threshold}")

    for u in unit_files:
        try:
            img = Image.open(u).convert('RGB')
        except Exception as e:
            print("⚠️ Failed to open", u, ":", e)
            continue
        q_emb = extractor.embed(img)  # (1, d) normalized

        if use_faiss:
            sims, idx = query_with_faiss(index, q_emb, top_k=top_k)
        else:
            sims, idx = query_bruteforce(all_embeddings, q_emb, top_k=top_k)

        top_labels = [str(all_labels[i]) for i in idx]
        top_paths = [str(all_paths[i]) for i in idx]
        top_scores = [float(s) for s in sims]

        best_score = top_scores[0]
        best_label = top_labels[0] if best_score >= threshold else "UNKNOWN"
        if verbose:
            print(f"{os.path.basename(u)} -> {best_label} (score={best_score:.4f})  top1_path={top_paths[0]}")
            # For debugging / inspection print top_k:
            for k in range(len(top_labels)):
                print(f"   {k+1:02d}. {top_labels[k]:20s} score={top_scores[k]:.4f}  db_path={top_paths[k]}")

        results.append({
            "unit_path": u,
            "pred_label": best_label,
            "pred_score": best_score,
            "top_k_labels": "|".join(top_labels),
            "top_k_scores": "|".join([f"{s:.4f}" for s in top_scores]),
            "top_k_paths": "|".join(top_paths)
        })

    df = pd.DataFrame(results)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print("✅ Predictions saved to", out_csv)
    
    # 🔽 Return distinct unit names (excluding "UNKNOWN")
    distinct_units = sorted(set([r["pred_label"] for r in results if r["pred_label"] != "UNKNOWN"]))
    return distinct_units

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--units_dir", required=True, help="Directory with cropped unit images (e.g. ../output/temp/units)")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.65,
                   help="Minimum cosine similarity to accept top match; below this -> UNKNOWN")
    p.add_argument("--out_csv", default=None, help="Path to CSV where predictions will be saved")
    p.add_argument("--device", default=None, help="torch device override, e.g. cpu or cuda:0")
    args = p.parse_args()
    predict_all(args.units_dir, args.top_k, args.threshold, args.out_csv, device=args.device)

if __name__ == "__main__":
    cli()
