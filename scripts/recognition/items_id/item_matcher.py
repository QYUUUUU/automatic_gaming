import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

# --- CONFIGURATION ---

# Folder with cropped items
cropped_folder = "cropped_items"

# Folder with reference items from MetaTFT
reference_folder = "tft_items"

# Optional: similarity threshold (0-1)
THRESHOLD = 0.2

# --- LOAD REFERENCE ITEMS ---
reference_items = {}
for filename in os.listdir(reference_folder):
    if filename.endswith(".png"):
        name = os.path.splitext(filename)[0]  # e.g., "BFSword"
        path = os.path.join(reference_folder, filename)
        ref_img = cv2.imread(path)
        reference_items[name] = ref_img

print(f"Loaded {len(reference_items)} reference items.")

# --- HELPER FUNCTION ---
def compare_images(img1, img2):
    """Compute SSIM similarity score between two images"""
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_resized = cv2.resize(img2, (img1_gray.shape[1], img1_gray.shape[0]))
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    score = ssim(img1_gray, img2_gray)
    return score

# --- MATCH CROPPED ITEMS ---
cropped_files = sorted(os.listdir(cropped_folder))  # ensure order
matches = []

for filename in cropped_files:
    if not filename.endswith(".png"):
        continue
    crop_path = os.path.join(cropped_folder, filename)
    crop_img = cv2.imread(crop_path)

    best_match = None
    best_score = -1

    for name, ref_img in reference_items.items():
        score = compare_images(crop_img, ref_img)
        if score > best_score:
            best_match = name
            best_score = score

    if best_score >= THRESHOLD:
        print(f"{filename} → {best_match} (score: {best_score:.2f})")
        matches.append((filename, best_match, best_score))
    else:
        print(f"{filename} → Unknown (best score: {best_score:.2f})")
        matches.append((filename, None, best_score))
