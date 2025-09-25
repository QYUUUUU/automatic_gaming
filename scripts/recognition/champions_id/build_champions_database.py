import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import faiss
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
OUTPUT_DIR = "../../../data/labeled"


def make_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

class FeatureExtractor:
    def __init__(self, device=DEVICE):
        self.device = device
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]  # remove final fc
        self.model = torch.nn.Sequential(*modules).to(self.device).eval()
        self.transform = make_transform()
    
    @torch.no_grad()
    def embed(self, pil_img):
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(x).view(1,-1).cpu().numpy()
        feat /= (np.linalg.norm(feat)+1e-8)
        return feat

extractor = FeatureExtractor()

all_embeddings = []
all_labels = []
all_paths = []

for label in os.listdir(OUTPUT_DIR):
    folder = os.path.join(OUTPUT_DIR, label)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            emb = extractor.embed(img)
            all_embeddings.append(emb)
            all_labels.append(label)
            all_paths.append(fpath)
        except:
            print("⚠️ Failed to load:", fpath)

all_embeddings = np.vstack(all_embeddings).astype('float32')
all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

# Build FAISS index
d = all_embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # cosine similarity = inner product on normalized vectors
index.add(all_embeddings)
print("✅ FAISS index built with", len(all_embeddings), "images")

# Save everything for later
np.save("../../../data/active_learning_ressources/embeddings_db.npy", all_embeddings)
np.save("../../../data/active_learning_ressources/paths_db.npy", np.array(all_paths))
np.save("../../../data/active_learning_ressources/labels_db.npy", np.array(all_labels))
faiss.write_index(index, "../../../data/active_learning_ressources/faiss_db.index")
print("✅ Saved embeddings, labels, paths, and FAISS index")
