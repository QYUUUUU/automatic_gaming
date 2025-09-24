import os
import shutil
import pandas as pd

LABELS_FILE = "labels.csv"
BASE_DIR = "labeled"  # output folder

df = pd.read_csv(LABELS_FILE)

for idx, row in df.iterrows():
    path, label = row['path'], row['label']
    
    out_dir = os.path.join(BASE_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(path)
    shutil.copy2(path, os.path.join(out_dir, fname))
