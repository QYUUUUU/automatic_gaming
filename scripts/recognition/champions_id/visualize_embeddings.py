import numpy as np
from sklearn.manifold import TSNE
from PIL import Image
import base64, io
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool

# --- Load data ---
embeddings = np.load("embeddings.npy")
paths = np.load("paths.npy")

# --- Dim reduction ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
coords = tsne.fit_transform(embeddings)

# --- Encode images as base64 ---
def img_to_b64(path, max_size=64):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        return None

b64_images = [img_to_b64(p) for p in paths]

# --- Build Bokeh source ---
source = ColumnDataSource(data=dict(
    x=coords[:,0],
    y=coords[:,1],
    path=paths,    img=b64_images
))

# --- Plot ---
p = figure(title="Champion embeddings (t-SNE)", tools="pan,wheel_zoom,reset,hover,save", width=900, height=700)
p.scatter("x", "y", source=source, size=8, color="navy", alpha=0.6)

# --- Custom hover ---
hover = p.select(dict(type=HoverTool))
hover.tooltips = """
<div>
    <div><img src="@img" style="width:64px;height:64px;"></div>
    <div><b>Cluster:</b> @label</div>
    <div><b>Path:</b> @path</div>
</div>
"""

output_file("tsne_clusters.html")
show(p)
