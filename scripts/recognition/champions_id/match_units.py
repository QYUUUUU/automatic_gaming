def recognize_image(image_path, top_k=3):
    img = Image.open(image_path).convert("RGB")
    emb = extractor.embed(img).astype('float32')
    emb /= np.linalg.norm(emb)
    D, I = index.search(emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append((all_paths[idx], all_labels[idx], float(score)))
    return results

# Example
new_screenshot = "test_screenshot.png"
matches = recognize_image(new_screenshot)
for path, label, score in matches:
    print(f"{label} ({score:.3f}) -> {path}")
