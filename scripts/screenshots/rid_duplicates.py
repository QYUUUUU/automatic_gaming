import hashlib
from PIL import Image

def ahash(image_path, hash_size=8):
    """Compute average hash (aHash) for an image."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return hex(int(bits, 2))

def sha256_file(file_path):
    """Compute SHA-256 hash of a file (byte-for-byte)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

file1 = "C:\\Users\\Alex\\Desktop\\automatic\\tft_recognition\\data\\shop_crops\\20250925_121021.png"
file2 = "C:\\Users\\Alex\\Desktop\\automatic\\tft_recognition\\data\\shop_crops\\20250925_121022.png"

print("aHash of file1:", ahash(file1))
print("aHash of file2:", ahash(file2))
print("SHA-256 of file1:", sha256_file(file1))
print("SHA-256 of file2:", sha256_file(file2))
