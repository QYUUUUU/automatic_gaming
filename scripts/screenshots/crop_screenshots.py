import cv2
import os
import hashlib
from PIL import Image

# Default crop region (x, y, width, height)
BOARD_CROP = (320, 350, 1100, 520)
SHOP_CROP = (480, 925, 1000, 150)


def crop_image(image_path, save_dir, crop_region="BOARD_CROP"):
    """
    Crop a single image and save it to a directory.
    
    Args:
        image_path (str): Path to the input image.
        save_dir (str): Directory where cropped image will be saved.
        crop_region (tuple): (x, y, width, height).
    
    Returns:
        str: Path to the cropped image.
    """
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    if crop_region == "BOARD_CROP":
        crop_region = BOARD_CROP
    elif crop_region == "SHOP_CROP":
        crop_region = SHOP_CROP
    x, y, w, h = crop_region
    crop = img[y:y + h, x:x + w]

    filename = "cropped_" + os.path.basename(image_path)
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, crop)

    print(f"✅ Cropped image saved to {save_path}")
    return save_path


def crop_images_in_dir(input_dir, save_dir, crop_region=BOARD_CROP):
    """
    Crop all .png images in a directory.
    
    Args:
        input_dir (str): Directory containing images to crop.
        save_dir (str): Directory where cropped images will be saved.
        crop_region (tuple): (x, y, width, height).
    
    Returns:
        list[str]: List of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            file_path = os.path.join(input_dir, filename)
            try:
                saved_files.append(crop_image(file_path, save_dir, crop_region))
            except Exception as e:
                print(f"❌ Failed to crop {file_path}: {e}")

    return saved_files

def ahash(image_path, hash_size=8):
    """Compute average hash (aHash) for an image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = ''.join('1' if p > avg else '0' for p in pixels)
            return hex(int(bits, 2))
    except Exception as e:
        print(f"⚠️ Could not hash {image_path}: {e}")
        return None

def remove_duplicate_images(folder_path, recursive=False):
    """Remove duplicate images in a folder (and subfolders if recursive=True)."""
    seen_hashes = {}
    deleted_files = []

    if recursive:
        walker = (os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files)
    else:
        walker = (os.path.join(folder_path, f) for f in os.listdir(folder_path))

    for file_path in walker:
        if not os.path.isfile(file_path):
            continue

        file_hash = ahash(file_path)
        if not file_hash:
            continue

        if file_hash in seen_hashes:
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
                print(f"🗑️ Deleted duplicate: {file_path}")
            except Exception as e:
                print(f"❌ Could not delete {file_path}: {e}")
        else:
            seen_hashes[file_hash] = file_path

    print(f"\n✅ Total duplicates deleted: {len(deleted_files)}")
    return deleted_files

if __name__ == "__main__":
    # Default behavior: crop all screenshots from raw_screenshots
    input_dir = "../../data/raw_screenshots"
    board_dir = "../../data/board_crops"
    shop_dir = "../../data/shop_crops"

    crop_images_in_dir(input_dir, board_dir, crop_region=BOARD_CROP)
    crop_images_in_dir(input_dir, shop_dir, crop_region=SHOP_CROP)

    remove_duplicate_images(shop_dir)

    print(f"✅ Cropped images saved to {board_dir}/")
    print(f"✅ Cropped images saved to {shop_dir}/")