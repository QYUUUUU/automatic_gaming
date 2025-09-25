import cv2
import os

# Default crop region (x, y, width, height)
BOARD_CROP = (320, 350, 1100, 520)


def crop_image(image_path, save_dir, crop_region=BOARD_CROP):
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

    x, y, w, h = crop_region
    crop = img[y:y + h, x:x + w]

    filename = os.path.basename("cropped_" + image_path)
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


if __name__ == "__main__":
    # Default behavior: crop all screenshots from raw_screenshots
    input_dir = "../../data/raw_screenshots"
    save_dir = "../../data/board_crops"

    crop_images_in_dir(input_dir, save_dir, crop_region=BOARD_CROP)
    print(f"✅ Cropped images saved to {save_dir}/")