import cv2
import numpy as np
import os
from typing import Optional, Union
from pathlib import Path

# ==============================
# DEFAULT CONFIGURATION
# ==============================

DEFAULT_INPUT_DIR = "../../data/board_crops"
DEFAULT_EMPTY_BOARD_IMG = "../data/reference_board/cropped_empty_board.png"
DEFAULT_OUTPUT_DIR = "../../data/units"

# Board grid
ROWS, COLS = 4, 7
BOARD_W, BOARD_H = 700, 400
SRC_POINTS = np.float32([[244, 86], [1000, 84], [1075, 450], [170, 450]])
HEX_RADIUS = 61

# Bench grid
BENCH_COLS = 9
BENCH_W, BENCH_H = 1040, 165
BENCH_SRC_POINTS = np.float32([[60, 390], [1090, 390], [1105, 480], [35, 480]])

UNIT_HEIGHT = 165          # Fixed crop height for champions
AREA_THRESHOLD = 0.08      # Minimum fraction of pixels to consider unit present
BOTTOM_OFFSET = 15         # Keep bottom 15px above original hex bottom

# ==============================
# FUNCTIONS
# ==============================

def generate_hex_centers(rows, cols, radius):
    centers = []
    dx = radius * np.sqrt(3)
    dy = radius * 1.5
    for r in range(rows):
        for c in range(cols):
            x = c * dx
            y = r * dy
            if r % 2 == 1:
                x += dx / 2
            centers.append((x, y))
    return centers

def hex_corners(center, radius):
    cx, cy = center
    return np.array([[cx + radius * np.cos(np.pi/3*i + np.pi/6),
                      cy + radius * np.sin(np.pi/3*i + np.pi/6)] for i in range(6)], dtype=np.float32)

def project_points(H_inv, points):
    n = len(points)
    pts_hom = np.hstack([points, np.ones((n,1))]).T
    projected = H_inv @ pts_hom
    projected /= projected[2]
    return projected[:2].T.astype(int)

def generate_bench_centers(cols, square_width):
    centers = []
    for c in range(cols):
        x = square_width * (c + 0.5)
        y = BENCH_H / 2  # full rectified bench height
        centers.append((x, y))
    return centers

def square_corners(center, width, height):
    cx, cy = center
    hw, hh = width/2, height/2
    return np.array([[cx-hw, cy-hh], [cx+hw, cy-hh], [cx+hw, cy+hh], [cx-hw, cy+hh]], dtype=np.float32)

def remove_board_background(board_img, empty_board_img, threshold=30):
    diff = cv2.absdiff(board_img, empty_board_img)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    result = cv2.bitwise_and(board_img, board_img, mask=mask)
    return result, mask

def keep_largest_blob(mask):
    """Keep only the largest connected component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
    return np.uint8(labels == largest_label) * 255

def crop_unit_if_present(img, mask, x_min, y_min, x_max, y_max, area_threshold):
    crop_mask = mask[y_min:y_max+1, x_min:x_max+1]
    crop_img = img[y_min:y_max+1, x_min:x_max+1]
    if crop_mask is None or crop_mask.size == 0:
        return None

    # Keep only largest blob
    crop_mask = keep_largest_blob(crop_mask)

    # Center overlap check
    h, w = crop_mask.shape
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.15)  # central 15% region
    center_mask = np.zeros_like(crop_mask)
    cv2.circle(center_mask, (cx, cy), radius, 255, -1)

    overlap = cv2.bitwise_and(crop_mask, center_mask)
    if np.count_nonzero(overlap) == 0:
        return None

    # Area fraction check
    nonzero = np.count_nonzero(crop_mask)
    total = crop_mask.size
    fraction = nonzero / total
    if fraction >= area_threshold:
        crop_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)
        return crop_img
    else:
        return None

# ==============================
# MAIN PIPELINE
# ==============================

def process_file(filename: str, empty_board: np.ndarray, board_dir: str, bench_dir: str) -> None:
    """Process a single image file to extract units from board and bench positions.
    
    Args:
        filename: Path to the input image file
        empty_board: The empty board reference image
        board_dir: Output directory for board units
        bench_dir: Output directory for bench units
    """
    img = cv2.imread(filename)
    if img is None:
        print(f"⚠️ Skipping {filename} (could not load)")
        return

    basename = os.path.splitext(os.path.basename(filename))[0]
    units_img, units_mask = remove_board_background(img, empty_board)

    # ------------------ Board hexes ------------------
    board_dst = np.float32([[0,0],[BOARD_W,0],[BOARD_W,BOARD_H],[0,BOARD_H]])
    H_board = cv2.getPerspectiveTransform(SRC_POINTS, board_dst)
    H_board_inv = np.linalg.inv(H_board)

    centers = generate_hex_centers(ROWS, COLS, HEX_RADIUS)
    projected_centers = project_points(H_board_inv, np.array(centers))

    for i, center in enumerate(projected_centers):
        corners = hex_corners(centers[i], HEX_RADIUS)
        projected_corners = project_points(H_board_inv, corners)
        x_min, x_max = projected_corners[:,0].min(), projected_corners[:,0].max()
        y_min, y_max = projected_corners[:,1].min(), projected_corners[:,1].max()

        y_max_crop = max(y_max - BOTTOM_OFFSET, 0)
        y_min_crop = max(y_max_crop - UNIT_HEIGHT, 0)

        crop = crop_unit_if_present(units_img, units_mask, x_min, y_min_crop, x_max, y_max_crop, AREA_THRESHOLD)
        if crop is not None:
            filename_out = os.path.join(board_dir, f"{basename}_unit_{i+1}.png")
            cv2.imwrite(filename_out, crop)

    # ------------------ Bench ------------------
    bench_dst = np.float32([[0,0],[BENCH_W,0],[BENCH_W,BENCH_H],[0,BENCH_H]])
    H_bench = cv2.getPerspectiveTransform(BENCH_SRC_POINTS, bench_dst)
    H_bench_inv = np.linalg.inv(H_bench)
    bench_cell_width = BENCH_W / BENCH_COLS
    bench_cell_height = BENCH_H

    for i in range(BENCH_COLS):
        cx = bench_cell_width * (i + 0.5)
        cy = bench_cell_height / 2
        corners = np.array([
            [cx - bench_cell_width/2, cy - bench_cell_height/2],
            [cx + bench_cell_width/2, cy - bench_cell_height/2],
            [cx + bench_cell_width/2, cy + bench_cell_height/2],
            [cx - bench_cell_width/2, cy + bench_cell_height/2]
        ], dtype=np.float32)

        projected_corners = project_points(H_bench_inv, corners)
        x_min, x_max = projected_corners[:,0].min(), projected_corners[:,0].max()
        y_min, y_max = projected_corners[:,1].min(), projected_corners[:,1].max()

        y_min_expanded = max(0, y_min - 70)
        y_max_expanded = min(units_img.shape[0], y_max - 15)

        crop = crop_unit_if_present(units_img, units_mask, x_min, y_min_expanded, x_max, y_max_expanded, AREA_THRESHOLD)
        if crop is not None:
            filename_out = os.path.join(bench_dir, f"{basename}_bench_unit_{i+1}.png")
            cv2.imwrite(filename_out, crop)

def preview_grids(filename: str, empty_board: np.ndarray, show_units: bool = True, show_empty: bool = True) -> None:
    """Overlay the board/bench grids and optionally detected units + empty slots on the given image.
    
    Args:
        filename: Path to the input image
        empty_board: The empty board reference image
        show_units: Whether to show detected units
        show_empty: Whether to show empty slots
    """
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Could not load {filename}")

    overlay = img.copy()

    # ------------------ Board homography ------------------
    board_dst = np.float32([[0,0],[BOARD_W,0],[BOARD_W,BOARD_H],[0,BOARD_H]])
    H_board = cv2.getPerspectiveTransform(SRC_POINTS, board_dst)
    H_board_inv = np.linalg.inv(H_board)

    centers = generate_hex_centers(ROWS, COLS, HEX_RADIUS)
    projected_centers = project_points(H_board_inv, np.array(centers))

    # Draw red hex grid for reference
    for center in centers:
        corners = hex_corners(center, HEX_RADIUS)
        projected_corners = project_points(H_board_inv, corners)
        cv2.polylines(overlay, [projected_corners], True, (0,0,255), 1)

    # ------------------ Bench homography ------------------
    bench_dst = np.float32([[0,0],[BENCH_W,0],[BENCH_W,BENCH_H],[0,BENCH_H]])
    H_bench = cv2.getPerspectiveTransform(BENCH_SRC_POINTS, bench_dst)
    H_bench_inv = np.linalg.inv(H_bench)
    bench_cell_width = BENCH_W / BENCH_COLS
    bench_cell_height = BENCH_H

    for i in range(BENCH_COLS):
        cx = bench_cell_width * (i + 0.5)
        cy = bench_cell_height / 2
        corners = np.array([
            [cx - bench_cell_width/2, cy - bench_cell_height/2],
            [cx + bench_cell_width/2, cy - bench_cell_height/2],
            [cx + bench_cell_width/2, cy + bench_cell_height/2],
            [cx - bench_cell_width/2, cy + bench_cell_height/2]
        ], dtype=np.float32)

        projected_corners = project_points(H_bench_inv, corners)
        cv2.polylines(overlay, [projected_corners], True, (0,255,0), 1)

    # ------------------ Optional unit detection ------------------
    if show_units or show_empty:
        units_img, units_mask = remove_board_background(img, empty_board)

        # ---- Board units ----
        for i, center in enumerate(projected_centers):
            corners = hex_corners(centers[i], HEX_RADIUS)
            projected_corners = project_points(H_board_inv, corners)
            x_min, x_max = projected_corners[:,0].min(), projected_corners[:,0].max()
            y_min, y_max = projected_corners[:,1].min(), projected_corners[:,1].max()

            y_max_crop = max(y_max - BOTTOM_OFFSET, 0)
            y_min_crop = max(y_max_crop - UNIT_HEIGHT, 0)

            crop = crop_unit_if_present(units_img, units_mask,
                                        x_min, y_min_crop, x_max, y_max_crop, AREA_THRESHOLD)

            if crop is not None and show_units:
                cv2.rectangle(overlay, (x_min, y_min_crop), (x_max, y_max_crop), (0,128,255), 2)  # orange = detected unit
            elif crop is None and show_empty:
                cv2.rectangle(overlay, (x_min, y_min_crop), (x_max, y_max_crop), (128,128,255), 1)  # gray = empty slot

        # ---- Bench units ----
        for i in range(BENCH_COLS):
            cx = bench_cell_width * (i + 0.5)
            cy = bench_cell_height / 2
            corners = np.array([
                [cx - bench_cell_width/2, cy - bench_cell_height/2],
                [cx + bench_cell_width/2, cy - bench_cell_height/2],
                [cx + bench_cell_width/2, cy + bench_cell_height/2],
                [cx - bench_cell_width/2, cy + bench_cell_height/2]
            ], dtype=np.float32)

            projected_corners = project_points(H_bench_inv, corners)
            x_min, x_max = projected_corners[:,0].min(), projected_corners[:,0].max()
            y_min, y_max = projected_corners[:,1].min(), projected_corners[:,1].max()

            y_min_expanded = max(0, y_min - 70)
            y_max_expanded = min(units_img.shape[0], y_max - 15)

            crop = crop_unit_if_present(units_img, units_mask,
                                        x_min, y_min_expanded, x_max, y_max_expanded, AREA_THRESHOLD)

            if crop is not None and show_units:
                cv2.rectangle(overlay, (x_min, y_min_expanded), (x_max, y_max_expanded), (0,255,255), 2)  # yellow = detected bench unit
            elif crop is None and show_empty:
                cv2.rectangle(overlay, (x_min, y_min_expanded), (x_max, y_max_expanded), (128,128,128), 1)  # gray = empty bench slot

    # ------------------ Show result ------------------
    cv2.imshow("Grid + Units Preview", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def isolate_units(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    preview_image: Optional[Union[str, Path]] = None,
    show_preview: bool = False,
    show_units: bool = True,
    show_empty: bool = True
) -> None:
    """Main function to process images and extract units from board and bench positions.
    
    Args:
        input_path: Path to a single image file or directory containing images
        output_dir: Directory where extracted units will be saved
        preview_image: Specific image to preview (if None, uses first available image)
        show_preview: Whether to show the grid preview
        show_units: Whether to show detected units in preview
        show_empty: Whether to show empty slots in preview
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If input paths don't exist
        ValueError: If no valid image files found in input directory
    """
    # Convert paths to Path objects for easier handling
    input_path = Path(input_path)
    empty_board_path = Path(DEFAULT_EMPTY_BOARD_IMG)
    output_dir = Path(output_dir)
    
    # Validate input paths
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not empty_board_path.exists():
        raise FileNotFoundError(f"Empty board image does not exist: {empty_board_path}")
    
    # Load empty board reference
    empty_board = cv2.imread(str(empty_board_path))
    if empty_board is None:
        raise FileNotFoundError(f"Could not load empty board image: {empty_board_path}")
    
    # Create output directories
    board_dir = output_dir / "board"
    bench_dir = output_dir / "bench"
    board_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image files to process
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    if input_path.is_file():
        if input_path.suffix.lower() not in image_extensions:
            raise ValueError(f"Input file is not a supported image format: {input_path}")
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        if not image_files:
            raise ValueError(f"No valid image files found in directory: {input_path}")
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")
    
    # Show preview if requested
    if show_preview:
        if preview_image:
            preview_path = Path(preview_image)
            if not preview_path.exists():
                raise FileNotFoundError(f"Preview image does not exist: {preview_path}")
        else:
            preview_path = image_files[0]
        
        print(f"🔍 Previewing grid on: {preview_path}")
        preview_grids(str(preview_path), empty_board, show_units, show_empty)
    
    # Process all images
    print(f"🚀 Processing {len(image_files)} image(s)...")
    processed_count = 0
    
    for image_file in image_files:
        print(f"🔍 Processing {image_file.name}...")
        process_file(str(image_file), empty_board, str(board_dir), str(bench_dir))
        processed_count += 1
    
    print(f"✅ Processed {processed_count} images.")

def main():
    """Legacy main function for backward compatibility."""
    # Use default paths for backward compatibility
    input_dir = DEFAULT_INPUT_DIR
    empty_board_path = DEFAULT_EMPTY_BOARD_IMG
    output_dir = DEFAULT_OUTPUT_DIR
    
    isolate_units(
        input_path=input_dir,
        empty_board_path=empty_board_path,
        output_dir=output_dir,
        show_preview=True
    )

if __name__ == "__main__":
    # Example usage:
    
    # Process a single image
    # isolate_units(
    #     input_path="../../data/board_crops/20250923_173041.png",
    #     empty_board_path="../../data/reference_board/cropped_empty_board.png",
    #     output_dir="../../data/units_single",
    #     show_preview=True
    # )
    
    # Process a folder of images
    # isolate_units(
    #     input_path="../../data/board_crops",
    #     empty_board_path="../../data/reference_board/cropped_empty_board.png", 
    #     output_dir="../../data/units_batch",
    #     show_preview=True,
    #     preview_image="../../data/board_crops/20250923_173041.png"
    # )
    
    # Use legacy main function for backward compatibility
    main()
