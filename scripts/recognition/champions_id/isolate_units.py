import cv2
import numpy as np
import os

# ==============================
# CONFIGURATION
# ==============================

INPUT_DIR = "board_crops"   
EMPTY_BOARD_IMG = "board.png" 

OUTPUT_DIR = "output"
BOARD_DIR = os.path.join(OUTPUT_DIR, "board")
BENCH_DIR = os.path.join(OUTPUT_DIR, "bench")
os.makedirs(BOARD_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)

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
AREA_THRESHOLD = 0.1      # Minimum fraction of pixels to consider unit present
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

def process_file(filename, empty_board):
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
            filename_out = os.path.join(BOARD_DIR, f"{basename}_unit_{i+1}.png")
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
            filename_out = os.path.join(BENCH_DIR, f"{basename}_bench_unit_{i+1}.png")
            cv2.imwrite(filename_out, crop)

def preview_grids(filename, empty_board, show_units=True, show_empty=True):
    """Overlay the board/bench grids and optionally detected units + empty slots on the given image."""
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



def main():
    empty_board = cv2.imread(EMPTY_BOARD_IMG)
    if empty_board is None:
        raise FileNotFoundError(f"Could not load {EMPTY_BOARD_IMG}")

    # Pick one file to preview
    test_img = os.path.join(INPUT_DIR, "20250923_173041.png")

    # Show grids
    #preview_grids(test_img, empty_board, show_units=True, show_empty=True)

    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(INPUT_DIR, file)
            print(f"🔍 Processing {filepath}...")
            process_file(filepath, empty_board)

    print(f"✅ Done! Units saved in '{BOARD_DIR}' and '{BENCH_DIR}'.")

if __name__ == "__main__":
    main()
