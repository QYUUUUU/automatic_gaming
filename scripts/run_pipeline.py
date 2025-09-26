"""
scripts/run_pipeline.py

Main entry point for TFT recognition.
This pipeline can:
  1. Capture & crop units from screenshots
  2. Mask background
  3. Label champions interactively
  4. Compute embeddings
  5. Build FAISS database
  6. Recognize champions in new screenshots
"""

import os
from screenshots.screenshoot import capture_monitor as capture
from screenshots.crop_screenshots import crop_image
from recognition.champions_id.isolate_units import isolate_units as isolate
# Add at the top with other imports
from recognition.champions_id.query_db import DB_DIR, load_db, predict_all
from recognition.champions_id.cleanup import clean_folder, FOLDERS_TO_CLEAN
from concurrent.futures import ThreadPoolExecutor

from recognition.shop_id.read_shop import read_tft_shop

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "../data"
OUTPUT_DIR = "../output"

RAW_DIR = os.path.join(DATA_DIR, "raw_screenshots")
BOARD_CROPS = os.path.join(DATA_DIR, "board_crops")
BENCH_CROPS = os.path.join(DATA_DIR, "bench_crops")
MASKED_DIR = os.path.join(DATA_DIR, "masked")
LABELED_DIR = os.path.join(DATA_DIR, "labeled")

EMBED_DIR = os.path.join(OUTPUT_DIR, "embeddings")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
FAISS_INDEX = os.path.join(EMBED_DIR, "faiss.index")

SCREENSHOT_DIR = os.path.join(TEMP_DIR, "raw_screenshot")
CROPPED_DIR = os.path.join(TEMP_DIR, "cropped_screenshot")
UNITS_DIR = os.path.join(TEMP_DIR, "units")

# =========================
# PIPELINE FUNCTIONS
# =========================

def capture_and_crop():
    """Take screenshots and crop into board/bench units."""

    # Capture screenshot from left monitor
    save_path = capture(monitor_index=1, save_dir=SCREENSHOT_DIR)
    # Crop units from screenshot
    board_crop = crop_image(image_path=save_path, save_dir=SCREENSHOT_DIR)
    # shop_crop = crop_image(image_path=save_path, save_dir=CROPPED_DIR, crop_region="SHOP_CROP")


    return save_path, board_crop

def isolate_units(save_path):
    """Run unit isolation on cropped screenshot."""
    isolate(
        input_path=save_path,
        output_dir=UNITS_DIR,
        show_preview=False
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    with ThreadPoolExecutor() as pool:
        future_db = pool.submit(load_db, DB_DIR)

        main_screenshot, board_crop = capture_and_crop()
        isolate_units(board_crop)

        db_data = future_db.result()  # join here
        unit_list = predict_all(UNITS_DIR, db_data, top_k=5, threshold=0.9, out_csv=os.path.join(TEMP_DIR, "predictions.csv"), verbose=False)    
        print("Detected units:", unit_list)

        # Step 3: Extract shop texts
        results = read_tft_shop(main_screenshot)
        print("Shop characters:", results)

        for folder in FOLDERS_TO_CLEAN:
            clean_folder(folder)
        print("🧹 Temp folders cleaned.")
