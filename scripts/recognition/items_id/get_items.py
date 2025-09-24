import json
import cv2
import os

# === CONFIGURATION ===

# Screenshot path
screenshot_path = "left_monitor.png"

# Top-left of the first item slot
first_item = (10, 275)  # <-- update to your actual first item

# Offsets between consecutive items
x_offset = 0  # horizontal distance between slots
y_offset = 51  # vertical distance between slots

# Number of items
num_items = 10

# Width & height of each item crop
item_width = 41
item_height = 42

# Output folder for cropped items
output_dir = "../../../output/temp/cropped_items"
os.makedirs(output_dir, exist_ok=True)

# === GENERATE COORDINATES ===
item_coords = []
for i in range(num_items):
    x = first_item[0]
    y = first_item[1] + i * y_offset
    item_coords.append((x, y, item_width, item_height))

print(f"✅ Generated coordinates for {num_items} item slots.")

# === CROP ITEMS ===
# Load screenshot
img = cv2.imread(screenshot_path)

cropped_files = []
for idx, (x, y, w, h) in enumerate(item_coords, start=1):
    crop = img[y:y+h, x:x+w]
    filename = os.path.join(output_dir, f"item_{idx}.png")
    cv2.imwrite(filename, crop)
    cropped_files.append(filename)
    print(f"Cropped item {idx} → {filename}")

print(f"✅ Cropped {len(cropped_files)} items to {output_dir}/")
