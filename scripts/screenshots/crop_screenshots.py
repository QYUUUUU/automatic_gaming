import cv2
import os

board_crop = (320, 350, 1100, 520)  # x, y, width, height (example)
save_cropped_dir = "board_crops"
os.makedirs(save_cropped_dir, exist_ok=True)

for filename in os.listdir("board_screenshots"):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join("board_screenshots", filename))
        x, y, w, h = board_crop
        crop = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(save_cropped_dir, filename), crop)
