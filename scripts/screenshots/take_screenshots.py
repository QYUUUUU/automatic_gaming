import mss
import cv2
import numpy as np
from pynput import keyboard
import os
from datetime import datetime

# --- CONFIGURATION ---
save_dir = "board_screenshots"
os.makedirs(save_dir, exist_ok=True)

monitor_index = 1  # left monitor

def capture_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        img = np.array(sct.grab(monitor))
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
        print(f"✅ Screenshot saved: {filepath}")

def on_press(key):
    try:
        if key.char == 'p':  # p key
            capture_screenshot()
        elif key.char == 'o':  # o key to quit
            return False
    except AttributeError:
        # Special keys (like ESC) that don't have a char attribute
        if key == keyboard.Key.esc:
            return False

print("Press 'p' to take a screenshot. Press 'o' to quit.")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
