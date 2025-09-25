import mss
import cv2
import numpy as np
import os
from datetime import datetime


def capture_monitor(monitor_index=1, save_dir=".", prefix="screenshot"):
    """
    Capture a screenshot from a given monitor index and save with a timestamped filename.
    
    Args:
        monitor_index (int): Index of the monitor to capture.
                             0 = all monitors (entire virtual screen).
                             1 = first monitor (leftmost screen).
        save_dir (str): Directory where the screenshot will be saved.
        prefix (str): Filename prefix before the timestamp.
    
    Returns:
        str: The full path of the saved screenshot file.
    """
    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)

    with mss.mss() as sct:
        monitors = sct.monitors

        # If only one monitor is available, force monitor_index = 1
        if len(monitors) == 2 and monitor_index > 1:
            print("⚠️ Only one monitor detected. Defaulting to monitor 1.")
            monitor_index = 1

        if monitor_index >= len(monitors):
            raise ValueError(
                f"Monitor index {monitor_index} out of range. "
                f"Available indices: 0 to {len(monitors)-1}"
            )

        monitor = monitors[monitor_index]
        img = np.array(sct.grab(monitor))

        # Save screenshot
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
        print(f"✅ Screenshot saved to {save_path}")

    return save_path


def preview_image(img, window_name="Screenshot"):
    """
    Display a screenshot in an OpenCV preview window.
    
    Args:
        img (np.ndarray): The image to display (BGRA format).
        window_name (str): Window title.
    """
    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        file_path = capture_monitor(monitor_index=1, save_dir=".", prefix="left_monitor")
        img = cv2.imread(file_path)  # Reload for preview
        preview_image(img, "Left Monitor Screenshot")
    except Exception as e:
        print(f"❌ Error: {e}")
