import mss
import cv2
import numpy as np

def capture_monitor(monitor_index=1, save_path="screenshot.png"):
    """
    Capture a screenshot from a given monitor index.
    monitor_index=1 usually means leftmost screen (0 is all monitors).
    """
    with mss.mss() as sct:
        # Get list of monitors
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} out of range. Available: {len(monitors)-1}")
        
        # Select monitor
        monitor = monitors[monitor_index]

        # Grab screenshot
        img = np.array(sct.grab(monitor))

        # Save to file
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
        print(f"✅ Screenshot saved to {save_path}")

        return img

if __name__ == "__main__":
    # Capture from left monitor (monitor index = 1)
    img = capture_monitor(monitor_index=1, save_path="left_monitor.png")

    # Show it in a preview window
    cv2.imshow("Left Monitor Screenshot", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
