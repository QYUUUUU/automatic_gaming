#!/usr/bin/env python3
"""
scripts/cleanup.py

Utility to clean up temp/output directories after pipeline runs.
By default, removes unit crops and raw screenshots.
"""

import os
import shutil

# Add more if you want
FOLDERS_TO_CLEAN = [
    "C:\\Users\\Alex\\Desktop\\automatic\\tft_recognition\\output\\temp\\units",
    "C:\\Users\\Alex\\Desktop\\automatic\\tft_recognition\\output\\temp\\raw_screenshot",
]

def clean_folder(path):
    """Delete a folder and recreate it empty."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    for folder in FOLDERS_TO_CLEAN:
        clean_folder(folder)
    print("✅ Cleanup complete.")
