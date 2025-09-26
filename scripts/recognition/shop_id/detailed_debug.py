from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import os

def save_preprocessing_steps(filepath, output_dir="debug_output"):
    """
    Save each preprocessing step as individual images for detailed examination.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img = Image.open(filepath)
    
    shop_coords = [
        (485, 1045, 105, 20),  # slot 1
        (685, 1045, 105, 20),  # slot 2
        (885, 1045, 105, 20),  # slot 3
        (1090, 1045, 105, 20),  # slot 4
        (1290, 1045, 105, 20),  # slot 5
    ]
    
    results = []
    
    for i, (x, y, w, h) in enumerate(shop_coords):
        print(f"\n--- Processing Slot {i+1} ---")
        print(f"Coordinates: x={x}, y={y}, w={w}, h={h}")
        
        # Step 1: Crop the region
        crop = img.crop((x, y, x+w, y+h))
        crop.save(os.path.join(output_dir, f"slot_{i+1}_01_original.png"))
        
        # Step 2: Convert to grayscale
        gray = ImageOps.grayscale(crop)
        gray.save(os.path.join(output_dir, f"slot_{i+1}_02_grayscale.png"))
        
        # Step 3: Enhance contrast
        enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
        enhanced.save(os.path.join(output_dir, f"slot_{i+1}_03_enhanced.png"))
        
        # Step 4: Invert colors
        inverted = ImageOps.invert(enhanced)
        inverted.save(os.path.join(output_dir, f"slot_{i+1}_04_inverted.png"))
        
        # Step 5: Binarize
        bw = inverted.point(lambda p: 0 if p < 128 else 255)
        bw.save(os.path.join(output_dir, f"slot_{i+1}_05_binarized.png"))
        
        # Try OCR on the final processed image with different configurations
        configs = [
            "--psm 8",  # Single word
            "--psm 7",  # Single text line
            "--psm 6",  # Single uniform block
            "--psm 13", # Single word, bypassing layout analysis
        ]
        
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            try:
                # Get text with confidence
                data = pytesseract.image_to_data(bw, lang="eng", config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                texts = [text for text in data['text'] if text.strip()]
                
                if confidences and texts:
                    avg_confidence = sum(confidences) / len(confidences)
                    combined_text = " ".join(texts).strip()
                    
                    print(f"  Config {config}: '{combined_text}' (confidence: {avg_confidence:.1f})")
                    
                    if avg_confidence > best_confidence and combined_text:
                        best_confidence = avg_confidence
                        best_text = combined_text
                
            except Exception as e:
                print(f"  Config {config}: Error - {e}")
        
        # Fallback to simple string extraction
        if not best_text:
            best_text = pytesseract.image_to_string(bw, lang="eng").strip()
            print(f"  Fallback: '{best_text}'")
        
        results.append(best_text)
        print(f"  Final result: '{best_text}'")
        
        # Also try different binarization thresholds
        for threshold in [100, 128, 150, 180]:
            thresh_bw = inverted.point(lambda p: 0 if p < threshold else 255)
            thresh_text = pytesseract.image_to_string(thresh_bw, lang="eng", config="--psm 8").strip()
            if threshold == 128:
                continue  # Skip the default one we already tested
            print(f"  Threshold {threshold}: '{thresh_text}'")
            
            # Save the alternative threshold version if it gives different results
            if thresh_text != best_text and thresh_text:
                thresh_bw.save(os.path.join(output_dir, f"slot_{i+1}_06_threshold_{threshold}.png"))
    
    print(f"\nAll processed images saved to: {output_dir}")
    print(f"Final OCR results: {results}")
    return results

def test_different_coordinates(filepath):
    """
    Test slightly different coordinate variations to see if positioning affects results.
    """
    img = Image.open(filepath)
    
    base_coords = [
        (485, 1045, 105, 20),  # slot 1
        (685, 1045, 105, 20),  # slot 2
        (885, 1045, 105, 20),  # slot 3
        (1090, 1045, 105, 20),  # slot 4
        (1290, 1045, 105, 20),  # slot 5
    ]
    
    print("\n=== Testing coordinate variations ===")
    
    for i, (base_x, base_y, base_w, base_h) in enumerate(base_coords):
        print(f"\nSlot {i+1} coordinate tests:")
        
        # Test variations: slightly up/down, left/right, wider/taller
        variations = [
            (base_x, base_y, base_w, base_h, "original"),
            (base_x, base_y - 2, base_w, base_h, "up_2px"),
            (base_x, base_y + 2, base_w, base_h, "down_2px"),
            (base_x - 3, base_y, base_w, base_h, "left_3px"),
            (base_x + 3, base_y, base_w, base_h, "right_3px"),
            (base_x, base_y, base_w + 10, base_h, "wider"),
            (base_x, base_y, base_w, base_h + 5, "taller"),
            (base_x - 5, base_y - 2, base_w + 10, base_h + 4, "expanded"),
        ]
        
        for x, y, w, h, desc in variations:
            crop = img.crop((x, y, x+w, y+h))
            
            # Quick preprocessing
            gray = ImageOps.grayscale(crop)
            enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
            inverted = ImageOps.invert(enhanced)
            bw = inverted.point(lambda p: 0 if p < 128 else 255)
            
            text = pytesseract.image_to_string(bw, lang="eng", config="--psm 8").strip()
            print(f"  {desc:12}: '{text}'")

if __name__ == "__main__":
    # Use multiple screenshots to test
    screenshots = [
        "C:\\Users\\Alex\\Desktop\\automatic\\tft_recognition\\data\\raw_screenshots\\20250923_172855.png"
    ]
    
    # Check if there are more recent screenshots to test
    screenshot_dir = "C:\\Users\\Alex\\Desktop\\automatic\\data\\raw_screenshots"
    if os.path.exists(screenshot_dir):
        all_screenshots = [os.path.join(screenshot_dir, f) for f in os.listdir(screenshot_dir) if f.endswith('.png')]
        all_screenshots.sort()  # Get them in chronological order
        
        # Test with the most recent few screenshots
        recent_screenshots = all_screenshots[-3:] if len(all_screenshots) >= 3 else all_screenshots
        screenshots.extend(recent_screenshots)
    
    for screenshot in screenshots:
        if os.path.exists(screenshot):
            print(f"\n{'='*60}")
            print(f"Testing screenshot: {os.path.basename(screenshot)}")
            print(f"{'='*60}")
            
            # Create output directory for this screenshot
            base_name = os.path.splitext(os.path.basename(screenshot))[0]
            output_dir = f"debug_output_{base_name}"
            
            results = save_preprocessing_steps(screenshot, output_dir)
            test_different_coordinates(screenshot)
        else:
            print(f"Screenshot not found: {screenshot}")