from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

def debug_shop_preprocessing(filepath):
    """
    Debug function to visualize the preprocessing steps for each shop slot.
    Shows original crop, grayscale, enhanced contrast, inverted, and binarized versions.
    """
    img = Image.open(filepath)
    
    shop_coords = [
        (485, 1045, 105, 20),  # slot 1
        (685, 1045, 105, 20),  # slot 2
        (885, 1045, 105, 20),  # slot 3
        (1090, 1045, 105, 20),  # slot 4
        (1290, 1045, 105, 20),  # slot 5
    ]
    
    # Create a figure with subplots for each slot
    fig, axes = plt.subplots(len(shop_coords), 5, figsize=(20, len(shop_coords) * 3))
    fig.suptitle('TFT Shop Preprocessing Debug', fontsize=16)
    
    # Column titles
    col_titles = ['Original Crop', 'Grayscale', 'Enhanced Contrast', 'Inverted', 'Binarized']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight='bold')
    
    results = []
    
    for i, (x, y, w, h) in enumerate(shop_coords):
        # Step 1: Crop the region
        crop = img.crop((x, y, x+w, y+h))
        
        # Step 2: Convert to grayscale
        gray = ImageOps.grayscale(crop)
        
        # Step 3: Enhance contrast
        enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
        
        # Step 4: Invert colors
        inverted = ImageOps.invert(enhanced)
        
        # Step 5: Binarize
        bw = inverted.point(lambda p: 0 if p < 128 else 255)
        
        # Display all steps
        axes[i, 0].imshow(crop)
        axes[i, 0].set_ylabel(f'Slot {i+1}', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(enhanced, cmap='gray')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(inverted, cmap='gray')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(bw, cmap='gray')
        axes[i, 4].axis('off')
        
        # Try OCR on the final processed image
        text = pytesseract.image_to_string(bw, lang="eng")
        results.append(text.strip())
        
        # Add OCR result as text below the binarized image
        axes[i, 4].text(0.5, -0.1, f"OCR: '{text.strip()}'", 
                       transform=axes[i, 4].transAxes, 
                       ha='center', va='top', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.show()
    
    return results

def try_alternative_preprocessing(filepath):
    """
    Try alternative preprocessing approaches to see if we can get better results.
    """
    img = Image.open(filepath)
    
    shop_coords = [
        (485, 1045, 105, 20),  # slot 1
        (685, 1045, 105, 20),  # slot 2
        (885, 1045, 105, 20),  # slot 3
        (1090, 1045, 105, 20),  # slot 4
        (1290, 1045, 105, 20),  # slot 5
    ]
    
    # Try different preprocessing approaches
    approaches = [
        ("Current Method", lambda crop: preprocess_current(crop)),
        ("Higher Contrast", lambda crop: preprocess_higher_contrast(crop)),
        ("Gaussian Blur + Sharpen", lambda crop: preprocess_blur_sharpen(crop)),
        ("Threshold Only", lambda crop: preprocess_threshold_only(crop)),
        ("Adaptive Threshold", lambda crop: preprocess_adaptive_threshold(crop)),
    ]
    
    fig, axes = plt.subplots(len(shop_coords), len(approaches), figsize=(25, len(shop_coords) * 3))
    fig.suptitle('Alternative Preprocessing Methods Comparison', fontsize=16)
    
    # Column titles
    for j, (method_name, _) in enumerate(approaches):
        axes[0, j].set_title(method_name, fontweight='bold')
    
    all_results = {method: [] for method, _ in approaches}
    
    for i, (x, y, w, h) in enumerate(shop_coords):
        crop = img.crop((x, y, x+w, y+h))
        
        axes[i, 0].set_ylabel(f'Slot {i+1}', fontweight='bold')
        
        for j, (method_name, preprocess_func) in enumerate(approaches):
            processed = preprocess_func(crop)
            
            # Display processed image
            axes[i, j].imshow(processed, cmap='gray')
            axes[i, j].axis('off')
            
            # Try OCR
            text = pytesseract.image_to_string(processed, lang="eng", config='--psm 8')
            all_results[method_name].append(text.strip())
            
            # Add OCR result
            axes[i, j].text(0.5, -0.1, f"'{text.strip()}'", 
                           transform=axes[i, j].transAxes, 
                           ha='center', va='top', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.show()
    
    return all_results

def preprocess_current(crop):
    """Current preprocessing method"""
    gray = ImageOps.grayscale(crop)
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    inverted = ImageOps.invert(enhanced)
    bw = inverted.point(lambda p: 0 if p < 128 else 255)
    return bw

def preprocess_higher_contrast(crop):
    """Higher contrast enhancement"""
    gray = ImageOps.grayscale(crop)
    enhanced = ImageEnhance.Contrast(gray).enhance(3.0)
    inverted = ImageOps.invert(enhanced)
    bw = inverted.point(lambda p: 0 if p < 100 else 255)
    return bw

def preprocess_blur_sharpen(crop):
    """Gaussian blur followed by sharpening"""
    from PIL import ImageFilter
    gray = ImageOps.grayscale(crop)
    # Slight blur to reduce noise
    blurred = gray.filter(ImageFilter.GaussianBlur(0.5))
    # Sharpen
    sharpened = blurred.filter(ImageFilter.SHARPEN)
    enhanced = ImageEnhance.Contrast(sharpened).enhance(2.5)
    inverted = ImageOps.invert(enhanced)
    bw = inverted.point(lambda p: 0 if p < 128 else 255)
    return bw

def preprocess_threshold_only(crop):
    """Simple thresholding without inversion"""
    gray = ImageOps.grayscale(crop)
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    # Try without inversion - maybe text is already dark
    bw = enhanced.point(lambda p: 0 if p < 128 else 255)
    return bw

def preprocess_adaptive_threshold(crop):
    """Adaptive thresholding using numpy"""
    gray = ImageOps.grayscale(crop)
    enhanced = ImageEnhance.Contrast(gray).enhance(2.0)
    
    # Convert to numpy array
    np_img = np.array(enhanced)
    
    # Simple adaptive threshold (mean of neighborhood)
    threshold = np.mean(np_img)
    adaptive_bw = np.where(np_img > threshold, 255, 0).astype(np.uint8)
    
    return Image.fromarray(adaptive_bw)

if __name__ == "__main__":
    # Use the screenshot from your script
    screenshot = "C:\\Users\\Alex\\Desktop\\automatic\\tft_recognition\\data\\raw_screenshots\\20250923_172855.png"
    
    print("=== Debugging Current Preprocessing ===")
    results = debug_shop_preprocessing(screenshot)
    print("Current method results:", results)
    
    print("\n=== Trying Alternative Preprocessing Methods ===")
    alt_results = try_alternative_preprocessing(screenshot)
    
    print("\nComparison of all methods:")
    for method, results in alt_results.items():
        print(f"{method}: {results}")