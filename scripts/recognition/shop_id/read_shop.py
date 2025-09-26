from PIL import Image, ImageEnhance, ImageOps
import pytesseract
import matplotlib.pyplot as plt

def correct_common_mistakes(text):
    """
    Corrects common OCR mistakes for TFT champion names.
    """
    if not text:
        return text
    
    # List of valid TFT champion names (add more as needed)
    valid_champions = [
        'Aatrox', 'Akali', 'Ashe', 'Azir', 'Bard', 'Blitzcrank', 'Brand', 'Braum',
        'Camille', 'Cassiopeia', 'Cho\'Gath', 'Darius', 'Diana', 'Dr. Mundo', 'Draven',
        'Elise', 'Ezreal', 'Fiora', 'Galio', 'Garen', 'Gnar', 'Graves', 'Gwen',
        'Hecarim', 'Heimerdinger', 'Janna', 'Jayce', 'Jhin', 'Jinx', 'Kalista', 'Karma',
        'Karthus', 'Kassadin', 'Katarina', 'Kayle', 'Kennen', 'Kog\'Maw', 'LeBlanc',
        'Lee Sin', 'Leona', 'Lissandra', 'Lucian', 'Lulu', 'Lux', 'Malphite', 'Maokai',
        'Miss Fortune', 'Mordekaiser', 'Morgana', 'Nami', 'Nasus', 'Neeko', 'Nilah',
        'Nocturne', 'Nunu', 'Olaf', 'Ornn', 'Poppy', 'Pyke', 'Rakan', 'Rek\'Sai',
        'Rell', 'Renekton', 'Riven', 'Rumble', 'Ryze', 'Sejuani', 'Senna', 'Seraphine',
        'Sett', 'Shen', 'Shyvana', 'Sivir', 'Skarner', 'Swain', 'Syndra', 'Tahm Kench',
        'Talon', 'Taric', 'Thresh', 'Tristana', 'Twisted Fate', 'Urgot', 'Varus',
        'Vayne', 'Veigar', 'Vel\'Koz', 'Vi', 'Viego', 'Viktor', 'Vladimir', 'Volibear',
        'Warwick', 'Xayah', 'Xerath', 'Xin Zhao', 'Yasuo', 'Yone', 'Yuumi', 'Zac',
        'Zed', 'Ziggs', 'Zilean', 'Zoe', 'Zyra', 'Kobuko', "Smolder"
    ]
    
    # Direct corrections for known mistakes
    corrections = {
        'kayley': 'Kayle',
        'Kayley': 'Kayle',
        'KAYLEY': 'Kayle',
    }
    
    # First try direct corrections
    for mistake, correction in corrections.items():
        if mistake.lower() == text.lower():
            return correction
    
    # Then try fuzzy matching against valid champion names
    text_lower = text.lower()
    for champion in valid_champions:
        champion_lower = champion.lower()
        # Simple similarity check - if most characters match
        if len(text) > 0 and len(champion) > 0:
            # Calculate simple character similarity
            matches = sum(1 for a, b in zip(text_lower, champion_lower) if a == b)
            similarity = matches / max(len(text_lower), len(champion_lower))
            
            # If similarity is high enough, use the correct champion name
            if similarity >= 0.7:  # 70% similarity threshold
                return champion
    
    return text

def read_tft_shop(filepath):
    """
    Reads the 5 TFT shop slots from a screenshot using pytesseract,
    with preprocessing to make light text on dark background more readable.
    
    :param filepath: Path to the screenshot
    :return: List of extracted texts
    """
    img = Image.open(filepath)
    results = []

    shop_coords = [
        (485, 1045, 105, 20),  # slot 1
        (685, 1045, 105, 20),  # slot 2
        (885, 1045, 105, 20),  # slot 3
        (1090, 1045, 105, 20),  # slot 4
        (1290, 1045, 105, 20),  # slot 5
    ]
    
    for (x, y, w, h) in shop_coords:
        crop = img.crop((x, y, x+w, y+h))
        
        # Improved preprocessing for better OCR accuracy
        # Scale up the image for better OCR
        scale_factor = 3
        crop = crop.resize((w * scale_factor, h * scale_factor), Image.LANCZOS)
        
        gray = ImageOps.grayscale(crop)
        enhanced = ImageEnhance.Contrast(gray).enhance(2.5)  # Slightly higher contrast
        
        # Try adaptive thresholding or different threshold value
        bw = enhanced.point(lambda p: 0 if p < 140 else 255)  # Adjusted threshold
        
        # Try multiple PSM modes and configurations
        configs = [
            '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            '--psm 8',
            '--psm 13'
        ]
        
        # Try each config and pick the best result
        best_text = ""
        best_confidence = 0
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(bw, lang="eng", config=config).strip()
                if text and len(text) > len(best_text):  # Prefer longer, more complete text
                    best_text = text
            except:
                continue
        
        # Post-process common OCR mistakes
        text = correct_common_mistakes(best_text)
        results.append(text)
    return results

def preview_coordinates(filepath, shop_coords):
    """
    Draws rectangles around the shop slots to help find correct coordinates.
    """
    img = Image.open(filepath)
    plt.imshow(img)
    
    ax = plt.gca()
    for (x, y, w, h) in shop_coords:
        rect = plt.Rectangle((x, y), w, h, edgecolor="red", facecolor="none", linewidth=2)
        ax.add_patch(rect)
    
    plt.show()