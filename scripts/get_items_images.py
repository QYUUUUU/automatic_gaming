from playwright.sync_api import sync_playwright
import os
import requests

URL = "https://www.metatft.com/items"
OUTPUT_DIR = "tft_items"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    return None

def scrape_items():
    items_downloaded = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, timeout=60000)
        page.wait_for_selector("table tbody tr")

        rows = page.query_selector_all("table tbody tr")
        for row in rows:
            img = row.query_selector("td:nth-child(1) img")
            if img:
                src = img.get_attribute("src")
                alt = img.get_attribute("alt")
                if src and alt:
                    # Strip CDN resizing prefix if present
                    if "cdn-cgi/image" in src:
                        src = src.split("/https://")[-1]
                        src = "https://" + src
                    filename = f"{alt.replace(' ', '_')}.png"
                    filepath = download_image(src, filename)
                    if filepath:
                        print(f"Downloaded {alt} → {filepath}")
                        items_downloaded.append(filepath)

        browser.close()
    return items_downloaded

if __name__ == "__main__":
    items = scrape_items()
    print(f"✅ Downloaded {len(items)} items")
