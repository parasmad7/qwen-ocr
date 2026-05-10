import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import json
from tqdm import tqdm
import numpy as np
import textwrap

try:
    from datasets import load_dataset
    print("Loading Wikipedia text corpus for diverse vocabulary...")
    wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    WIKI_TEXTS = [t.strip() for t in wiki_dataset["text"] if len(t.strip()) > 50]
except Exception as e:
    print(f"Failed to load wikitext: {e}. Falling back to limited vocabulary.")
    WIKI_TEXTS = []

# Configuration
FONT_DIR = "data/fonts"
PAGE_SIZE = (1024, 1280) # Width, Height

# List of some popular handwriting fonts from Google Fonts (direct links)
FONTS = {
    "Caveat": "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat-Regular.ttf",
    "DancingScript": "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript-Regular.ttf",
    "IndieFlower": "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "ShadowsIntoLight": "https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "PatrickHand": "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"
}



def download_fonts():
    os.makedirs(FONT_DIR, exist_ok=True)
    font_paths = []
    for name, url in FONTS.items():
        path = os.path.join(FONT_DIR, f"{name}.ttf")
        if not os.path.exists(path):
            print(f"Downloading {name} font...")
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(r.content)
                font_paths.append(path)
            except Exception as e:
                print(f"Failed to download {name} font from {url}: {e}")
        else:
            font_paths.append(path)
    
    return font_paths

def generate_random_text():
    num_lines = random.randint(10, 25)
    
    if WIKI_TEXTS:
        return "\n".join(random.choices(WIKI_TEXTS, k=num_lines))
        
    # Fallback to limited corpus if datasets fails
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Handwriting is a lost art in the digital age.",
        "Optical Character Recognition has come a long way.",
        "Full page transcription is a challenging task for VLMs.",
        "We are training Qwen3.5 to recognize various handwriting styles.",
        "The sun sets over the horizon, painting the sky in shades of orange.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        "In the middle of every difficulty lies opportunity.",
        "Imagination is more important than knowledge.",
        "The only way to do great work is to love what you do.",
        "Life is what happens when you're busy making other plans.",
        "The future belongs to those who believe in the beauty of their dreams.",
        "It does not matter how slowly you go as long as you do not stop.",
        "Everything you've ever wanted is on the other side of fear.",
        "Believe you can and you're halfway there.",
        "Do not go where the path may lead, go instead where there is no path and leave a trail.",
        "What you get by achieving your goals is not as important as what you become by achieving your goals.",
        "The best way to predict the future is to create it.",
        "The only limit to our realization of tomorrow will be our doubts of today.",
        "Keep your face always toward the sunshine - and shadows will fall behind you."
    ]
    return "\n".join(random.choices(sentences, k=num_lines))

def create_synthetic_page(text, font_path, output_path):
    # 1. Create a blank "paper" image (White paper)
    bg_color = (255, 255, 255)
    image = Image.new("RGB", PAGE_SIZE, bg_color)
    draw = ImageDraw.Draw(image)
    
    # 2. Select font and size
    font_size = random.randint(28, 40)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    # 3. Draw "Rules" (Lines)
    line_spacing = font_size + 20
    margin_top = 80
    margin_left = 100
    
    # Draw horizontal blue lines
    rule_color = (200, 220, 255) # Light blue
    for y in range(margin_top + line_spacing, PAGE_SIZE[1], line_spacing):
        draw.line([(0, y), (PAGE_SIZE[0], y)], fill=rule_color, width=1)
        
    # Draw vertical red margin line
    margin_color = (255, 200, 200) # Light red
    draw.line([(margin_left, 0), (margin_left, PAGE_SIZE[1])], fill=margin_color, width=2)
    
    # 4. Wrap and Draw text aligned to lines
    margin_right = 50
    wrap_width = (PAGE_SIZE[0] - margin_left - margin_right) // (font_size // 2) # Rough estimate in chars
    
    wrapped_lines = []
    for line in text.split("\n"):
        wrapped_lines.extend(textwrap.wrap(line, width=max(20, wrap_width)))
    
    text_elements = []
    current_y = margin_top + line_spacing # Start at the first rule
    
    for line in wrapped_lines:
        if not line.strip():
            current_y += line_spacing
            continue

        # Create a temporary image for the line to allow individual line rotation
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        w, h = bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 40
        
        line_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        line_draw = ImageDraw.Draw(line_img)
        
        # Random ink color
        ink_color = (random.randint(0, 30), random.randint(0, 30), random.randint(40, 100))
        
        # Draw text slightly above the bottom of line_img to allow for descenders
        line_draw.text((20, 20), line, font=font, fill=ink_color)
        
        # Rotation drift: more tilt as we approach the right side (or just random variance)
        # We also add a subtle "vertical-ish" drift for the whole line if it's long
        line_angle = random.uniform(-1.2, 0.8) 
        if len(line) > wrap_width * 0.8:
            line_angle += random.uniform(-0.5, 0.5) # Extra drift for full lines
            
        line_img = line_img.rotate(line_angle, resample=Image.BICUBIC, expand=True)
        
        # Alignment: Position the line so its baseline sits ON the rule
        # The line_img height includes padding, so we adjust current_y
        x_offset = random.randint(5, 15)
        y_offset = random.randint(-4, 2)
        
        # Adjust y to sit on the rule line
        # Rule is at current_y. Text is drawn at 20px in line_img.
        # We want the baseline (roughly h-20) to be at current_y.
        paste_y = current_y - (h - 20) + y_offset
        
        # Irregular Inter-line spacing
        # Randomly skip a line or add extra space
        if random.random() > 0.9:
            current_y += line_spacing # Skip a rule
            
        # Paste the rotated line onto the main image
        image.paste(line_img, (margin_left + x_offset, paste_y), line_img)
        text_elements.append({'text': line, 'x': margin_left + x_offset, 'y': paste_y})
        
        current_y += line_spacing + random.randint(-2, 2) # Slight spacing jitter
        
        if current_y > PAGE_SIZE[1] - 50:
            break
            
    # 5. Add Marginalia (2-3 words stacked vertically on the extreme right)
    if random.random() > 0.3:
        side_words = random.choices(text.split(), k=random.randint(2, 3))
        side_y = random.randint(margin_top, PAGE_SIZE[1] // 2)
        side_x = PAGE_SIZE[0] - random.randint(40, 80)
        
        for word in side_words:
            # Create a small image for the side word
            dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bbox = dummy_draw.textbbox((0, 0), word, font=font)
            sw, sh = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20
            
            sword_img = Image.new("RGBA", (sw, sh), (255, 255, 255, 0))
            sword_draw = ImageDraw.Draw(sword_img)
            
            # Use a slightly different ink color for marginalia
            s_ink = (random.randint(100, 150), 0, 0) # Reddish ink for notes
            sword_draw.text((10, 10), word, font=font, fill=s_ink)
            
            # Rotate vertically (90 degrees)
            sword_img = sword_img.rotate(random.choice([90, -90, 0]), resample=Image.BICUBIC, expand=True)
            
            image.paste(sword_img, (side_x, side_y), sword_img)
            text_elements.append({'text': word, 'x': side_x, 'y': side_y})
            side_y += sword_img.height + 10
            
    # 6. Add slight global page rotation
    angle = random.uniform(-1.0, 1.0)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    
    # 6. Apply camera/lighting effects
    # Brightness and Contrast jitter
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.9, 1.1))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.9, 1.1))
    
    # Very subtle Gaussian Blur (simulating focus)
    if random.random() > 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.5)))
    
    # 7. Save the image
    image.save(output_path)
    
    # Sort elements top-to-bottom
    text_elements.sort(key=lambda e: e['y'])
    
    lines = []
    current_line = []
    current_y = None
    y_threshold = line_spacing / 2
    
    for el in text_elements:
        if current_y is None:
            current_y = el['y']
            current_line.append(el)
        elif abs(el['y'] - current_y) <= y_threshold:
            current_line.append(el)
        else:
            current_line.sort(key=lambda e: e['x'])
            lines.append(current_line)
            current_line = [el]
            current_y = el['y']
            
    if current_line:
        current_line.sort(key=lambda e: e['x'])
        lines.append(current_line)
        
    final_lines_text = []
    avg_char_width = font_size * 0.5
    for line_elements in lines:
        line_str = ""
        last_x_end = margin_left
        
        for el in line_elements:
            gap = el['x'] - last_x_end
            if gap > avg_char_width * 2 and line_str != "":
                num_spaces = int(gap / avg_char_width)
                line_str += " " * min(num_spaces, 20)
            elif line_str != "":
                line_str += " "
                
            line_str += el['text']
            
            dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bbox = dummy_draw.textbbox((0, 0), el['text'], font=font)
            last_x_end = el['x'] + (bbox[2] - bbox[0])
            
        final_lines_text.append(line_str.strip())
        
    return "\n".join(final_lines_text)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/synthetic_train")
    parser.add_argument("--num_samples", type=int, default=5000)
    args = parser.parse_args()
    
    output_dir = args.output_dir
    num_samples = args.num_samples

    font_paths = download_fonts()
    if not font_paths:
        print("Error: No fonts found! Please download fonts manually or ensure system fonts are available.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []
    
    print(f"Generating {num_samples} synthetic pages in {output_dir}...")
    for i in tqdm(range(num_samples)):
        text = generate_random_text()
        font_path = random.choice(font_paths)
        filename = f"page_{i:04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        final_text = create_synthetic_page(text, font_path, output_path)
        
        metadata.append({
            "image": filename,
            "text": final_text
        })
        
    with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Generation complete. Files saved to {output_dir}")

if __name__ == "__main__":
    main()
