import os
import random
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import json
from tqdm import tqdm
import numpy as np
import textwrap
import argparse

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

# List of handwriting fonts (Already downloaded in data/fonts)
# FONTS = {
#     "Caveat": "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat-Regular.ttf",
#     "DancingScript": "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript-Regular.ttf",
#     "IndieFlower": "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf",
#     "ShadowsIntoLight": "https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf",
#     "PatrickHand": "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"
# }

def get_font_paths():
    os.makedirs(FONT_DIR, exist_ok=True)
    font_paths = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(".ttf")]
    return font_paths

def generate_random_text():
    num_lines = random.randint(10, 25)
    if WIKI_TEXTS:
        return "\n".join(random.choices(WIKI_TEXTS, k=num_lines))
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Handwriting is a lost art in the digital age.",
        "Optical Character Recognition has come a long way.",
        "Selective OCR involves ignoring crossed out text.",
        "We are training the model to be robust to deletions.",
        "Success is not final, failure is not fatal.",
        "Believe you can and you're halfway there."
    ]
    return "\n".join(random.choices(sentences, k=num_lines))

def draw_strikeout(draw, bbox, style="line", color=(0, 0, 0)):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    width = random.randint(2, 4)
    
    if style == "line":
        y_mid = y1 + h/2 + random.randint(-2, 2)
        draw.line([(x1-2, y_mid), (x2+2, y_mid)], fill=color, width=width)
    
    elif style == "double_line":
        y_mid = y1 + h/2
        draw.line([(x1-2, y_mid - 3), (x2+2, y_mid - 3)], fill=color, width=width-1)
        draw.line([(x1-2, y_mid + 3), (x2+2, y_mid + 3)], fill=color, width=width-1)
        
    elif style == "wavy":
        num_points = max(5, int(w / 10))
        points = []
        for i in range(num_points + 1):
            px = x1 + (w * i / num_points)
            py = y1 + h/2 + random.randint(-8, 8)
            points.append((px, py))
        draw.line(points, fill=color, width=width, joint="curve")
        
    elif style == "cross":
        step = w / random.randint(2, 4)
        for i in np.arange(x1, x2, step):
            draw.line([(i, y1), (min(i+step, x2), y2)], fill=color, width=width-1)
            draw.line([(min(i+step, x2), y1), (i, y2)], fill=color, width=width-1)
            
    elif style == "scribble":
        for _ in range(random.randint(5, 10)):
            pts = [(random.randint(int(x1), int(x2)), random.randint(int(y1), int(y2))) for _ in range(4)]
            draw.line(pts, fill=color, width=width-1, joint="curve")

def create_selective_page(text, font_path, output_path, word_strikeout_prob=0.2, page_has_strikeout=True):
    image = Image.new("RGB", PAGE_SIZE, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    font_size = random.randint(30, 42)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        
    line_spacing = font_size + 24
    margin_top, margin_left, margin_right = 80, 100, 60
    
    for y in range(margin_top + line_spacing, PAGE_SIZE[1], line_spacing):
        draw.line([(0, y), (PAGE_SIZE[0], y)], fill=(210, 230, 255), width=1)
    draw.line([(margin_left, 0), (margin_left, PAGE_SIZE[1])], fill=(255, 210, 210), width=2)
    
    wrap_width = (PAGE_SIZE[0] - margin_left - margin_right) // (font_size // 2)
    wrapped_lines = []
    for raw_line in text.split("\n"):
        wrapped_lines.extend(textwrap.wrap(raw_line, width=max(20, wrap_width)))
        
    text_elements = []
    current_y = margin_top + line_spacing
    
    for line in wrapped_lines:
        if not line.strip() or current_y > PAGE_SIZE[1] - 60:
            current_y += line_spacing
            continue
            
        words = line.split()
        x_cursor = margin_left + random.randint(5, 15)
        ink_color = (random.randint(0, 40), random.randint(0, 40), random.randint(50, 120))
        
        for word in words:
            # Word dimensions for the temporary canvas
            dummy_bbox = draw.textbbox((0, 0), word, font=font)
            w, h = dummy_bbox[2] - dummy_bbox[0], dummy_bbox[3] - dummy_bbox[1]
            
            # Create word image with padding
            word_img = Image.new("RGBA", (w + 40, h + 40), (255, 255, 255, 0))
            word_draw = ImageDraw.Draw(word_img)
            
            # Draw text at a fixed offset
            text_x, text_y = 20, 20
            word_draw.text((text_x, text_y), word, font=font, fill=ink_color)
            
            # Get PRECISE bounding box of the actual characters
            char_bbox = word_draw.textbbox((text_x, text_y), word, font=font)
            
            is_crossed = page_has_strikeout and (random.random() < word_strikeout_prob)
            
            if is_crossed:
                style = random.choice(["line", "double_line", "wavy", "cross", "scribble"])
                # Draw strikeout exactly over the characters
                draw_strikeout(word_draw, char_bbox, style=style, color=ink_color)
            else:
                # Add to ground truth
                text_elements.append({'text': word, 'x': x_cursor, 'y': current_y, 'w': w})
            
            angle = random.uniform(-2, 2)
            word_img = word_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            paste_y = current_y - h + random.randint(-2, 2)
            image.paste(word_img, (x_cursor, paste_y), word_img)
            x_cursor += word_img.width - 10
            
        current_y += line_spacing + random.randint(-2, 2)
        
    image = image.rotate(random.uniform(-0.5, 0.5), resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.95, 1.05))
    image.save(output_path)
    
    text_elements.sort(key=lambda e: e['y'])
    final_lines = []
    if text_elements:
        curr_line_words = [text_elements[0]]
        for i in range(1, len(text_elements)):
            if abs(text_elements[i]['y'] - curr_line_words[-1]['y']) < line_spacing / 2:
                curr_line_words.append(text_elements[i])
            else:
                curr_line_words.sort(key=lambda e: e['x'])
                final_lines.append(" ".join([w['text'] for w in curr_line_words]))
                curr_line_words = [text_elements[i]]
        if curr_line_words:
            curr_line_words.sort(key=lambda e: e['x'])
            final_lines.append(" ".join([w['text'] for w in curr_line_words]))
            
    return "\n".join(final_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base", type=str, default="data/selective")
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--word_strikeout_prob", type=float, default=0.2, help="Prob of a word being crossed out if the page allows it")
    parser.add_argument("--page_strikeout_prob", type=float, default=0.7, help="Prob of a page containing any strikeouts at all")
    args = parser.parse_args()
    
    font_paths = get_font_paths()
    
    for split, num_samples in [("train", args.num_train), ("test", args.num_test)]:
        output_dir = os.path.join(args.output_base, split)
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = []
        print(f"Generating {num_samples} selective OCR pages for {split}...")
        
        for i in tqdm(range(num_samples)):
            text = generate_random_text()
            font_path = random.choice(font_paths)
            filename = f"selective_{i:04d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Decide if this page should have any strikeouts
            has_strikeout = random.random() < args.page_strikeout_prob
            
            final_text = create_selective_page(
                text, font_path, output_path, 
                word_strikeout_prob=args.word_strikeout_prob, 
                page_has_strikeout=has_strikeout
            )
            
            metadata.append({
                "image": filename,
                "text": final_text
            })
            
        with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")
                
    print(f"Complete. Dataset saved to {args.output_base}")

if __name__ == "__main__":
    main()
