import json
import base64
from PIL import Image
import io
import google.generativeai as genai
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()
# --- CONFIGURATION ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
INPUT_FILE = "tds_discourse_data.json"
OUTPUT_FILE = "tds_discourse_with_ocr_RESUMED.json"

# --- OCR Function with Retry on Quota Error ---
def run_ocr_on_image(base64_string):
    image_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_data))

    if img.width < 32 or img.height < 32:
        return ""

    model = genai.GenerativeModel("gemini-2.0-flash")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    while True:
        try:
            response = model.generate_content([
                {"mime_type": "image/png", "data": image_bytes},
                """ You are an OCR and image-context assistant. Return exactly two sentences and nothing else.
1. First sentence: verbatim extract of all text in the image.
2. Second sentence: a concise explanation of the image’s context.
Do not include headings, labels, bullet points, or extra commentary."""
            ])
            time.sleep(3)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e):
                print("[Retry] Rate limit hit. Waiting 30 seconds...")
                time.sleep(30)
            else:
                print(f"[Error] OCR failed: {e}")
                return ""

# --- LOAD INPUT FILE ---
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- PROCESS ONLY UNFINISHED POSTS ---
updated_count = 0
for post in tqdm(data, desc="Resuming OCR"):
    if post.get("extracted_image_text"):  # Already processed
        continue

    extracted_texts = []
    for b64img in post.get("images_base64", []):
        try:
            ocr_result = run_ocr_on_image(b64img)
            if ocr_result:
                extracted_texts.append(ocr_result)
        except Exception as e:
            print(f"[Error] OCR failed: {e}")
    post["extracted_image_text"] = "\n---\n".join(extracted_texts)
    updated_count += 1

# --- SAVE FINAL OUTPUT ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ OCR resumed and completed. {updated_count} new posts processed and saved to {OUTPUT_FILE}.")
