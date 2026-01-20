import os
from utils import process_single_image

# ==========================
# PATH SETUP (AMAN)
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_COLOR_DIR = os.path.join(BASE_DIR, "data_raw", "batik_color")
RAW_OUTLINE_DIR = os.path.join(BASE_DIR, "data_raw", "batik_outline")

os.makedirs(RAW_OUTLINE_DIR, exist_ok=True)

files = sorted(os.listdir(RAW_COLOR_DIR))

for fname in files:
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    input_path = os.path.join(RAW_COLOR_DIR, fname)
    color_out = os.path.join(RAW_COLOR_DIR, fname)
    outline_out = os.path.join(RAW_OUTLINE_DIR, fname)

    process_single_image(
        input_path=input_path,
        output_color_path=color_out,
        output_outline_path=outline_out
    )

    print(f"✔ Processed {fname}")

print("✅ Dataset outline selesai dibuat.")
