import cv2
import os
import numpy as np

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 256

# ==========================
# 1. LOAD IMAGE
# ==========================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ==========================
# 2. RESIZE IMAGE
# ==========================
def resize_image(img, size=IMG_SIZE):
    return cv2.resize(img, (size, size))


# ==========================
# 3. SEMI-AUTOMATIC TRACING
# ==========================
def generate_outline(img):
    """
    Input  : RGB image
    Output : outline image (hitam-putih)
    """
    # Convert ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur ringan untuk mengurangi noise kain
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection (Canny)
    edges = cv2.Canny(
        blur,
        threshold1=50,
        threshold2=150
    )

    # Invert: garis hitam, background putih
    outline = 255 - edges

    return outline


# ==========================
# 4. NORMALIZATION
# ==========================
def normalize(img):
    """
    Convert ke range [-1, 1]
    """
    img = img.astype(np.float32) / 127.5 - 1.0
    return img


# ==========================
# 5. SAVE IMAGE
# ==========================
def save_image(path, img):
    """
    img: RGB atau grayscale
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, img)


# ==========================
# 6. FULL PIPELINE (SATU GAMBAR)
# ==========================
def process_single_image(
    input_path,
    output_color_path,
    output_outline_path
):
    # Load
    img = load_image(input_path)

    # Resize
    img_resized = resize_image(img)

    # Generate outline
    outline = generate_outline(img_resized)

    # Save
    save_image(output_color_path, img_resized)
    save_image(output_outline_path, outline)
