import os
import random
import shutil

# ==========================
# CONFIG
# ==========================
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# ==========================
# PATH SETUP (AMAN)
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_COLOR_DIR   = os.path.join(BASE_DIR, "data_raw", "batik_color")
RAW_OUTLINE_DIR = os.path.join(BASE_DIR, "data_raw", "batik_outline")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ==========================
# BUAT FOLDER OUTPUT
# ==========================
def make_dirs():
    for split in ["train", "val", "test"]:
        for sub in ["A", "B"]:
            os.makedirs(os.path.join(DATASET_DIR, split, sub), exist_ok=True)

# ==========================
# MAIN SPLIT FUNCTION
# ==========================
def split_dataset():
    files = sorted([
        f for f in os.listdir(RAW_COLOR_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    assert len(files) > 0, "❌ Tidak ada gambar di batik_color!"
    
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end   = train_end + int(total * VAL_RATIO)

    splits = {
        "train": files[:train_end],
        "val":   files[train_end:val_end],
        "test":  files[val_end:]
    }

    for split, file_list in splits.items():
        for fname in file_list:
            src_A = os.path.join(RAW_COLOR_DIR, fname)
            src_B = os.path.join(RAW_OUTLINE_DIR, fname)

            if not os.path.exists(src_B):
                print(f"⚠ Outline tidak ditemukan untuk {fname}, dilewati")
                continue

            dst_A = os.path.join(DATASET_DIR, split, "A", fname)
            dst_B = os.path.join(DATASET_DIR, split, "B", fname)

            shutil.copy(src_A, dst_A)
            shutil.copy(src_B, dst_B)

        print(f"✅ {split}: {len(file_list)} pasangan")

    print("\n🎉 Dataset berhasil di-split!")

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    make_dirs()
    split_dataset()
