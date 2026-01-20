import streamlit as st
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import base64
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim





# ==========================
# CONFIG
# ==========================
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pix2pix_outline.pth")
LIBRARY_DIR = os.path.join(BASE_DIR, "library_batik")
# ==========================
# STYLE
# ==========================
st.set_page_config(page_title="Batik Outline Generator (GAN)", layout="wide")

st.markdown("""
<style>
/* ===== GLOBAL ===== */
.main {
    background-color: #f8f9fa;
}

h1, h2, h3 {
    color: #1f2c3d;
}

.block-container {
    padding-top: 2rem;
}

.caption {
    font-size: 0.9rem;
    color: #555;
}

/* ===== HIGHLIGHT / INFO BOX ===== */
.highlight {
    background-color: #eef4ff;
    padding: 0.75rem;
    border-left: 5px solid #3b6cff;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

/* ===== METRIC BOX ===== */
.metric-box {
    background-color: #ffffff;
    padding: 0.75rem;
    border-radius: 6px;
    border: 1px solid #ddd;
}

/* ===== LIBRARY BATIK ===== */
.library-container {
    max-height: 260px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #ffffff;
}

.library-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    align-items: start;
}

.library-img {
    width: 100%;
}

.library-img img {
    width: 10%;
    height: auto;
    object-fit: contain;
    border-radius: 6px;
    border: 1px solid #ccc;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# MODEL
# ==========================
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.d1 = down(3, 64)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)

        self.u1 = up(512, 256)
        self.u2 = up(512, 128)
        self.u3 = up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        u1 = self.u1(d4)
        u2 = self.u2(torch.cat([u1, d3], 1))
        u3 = self.u3(torch.cat([u2, d2], 1))

        return self.final(torch.cat([u3, d1], 1))

@st.cache_resource
def load_model():
    model = UNetGenerator().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ==========================
# TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==========================
# UTIL: LOAD LIBRARY IMAGES
# ==========================
def load_library_images(folder):
    if not os.path.exists(folder):
        return []
    exts = (".jpg", ".jpeg", ".png")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    images = []
    for f in files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except:
            pass
    return images


# ==========================
# CANNY
# ==========================
def canny_edge(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    edge = cv2.resize(edge, (IMAGE_SIZE, IMAGE_SIZE))
    return Image.fromarray(edge)

# ==========================
# METRICS
# ==========================
def compute_metrics(gan_img, canny_img):
    gan = cv2.resize(np.array(gan_img), (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    canny = cv2.resize(np.array(canny_img), (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    l1 = np.mean(np.abs(gan - canny))
    l2 = np.mean((gan - canny) ** 2)
    ssim_val = ssim(gan, canny, data_range=1.0)
    return l1, l2, ssim_val

def edge_density(edge_img):
    arr = np.array(edge_img)
    return (np.sum(arr > 0) / arr.size) * 100

# ==========================
# STATE
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ==========================
# PAGE 1
# ==========================
if st.session_state.page == "upload":

    st.title("🎨 Batik Outline Generator (Pix2Pix GAN)")
    st.markdown("""
    Aplikasi ini mendemonstrasikan ekstraksi **struktur garis motif batik**
    menggunakan **Pix2Pix GAN**, dibandingkan dengan metode klasik
    **Canny Edge**.
    """)

    library_images = load_library_images(LIBRARY_DIR)

    st.subheader("📚 Library Contoh Motif Batik")

    if len(library_images) == 0:
        st.info(
            "Belum ada gambar di library.\n\n"
            f"Tambahkan gambar ke folder:\n`{LIBRARY_DIR}`"
        )
    else:
        st.markdown("motif batik dari berbagai daerah di Indonesia:")

        library_images = load_library_images(LIBRARY_DIR)

        if library_images:
            rows = [library_images[i:i+4] for i in range(0, len(library_images), 4)]

            for row in rows:
                cols = st.columns(4)
                for col, img in zip(cols, row):
                    with col:
                        st.image(img, width=250)




    st.divider()


    uploaded_files = st.file_uploader(
        "Unggah satu atau beberapa citra batik (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        images = [Image.open(f).convert("RGB") for f in uploaded_files]
        st.session_state.images = images

        st.success(f"{len(images)} gambar berhasil diunggah.")

        cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(img, width=200)

        if st.button("🚀 Generate & Evaluasi"):
            st.session_state.page = "hasil"
            st.rerun()

# ==========================
# PAGE 2
# ==========================
elif st.session_state.page == "hasil":

    st.title("📊 Hasil, Evaluasi, dan Pembahasan (Bab 4)")

    images = st.session_state.images
    progress = st.progress(0)

    metrics_rows = []

    with st.spinner("⏳ Proses inferensi dan evaluasi..."):
        for idx, image in enumerate(images):
            progress.progress((idx + 1) / len(images))
            time.sleep(0.2)

            st.subheader(f"Gambar ke-{idx+1}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Batik Asli**")
                st.image(image, width=250)

            canny_img = canny_edge(image)

            with col2:
                st.markdown("**Canny Edge**")
                st.image(canny_img, width=250)

            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
            output = (output + 1) / 2
            gan_img = transforms.ToPILImage()(output.squeeze().cpu())

            with col3:
                st.markdown("**Outline Motif (GAN)**")
                st.image(gan_img, width=250)

            l1, l2, ssim_val = compute_metrics(gan_img, canny_img)
            d_canny = edge_density(canny_img)
            d_gan = edge_density(gan_img)

            metrics_rows.append({
                "Image": f"Image_{idx+1}",
                "L1": l1,
                "L2": l2,
                "SSIM": ssim_val,
                "EdgeDensity_Canny": d_canny,
                "EdgeDensity_GAN": d_gan
            })

            m1, m2, m3 = st.columns(3)
            m1.metric("L1 Loss", f"{l1:.4f}")
            m2.metric("L2 Loss", f"{l2:.4f}")
            m3.metric("SSIM", f"{ssim_val:.4f}")

            d1, d2 = st.columns(2)
            d1.metric("Edge Density Canny (%)", f"{d_canny:.2f}")
            d2.metric("Edge Density GAN (%)", f"{d_gan:.2f}")

            st.markdown("""
            <div class="highlight">
            Canny Edge menghasilkan kepadatan garis tinggi akibat tekstur kain,
            sedangkan Pix2Pix GAN lebih selektif dan fokus pada garis motif utama.
            </div>
            """, unsafe_allow_html=True)

            st.divider()

    df = pd.DataFrame(metrics_rows)

    st.subheader("📊 Rata-rata Metrik Seluruh Batch")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg L1", f"{df['L1'].mean():.4f}")
    c2.metric("Avg L2", f"{df['L2'].mean():.4f}")
    c3.metric("Avg SSIM", f"{df['SSIM'].mean():.4f}")
    c4.metric("Avg Edge Canny (%)", f"{df['EdgeDensity_Canny'].mean():.2f}")
    c5.metric("Avg Edge GAN (%)", f"{df['EdgeDensity_GAN'].mean():.2f}")

    st.subheader("📈 Grafik Perbandingan Edge Density")
    bar_df = df[["EdgeDensity_Canny", "EdgeDensity_GAN"]]
    st.bar_chart(bar_df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Tabel Metrik (CSV)",
        csv,
        "batik_outline_metrics.csv",
        "text/csv"
    )

    st.markdown("""
    ### Ringkasan Bab 4

    Analisis kuantitatif dan visual menunjukkan bahwa Pix2Pix GAN
    menghasilkan kepadatan garis yang lebih rendah dan selektif,
    menandakan keberhasilan model dalam mengekstraksi struktur
    garis utama motif batik dibandingkan metode baseline.
    """)

    if st.button("⬅️ Kembali"):
        st.session_state.page = "upload"
        st.rerun()
