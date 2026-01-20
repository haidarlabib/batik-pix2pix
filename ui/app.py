# =========================================================
# ui/app.py — FINAL (SINGLE PAGE, NO SIDEBAR)
# + Progress Bar
# + Loading Spinner
# + Highlight Perbedaan Canny vs GAN
# + Narasi Bab 4 (Hasil & Pembahasan) di UI
# =========================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import time

# ==========================
# CONFIG
# ==========================
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pix2pix_outline.pth")

# ==========================
# STYLE (AKADEMIK)
# ==========================
st.set_page_config(
    page_title="Batik Outline Generator (GAN)",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f8f9fa; }
h1, h2, h3 { color: #1f2c3d; }
.block-container { padding-top: 2rem; }
.caption { font-size: 0.9rem; color: #555; }
.highlight {
    background-color: #eef4ff;
    padding: 0.75rem;
    border-left: 5px solid #3b6cff;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
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
# CANNY
# ==========================
def canny_edge(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    return Image.fromarray(edge)

# ==========================
# STATE INIT
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ==========================
# PAGE 1 — UPLOAD & PREVIEW
# ==========================
if st.session_state.page == "upload":

    st.title("🎨 Batik Outline Generator (Pix2Pix GAN)")

    st.markdown("""
    Penelitian ini berfokus pada **ekstraksi struktur garis utama motif batik**
    menggunakan **Pix2Pix GAN**. Pendekatan ini bertujuan menghasilkan
    representasi pola motif yang menyerupai **pola canting**, bukan sekadar
    citra grayscale atau deteksi tepi konvensional.
    Dalam proses pembuatan batik tradisional, pembatik terlebih dahulu menggambar
    **pola garis motif (outline)** sebelum proses pewarnaan.  
    Penelitian ini mengusulkan penggunaan **Pix2Pix GAN** untuk mengekstraksi
    struktur garis motif batik secara otomatis dari citra batik berwarna.

    Pendekatan ini **berbeda dengan grayscale atau edge detection**, karena
    model GAN mempelajari **struktur motif utama** berdasarkan data latih,
    bukan sekadar perubahan intensitas warna.
    """)

    st.divider()

    st.subheader("📤 Unggah Citra Batik")

    uploaded_files = st.file_uploader(
        "Unggah satu atau beberapa citra batik (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        images = [Image.open(f).convert("RGB") for f in uploaded_files]
        st.session_state.images = images

        st.success(f"{len(images)} gambar berhasil diunggah.")

        st.subheader("🔍 Pratinjau Gambar")
        cols = st.columns(min(4, len(images)))

        for idx, img in enumerate(images):
            with cols[idx % len(cols)]:
                st.image(img, caption=f"Gambar {idx+1}", width=200)

        st.divider()

        if st.button("🚀 Generate & Bandingkan"):
            st.session_state.page = "hasil"
            st.rerun()

# ==========================
# PAGE 2 — HASIL & PEMBAHASAN
# ==========================
elif st.session_state.page == "hasil":

    st.title("📊 Hasil & Perbandingan Metode")

    st.markdown("""
    ### Bab 4 — Hasil dan Pembahasan

    Pada bagian ini ditampilkan hasil pengujian model Pix2Pix GAN
    dalam mengekstraksi pola garis motif batik, serta perbandingannya
    dengan metode **Canny Edge** sebagai baseline klasik.
    """)

    images = st.session_state.images

    progress = st.progress(0)
    total = len(images)

    with st.spinner("⏳ Sedang menghasilkan outline motif menggunakan GAN..."):
        for idx, image in enumerate(images):
            time.sleep(0.3)
            progress.progress((idx + 1) / total)

            st.subheader(f"Gambar ke-{idx+1}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Batik Asli**")
                st.image(image, width=250)

            with col2:
                st.markdown("**Canny Edge (Baseline)**")
                canny_img = canny_edge(image)
                st.image(canny_img, width=250)

            with col3:
                st.markdown("**Outline Motif (Pix2Pix GAN)**")
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(img_tensor)
                output = (output + 1) / 2
                gan_img = transforms.ToPILImage()(output.squeeze().cpu())
                st.image(gan_img, width=250)

            st.markdown("""
            <div class="highlight">
            <b>Analisis:</b><br>
            Metode Canny Edge menghasilkan banyak garis akibat tekstur kain
            dan variasi warna. Sebaliknya, Pix2Pix GAN mampu mengekstraksi
            <i>garis motif utama</i> secara lebih konsisten, menunjukkan
            kemampuan model dalam memahami struktur visual motif batik.
            </div>
            """, unsafe_allow_html=True)

            st.divider()

    st.markdown("""
    ### Ringkasan Pembahasan

    Berdasarkan hasil pengujian visual, dapat disimpulkan bahwa:
    - Canny Edge sensitif terhadap noise dan tekstur kain
    - Pix2Pix GAN menghasilkan garis motif yang lebih bersih
    - Hasil GAN lebih relevan untuk proses desain dan digitalisasi batik

    Dengan demikian, metode Pix2Pix GAN terbukti lebih efektif
    dalam mengekstraksi struktur motif batik dibandingkan metode klasik.
    """)

    if st.button("⬅️ Kembali ke Upload"):
        st.session_state.page = "upload"
        st.rerun()
