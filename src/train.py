import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
LR = 2e-4
LAMBDA_L1 = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")
RESULT_DIR = os.path.join(BASE_DIR, "results", "samples")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================
# DATASET
# ==========================
class BatikDataset(Dataset):
    def __init__(self, root):
        self.A = os.path.join(root, "A")
        self.B = os.path.join(root, "B")
        self.files = sorted(os.listdir(self.A))

        self.transform_A = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5])
        ])

        self.transform_B = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],
                            [0.5])
        ])


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_A = Image.open(os.path.join(self.A, fname)).convert("RGB")
        img_B = Image.open(os.path.join(self.B, fname)).convert("L")

        img_A = self.transform_A(img_A)
        img_B = self.transform_B(img_B)


        return img_A, img_B

# ==========================
# GENERATOR (U-NET)
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

        out = self.final(torch.cat([u3, d1], 1))
        return out

# ==========================
# DISCRIMINATOR (PATCHGAN)
# ==========================
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))

# ==========================
# TRAINING
# ==========================
def train():
    dataset = BatikDataset(DATASET_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = UNetGenerator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()

    start_time = time.time()
    epoch_times = []

    print(f"🚀 Training dimulai | Device: {DEVICE}")
    print("-" * 50)

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # ======================
            # Train Discriminator
            # ======================
            fake = G(x)

            D_real = D(x, y)
            D_fake = D(x, fake.detach())

            loss_D = (criterion_GAN(D_real, torch.ones_like(D_real)) +
                      criterion_GAN(D_fake, torch.zeros_like(D_fake))) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ======================
            # Train Generator
            # ======================
            D_fake = D(x, fake)
            loss_G = (criterion_GAN(D_fake, torch.ones_like(D_fake)) +
                      LAMBDA_L1 * criterion_L1(fake, y))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(
                loss_D=f"{loss_D.item():.4f}",
                loss_G=f"{loss_G.item():.4f}"
            )

        # ======================
        # END EPOCH
        # ======================
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)

        # SAVE SAMPLE & MODEL
        save_image((fake + 1) / 2,
                   f"{RESULT_DIR}/epoch_{epoch+1}.png")

        torch.save(G.state_dict(),
                   f"{MODEL_DIR}/pix2pix_outline.pth")

        # ======================
        # LOG SETIAP 10 EPOCH
        # ======================
        if (epoch + 1) % 10 == 0:
            avg_epoch_time = sum(epoch_times[-10:]) / 10
            elapsed = time.time() - start_time
            remaining = avg_epoch_time * (EPOCHS - (epoch + 1))

            print("\n⏱️ TRAINING STATUS")
            print(f"  Epoch          : {epoch+1}/{EPOCHS}")
            print(f"  Rata-rata/epoch: {avg_epoch_time/60:.2f} menit")
            print(f"  Waktu berjalan : {elapsed/60:.2f} menit")
            print(f"  Estimasi sisa  : {remaining/60:.2f} menit")
            print("-" * 50)

    # ======================
    # TOTAL TIME
    # ======================
    total_time = time.time() - start_time
    print("\n🎉 TRAINING SELESAI")
    print(f"🕒 Total waktu training: {total_time/60:.2f} menit")
    print(f"🕒 Total waktu training: {total_time/3600:.2f} jam")


if __name__ == "__main__":
    train()
