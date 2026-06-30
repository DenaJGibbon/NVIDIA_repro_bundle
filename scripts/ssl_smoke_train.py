import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------

assert torch.cuda.is_available(), "CUDA GPU not available"

device = "cuda"

print("Using GPU:")
print(torch.cuda.get_device_name(0))

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

manifest_csv = "/home/nvidia/data/ssl_v1/SSL_Clips_small/ssl_smoke_manifest_brev.csv"

output_dir = Path("/home/nvidia/test_run/ssl_outputs")
output_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------

manifest = pd.read_csv(manifest_csv)

print(f"Loaded {len(manifest)} clips")

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class SSLDataset(Dataset):

    def __init__(self, manifest_df):

        self.df = manifest_df

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        wav_path = self.df.iloc[idx]["OutputPath"]

        audio, sr = sf.read(wav_path)

        # Convert stereo -> mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        # Fixed length
        target_length = sr * 12

        if len(audio) < target_length:
            pad = target_length - len(audio)
            audio = np.pad(audio, (0, pad))

        audio = audio[:target_length]

        audio = torch.tensor(audio)

        return audio

# ---------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------

dataset = SSLDataset(manifest)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ---------------------------------------------------------------------
# Spectrogram helper
# ---------------------------------------------------------------------

def waveform_to_spec(x):

    spec = torch.stft(
        x,
        n_fft=1024,
        hop_length=320,
        win_length=1024,
        return_complex=True
    )

    spec = torch.abs(spec)

    spec = torch.log1p(spec)

    return spec.unsqueeze(1)

# ---------------------------------------------------------------------
# Masking function
# ---------------------------------------------------------------------

def mask_spec(spec, mask_fraction=0.25):

    masked = spec.clone()

    batch_size, _, freq_bins, time_bins = spec.shape

    n_mask = int(time_bins * mask_fraction)

    for i in range(batch_size):

        idx = torch.randperm(time_bins)[:n_mask]

        masked[i, :, :, idx] = 0

    return masked

# ---------------------------------------------------------------------
# SSL model
# ---------------------------------------------------------------------

class SmallSSLModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                64,
                32,
                4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                16,
                4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                16,
                1,
                4,
                stride=2,
                padding=1
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        out = self.decoder(z)

        out = out[:, :, :x.shape[2], :x.shape[3]]

        return out

# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------

model = SmallSSLModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.MSELoss()

# ---------------------------------------------------------------------
# Track training loss
# ---------------------------------------------------------------------

epoch_losses = []

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

n_epochs = 5

print("Starting SSL training...")

for epoch in range(n_epochs):

    model.train()

    running_loss = 0

    start = time.time()

    for batch_idx, waveform in enumerate(loader):

        batch_start = time.time()

        waveform = waveform.to(device)

        spec = waveform_to_spec(waveform)

        if epoch == 0 and batch_idx == 0:

            print("\nWaveform shape:", waveform.shape)
            print("Spectrogram shape:", spec.shape)

        masked_spec = mask_spec(spec)

        pred_spec = model(masked_spec)

        loss = loss_fn(pred_spec, spec)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        batch_time = time.time() - batch_start

        if batch_idx % 5 == 0:

            gpu_mem = (
                torch.cuda.memory_allocated() / 1024**3
            )

            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss {loss.item():.4f} | "
                f"GPU Mem {gpu_mem:.2f} GB | "
                f"Batch Time {batch_time:.2f} sec"
            )

    mean_loss = running_loss / len(loader)

    epoch_losses.append(mean_loss)

    elapsed = time.time() - start

    print(
        f"\nEpoch {epoch+1} complete | "
        f"Mean loss = {mean_loss:.4f} | "
        f"Epoch Time = {elapsed:.1f} sec\n"
    )

    torch.save(
        model.state_dict(),
        output_dir / f"ssl_epoch_{epoch+1}.pt"
    )

# ---------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------

torch.save(
    model.state_dict(),
    output_dir / "ssl_final_model.pt"
)

# ---------------------------------------------------------------------
# Plot training curve
# ---------------------------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(
    range(1, n_epochs + 1),
    epoch_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Training Loss")
plt.title("SSL Training Loss")

plot_path = output_dir / "training_loss.png"

plt.savefig(plot_path, dpi=300)

print("\nTraining complete")
print(f"Outputs saved to: {output_dir}")
print(f"Training plot saved to: {plot_path}")
