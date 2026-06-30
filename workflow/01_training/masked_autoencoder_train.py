import os
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T


def list_audio_files(root):
    root = Path(root)
    return sorted([str(p) for p in root.rglob("*.wav")])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FixedClipMelDataset(Dataset):
    def __init__(
        self,
        files,
        sample_rate=32000,
        clip_sec=12.0,
        n_mels=128,
        fmin=50,
        fmax=14000,
    ):
        self.files = files
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_sec)

        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            win_length=1024,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        if wav.size(1) < self.clip_len:
            wav = F.pad(wav, (0, self.clip_len - wav.size(1)))
        else:
            wav = wav[:, :self.clip_len]

        spec = self.melspec(wav)
        spec = torch.log(spec.clamp_min(1e-10))
        spec = (spec - spec.mean()) / spec.std().clamp_min(1e-6)

        return spec, path


def make_mask(spec, mask_fraction=0.35, block_width=8):
    masked = spec.clone()
    mask = torch.zeros_like(spec)

    batch_size, channels, n_mels, time_bins = spec.shape

    n_blocks = max(1, int((time_bins * mask_fraction) / block_width))

    for b in range(batch_size):
        for _ in range(n_blocks):
            start = random.randint(0, max(0, time_bins - block_width))
            end = min(time_bins, start + block_width)

            masked[b, :, :, start:end] = 0
            mask[b, :, :, start:end] = 1

    return masked, mask


class MaskedConvAutoencoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding_head = nn.Linear(256, embedding_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        z_map = self.encoder(x)

        pooled = self.pool(z_map).squeeze(-1).squeeze(-1)
        embedding = self.embedding_head(pooled)

        recon = self.decoder(z_map)

        recon = recon[:, :, :x.shape[2], :x.shape[3]]

        return recon, embedding


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_sec", type=float, default=12.0)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--mask_fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    files = list_audio_files(args.data_root)

    if args.max_files > 0:
        random.shuffle(files)
        files = files[:args.max_files]

    print("Found files:", len(files))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = FixedClipMelDataset(
        files=files,
        sample_rate=args.sr,
        clip_sec=args.clip_sec,
        n_mels=args.n_mels,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 2),
        pin_memory=True,
        drop_last=True,
    )

    model = MaskedConvAutoencoder(
        embedding_dim=512
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    losses = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (spec, paths) in enumerate(pbar):
            spec = spec.to(device, non_blocking=True)

            masked, mask = make_mask(
                spec,
                mask_fraction=args.mask_fraction,
            )

            recon, embedding = model(masked)

            # Loss only on masked regions
            loss = ((recon - spec) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            mean_loss = running / (i + 1)

            gpu_mem = 0
            if device == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1024**3

            pbar.set_postfix(loss=mean_loss, gpu_gb=f"{gpu_mem:.2f}")

        epoch_loss = running / len(loader)
        losses.append(epoch_loss)

        ckpt_path = outdir / f"masked_autoencoder_epoch{epoch + 1}.pt"

        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "losses": losses,
            },
            ckpt_path,
        )

        print(f"Saved checkpoint: {ckpt_path}")
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    final_path = outdir / "masked_autoencoder_final.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "losses": losses,
        },
        final_path,
    )

    with open(outdir / "losses.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Masked reconstruction loss")
    plt.title("Masked autoencoder training loss")
    plt.tight_layout()
    plt.savefig(outdir / "training_loss.png", dpi=300)

    print("Done.")
    print("Saved final model:", final_path)


if __name__ == "__main__":
    main()
