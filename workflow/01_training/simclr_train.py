import os
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
from torchvision.models import resnet18


def list_audio_files(root, exts=(".wav", ".WAV")):
    files = []
    root = Path(root)
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted([str(f) for f in files])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SpecAugment(nn.Module):
    def __init__(self, time_mask_param=40, freq_mask_param=14):
        super().__init__()
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x):
        x = self.time_mask(x)
        x = self.time_mask(x)
        x = self.freq_mask(x)
        x = self.freq_mask(x)
        return x


def random_gain(wav, low=0.7, high=1.3):
    return wav * random.uniform(low, high)


def add_noise(wav, snr_db_low=10, snr_db_high=25):
    snr_db = random.uniform(snr_db_low, snr_db_high)
    sig_power = wav.pow(2).mean().clamp_min(1e-12)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


class FixedClipSpecDataset(Dataset):
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

        self.specaug = SpecAugment()

    def __len__(self):
        return len(self.files)

    def load_clip(self, path):
        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        if wav.size(1) < self.clip_len:
            wav = F.pad(wav, (0, self.clip_len - wav.size(1)))
        else:
            wav = wav[:, :self.clip_len]

        return wav

    def wav_to_logmel(self, wav):
        spec = self.melspec(wav)
        spec = torch.log(spec.clamp_min(1e-10))
        spec = (spec - spec.mean()) / spec.std().clamp_min(1e-6)
        return spec

    def make_view(self, wav):
        wav = random_gain(wav)
        wav = add_noise(wav)
        spec = self.wav_to_logmel(wav)
        spec = self.specaug(spec)
        return spec

    def __getitem__(self, idx):
        wav = self.load_clip(self.files[idx])
        v1 = self.make_view(wav.clone())
        v2 = self.make_view(wav.clone())
        return v1, v2


class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        feat_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.encoder = base

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


def nt_xent(z1, z2, temp=0.2):
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)

    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = (z @ z.t()) / temp

    mask = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e4)

    positives = torch.cat(
        [torch.diag(sim, n), torch.diag(sim, -n)],
        dim=0,
    )

    denom = torch.logsumexp(sim, dim=1)

    loss = -(positives - denom).mean()

    return loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="/home/nvidia/test_run/simclr_outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_sec", type=float, default=12.0)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    files = list_audio_files(args.data_root)

    if args.max_files > 0:
        random.shuffle(files)
        files = files[: args.max_files]

    if len(files) == 0:
        raise RuntimeError("No WAV files found.")

    print(f"Found {len(files)} audio files")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = FixedClipSpecDataset(
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

    model = SimCLR(proj_dim=128).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    epoch_losses = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (x1, x2) in enumerate(pbar):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, z1 = model(x1)
                _, z2 = model(x2)
                loss = nt_xent(z1, z2, temp=0.2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            mean_loss = running / (i + 1)

            gpu_mem = 0
            if device == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1024**3

            pbar.set_postfix(loss=mean_loss, gpu_gb=f"{gpu_mem:.2f}")

        epoch_loss = running / len(loader)
        epoch_losses.append(epoch_loss)

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch + 1,
            "epoch_loss": epoch_loss,
        }

        ckpt_path = outdir / f"simclr_resnet18_epoch{epoch + 1}.pt"
        torch.save(ckpt, ckpt_path)

        print(f"Saved checkpoint: {ckpt_path}")
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch_losses": epoch_losses,
        },
        outdir / "simclr_resnet18_final.pt",
    )

    with open(outdir / "losses.txt", "w") as f:
        for loss in epoch_losses:
            f.write(f"{loss}\n")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent loss")
    plt.title("SimCLR SSL training loss")
    plt.tight_layout()
    plt.savefig(outdir / "training_loss.png", dpi=300)

    print("Done.")
    print(f"Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()