import os
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchaudio
import torchaudio.transforms as T


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files_by_root(data_roots, exts=(".wav", ".WAV", ".flac", ".FLAC")):
    records = []

    for root in data_roots:
        root = Path(root)
        source = root.name

        files = []
        for ext in exts:
            files.extend(root.rglob(f"*{ext}"))

        for f in files:
            records.append({
                "path": str(f),
                "source": source,
            })

    return sorted(records, key=lambda x: x["path"])


class FixedClipMelDataset(Dataset):
    def __init__(
        self,
        records,
        sample_rate=32000,
        clip_sec=12.0,
        n_mels=128,
        fmin=50,
        fmax=14000,
    ):
        self.records = records
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
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec["path"]

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

        return spec, rec["source"], path


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


def make_source_weights(records, source_weight_mode="equal_source"):
    source_counts = Counter([r["source"] for r in records])

    print("\nSource counts:")
    for source, n in source_counts.items():
        print(source, n)

    if source_weight_mode == "none":
        return None

    if source_weight_mode == "equal_source":
        weights = [
            1.0 / source_counts[r["source"]]
            for r in records
        ]
        return torch.DoubleTensor(weights)

    raise ValueError(f"Unknown source_weight_mode: {source_weight_mode}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_roots",
        nargs="+",
        required=True,
        help="One or more audio folders."
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_sec", type=float, default=12.0)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--mask_fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=0)

    parser.add_argument(
        "--source_weight_mode",
        type=str,
        default="equal_source",
        choices=["equal_source", "none"]
    )

    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=0,
        help="If >0, controls epoch length for weighted sampling."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    records = list_audio_files_by_root(args.data_roots)

    if args.max_files > 0:
        random.shuffle(records)
        records = records[:args.max_files]

    if len(records) == 0:
        raise RuntimeError("No audio files found.")

    print("\nFound files:", len(records))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = FixedClipMelDataset(
        records=records,
        sample_rate=args.sr,
        clip_sec=args.clip_sec,
        n_mels=args.n_mels,
    )

    weights = make_source_weights(
        records,
        source_weight_mode=args.source_weight_mode
    )

    if weights is not None:
        if args.steps_per_epoch > 0:
            num_samples = args.steps_per_epoch * args.batch_size
        else:
            num_samples = len(records)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=min(8, os.cpu_count() or 2),
            pin_memory=True,
            drop_last=True,
        )

        print("\nUsing WeightedRandomSampler")
        print("Samples per epoch:", num_samples)

    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 2),
            pin_memory=True,
            drop_last=True,
        )

        print("\nUsing normal shuffled sampling")

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
        source_seen = Counter()

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (spec, sources, paths) in enumerate(pbar):
            spec = spec.to(device, non_blocking=True)

            source_seen.update(list(sources))

            masked, mask = make_mask(
                spec,
                mask_fraction=args.mask_fraction,
            )

            recon, embedding = model(masked)

            loss = (
                ((recon - spec) ** 2 * mask).sum()
                / mask.sum().clamp_min(1.0)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            mean_loss = running / (i + 1)

            gpu_mem = 0
            if device == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1024**3

            pbar.set_postfix(
                loss=mean_loss,
                gpu_gb=f"{gpu_mem:.2f}"
            )

        epoch_loss = running / len(loader)
        losses.append(epoch_loss)

        print("\nSource samples seen this epoch:")
        for source, n in source_seen.items():
            print(source, n)

        ckpt_path = outdir / f"masked_autoencoder_epoch{epoch + 1}.pt"

        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "losses": losses,
                "source_counts": dict(Counter([r["source"] for r in records])),
                "source_seen_epoch": dict(source_seen),
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
            "source_counts": dict(Counter([r["source"] for r in records])),
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
    plt.title("Masked autoencoder tropical weighted training loss")
    plt.tight_layout()
    plt.savefig(outdir / "training_loss.png", dpi=300)

    print("Done.")
    print("Saved final model:", final_path)


if __name__ == "__main__":
    main()