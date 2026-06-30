import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_sec", type=float, default=12.0)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    files = sorted([str(p) for p in Path(args.data_root).rglob("*.wav")])
    print("Found files:", len(files))

    dataset = FixedClipMelDataset(
        files=files,
        sample_rate=args.sr,
        clip_sec=args.clip_sec,
        n_mels=args.n_mels,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = MaskedConvAutoencoder(embedding_dim=512).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    rows = []

    with torch.no_grad():
        for specs, paths in tqdm(loader):
            specs = specs.to(device, non_blocking=True)

            _, emb = model(specs)

            emb = emb.detach().cpu().numpy()

            for i, path in enumerate(paths):
                row = {
                    "OutputPath": path,
                    "ClipID": Path(path).stem,
                }

                for j, val in enumerate(emb[i]):
                    row[f"emb_{j}"] = val

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv)
    print(df.shape)


if __name__ == "__main__":
    main()