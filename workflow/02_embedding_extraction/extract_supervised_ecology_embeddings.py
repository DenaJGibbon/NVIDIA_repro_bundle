import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class AudioDataset(Dataset):
    def __init__(
        self,
        files,
        sample_rate=32000,
        clip_sec=12,
        n_mels=128,
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
            f_min=50,
            f_max=14000,
            power=2.0,
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        try:
            wav, sr = torchaudio.load(path)

            if wav.numel() == 0 or wav.size(1) == 0:
                raise RuntimeError("empty audio")

            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(
                    wav,
                    sr,
                    self.sample_rate
                )

            if wav.size(1) < self.clip_len:
                wav = F.pad(
                    wav,
                    (0, self.clip_len - wav.size(1))
                )
            else:
                wav = wav[:, :self.clip_len]

            spec = self.melspec(wav)

            spec = torch.log(
                spec.clamp_min(1e-10)
            )

            spec = (
                spec - spec.mean()
            ) / spec.std().clamp_min(1e-6)

        except Exception:
            spec = torch.zeros(
                1,
                128,
                int(self.clip_len / 320) + 1
            )

        return spec, path


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class SupervisedEcoEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
        n_sat_pcs=8,
        n_monsoon=2,
    ):
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

        self.embedding = nn.Linear(
            256,
            embedding_dim
        )

        self.time_head = nn.Linear(
            embedding_dim,
            2
        )

        self.sat_head = nn.Linear(
            embedding_dim,
            n_sat_pcs
        )

        self.monsoon_head = nn.Linear(
            embedding_dim,
            n_monsoon
        )

    def forward(self, x):

        z = self.encoder(x)

        z = self.pool(z)

        z = z.squeeze(-1).squeeze(-1)

        emb = self.embedding(z)

        return {
            "embedding": emb,
            "time": self.time_head(emb),
            "sat": self.sat_head(emb),
            "monsoon": self.monsoon_head(emb),
        }


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def list_audio_files(root):

    exts = [".wav", ".flac"]

    files = []

    for ext in exts:
        files.extend(
            Path(root).rglob(f"*{ext}")
        )

    return sorted([str(x) for x in files])


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--out_csv",
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--clip_sec",
        type=float,
        default=12
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512
    )

    parser.add_argument(
        "--n_sat_pcs",
        type=int,
        default=8
    )

    args = parser.parse_args()

    print("\nListing audio files...")

    files = list_audio_files(
        args.data_root
    )

    print("Files found:", len(files))

    dataset = AudioDataset(
        files=files,
        clip_sec=args.clip_sec,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device
    )

    model = SupervisedEcoEncoder(
        embedding_dim=args.embedding_dim,
        n_sat_pcs=args.n_sat_pcs,
        n_monsoon=2,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model = model.to(device)

    model.eval()

    rows = []

    with torch.no_grad():

        pbar = tqdm(loader)

        for spec, paths in pbar:

            spec = spec.to(
                device,
                non_blocking=True
            )

            out = model(spec)

            emb = (
                out["embedding"]
                .cpu()
                .numpy()
            )

            for i, path in enumerate(paths):

                row = {
                    "OutputPath": path,
                    "ClipID": Path(path).stem,
                }

                for j in range(emb.shape[1]):
                    row[f"emb_{j}"] = emb[i, j]

                rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        out_csv,
        index=False
    )

    print("\nSaved:")
    print(out_csv)

    print("\nRows:", len(df))
    print("Embedding dimensions:", emb.shape[1])


if __name__ == "__main__":
    main()
