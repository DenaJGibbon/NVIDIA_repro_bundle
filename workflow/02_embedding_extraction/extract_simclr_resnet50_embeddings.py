import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.transforms as T
from torchvision.models import resnet50


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
            spec = torch.log(spec.clamp_min(1e-10))
            spec = (spec - spec.mean()) / spec.std().clamp_min(1e-6)

        except Exception:
            spec = torch.zeros(
                1,
                128,
                int(self.clip_len / 320) + 1
            )

        return spec, path


# ------------------------------------------------------------
# SimCLR ResNet50 model
# ------------------------------------------------------------

class SimCLRResNet50(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        base = resnet50(weights=None)

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
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def list_audio_files(root):
    root = Path(root)

    files = []
    for ext in [".wav", ".WAV", ".flac", ".FLAC"]:
        files.extend(root.rglob(f"*{ext}"))

    return sorted([str(x) for x in files])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_sec", type=float, default=12)
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)

    args = parser.parse_args()

    print("\nListing audio files...")
    files = list_audio_files(args.data_root)
    print("Files found:", len(files))

    dataset = AudioDataset(
        files=files,
        sample_rate=args.sample_rate,
        clip_sec=args.clip_sec,
        n_mels=args.n_mels,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device
    )

    model = SimCLRResNet50(
        proj_dim=128
    )

    state = ckpt["model"]

    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    rows = []

    with torch.no_grad():
        pbar = tqdm(loader)

        for specs, paths in pbar:
            specs = specs.to(
                device,
                non_blocking=True
            )

            h, z = model(specs)

            emb = h.cpu().numpy()

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
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(out_csv)
    print("\nRows:", len(df))
    print("Embedding dimensions:", emb.shape[1])


if __name__ == "__main__":
    main()
