import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from tqdm import tqdm


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class FixedClipDataset(Dataset):

    def __init__(
        self,
        files,
        sample_rate=32000,
        clip_sec=12.0,
        n_mels=128
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
            power=2.0
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        wav, sr = torchaudio.load(path)

        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample
        if sr != self.sample_rate:

            wav = torchaudio.functional.resample(
                wav,
                sr,
                self.sample_rate
            )

        # fixed length
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

        return spec, path


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

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
            bias=False
        )

        feat_dim = base.fc.in_features

        base.fc = nn.Identity()

        self.encoder = base

        self.projector = nn.Sequential(

            nn.Linear(feat_dim, 512),

            nn.ReLU(inplace=True),

            nn.Linear(512, proj_dim)
        )

    def forward(self, x):

        h = self.encoder(x)

        z = self.projector(h)

        z = F.normalize(z, dim=1)

        return h, z


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

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

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------------
    # Files
    # --------------------------------------------------------------

    files = sorted([
        str(p)
        for p in Path(args.data_root).rglob("*.wav")
    ])

    print(f"Found {len(files)} WAV files")

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    dataset = FixedClipDataset(files)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    model = SimCLR().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    if "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"]
        )

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    # --------------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------------

    rows = []

    with torch.no_grad():

        for specs, paths in tqdm(loader):

            specs = specs.to(device)

            h, z = model(specs)

            h = h.cpu().numpy()
            z = z.cpu().numpy()

            for i, path in enumerate(paths):

                row = {

                    "OutputPath": path,

                    "ClipID": Path(path).stem
                }

                # encoder embeddings
                for j in range(h.shape[1]):

                    row[f"emb_{j}"] = h[i, j]

                # projection embeddings
                for j in range(z.shape[1]):

                    row[f"proj_{j}"] = z[i, j]

                rows.append(row)

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    df = pd.DataFrame(rows)

    df.to_csv(
        args.out_csv,
        index=False
    )

    print(f"Saved embeddings to:\n{args.out_csv}")


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

if __name__ == "__main__":

    main()