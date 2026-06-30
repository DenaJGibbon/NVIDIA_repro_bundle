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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

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

        label = Path(path).parent.name

        return spec, path, label


# ------------------------------------------------------------
# SimCLR model
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_sec", type=float, default=12.0)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    files = sorted([
        str(p)
        for p in Path(args.data_root).rglob("*.wav")
    ])

    print("Found WAV files:", len(files))

    dataset = FixedClipDataset(
        files=files,
        clip_sec=args.clip_sec
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SimCLR().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    rows = []

    with torch.no_grad():

        for specs, paths, labels in tqdm(loader):

            specs = specs.to(device, non_blocking=True)

            h, z = model(specs)

            h = h.cpu().numpy()

            for i, path in enumerate(paths):

                row = {
                    "OutputPath": path,
                    "ClipID": Path(path).stem,
                    "Label": labels[i]
                }

                for j in range(h.shape[1]):
                    row[f"emb_{j}"] = h[i, j]

                rows.append(row)

    emb_df = pd.DataFrame(rows)

    emb_csv = outdir / "simclr_detection_embeddings.csv"

    emb_df.to_csv(emb_csv, index=False)

    print("Saved embeddings:")
    print(emb_csv)

    print("\nLabel counts:")
    print(emb_df["Label"].value_counts())

    # ------------------------------------------------------------
    # Linear probe: gibbon vs noise
    # ------------------------------------------------------------

    feature_cols = [
        c for c in emb_df.columns
        if c.startswith("emb_")
    ]

    X = emb_df[feature_cols].to_numpy()

    y = (
        emb_df["Label"]
        .astype(str)
        .to_numpy(dtype=str)
    )

    X = StandardScaler().fit_transform(X)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy"
    )

    print("\nLinear probe: gibbon vs noise")
    print(f"Balanced accuracy mean: {scores.mean():.3f}")
    print(f"Balanced accuracy sd:   {scores.std():.3f}")
    print(f"Chance level approx:    {1 / len(pd.unique(y)):.3f}")

    summary = pd.DataFrame({
        "Model": ["SimCLR_soundscape_frozen"],
        "Task": ["Gibbon_vs_noise"],
        "BalancedAccuracyMean": [scores.mean()],
        "BalancedAccuracySD": [scores.std()],
        "ChanceApprox": [1 / len(pd.unique(y))],
        "N": [len(y)],
        "NClasses": [len(pd.unique(y))],
        "NFeatures": [len(feature_cols)]
    })

    summary_csv = outdir / "simclr_detection_probe_summary.csv"

    summary.to_csv(summary_csv, index=False)

    print("\nSaved summary:")
    print(summary_csv)


if __name__ == "__main__":

    main()
