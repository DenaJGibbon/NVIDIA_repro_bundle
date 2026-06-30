import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
import torchaudio.transforms as T

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def list_audio_files(root):

    exts = [".wav", ".flac", ".mp3"]

    return sorted([
        str(p)
        for p in Path(root).rglob("*")
        if p.suffix.lower() in exts
    ])


def time_circular(hour):

    angle = 2 * np.pi * hour / 24

    return np.sin(angle), np.cos(angle)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class AudioMetadataDataset(Dataset):

    def __init__(
        self,
        df,
        sample_rate=32000,
        clip_sec=12,
        n_mels=128
    ):

        self.df = df.reset_index(drop=True)

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

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["FilePath"]

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

        y_time = torch.tensor(
            [row["TimeSin"], row["TimeCos"]],
            dtype=torch.float32
        )

        y_sat = torch.tensor(
            row[self.sat_cols].values.astype(np.float32)
        )

        y_habitat = torch.tensor(
            int(row["HabitatClass"]),
            dtype=torch.long
        )

        y_rain = torch.tensor(
            [row["RainTarget"]],
            dtype=torch.float32
        )

        return (
            spec,
            y_time,
            y_sat,
            y_habitat,
            y_rain,
            path
        )


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class SupervisedEcoEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        n_sat=8,
        n_habitats=4
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                3,
                stride=2,
                padding=1
            ),
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
            n_sat
        )

        self.habitat_head = nn.Linear(
            embedding_dim,
            n_habitats
        )

        self.rain_head = nn.Linear(
            embedding_dim,
            1
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
            "habitat": self.habitat_head(emb),
            "rain": self.rain_head(emb),
        }


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
        "--metadata_csv",
        required=True
    )

    parser.add_argument(
        "--satellite_csv",
        required=True
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--clip_sec",
        type=float,
        default=12
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--rain_col",
        default="Rain_3day_mm"
    )

    parser.add_argument(
        "--rain_weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--time_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--sat_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--habitat_weight",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Audio files
    # ------------------------------------------------------------

    files = list_audio_files(
        args.data_root
    )

    print("Found audio files:", len(files))

    file_df = pd.DataFrame({
        "FilePath": files,
        "FileName": [
            Path(f).name for f in files
        ],
        "ClipID": [
            Path(f).stem for f in files
        ],
        "FileStem": [
            Path(f).stem.split("_clip")[0] for f in files
        ],
    })

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    meta = pd.read_csv(
        args.metadata_csv,
        low_memory=False
    )

    sat = pd.read_csv(args.satellite_csv, low_memory=False)

    possible_plot_cols = ["Plot", "plot", "Site", "site", "Transect", "transect"]

    sat_plot_col = None
    for c in possible_plot_cols:
        if c in sat.columns:
            sat_plot_col = c
            break

    if sat_plot_col is None:
        raise ValueError("Could not find plot/site/transect column in satellite CSV.")

    sat = sat.rename(columns={sat_plot_col: "Transect"})

    sat["Transect"] = sat["Transect"].astype(str).str.upper()
    sat["Transect"] = sat["Transect"].str.extract(r"(T\d{2})")[0]
    sat["Transect"] = "WA-" + sat["Transect"]

    sat_cols = [
        c for c in sat.columns
        if c.startswith("A")
    ]

    print("Using satellite embedding columns:")
    print(sat_cols[:5], "...", sat_cols[-5:])
    print("N satellite features:", len(sat_cols))

    if len(sat_cols) == 0:
        raise ValueError("No satellite embedding columns found in satellite CSV.")

    meta["Transect"] = meta["Transect"].astype(str).str.upper()
    meta["Transect"] = meta["Transect"].str.extract(r"(T\d{2})")[0]
    meta["Transect"] = "WA-" + meta["Transect"]

    print("Metadata transects:")
    print(sorted(meta["Transect"].dropna().unique())[:10])

    print("Satellite transects:")
    print(sorted(sat["Transect"].dropna().unique())[:10])

    meta = meta.merge(
        sat[["Transect"] + sat_cols],
        on="Transect",
        how="left"
    )

    print("Rows with satellite data after merge:")
    print(meta[sat_cols].notna().all(axis=1).sum(), "of", len(meta))

    if len(sat_cols) == 0:
        print("No SatPC columns found; using Longitude and Latitude as spatial targets.")
        sat_cols = ["Longitude", "Latitude"]

    required_cols = [
        "FileName",
        "Hour",
        "HabitatType",
        args.rain_col,
    ] + sat_cols

    missing = [
        c for c in required_cols
        if c not in meta.columns
    ]

    if len(missing) > 0:

        raise ValueError(
            f"Missing metadata columns: {missing}"
        )

    meta["FileName"] = (
        meta["FileName"]
        .astype(str)
    )

    meta["FileStem"] = meta["FileName"].apply(
        lambda x: Path(str(x)).stem
    )

    df = file_df.merge(
        meta,
        on="FileStem",
        how="inner",
        suffixes=("", "_meta")
    )

    print("After metadata merge:", len(df))

    if len(df) == 0:

        raise ValueError(
            "No matching metadata rows."
        )

    df = df.dropna(
        subset=[
            "Hour",
            "HabitatType",
            args.rain_col,
        ] + sat_cols
    ).copy()

    # ------------------------------------------------------------
    # Time targets
    # ------------------------------------------------------------

    df["Hour"] = df["Hour"].astype(int)

    time_vals = df["Hour"].apply(
        time_circular
    )

    df["TimeSin"] = [
        x[0] for x in time_vals
    ]

    df["TimeCos"] = [
        x[1] for x in time_vals
    ]

    # ------------------------------------------------------------
    # Habitat targets
    # ------------------------------------------------------------

    habitat_encoder = LabelEncoder()

    df["HabitatClass"] = (
        habitat_encoder.fit_transform(
            df["HabitatType"].astype(str)
        )
    )

    # ------------------------------------------------------------
    # Rainfall target
    # ------------------------------------------------------------

    df["RainRaw"] = (
        df[args.rain_col]
        .astype(float)
    )

    df["RainTarget"] = np.log1p(
        df["RainRaw"]
    )

    # ------------------------------------------------------------
    # Scale satellite PCs
    # ------------------------------------------------------------

    scaler = StandardScaler()

    df[sat_cols] = scaler.fit_transform(
        df[sat_cols]
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset = AudioMetadataDataset(
        df=df,
        sample_rate=32000,
        clip_sec=args.clip_sec,
    )

    dataset.sat_cols = sat_cols

    n_train = int(0.8 * len(dataset))

    n_val = len(dataset) - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = SupervisedEcoEncoder(
        embedding_dim=512,
        n_sat=len(sat_cols),
        n_habitats=df["HabitatClass"].nunique()
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    history = []

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for (
            spec,
            y_time,
            y_sat,
            y_habitat,
            y_rain,
            paths
        ) in pbar:

            spec = spec.to(device)

            y_time = y_time.to(device)

            y_sat = y_sat.to(device)

            y_habitat = y_habitat.to(device)

            y_rain = y_rain.to(device)

            out = model(spec)

            loss_time = F.mse_loss(
                out["time"],
                y_time
            )

            loss_sat = F.mse_loss(
                out["sat"],
                y_sat
            )

            loss_habitat = F.cross_entropy(
                out["habitat"],
                y_habitat
            )

            loss_rain = F.mse_loss(
                out["rain"],
                y_rain
            )

            loss = (
                args.time_weight * loss_time
                + args.sat_weight * loss_sat
                + args.habitat_weight * loss_habitat
                + args.rain_weight * loss_rain
            )

            opt.zero_grad(
                set_to_none=True
            )

            loss.backward()

            opt.step()

            running_loss += loss.item()

            pbar.set_postfix(
                loss=running_loss / (pbar.n + 1)
            )

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": (
                running_loss / len(train_loader)
            )
        }

        history.append(epoch_summary)

        print(epoch_summary)

        ckpt = (
            outdir /
            f"supervised_eco_encoder_epoch{epoch+1}.pt"
        )

        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "history": history,
                "sat_cols": sat_cols,
                "habitat_classes":
                    list(habitat_encoder.classes_),
            },
            ckpt
        )

        print("Saved:", ckpt)

    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------

    final_path = (
        outdir /
        "supervised_eco_encoder_final.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "sat_cols": sat_cols,
            "habitat_classes":
                list(habitat_encoder.classes_),
        },
        final_path
    )

    pd.DataFrame(history).to_csv(
        outdir / "training_history.csv",
        index=False
    )

    print("Done.")
    print("Saved final:", final_path)


if __name__ == "__main__":
    main()