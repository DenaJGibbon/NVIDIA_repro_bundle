import argparse
import re
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def list_audio_files(root):
    return sorted([str(p) for p in Path(root).rglob("*.wav")])


def parse_clip_metadata(path):
    name = Path(path).stem

    plot = None
    m = re.search(r"(WA-T\d{2})", name)
    if m:
        plot = m.group(1)

    date = None
    hour = None

    m = re.search(r"_(\d{8})_(\d{6})\+0700", name)
    if m:
        date = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
        hour = int(m.group(2)[:2])

    return plot, date, hour


def time_circular(hour):
    angle = 2 * np.pi * hour / 24
    return np.sin(angle), np.cos(angle)


def monsoon_from_month(month):
    # Cambodia default fallback if rainfall CSV not provided.
    # Adjust later if rainfall-based labels are available.
    if month in [5, 6, 7, 8, 9, 10]:
        return "Wet"
    return "Dry"


class AudioMetadataDataset(Dataset):
    def __init__(self, df, sample_rate=32000, clip_sec=12, n_mels=128):
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
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

            if wav.size(1) < self.clip_len:
                wav = F.pad(wav, (0, self.clip_len - wav.size(1)))
            else:
                wav = wav[:, :self.clip_len]

            spec = self.melspec(wav)
            spec = torch.log(spec.clamp_min(1e-10))
            spec = (spec - spec.mean()) / spec.std().clamp_min(1e-6)

        except Exception:
            spec = torch.zeros(1, 128, int(self.clip_len / 320) + 1)

        y_time = torch.tensor(
            [row["TimeSin"], row["TimeCos"]],
            dtype=torch.float32,
        )

        sat_cols = [c for c in self.df.columns if c.startswith("SatPC")]
        y_sat = torch.tensor(
            row[sat_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        y_monsoon = torch.tensor(
            int(row["MonsoonClass"]),
            dtype=torch.long,
        )

        return spec, y_time, y_sat, y_monsoon, path


class SupervisedEcoEncoder(nn.Module):
    def __init__(self, embedding_dim=512, n_sat_pcs=8, n_monsoon=2):
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
        self.embedding = nn.Linear(256, embedding_dim)

        self.time_head = nn.Linear(embedding_dim, 2)
        self.sat_head = nn.Linear(embedding_dim, n_sat_pcs)
        self.monsoon_head = nn.Linear(embedding_dim, n_monsoon)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pool(z).squeeze(-1).squeeze(-1)
        emb = self.embedding(z)

        return {
            "embedding": emb,
            "time": self.time_head(emb),
            "sat": self.sat_head(emb),
            "monsoon": self.monsoon_head(emb),
        }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--satellite_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--clip_sec", type=float, default=12)
    parser.add_argument("--n_sat_pcs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(args.data_root)
    print("Found audio files:", len(files))

    rows = []
    for f in files:
        plot, date, hour = parse_clip_metadata(f)

        if plot is None or pd.isna(date) or hour is None:
            continue

        t_sin, t_cos = time_circular(hour)

        rows.append({
            "FilePath": f,
            "ClipID": Path(f).stem,
            "Plot": plot,
            "Date": date,
            "Hour": hour,
            "Month": date.month,
            "TimeSin": t_sin,
            "TimeCos": t_cos,
            "Monsoon": monsoon_from_month(date.month),
        })

    df = pd.DataFrame(rows)
    print("Parsed clips:", len(df))

    df["MonsoonClass"] = df["Monsoon"].map({"Dry": 0, "Wet": 1})

    sat = pd.read_csv(args.satellite_csv)

    possible_plot_cols = ["Plot", "plot", "Site", "site", "Transect", "transect"]
    sat_plot_col = None
    for c in possible_plot_cols:
        if c in sat.columns:
            sat_plot_col = c
            break

    if sat_plot_col is None:
        raise ValueError("Could not find plot/site column in satellite CSV.")

    sat = sat.rename(columns={sat_plot_col: "Plot"})
    # Normalize satellite plot IDs: T01 -> WA-T01
    sat["Plot"] = sat["Plot"].astype(str).str.upper()
    sat["Plot"] = sat["Plot"].str.extract(r"(T\d{2})")[0]
    sat["Plot"] = "WA-" + sat["Plot"]
    
    numeric_cols = [
        c for c in sat.columns
        if c != "Plot" and pd.api.types.is_numeric_dtype(sat[c])
    ]

    sat_dat = sat[["Plot"] + numeric_cols].dropna().copy()

    scaler = StandardScaler()
    X_sat = scaler.fit_transform(sat_dat[numeric_cols])

    pca = PCA(n_components=args.n_sat_pcs)
    pcs = pca.fit_transform(X_sat)

    pc_cols = [f"SatPC{i+1}" for i in range(args.n_sat_pcs)]

    sat_pc = pd.DataFrame(pcs, columns=pc_cols)
    sat_pc["Plot"] = sat_dat["Plot"].values

    df = df.merge(sat_pc, on="Plot", how="left")
    df = df.dropna(subset=pc_cols).copy()

    print("After satellite merge:", len(df))
    print("Plots:")
    print(df["Plot"].value_counts())
    print("Monsoon:")
    print(df["Monsoon"].value_counts())

    df.to_csv(outdir / "supervised_training_manifest.csv", index=False)

    dataset = AudioMetadataDataset(
        df=df,
        clip_sec=args.clip_sec,
    )

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = SupervisedEcoEncoder(
        embedding_dim=512,
        n_sat_pcs=args.n_sat_pcs,
        n_monsoon=2,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    history = []

    for epoch in range(args.epochs):
        model.train()
        running = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for spec, y_time, y_sat, y_monsoon, paths in pbar:
            spec = spec.to(device, non_blocking=True)
            y_time = y_time.to(device, non_blocking=True)
            y_sat = y_sat.to(device, non_blocking=True)
            y_monsoon = y_monsoon.to(device, non_blocking=True)

            out = model(spec)

            loss_time = F.mse_loss(out["time"], y_time)
            loss_sat = F.mse_loss(out["sat"], y_sat)
            loss_monsoon = F.cross_entropy(out["monsoon"], y_monsoon)

            loss = loss_time + loss_sat + loss_monsoon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            pbar.set_postfix(loss=running / (len(history) + 1 + pbar.n))

        epoch_loss = running / len(train_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
        })

        ckpt = outdir / f"supervised_eco_encoder_epoch{epoch+1}.pt"

        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "history": history,
                "pc_cols": pc_cols,
            },
            ckpt,
        )

        print("Saved:", ckpt)

    final_path = outdir / "supervised_eco_encoder_final.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "pc_cols": pc_cols,
        },
        final_path,
    )

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    print("Done.")
    print("Saved final:", final_path)


if __name__ == "__main__":
    main()
