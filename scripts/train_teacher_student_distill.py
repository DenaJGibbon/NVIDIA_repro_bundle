import os
import random
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
from torchvision.models import resnet18, resnet50


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files(data_roots, exts=(".wav", ".WAV", ".flac", ".FLAC")):
    files = []

    for root in data_roots:
        root = Path(root)
        for ext in exts:
            files.extend(root.rglob(f"*{ext}"))

    return sorted([str(f) for f in files])


class TeacherStudentDataset(Dataset):
    def __init__(
        self,
        df,
        feature_cols,
        sample_rate=32000,
        clip_sec=12,
        n_mels=128,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
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

        teacher = torch.tensor(
            row[self.feature_cols].values.astype(np.float32)
        )

        return spec, teacher, path


class StudentEncoder(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        student_dim=512,
        teacher_dim=1024,
    ):
        super().__init__()

        if backbone == "resnet18":
            base = resnet18(weights=None)
        elif backbone == "resnet50":
            base = resnet50(weights=None)
        else:
            raise ValueError("backbone must be resnet18 or resnet50")

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

        self.student_embedding = nn.Linear(
            feat_dim,
            student_dim
        )

        self.teacher_head = nn.Sequential(
            nn.Linear(student_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, teacher_dim),
        )

    def forward(self, x):
        h = self.encoder(x)

        student = self.student_embedding(h)

        pred_teacher = self.teacher_head(student)

        return student, pred_teacher


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_roots",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--teacher_csv",
        required=True,
        help="CSV with ClipID and emb_* columns from BirdNET or Perch."
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    parser.add_argument(
        "--teacher_name",
        default="BirdNET",
    )

    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50"],
    )

    parser.add_argument("--student_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_sec", type=float, default=12)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nListing audio files...")
    files = list_audio_files(args.data_roots)

    if args.max_files > 0:
        random.shuffle(files)
        files = files[:args.max_files]

    file_df = pd.DataFrame({
        "FilePath": files,
        "ClipID": [Path(f).stem for f in files],
    })

    print("Audio files:", len(file_df))

    print("\nLoading teacher embeddings...")
    teacher = pd.read_csv(args.teacher_csv)

    feature_cols = [
        c for c in teacher.columns
        if c.startswith("emb_")
    ]

    if len(feature_cols) == 0:
        raise ValueError("No emb_* columns found in teacher CSV.")

    print("Teacher rows:", len(teacher))
    print("Teacher dim:", len(feature_cols))

    teacher = teacher[["ClipID"] + feature_cols].copy()

    df = file_df.merge(
        teacher,
        on="ClipID",
        how="inner"
    )

    print("Matched audio + teacher rows:", len(df))

    if len(df) == 0:
        raise ValueError("No matching ClipID values between audio and teacher CSV.")

    manifest_path = outdir / "distillation_manifest.csv"
    df[["FilePath", "ClipID"]].to_csv(manifest_path, index=False)
    print("Saved manifest:", manifest_path)

    dataset = TeacherStudentDataset(
        df=df,
        feature_cols=feature_cols,
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

    model = StudentEncoder(
        backbone=args.backbone,
        student_dim=args.student_dim,
        teacher_dim=len(feature_cols),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device == "cuda")
    )

    losses = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        for i, (spec, teacher_emb, paths) in enumerate(pbar):
            spec = spec.to(device, non_blocking=True)
            teacher_emb = teacher_emb.to(device, non_blocking=True)

            teacher_norm = F.normalize(
                teacher_emb.float(),
                dim=1
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                student_emb, pred_teacher = model(spec)

                pred_norm = F.normalize(
                    pred_teacher.float(),
                    dim=1
                )

                loss_cos = 1 - F.cosine_similarity(
                    pred_norm,
                    teacher_norm,
                    dim=1
                ).mean()

                loss_mse = F.mse_loss(
                    pred_norm,
                    teacher_norm
                )

                loss = loss_cos + 0.25 * loss_mse

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        ckpt_path = outdir / f"student_{args.teacher_name}_{args.backbone}_epoch{epoch+1}.pt"

        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
                "losses": losses,
                "teacher_dim": len(feature_cols),
                "student_dim": args.student_dim,
            },
            ckpt_path,
        )

        print("Saved:", ckpt_path)
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    final_path = outdir / f"student_{args.teacher_name}_{args.backbone}_final.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "losses": losses,
            "teacher_dim": len(feature_cols),
            "student_dim": args.student_dim,
        },
        final_path,
    )

    pd.DataFrame({
        "epoch": range(1, len(losses) + 1),
        "loss": losses,
    }).to_csv(outdir / "losses.csv", index=False)

    print("\nDone.")
    print("Saved final:", final_path)


if __name__ == "__main__":
    main()