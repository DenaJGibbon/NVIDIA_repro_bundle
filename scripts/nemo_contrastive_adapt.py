import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import nemo.collections.asr as nemo_asr


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def list_audio_files(root):
    root = Path(root)
    files = []
    for ext in ["*.wav", "*.WAV"]:
        files.extend(root.rglob(ext))
    return sorted([str(f) for f in files])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_gain(wav, low=0.7, high=1.3):
    return wav * random.uniform(low, high)


def add_noise(wav, snr_db_low=10, snr_db_high=25):
    snr_db = random.uniform(snr_db_low, snr_db_high)
    sig_power = wav.pow(2).mean().clamp_min(1e-12)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class AudioPairDataset(Dataset):

    def __init__(self, files, sample_rate=16000, clip_sec=12.0):
        self.files = files
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_sec)

    def __len__(self):
        return len(self.files)

    def load_clip(self, path):
        try:
            wav, sr = torchaudio.load(path)

            if wav.numel() == 0 or wav.size(1) == 0:
                raise RuntimeError(f"Empty audio file: {path}")

            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

            if wav.numel() == 0 or wav.size(1) == 0:
                raise RuntimeError(f"Empty audio after resampling: {path}")

            if wav.size(1) < self.clip_len:
                wav = F.pad(wav, (0, self.clip_len - wav.size(1)))
            else:
                wav = wav[:, :self.clip_len]

            return wav.squeeze(0)

        except Exception as e:
            print(f"Skipping bad file: {path} | {e}")
            return None
            
    def make_view(self, wav):
        wav = wav.clone()
        wav = random_gain(wav)
        wav = add_noise(wav)

        # Light temporal masking directly on waveform
        if random.random() < 0.5:
            n = wav.numel()
            mask_len = int(0.05 * n)
            start = random.randint(0, max(0, n - mask_len))
            wav[start:start + mask_len] = 0

        return wav

    def __getitem__(self, idx):
        wav = self.load_clip(self.files[idx])

        if wav is None:
            new_idx = random.randint(0, len(self.files) - 1)
            wav = self.load_clip(self.files[new_idx])

        if wav is None:
            wav = torch.zeros(self.clip_len)

        v1 = self.make_view(wav)
        v2 = self.make_view(wav)

        return v1, v2
# ---------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------

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
        dim=0
    )

    denom = torch.logsumexp(sim, dim=1)

    loss = -(positives - denom).mean()

    return loss


# ---------------------------------------------------------------------
# NeMo encoder wrapper
# ---------------------------------------------------------------------

class NeMoContrastiveModel(nn.Module):

    def __init__(self, nemo_model_name="stt_en_conformer_ctc_large", proj_dim=128):
        super().__init__()

        self.nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=nemo_model_name
        )

        # We use the preprocessor + encoder only
        self.preprocessor = self.nemo_model.preprocessor
        self.encoder = self.nemo_model.encoder

        # Infer encoder dimension with dummy input later if needed
        self.projector = None
        self.proj_dim = proj_dim

    def build_projector(self, encoder_dim, device):
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.proj_dim)
        ).to(device)

    def forward(self, wav):
        # wav shape: batch x samples
        lengths = torch.full(
            size=(wav.shape[0],),
            fill_value=wav.shape[1],
            dtype=torch.long,
            device=wav.device
        )

        processed_signal, processed_length = self.preprocessor(
            input_signal=wav,
            length=lengths
        )

        encoded, encoded_len = self.encoder(
            audio_signal=processed_signal,
            length=processed_length
        )

        # encoded is usually batch x features x time
        pooled = encoded.mean(dim=2)

        if self.projector is None:
            self.build_projector(
                encoder_dim=pooled.shape[1],
                device=pooled.device
            )

        z = self.projector(pooled)
        z = F.normalize(z, dim=1)

        return pooled, z


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip_sec", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="stt_en_conformer_ctc_large")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    files = list_audio_files(args.data_root)

    if len(files) == 0:
        raise RuntimeError("No WAV files found.")

    print(f"Found {len(files)} audio files")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = AudioPairDataset(
        files=files,
        sample_rate=16000,
        clip_sec=args.clip_sec
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    model = NeMoContrastiveModel(
        nemo_model_name=args.model_name,
        proj_dim=128
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    losses = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (x1, x2) in enumerate(pbar):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = nt_xent(z1, z2, temp=0.2)

            loss.backward()
            optimizer.step()

            running += loss.item()
            mean_loss = running / (i + 1)

            gpu_mem = 0
            if device == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1024**3

            pbar.set_postfix(loss=mean_loss, gpu_gb=f"{gpu_mem:.2f}")

        epoch_loss = running / len(loader)
        losses.append(epoch_loss)

        ckpt_path = outdir / f"nemo_contrastive_epoch{epoch + 1}.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "losses": losses,
                "args": vars(args)
            },
            ckpt_path
        )

        print(f"Saved: {ckpt_path}")
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

    final_path = outdir / "nemo_contrastive_final.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "losses": losses,
            "args": vars(args)
        },
        final_path
    )

    with open(outdir / "losses.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    print("Done.")
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()