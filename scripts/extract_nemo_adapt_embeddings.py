from pathlib import Path
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nemo.collections.asr as nemo_asr


class AudioDataset(Dataset):
    def __init__(self, files, sample_rate=16000, clip_sec=12.0):
        self.files = files
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_sec)

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

        return wav.squeeze(0), path


class NeMoContrastiveModel(nn.Module):
    def __init__(self, nemo_model_name="stt_en_conformer_ctc_large", proj_dim=128):
        super().__init__()

        self.nemo_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=nemo_model_name
        )

        self.preprocessor = self.nemo_model.preprocessor
        self.encoder = self.nemo_model.encoder

        self.projector = None
        self.proj_dim = proj_dim

    def build_projector(self, encoder_dim, device):
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.proj_dim)
        ).to(device)

    def forward(self, wav):
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

        pooled = encoded.mean(dim=2)

        if self.projector is None:
            self.build_projector(pooled.shape[1], pooled.device)

        z = self.projector(pooled)
        z = F.normalize(z, dim=1)

        return pooled, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_sec", type=float, default=12.0)
    parser.add_argument("--model_name", default="stt_en_conformer_ctc_large")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    files = sorted([str(p) for p in Path(args.data_root).rglob("*.wav")])
    print(f"Found {len(files)} WAV files")

    dataset = AudioDataset(files, sample_rate=16000, clip_sec=args.clip_sec)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = NeMoContrastiveModel(
        nemo_model_name=args.model_name,
        proj_dim=128
    ).to(device)

    # Build projector before loading checkpoint
    dummy = torch.zeros((1, int(16000 * args.clip_sec)), device=device)
    with torch.no_grad():
        model(dummy)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)

    model.eval()

    rows = []

    with torch.no_grad():
        for wavs, paths in tqdm(loader):
            wavs = wavs.to(device, non_blocking=True)

            pooled, z = model(wavs)

            pooled = pooled.cpu().numpy()
            z = z.cpu().numpy()

            for i, path in enumerate(paths):
                row = {
                    "OutputPath": path,
                    "ClipID": Path(path).stem
                }

                for j in range(pooled.shape[1]):
                    row[f"emb_{j}"] = pooled[i, j]

                for j in range(z.shape[1]):
                    row[f"proj_{j}"] = z[i, j]

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("Saved embeddings:")
    print(args.out_csv)


if __name__ == "__main__":
    main()
