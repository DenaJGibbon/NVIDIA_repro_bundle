from pathlib import Path
import argparse

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

import nemo.collections.asr as nemo_asr


# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", required=True)
parser.add_argument("--out_csv", required=True)

parser.add_argument(
    "--model_name",
    default="stt_en_conformer_ctc_large"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1
)

parser.add_argument(
    "--sample_rate",
    type=int,
    default=16000
)

parser.add_argument(
    "--clip_sec",
    type=float,
    default=6
)

parser.add_argument(
    "--max_files",
    type=int,
    default=None
)

args = parser.parse_args()


# ------------------------------------------------------------
# Paths / device
# ------------------------------------------------------------

data_root = Path(args.data_root)
out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name=args.model_name
)

model = model.to(device)
model.eval()


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

files = sorted([
    p for p in data_root.rglob("*")
    if p.suffix.lower() in [".wav", ".flac", ".mp3"]
])

if args.max_files is not None:
    files = files[:args.max_files]

print("Found files:", len(files))


# ------------------------------------------------------------
# Load audio helper
# ------------------------------------------------------------

max_len_samples = int(args.sample_rate * args.clip_sec)


def load_audio(path):

    wav, sr = torchaudio.load(str(path))

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != args.sample_rate:
        wav = torchaudio.functional.resample(
            wav,
            sr,
            args.sample_rate
        )

    wav = wav.squeeze(0)

    # Limit duration to avoid NeMo conformer OOM
    wav = wav[:max_len_samples]

    return wav


# ------------------------------------------------------------
# Extract embeddings
# ------------------------------------------------------------

rows = []

for start in tqdm(range(0, len(files), args.batch_size)):

    batch_files = files[start:start + args.batch_size]

    waves = []
    lengths = []
    valid_files = []

    for wav_path in batch_files:

        try:
            wav = load_audio(wav_path)

        except Exception as e:
            print(f"Failed: {wav_path}")
            print(e)
            continue

        waves.append(wav)
        lengths.append(wav.shape[0])
        valid_files.append(wav_path)

    if len(waves) == 0:
        continue

    max_len = max(lengths)

    padded = []

    for wav in waves:

        if wav.shape[0] < max_len:
            wav = F.pad(
                wav,
                (0, max_len - wav.shape[0])
            )

        padded.append(wav)

    audio = torch.stack(padded).to(device)

    length = torch.tensor(
        lengths,
        device=device
    )

    with torch.no_grad():

        processed_signal, processed_length = model.preprocessor(
            input_signal=audio,
            length=length
        )

        encoded, encoded_len = model.encoder(
            audio_signal=processed_signal,
            length=processed_length
        )

        emb = (
            encoded
            .mean(dim=2)
            .detach()
            .cpu()
            .numpy()
        )

    for i, wav_path in enumerate(valid_files):

        row = {
            "OutputPath": str(wav_path),
            "ClipID": wav_path.stem
        }

        for j, val in enumerate(emb[i]):
            row[f"emb_{j}"] = val

        rows.append(row)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Rows:", len(df))

if len(rows) > 0:
    print("Embedding dim:", emb.shape[1])