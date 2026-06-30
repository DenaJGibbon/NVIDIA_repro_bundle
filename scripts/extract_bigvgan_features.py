from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import librosa

import bigvgan
from meldataset import get_mel_spectrogram


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", required=True)
parser.add_argument("--out_csv", required=True)

parser.add_argument(
    "--model_name",
    default="nvidia/bigvgan_v2_24khz_100band_256x"
)

parser.add_argument("--device", default="cuda")
parser.add_argument("--max_files", type=int, default=None)

args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

data_root = Path(args.data_root)
out_csv = Path(args.out_csv)

out_csv.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

files = sorted([
    p for p in data_root.rglob("*")
    if p.suffix.lower() in [".wav", ".flac", ".mp3"]
])

if args.max_files is not None:
    files = files[:args.max_files]

print("Files found:", len(files))
print("Device:", device)


# ------------------------------------------------------------
# Load BigVGAN
# ------------------------------------------------------------

print("Loading BigVGAN:", args.model_name)

model = bigvgan.BigVGAN.from_pretrained(
    args.model_name,
    use_cuda_kernel=False
)

model.remove_weight_norm()
model = model.eval().to(device)

sr_model = model.h.sampling_rate

print("Sample rate:", sr_model)
print("Num mels:", model.h.num_mels)


# ------------------------------------------------------------
# Register hooks
# ------------------------------------------------------------

activations = {}

def make_hook(name):

    def hook_fn(module, inputs, output):

        if torch.is_tensor(output):
            activations[name] = output.detach()

    return hook_fn


hooked_layers = []

for name, module in model.named_modules():

    # Useful BigVGAN feature stages:
    # conv_pre = early mel projection
    # ups.* = intermediate generator hierarchy
    # resblocks.* = nonlinear residual generator features
    # conv_post = late waveform projection
        if name in [
            "conv_pre",
            "ups.0",
            "ups.1",
            "ups.2",
            "ups.3",
            "conv_post",
        ]:
                module.register_forward_hook(
            make_hook(name)
        )

        hooked_layers.append(name)

print("Hooked layers:", len(hooked_layers))

for name in hooked_layers[:20]:
    print(" ", name)

if len(hooked_layers) > 20:
    print(" ...")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_audio(path):

    audio, _ = librosa.load(
        str(path),
        sr=sr_model,
        mono=True
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        raise ValueError("Empty audio")

    # avoid extreme values
    peak = np.max(np.abs(audio))

    if peak > 1.0:
        audio = audio / peak

    audio = torch.FloatTensor(audio).unsqueeze(0).to(device)

    return audio


def summarize_tensor(x):

    # Expected shapes are usually:
    # batch x channels x time
    # or batch x time
    # We summarize over all non-batch/channel dimensions.

    if x.ndim == 3:

        mean = x.mean(dim=2).squeeze(0)
        sd = x.std(dim=2).squeeze(0)

        return torch.cat([mean, sd], dim=0)

    elif x.ndim == 2:

        mean = x.mean(dim=1)
        sd = x.std(dim=1)

        return torch.cat([mean, sd], dim=0)

    elif x.ndim == 1:

        return x

    else:

        flat = x.reshape(x.shape[0], -1)

        mean = flat.mean(dim=1)
        sd = flat.std(dim=1)

        return torch.cat([mean, sd], dim=0)


def extract_features(path):

    audio = load_audio(path)

    mel = get_mel_spectrogram(
        audio,
        model.h
    )

    activations.clear()

    with torch.no_grad():
        _ = model(mel)

    feature_parts = []

    # Always include mel summary
    feature_parts.append(
        summarize_tensor(mel)
    )

    # Add selected layer summaries
    for layer_name in hooked_layers:

        if layer_name not in activations:
            continue

        feature_parts.append(
            summarize_tensor(
                activations[layer_name]
            )
        )

    feat = torch.cat(
        feature_parts,
        dim=0
    )

    return feat.detach().cpu().numpy()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

rows = []

for path in tqdm(files):

    try:

        feat = extract_features(path)

        row = {
            "OutputPath": str(path),
            "ClipID": path.stem,
            "Class": path.parent.name,
        }

        for j, val in enumerate(feat):
            row[f"emb_{j}"] = val

        rows.append(row)

    except Exception as e:

        print("Failed:", path)
        print(e)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(
    out_csv,
    index=False
)

print("Saved:", out_csv)
print("Rows:", len(df))
print(
    "Embedding dim:",
    len([c for c in df.columns if c.startswith("emb_")])
)