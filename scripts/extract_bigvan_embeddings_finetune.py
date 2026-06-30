from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import librosa

import bigvgan
from meldataset import get_mel_spectrogram


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", required=True)
parser.add_argument("--out_csv", required=True)
parser.add_argument("--model_name", default="nvidia/bigvgan_v2_24khz_100band_256x")
parser.add_argument("--checkpoint_file", default=None)
parser.add_argument("--device", default="cuda")
parser.add_argument("--max_files", type=int, default=None)
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

data_root = Path(args.data_root)
out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

files = sorted([
    p for p in data_root.rglob("*")
    if p.suffix.lower() in [".wav", ".flac", ".mp3"]
])

if args.max_files is not None:
    files = files[:args.max_files]

print("Files found:", len(files))
print("Device:", device)

print("Loading BigVGAN:", args.model_name)
model = bigvgan.BigVGAN.from_pretrained(
    args.model_name,
    use_cuda_kernel=False
)

if args.checkpoint_file is not None:
    print("Loading fine-tuned checkpoint:", args.checkpoint_file)
    ckpt = torch.load(args.checkpoint_file, map_location=device)
    state = ckpt["generator"] if "generator" in ckpt else ckpt
    model.load_state_dict(state)

model.remove_weight_norm()
model = model.eval().to(device)

sr_model = model.h.sampling_rate
print("Sample rate:", sr_model)
print("Num mels:", model.h.num_mels)

activations = {}

def make_hook(name):
    def hook_fn(module, inputs, output):
        if torch.is_tensor(output):
            activations[name] = output.detach()
    return hook_fn

hooked_layers = []

target_layers = [
    "conv_pre",
    "ups.0",
    "ups.1",
    "ups.2",
    "ups.3",
    "conv_post",
]

for name, module in model.named_modules():
    if name in target_layers:
        module.register_forward_hook(make_hook(name))
        hooked_layers.append(name)

print("Hooked layers:", hooked_layers)

def load_audio(path):
    audio, _ = librosa.load(str(path), sr=sr_model, mono=True)
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0:
        raise ValueError("Empty audio")

    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    return torch.FloatTensor(audio).unsqueeze(0).to(device)

def summarize_tensor(x):
    if x.ndim == 3:
        mean = x.mean(dim=2).squeeze(0)
        sd = x.std(dim=2).squeeze(0)
        return torch.cat([mean, sd], dim=0)

    if x.ndim == 2:
        mean = x.mean(dim=1)
        sd = x.std(dim=1)
        return torch.cat([mean, sd], dim=0)

    if x.ndim == 1:
        return x

    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(dim=1)
    sd = flat.std(dim=1)
    return torch.cat([mean, sd], dim=0)

def extract_features(path):
    audio = load_audio(path)
    mel = get_mel_spectrogram(audio, model.h).to(device)

    activations.clear()

    with torch.inference_mode():
        _ = model(mel)

    feature_parts = [summarize_tensor(mel)]

    for layer_name in hooked_layers:
        if layer_name in activations:
            feature_parts.append(summarize_tensor(activations[layer_name]))

    feat = torch.cat(feature_parts, dim=0)
    return feat.detach().cpu().numpy()

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

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Rows:", len(df))
print("Embedding dim:", len([c for c in df.columns if c.startswith("emb_")]))
