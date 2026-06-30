from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import soundfile as sf
import librosa

import bigvgan
from meldataset import get_mel_spectrogram


parser = argparse.ArgumentParser()

parser.add_argument("--data_root", required=True)
parser.add_argument("--out_root", required=True)

parser.add_argument(
    "--model_name",
    default="nvidia/bigvgan_v2_24khz_100band_256x"
)

parser.add_argument("--target_per_class", type=int, default=100)
parser.add_argument("--max_originals_per_class", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", default="cuda")

parser.add_argument("--noise_sd", type=float, default=0.03)
parser.add_argument("--gain_min", type=float, default=0.85)
parser.add_argument("--gain_max", type=float, default=1.15)
parser.add_argument("--time_mask_frames", type=int, default=8)
parser.add_argument("--freq_mask_bins", type=int, default=8)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

device = args.device if torch.cuda.is_available() else "cpu"

data_root = Path(args.data_root)
out_root = Path(args.out_root)

aug_wav_root = out_root / "generated_wav"
feature_csv = out_root / "bigvgan_features.csv"

aug_wav_root.mkdir(parents=True, exist_ok=True)


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

print("Device:", device)
print("BigVGAN sample rate:", sr_model)
print("Mel bands:", model.h.num_mels)


# ------------------------------------------------------------
# Optional learned feature hooks
# ------------------------------------------------------------

activations = {}

def make_hook(name):
    def hook(module, inputs, output):
        activations[name] = output.detach()
    return hook

hook_names = []

for name, module in model.named_modules():
    if name in ["conv_pre"]:
        module.register_forward_hook(make_hook(name))
        hook_names.append(name)

print("Hooked feature layers:", hook_names)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

audio_exts = [".wav", ".flac", ".mp3"]


def list_class_files(root):

    class_files = {}

    for class_dir in sorted(root.iterdir()):

        if not class_dir.is_dir():
            continue

        files = sorted([
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in audio_exts
        ])

        if args.max_originals_per_class is not None:
            files = files[:args.max_originals_per_class]

        if len(files) > 0:
            class_files[class_dir.name] = files

    return class_files


def load_wav(path):

    wav, sr = librosa.load(
        str(path),
        sr=sr_model,
        mono=True
    )

    wav = np.asarray(wav, dtype=np.float32)

    if np.max(np.abs(wav)) > 0:
        wav = wav / max(1.0, np.max(np.abs(wav)))

    wav_t = torch.FloatTensor(wav).unsqueeze(0)

    return wav_t


def augment_mel(mel):

    mel_aug = mel.clone()

    gain = random.uniform(
        args.gain_min,
        args.gain_max
    )

    mel_aug = mel_aug * gain

    if args.noise_sd > 0:
        mel_aug = mel_aug + args.noise_sd * torch.randn_like(mel_aug)

    # Frequency mask
    if args.freq_mask_bins > 0:

        n_mels = mel_aug.shape[1]

        width = random.randint(
            1,
            min(args.freq_mask_bins, n_mels)
        )

        start = random.randint(
            0,
            max(0, n_mels - width)
        )

        mel_aug[:, start:start + width, :] = mel_aug.mean()

    # Time mask
    if args.time_mask_frames > 0:

        n_frames = mel_aug.shape[2]

        width = random.randint(
            1,
            min(args.time_mask_frames, n_frames)
        )

        start = random.randint(
            0,
            max(0, n_frames - width)
        )

        mel_aug[:, :, start:start + width] = mel_aug.mean()

    return mel_aug


def extract_bigvgan_features(mel):

    activations.clear()

    with torch.no_grad():
        wav_gen = model(mel)

    feature_parts = []

    # Mel summary features
    mel_mean = mel.mean(dim=2)
    mel_sd = mel.std(dim=2)

    feature_parts.append(mel_mean)
    feature_parts.append(mel_sd)

    # Learned generator activation summaries
    for name in hook_names:

        if name not in activations:
            continue

        act = activations[name]

        if act.ndim == 3:

            act_mean = act.mean(dim=2)
            act_sd = act.std(dim=2)

            feature_parts.append(act_mean)
            feature_parts.append(act_sd)

    feat = torch.cat(
        feature_parts,
        dim=1
    )

    return (
        wav_gen.squeeze().detach().cpu().numpy(),
        feat.squeeze().detach().cpu().numpy()
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

class_files = list_class_files(data_root)

print("\nClasses found:")
for cls, files in class_files.items():
    print(cls, len(files))

feature_rows = []

for cls, files in class_files.items():

    class_out = aug_wav_root / cls
    class_out.mkdir(parents=True, exist_ok=True)

    n_existing = len(files)
    n_to_generate = max(
        0,
        args.target_per_class - n_existing
    )

    print(
        f"\nClass: {cls} | existing={n_existing} | generating={n_to_generate}"
    )

    # ------------------------------------------------------------
    # Extract features for original clips
    # ------------------------------------------------------------

    for path in tqdm(files, desc=f"{cls} originals"):

        try:
            wav = load_wav(path)

            mel = get_mel_spectrogram(
                wav,
                model.h
            ).to(device)

            _, feat = extract_bigvgan_features(mel)

            row = {
                "Class": cls,
                "Type": "original",
                "SourcePath": str(path),
                "OutputPath": str(path),
            }

            for j, val in enumerate(feat):
                row[f"feat_{j}"] = val

            feature_rows.append(row)

        except Exception as e:
            print("Failed original:", path)
            print(e)

    # ------------------------------------------------------------
    # Generate augmented clips for underrepresented classes
    # ------------------------------------------------------------

    for i in tqdm(range(n_to_generate), desc=f"{cls} augmented"):

        src_path = random.choice(files)

        try:
            wav = load_wav(src_path)

            mel = get_mel_spectrogram(
                wav,
                model.h
            ).to(device)

            mel_aug = augment_mel(mel)

            wav_gen, feat = extract_bigvgan_features(mel_aug)

            wav_gen = np.asarray(wav_gen, dtype=np.float32)

            if np.max(np.abs(wav_gen)) > 0:
                wav_gen = wav_gen / max(1.0, np.max(np.abs(wav_gen)))

            out_wav = (
                class_out /
                f"{cls}_bigvgan_aug_{i:04d}_from_{src_path.stem}.wav"
            )

            sf.write(
                str(out_wav),
                wav_gen,
                sr_model
            )

            row = {
                "Class": cls,
                "Type": "bigvgan_augmented",
                "SourcePath": str(src_path),
                "OutputPath": str(out_wav),
            }

            for j, val in enumerate(feat):
                row[f"feat_{j}"] = val

            feature_rows.append(row)

        except Exception as e:
            print("Failed augmented:", src_path)
            print(e)


# ------------------------------------------------------------
# Save feature table
# ------------------------------------------------------------

feature_df = pd.DataFrame(feature_rows)

feature_df.to_csv(
    feature_csv,
    index=False
)

print("\nSaved generated wavs:", aug_wav_root)
print("Saved features:", feature_csv)
print("Rows:", len(feature_df))
