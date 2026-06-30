from pathlib import Path
import argparse

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

import bigvgan
from meldataset import get_mel_spectrogram


parser = argparse.ArgumentParser()

parser.add_argument("--input_wavs_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--model_name", default="nvidia/bigvgan_v2_24khz_100band_256x")
parser.add_argument("--device", default="cuda")
parser.add_argument("--max_files", type=int, default=None)

args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

input_dir = Path(args.input_wavs_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted([
    p for p in input_dir.rglob("*")
    if p.suffix.lower() in [".wav", ".flac", ".mp3"]
])

if args.max_files is not None:
    files = files[:args.max_files]

print("Files:", len(files))
print("Device:", device)

model = bigvgan.BigVGAN.from_pretrained(
    args.model_name,
    use_cuda_kernel=False
)

model.remove_weight_norm()
model = model.eval().to(device)

sr = model.h.sampling_rate

for path in tqdm(files):

    try:
        wav, _ = librosa.load(
            str(path),
            sr=sr,
            mono=True
        )

        wav = np.asarray(wav, dtype=np.float32)

        wav_t = torch.FloatTensor(wav).unsqueeze(0).to(device)

        mel = get_mel_spectrogram(
            wav_t,
            model.h
        ).to(device)

        with torch.inference_mode():
            wav_gen = model(mel)

        wav_gen = wav_gen.squeeze().detach().cpu().numpy()

        if np.max(np.abs(wav_gen)) > 0:
            wav_gen = wav_gen / max(1.0, np.max(np.abs(wav_gen)))

        out_path = output_dir / f"{path.stem}_bigvgan_resynth.wav"

        sf.write(
            str(out_path),
            wav_gen,
            sr
        )

    except Exception as e:
        print("Failed:", path)
        print(e)

print("Saved to:", output_dir)
