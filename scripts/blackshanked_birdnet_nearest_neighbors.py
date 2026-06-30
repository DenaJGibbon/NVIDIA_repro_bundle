from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import soundfile as sf

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed_root",
    default="/data/ssl_v1/training_kswsspecies/blackshanked"
)

parser.add_argument(
    "--search_embeddings",
    default="/data/ssl_v1/SSL_full_experiments/birdnet_outputs/birdnet_embeddings.csv"
)

parser.add_argument(
    "--out_root",
    default="/data/ssl_v1/ksws_agile_species_search/blackshanked_birdnet_nn"
)

parser.add_argument(
    "--clip_sec",
    type=float,
    default=3
)

parser.add_argument(
    "--top_k",
    type=int,
    default=1000
)

parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()

random.seed(args.seed)

seed_root = Path(args.seed_root)
out_root = Path(args.out_root)
out_root.mkdir(parents=True, exist_ok=True)

seed_3s_root = out_root / "seed_3s_clips"
seed_3s_root.mkdir(parents=True, exist_ok=True)

seed_embeddings_csv = out_root / "seed_birdnet_embeddings.csv"
candidates_csv = out_root / "blackshanked_nearest_neighbors.csv"


# ------------------------------------------------------------
# Step 1: Make 3-s padded seed clips
# ------------------------------------------------------------

audio_exts = [".wav", ".flac", ".mp3"]

seed_files = sorted([
    p for p in seed_root.rglob("*")
    if p.suffix.lower() in audio_exts
])

print("Seed files found:", len(seed_files))

failed = []

for in_path in seed_files:

    try:
        audio, sr = sf.read(str(in_path))

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        target_len = int(args.clip_sec * sr)

        if len(audio) >= target_len:
            start = random.randint(0, len(audio) - target_len)
            out_audio = audio[start:start + target_len]

        else:
            out_audio = np.zeros(target_len, dtype=audio.dtype)
            max_start = target_len - len(audio)
            insert_start = random.randint(0, max_start)

            out_audio[
                insert_start:insert_start + len(audio)
            ] = audio

        out_path = seed_3s_root / f"{in_path.stem}_3s.wav"

        sf.write(str(out_path), out_audio, sr)

    except Exception as e:
        failed.append({
            "FilePath": str(in_path),
            "Error": str(e)
        })

if failed:
    pd.DataFrame(failed).to_csv(
        out_root / "failed_seed_padding.csv",
        index=False
    )

print("Seed 3-s clips written:", len(list(seed_3s_root.glob('*.wav'))))


# ------------------------------------------------------------
# Step 2 reminder: seed embeddings must be extracted externally
# ------------------------------------------------------------

print("\nNow extract BirdNET embeddings for seed clips with:")
print(f"""
cd /home/nvidia/birdnet_test
source birdnet_env/bin/activate

python extract_birdnet_embeddings.py \\
  --data_root {seed_3s_root} \\
  --out_csv {seed_embeddings_csv} \\
  --batch_size 64 \\
  --n_workers 16
""")

print("\nAfter that finishes, rerun this script with:")
print("--skip_padding")
