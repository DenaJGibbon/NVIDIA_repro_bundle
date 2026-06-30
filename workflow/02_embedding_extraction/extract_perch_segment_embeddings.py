from pathlib import Path
import argparse
import time
import os

import numpy as np
import pandas as pd
import birdnet
import torch


# ------------------------------------------------------------
# Threading
# ------------------------------------------------------------

torch.set_num_threads(16)

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", required=True)
parser.add_argument("--out_csv", required=True)

parser.add_argument(
    "--device",
    default="CPU"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=16
)

parser.add_argument(
    "--n_workers",
    type=int,
    default=16
)

parser.add_argument(
    "--max_files",
    type=int,
    default=None
)

args = parser.parse_args()


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

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

print("Device:", args.device)
print("Found files:", len(files))


# ------------------------------------------------------------
# Load Perch
# ------------------------------------------------------------

model = birdnet.load_perch_v2(
    device=args.device
)

print("Embedding dim:", model.get_embeddings_dim())

print("Sample rate:", model.get_sample_rate())

print(
    "Segment size seconds:",
    model.get_segment_size_s()
)


# ------------------------------------------------------------
# Encode
# ------------------------------------------------------------

print("Starting Perch encoding...")

start_time = time.time()

result = model.encode(
    [str(f) for f in files],
    batch_size=args.batch_size,
    n_producers=1,
    n_workers=args.n_workers,
    device=args.device,
    show_stats="progress"
)

elapsed = time.time() - start_time

print(
    f"Encoding finished in "
    f"{elapsed / 60:.2f} minutes"
)

print("Result type:", type(result))

print(
    "Result attributes:",
    [x for x in dir(result) if not x.startswith("_")]
)


# ------------------------------------------------------------
# Extract embeddings
# ------------------------------------------------------------

if hasattr(result, "embeddings"):

    embeddings = result.embeddings

elif hasattr(result, "embedding"):

    embeddings = result.embedding

elif hasattr(result, "data"):

    embeddings = result.data

else:

    raise ValueError(
        "Could not find embeddings on result object."
    )

embeddings = np.asarray(embeddings)

print(
    "Raw embeddings shape:",
    embeddings.shape
)


# ------------------------------------------------------------
# Save window-level embeddings
# ------------------------------------------------------------

rows = []

if embeddings.ndim == 3:

    print(
        "Detected window-level embeddings:"
        f" {embeddings.shape[1]} windows per clip"
    )

    window_sec = model.get_segment_size_s()

    for file_i, f in enumerate(files):

        if file_i % 25 == 0:

            print(
                f"Processing file "
                f"{file_i + 1}/{len(files)}"
            )

        try:

            import soundfile as sf

            info = sf.info(str(f))

            duration_sec = (
                info.frames /
                info.samplerate
            )

        except Exception:

            duration_sec = np.nan

        for win_i in range(embeddings.shape[1]):

            emb = embeddings[file_i, win_i]

            row = {
                "OutputPath": str(f),

                "ClipID": f.stem,

                "DurationSec": duration_sec,

                "WindowIndex": win_i,

                "WindowStartSec":
                    win_i * window_sec,

                "WindowEndSec":
                    min(
                        (win_i + 1) * window_sec,
                        duration_sec
                    ),

                "WindowClipID": (
                    f"{f.stem}"
                    f"_win{win_i:03d}"
                    f"_{win_i * window_sec:.1f}-"
                    f"{min((win_i + 1) * window_sec, duration_sec):.1f}s"
                )
            }

            for j in range(len(emb)):

                row[f"emb_{j}"] = emb[j]

            rows.append(row)

elif embeddings.ndim == 2:

    print(
        "Detected clip-level embeddings"
    )

    for file_i, f in enumerate(files):

        try:

            import soundfile as sf

            info = sf.info(str(f))

            duration_sec = (
                info.frames /
                info.samplerate
            )

        except Exception:

            duration_sec = np.nan

        emb = embeddings[file_i]

        row = {
            "OutputPath": str(f),
            "ClipID": f.stem,
            "DurationSec": duration_sec
        }

        for j in range(len(emb)):

            row[f"emb_{j}"] = emb[j]

        rows.append(row)

else:

    raise ValueError(
        f"Unexpected embedding shape: "
        f"{embeddings.shape}"
    )


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(
    out_csv,
    index=False
)

print("Saved:", out_csv)

print("Rows:", len(df))

print(
    "Embedding dimension:",
    len(
        [c for c in df.columns
         if c.startswith("emb_")]
    )
)