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
# Convert segment-level embeddings to clip-level embeddings
# ------------------------------------------------------------

if embeddings.ndim == 3:

    # files x segments x embedding_dim
    embeddings = embeddings.mean(axis=1)

elif embeddings.ndim == 2:

    # already files x embedding_dim
    pass

else:

    raise ValueError(
        f"Unexpected embeddings shape: "
        f"{embeddings.shape}"
    )

print(
    "Clip-level embeddings shape:",
    embeddings.shape
)


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

rows = []

for i, f in enumerate(files):

    if i % 25 == 0:

        print(
            f"Writing row "
            f"{i + 1}/{len(files)}"
        )

    row = {
        "OutputPath": str(f),
        "ClipID": f.stem
    }

    for j in range(embeddings.shape[1]):

        row[f"emb_{j}"] = embeddings[i, j]

    rows.append(row)

df = pd.DataFrame(rows)

df.to_csv(
    out_csv,
    index=False
)

print("Saved:", out_csv)