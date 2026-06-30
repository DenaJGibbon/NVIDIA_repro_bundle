from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
import birdnet
import torch
import soundfile as sf


torch.set_num_threads(16)

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", required=True)
parser.add_argument("--out_csv", required=True)

parser.add_argument("--model_version", default="2.4")
parser.add_argument("--backend", default="tf")

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_workers", type=int, default=16)

parser.add_argument("--max_files", type=int, default=None)

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

print("Found files:", len(files))


# ------------------------------------------------------------
# Load BirdNET
# ------------------------------------------------------------

model = birdnet.load(
    "acoustic",
    args.model_version,
    args.backend
)

segment_sec = model.get_segment_size_s()

print("Embedding dim:", model.get_embeddings_dim())

print("Sample rate:", model.get_sample_rate())

print("Segment size seconds:", segment_sec)


# ------------------------------------------------------------
# Encode
# ------------------------------------------------------------

result = model.encode(
    [str(f) for f in files],
    batch_size=args.batch_size,
    n_workers=args.n_workers,
    show_stats="progress"
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

print("Raw embeddings shape:", embeddings.shape)


# ------------------------------------------------------------
# Save segment-level embeddings
# ------------------------------------------------------------

rows = []

if embeddings.ndim == 3:

    for file_i, f in enumerate(files):

        try:

            info = sf.info(str(f))

            duration_sec = (
                info.frames / info.samplerate
            )

            n_valid_segments = int(
                np.ceil(duration_sec / segment_sec)
            )

        except Exception:

            duration_sec = np.nan

            n_valid_segments = embeddings.shape[1]

        n_segments = min(
            embeddings.shape[1],
            n_valid_segments
        )

        for seg_i in range(n_segments):

            emb = embeddings[file_i, seg_i, :]

            row = {
                "OutputPath": str(f),

                "ClipID": f.stem,

                "SegmentIndex": seg_i,

                "SegmentStartSec": (
                    seg_i * segment_sec
                ),

                "SegmentEndSec": min(
                    (seg_i + 1) * segment_sec,
                    duration_sec
                ),

                "DurationSec": duration_sec,

                "SegmentClipID": (
                    f"{f.stem}"
                    f"_seg{seg_i:03d}"
                    f"_{seg_i * segment_sec:.1f}-"
                    f"{min((seg_i + 1) * segment_sec, duration_sec):.1f}s"
                ),
            }

            for j in range(emb.shape[0]):

                row[f"emb_{j}"] = emb[j]

            rows.append(row)

elif embeddings.ndim == 2:

    for file_i, f in enumerate(files):

        try:

            info = sf.info(str(f))

            duration_sec = (
                info.frames / info.samplerate
            )

        except Exception:

            duration_sec = segment_sec

        emb = embeddings[file_i, :]

        row = {
            "OutputPath": str(f),

            "ClipID": f.stem,

            "SegmentIndex": 0,

            "SegmentStartSec": 0.0,

            "SegmentEndSec": min(
                segment_sec,
                duration_sec
            ),

            "DurationSec": duration_sec,

            "SegmentClipID": (
                f"{f.stem}"
                f"_seg000"
                f"_0.0-{min(segment_sec, duration_sec):.1f}s"
            ),
        }

        for j in range(emb.shape[0]):

            row[f"emb_{j}"] = emb[j]

        rows.append(row)

else:

    raise ValueError(
        f"Unexpected embeddings shape: {embeddings.shape}"
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