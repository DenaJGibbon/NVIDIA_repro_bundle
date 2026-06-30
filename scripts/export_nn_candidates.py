from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import soundfile as sf

import matplotlib.pyplot as plt
from scipy.signal import spectrogram


parser = argparse.ArgumentParser()

parser.add_argument("--candidates_csv", required=True)
parser.add_argument("--out_dir", required=True)

parser.add_argument("--top_n", type=int, default=100)

args = parser.parse_args()


out_dir = Path(args.out_dir)

wav_dir = out_dir / "wav"
spec_dir = out_dir / "spectrograms"

wav_dir.mkdir(parents=True, exist_ok=True)
spec_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.candidates_csv)

df = df.head(args.top_n).copy()

print("Exporting:", len(df))


def make_spectrogram(audio, sr, out_png):

    f, t, Sxx = spectrogram(
        audio,
        fs=sr,
        nperseg=1024,
        noverlap=768
    )

    Sxx = 10 * np.log10(Sxx + 1e-10)

    plt.figure(figsize=(8, 4))

    plt.imshow(
        Sxx,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        out_png,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


rows = []

for i, row in df.iterrows():

    # ------------------------------------------------------------
    # Rank handling
    # ------------------------------------------------------------

    if "Rank" in row.index:
        rank = int(row["Rank"])

    elif "BestRank" in row.index:
        rank = int(row["BestRank"])

    else:
        rank = i + 1

    # ------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------

    if "OutputPath" in row.index:
        wav_path = Path(row["OutputPath"])

    elif "CandidatePath" in row.index:
        wav_path = Path(row["CandidatePath"])

    else:
        print("No path column found.")
        continue

    try:

        audio, sr = sf.read(str(wav_path))

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # ------------------------------------------------------------
        # Segment-level export
        # ------------------------------------------------------------

        if (
            "SegmentStartSec" in row.index
            and "SegmentEndSec" in row.index
        ):

            start_sec = float(row["SegmentStartSec"])
            end_sec = float(row["SegmentEndSec"])

            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            audio = audio[start_sample:end_sample]

            seg_string = (
                f"_seg{int(row['SegmentIndex']):03d}"
                f"_{start_sec:.1f}-{end_sec:.1f}s"
            )

        else:

            seg_string = ""

        # ------------------------------------------------------------
        # Similarity
        # ------------------------------------------------------------

        if "CentroidSimilarity" in row.index:
            sim = float(row["CentroidSimilarity"])

        elif "BestSimilarity" in row.index:
            sim = float(row["BestSimilarity"])

        else:
            sim = np.nan

        # ------------------------------------------------------------
        # Naming
        # ------------------------------------------------------------
# ------------------------------------------------------------
# Naming
# ------------------------------------------------------------
        
        clip_id = (
            row["ClipID"]
            if "ClipID" in row.index
            else row.get("CandidateClipID", wav_path.stem)
        )
        
        if np.isnan(sim):
            sim_str = "simNA"
        else:
            sim_str = f"sim{sim:.3f}"
        
        rank_str = str(rank).zfill(4)
        
        base_name = (
            f"{sim_str}"
            f"_rank{rank_str}"
            f"_{clip_id}"
            f"{seg_string}"
        )

        out_wav = wav_dir / f"{base_name}.wav"
        out_png = spec_dir / f"{base_name}.png"

        sf.write(
            str(out_wav),
            audio,
            sr
        )

        make_spectrogram(
            audio,
            sr,
            out_png
        )

        out_row = row.to_dict()

        out_row["ExportedWav"] = str(out_wav)
        out_row["ExportedSpectrogram"] = str(out_png)

        rows.append(out_row)

    except Exception as e:

        print("Failed:", wav_path)
        print(e)


export_df = pd.DataFrame(rows)

export_df.to_csv(
    out_dir / "exported_candidates_metadata.csv",
    index=False
)

print("\nDone.")
print("Output:", out_dir)