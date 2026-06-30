import re
import argparse
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--in_csv", required=True)
parser.add_argument("--out_csv", required=True)

args = parser.parse_args()


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

df = pd.read_csv(args.in_csv)

vector_cols = [c for c in df.columns if c.startswith("emb_")]

print("Vector columns:", vector_cols)


# ------------------------------------------------------------
# Parse vector strings
# ------------------------------------------------------------

float_pattern = re.compile(
    r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?"
)

def parse_vector(x):
    x = str(x)
    vals = [float(v) for v in float_pattern.findall(x)]
    return np.array(vals, dtype=float)


all_clip_embeddings = []

for idx, row in df.iterrows():

    segment_vectors = []

    for col in vector_cols:

        vec = parse_vector(row[col])

        if len(vec) > 0:
            segment_vectors.append(vec)

    if len(segment_vectors) == 0:
        raise ValueError(f"No vectors parsed for row {idx}")

    lengths = [len(v) for v in segment_vectors]

    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent vector lengths in row {idx}: {lengths}")

    clip_embedding = np.vstack(segment_vectors).mean(axis=0)

    all_clip_embeddings.append(clip_embedding)


# ------------------------------------------------------------
# Convert to wide embedding table
# ------------------------------------------------------------

X = np.vstack(all_clip_embeddings)

emb_df = pd.DataFrame(
    X,
    columns=[f"emb_{i}" for i in range(X.shape[1])]
)


out = pd.concat(
    [
        df[["OutputPath", "ClipID"]].reset_index(drop=True),
        emb_df.reset_index(drop=True)
    ],
    axis=1
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out.to_csv(args.out_csv, index=False)

print("Saved fixed embeddings:")
print(args.out_csv)
print(out.shape)