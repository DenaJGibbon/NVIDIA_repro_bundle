from pathlib import Path
import argparse

import pandas as pd
import numpy as np

import umap
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


parser = argparse.ArgumentParser()

parser.add_argument("--embeddings_csv", required=True)
parser.add_argument("--out_prefix", required=True)

args = parser.parse_args()


df = pd.read_csv(args.embeddings_csv)

if "Class" not in df.columns:
    raise ValueError(
        "Expected Class column."
    )

emb_cols = [
    c for c in df.columns
    if c.startswith("emb_")
]

print("Rows:", len(df))
print("Embedding dims:", len(emb_cols))
print("Classes:", df["Class"].nunique())

X = df[emb_cols].to_numpy(dtype=float)

X = normalize(X)

sil = silhouette_score(
    X,
    df["Class"]
)

print(
    f"Silhouette score: {sil:.3f}"
)

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.2,
    metric="cosine",
    random_state=42
)

coords = reducer.fit_transform(X)

df["UMAP1"] = coords[:, 0]
df["UMAP2"] = coords[:, 1]

coord_csv = (
    args.out_prefix +
    "_umap_coords.csv"
)

df.to_csv(
    coord_csv,
    index=False
)

print("Saved:", coord_csv)

plt.figure(
    figsize=(12, 10)
)

classes = sorted(
    df["Class"].unique()
)

for cls in classes:

    sub = df[
        df["Class"] == cls
    ]

    plt.scatter(
        sub["UMAP1"],
        sub["UMAP2"],
        s=20,
        alpha=0.7,
        label=cls
    )

plt.title(
    f"BigVGAN Features\n"
    f"Silhouette={sil:.3f}"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

png_file = (
    args.out_prefix +
    "_umap.png"
)

plt.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight"
)

print("Saved:", png_file)
