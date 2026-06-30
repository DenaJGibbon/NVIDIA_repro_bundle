from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


parser = argparse.ArgumentParser()

parser.add_argument("--seed_embeddings", required=True)
parser.add_argument("--search_embeddings", required=True)
parser.add_argument("--out_csv", required=True)

parser.add_argument("--seed_label", default="target_signal")
parser.add_argument("--top_n", type=int, default=2000)

args = parser.parse_args()


seed = pd.read_csv(args.seed_embeddings)
search = pd.read_csv(args.search_embeddings)


# ------------------------------------------------------------
# Standardize BirdNET and Perch columns
# ------------------------------------------------------------

rename_map = {
    "SegmentIndex": "WindowIndex",
    "SegmentStartSec": "WindowStartSec",
    "SegmentEndSec": "WindowEndSec",
    "SegmentClipID": "WindowClipID",
}

seed = seed.rename(columns=rename_map)
search = search.rename(columns=rename_map)

if (
    "WindowStartSec" in search.columns
    and "WindowEndSec" in search.columns
):

    search["WindowStartSec"] = pd.to_numeric(
        search["WindowStartSec"],
        errors="coerce"
    )

    search["WindowEndSec"] = pd.to_numeric(
        search["WindowEndSec"],
        errors="coerce"
    )

    search = search.dropna(
        subset=["WindowStartSec", "WindowEndSec"]
    )

    search = search[
        search["WindowEndSec"] >
        search["WindowStartSec"]
    ]

seed_cols = [
    c for c in seed.columns
    if c.startswith("emb_")
]

search_cols = [
    c for c in search.columns
    if c.startswith("emb_")
]

feature_cols = sorted(
    set(seed_cols).intersection(search_cols),
    key=lambda x: int(x.replace("emb_", ""))
)

seed = seed.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(subset=feature_cols)

search = search.replace(
    [np.inf, -np.inf],
    np.nan
).dropna(subset=feature_cols)

print("Seed rows:", len(seed))
print("Search rows:", len(search))
print("Embedding dims:", len(feature_cols))

# ------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------

seed_cols = [
    c for c in seed.columns
    if c.startswith("emb_")
]

search_cols = [
    c for c in search.columns
    if c.startswith("emb_")
]

feature_cols = sorted(
    set(seed_cols).intersection(search_cols),
    key=lambda x: int(x.replace("emb_", ""))
)

if len(feature_cols) == 0:
    raise ValueError("No shared emb_* columns found.")

print("Seed rows:", len(seed))
print("Search rows:", len(search))
print("Embedding dims:", len(feature_cols))


# ------------------------------------------------------------
# Remove rows with missing/non-finite embeddings
# ------------------------------------------------------------

seed = seed.replace([np.inf, -np.inf], np.nan)
search = search.replace([np.inf, -np.inf], np.nan)

seed = seed.dropna(subset=feature_cols).copy()
search = search.dropna(subset=feature_cols).copy()

print("Seed rows after NA filter:", len(seed))
print("Search rows after NA filter:", len(search))


# ------------------------------------------------------------
# Centroid similarity
# ------------------------------------------------------------

from sklearn.preprocessing import normalize

X_seed = seed[feature_cols].to_numpy(dtype=float)
X_search = search[feature_cols].to_numpy(dtype=float)

# L2 normalize embeddings
X_seed = normalize(X_seed)
X_search = normalize(X_search)

# centroid
centroid = X_seed.mean(
    axis=0,
    keepdims=True
)

centroid = normalize(centroid)

# cosine similarity
scores = cosine_similarity(
    X_search,
    centroid
).ravel()

out = search.copy()

out["SeedLabel"] = args.seed_label
out["CentroidSimilarity"] = scores

out = out.sort_values(
    "CentroidSimilarity",
    ascending=False
).reset_index(drop=True)

out["Rank"] = np.arange(1, len(out) + 1)

out = out.head(args.top_n).copy()

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

out.to_csv(out_csv, index=False)

print("\nSaved:")
print(out_csv)

print("\nTop 25:")

print_cols = [
    "Rank",
    "SeedLabel",
    "CentroidSimilarity",
    "ClipID",
    "WindowIndex",
    "WindowStartSec",
    "WindowEndSec",
    "OutputPath",
]

print_cols = [
    c for c in print_cols
    if c in out.columns
]

print(out[print_cols].head(25))
print_cols = [
    c for c in print_cols
    if c in out.columns
]

print(out[print_cols].head(25))