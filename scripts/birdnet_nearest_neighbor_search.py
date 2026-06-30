from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


parser = argparse.ArgumentParser()

parser.add_argument("--seed_embeddings", required=True)
parser.add_argument("--search_embeddings", required=True)
parser.add_argument("--out_csv", required=True)

parser.add_argument("--top_k", type=int, default=2000)
parser.add_argument("--metric", default="cosine")
parser.add_argument("--seed_label", default="target_signal")

args = parser.parse_args()


seed = pd.read_csv(args.seed_embeddings)
search = pd.read_csv(args.search_embeddings)

seed_cols = [c for c in seed.columns if c.startswith("emb_")]
search_cols = [c for c in search.columns if c.startswith("emb_")]

feature_cols = sorted(
    set(seed_cols).intersection(search_cols),
    key=lambda x: int(x.replace("emb_", ""))
)

if len(feature_cols) == 0:
    raise ValueError("No shared emb_* columns found.")

print("Seed rows:", len(seed))
print("Search rows:", len(search))
print("Embedding dims:", len(feature_cols))

X_seed = seed[feature_cols].to_numpy()
X_search = search[feature_cols].to_numpy()

scaler = StandardScaler()
X_search = scaler.fit_transform(X_search)
X_seed = scaler.transform(X_seed)

nn = NearestNeighbors(
    n_neighbors=min(args.top_k, len(search)),
    metric=args.metric,
)

nn.fit(X_search)

distances, indices = nn.kneighbors(X_seed)

rows = []

for seed_i in range(len(seed)):

    seed_clip = seed.loc[seed_i, "ClipID"]
    seed_path = seed.loc[seed_i, "OutputPath"]

    for rank, search_i in enumerate(indices[seed_i]):

        dist = distances[seed_i, rank]

        rows.append({
            "SeedLabel": args.seed_label,
            "SeedClipID": seed_clip,
            "SeedPath": seed_path,
            "RankWithinSeed": rank + 1,
            "Distance": dist,
            "Similarity": 1 - dist if args.metric == "cosine" else np.nan,
            "CandidateClipID": search.loc[search_i, "ClipID"],
            "CandidatePath": search.loc[search_i, "OutputPath"],
        })


raw = pd.DataFrame(rows)

agg = (
    raw
    .groupby(
        ["SeedLabel", "CandidateClipID", "CandidatePath"],
        as_index=False
    )
    .agg(
        BestDistance=("Distance", "min"),
        MeanDistance=("Distance", "mean"),
        BestSimilarity=("Similarity", "max"),
        MeanSimilarity=("Similarity", "mean"),
        NSeedMatches=("SeedClipID", "nunique"),
        BestRank=("RankWithinSeed", "min"),
    )
    .sort_values(
        ["BestDistance", "BestRank"],
        ascending=[True, True]
    )
)

out_csv = Path(args.out_csv)
out_csv.parent.mkdir(parents=True, exist_ok=True)

agg.to_csv(out_csv, index=False)

raw_csv = out_csv.with_name(out_csv.stem + "_raw_seed_matches.csv")
raw.to_csv(raw_csv, index=False)

print("Saved aggregated candidates:", out_csv)
print("Saved raw seed matches:", raw_csv)

print("\nTop 25:")
print(agg.head(25))
