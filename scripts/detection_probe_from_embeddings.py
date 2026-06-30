import argparse
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


parser = argparse.ArgumentParser()

parser.add_argument("--embedding_csv", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--feature_prefix", default="emb_")

args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.embedding_csv)

df["Label"] = df["OutputPath"].apply(
    lambda x: Path(x).parent.name
)

print("Label counts:")
print(df["Label"].value_counts())

feature_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]

X = df[feature_cols].to_numpy()
y = df["Label"].astype(str).to_numpy(dtype=str)

X = StandardScaler().fit_transform(X)

clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    clf,
    X,
    y,
    cv=cv,
    scoring="balanced_accuracy"
)

print(f"\nLinear probe: {args.name}")
print(f"Balanced accuracy mean: {scores.mean():.3f}")
print(f"Balanced accuracy sd:   {scores.std():.3f}")
print(f"Chance level approx:    {1 / len(pd.unique(y)):.3f}")

summary = pd.DataFrame({
    "Model": [args.name],
    "Task": ["Gibbon_vs_noise"],
    "BalancedAccuracyMean": [scores.mean()],
    "BalancedAccuracySD": [scores.std()],
    "ChanceApprox": [1 / len(pd.unique(y))],
    "N": [len(y)],
    "NClasses": [len(pd.unique(y))],
    "NFeatures": [len(feature_cols)]
})

summary.to_csv(outdir / f"{args.name}_detection_probe_summary.csv", index=False)
print("Saved:", outdir / f"{args.name}_detection_probe_summary.csv")
