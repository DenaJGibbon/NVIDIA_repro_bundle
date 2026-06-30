from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


parser = argparse.ArgumentParser()

parser.add_argument("--embedding_csv", required=True)
parser.add_argument("--outdir", required=True)
parser.add_argument("--feature_prefix", default="emb_")
parser.add_argument("--name", default="model")

args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.embedding_csv)

df["IndividualID"] = df["ClipID"].str.extract(r"^([A-Za-z]+_\d{2})")

print(df[["ClipID", "IndividualID"]].head())
print("\nIndividual counts:")
print(df["IndividualID"].value_counts(dropna=False))

feature_cols = [
    c for c in df.columns
    if c.startswith(args.feature_prefix)
]

if len(feature_cols) == 0:
    raise ValueError(f"No columns found with prefix: {args.feature_prefix}")

X = df[feature_cols].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

df["PC1"] = pcs[:, 0]
df["PC2"] = pcs[:, 1]

df.to_csv(outdir / f"{args.name}_individual_pca_scores.csv", index=False)

plt.figure(figsize=(8, 7))

for ind, sub in df.groupby("IndividualID"):
    plt.scatter(
        sub["PC1"],
        sub["PC2"],
        s=20,
        alpha=0.75,
        label=ind
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
plt.title(f"{args.name}: PCA colored by individual ID")

if df["IndividualID"].nunique() <= 20:
    plt.legend(frameon=False, fontsize=7)

plt.tight_layout()

plot_path = outdir / f"{args.name}_individual_pca.png"
plt.savefig(plot_path, dpi=300)

print("\nSaved PCA plot:")
print(plot_path)


dat = df.dropna(subset=["IndividualID"]).copy()

X_probe = dat[feature_cols].to_numpy()
y = dat["IndividualID"].astype(str).to_numpy(dtype=str)

X_probe = StandardScaler().fit_transform(X_probe)

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    clf,
    X_probe,
    y,
    cv=cv,
    scoring="balanced_accuracy"
)

print("\nLinear probe: IndividualID")
print(f"Balanced accuracy mean: {scores.mean():.3f}")
print(f"Balanced accuracy sd:   {scores.std():.3f}")
print(f"Chance level approx:    {1 / len(pd.unique(y)):.3f}")
print("Class counts:")
print(pd.Series(y).value_counts())

summary = pd.DataFrame({
    "Model": [args.name],
    "Task": ["IndividualID"],
    "BalancedAccuracyMean": [scores.mean()],
    "BalancedAccuracySD": [scores.std()],
    "ChanceApprox": [1 / len(pd.unique(y))],
    "N": [len(y)],
    "NFeatures": [len(feature_cols)],
    "NClasses": [len(pd.unique(y))]
})

summary_path = outdir / f"{args.name}_individual_probe_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nSaved summary:")
print(summary_path)