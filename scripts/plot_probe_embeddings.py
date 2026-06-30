import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def classify_time(hour):
    if 4 <= hour < 8:
        return "Dawn"
    elif 8 <= hour < 18:
        return "Day"
    elif 18 <= hour < 20:
        return "Dusk"
    else:
        return "Night"


def run_probe(df, feature_cols, label_col):
    dat = df.dropna(subset=[label_col]).copy()

    X = dat[feature_cols].to_numpy()
    y = dat[label_col].astype(str).to_numpy(dtype=str)

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

    print(f"\nLinear probe: {label_col}")
    print(f"Balanced accuracy mean: {scores.mean():.3f}")
    print(f"Balanced accuracy sd:   {scores.std():.3f}")
    print(f"Chance level approx:    {1 / len(pd.unique(y)):.3f}")
    print("Class counts:")
    print(pd.Series(y).value_counts())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_csv", required=True)
    parser.add_argument("--habitat_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--feature_prefix", default="emb_")
    parser.add_argument("--name", default="embeddings")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.embedding_csv)
    habitat = pd.read_csv(args.habitat_csv)

    df["Plot"] = df["ClipID"].str.extract(r"(WA-T\d{2})")

    df["Hour"] = (
        df["ClipID"]
        .str.extract(r"_(\d{2})\d{4}\+")[0]
        .astype(int)
    )

    df["TimeOfDay"] = df["Hour"].apply(classify_time)

    df = df.merge(habitat, on="Plot", how="left")

    print(df[["ClipID", "Plot", "HabitatType", "TimeOfDay"]].head())
    print("\nHabitat counts:")
    print(df["HabitatType"].value_counts(dropna=False))
    print("\nTime-of-day counts:")
    print(df["TimeOfDay"].value_counts(dropna=False))

    feature_cols = [
        c for c in df.columns
        if c.startswith(args.feature_prefix)
    ]

    if len(feature_cols) == 0:
        raise ValueError(f"No columns found with prefix: {args.feature_prefix}")

    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    df = df.copy()
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    scores_csv = outdir / f"{args.name}_pca_scores_metadata.csv"
    df.to_csv(scores_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for habitat_type, sub in df.groupby("HabitatType"):
        axes[0].scatter(
            sub["PC1"],
            sub["PC2"],
            s=25,
            alpha=0.75,
            label=f"Habitat {habitat_type}"
        )

    axes[0].set_title(f"{args.name}: PCA by habitat type")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[0].legend(frameon=False, fontsize=8)

    for tod in ["Dawn", "Day", "Dusk", "Night"]:
        sub = df[df["TimeOfDay"] == tod]
        if len(sub) > 0:
            axes[1].scatter(
                sub["PC1"],
                sub["PC2"],
                s=25,
                alpha=0.75,
                label=tod
            )

    axes[1].set_title(f"{args.name}: PCA by time of day")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[1].legend(frameon=False, fontsize=8)

    plt.tight_layout()

    plot_path = outdir / f"{args.name}_pca_habitat_timeofday.png"
    plt.savefig(plot_path, dpi=300)

    print("\nSaved:")
    print(plot_path)
    print(scores_csv)

    run_probe(df, feature_cols, "HabitatType")
    run_probe(df, feature_cols, "TimeOfDay")


if __name__ == "__main__":
    main()