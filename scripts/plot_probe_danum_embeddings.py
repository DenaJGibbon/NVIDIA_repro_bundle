import re
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def classify_time(hour):
    if pd.isna(hour):
        return None
    hour = int(hour)
    if 4 <= hour < 8:
        return "Dawn"
    elif 8 <= hour < 18:
        return "Day"
    elif 18 <= hour < 20:
        return "Dusk"
    else:
        return "Night"


def extract_location(x):
    m = re.search(r"_S(\d{2})_", str(x))
    if m:
        return f"S{m.group(1)}"
    m = re.search(r"(S\d{2})", str(x))
    if m:
        return m.group(1)
    return None


def extract_datetime_parts(x):
    x = str(x)

    m = re.search(r"_(\d{8})_(\d{6})Z", x)
    if m:
        date_raw = m.group(1)
        time_raw = m.group(2)
        return date_raw, time_raw

    m = re.search(r"_(\d{8})_(\d{6})", x)
    if m:
        date_raw = m.group(1)
        time_raw = m.group(2)
        return date_raw, time_raw

    return None, None


def run_probe(df, feature_cols, label_col, outdir, name):
    dat = df.dropna(subset=[label_col]).copy()

    if dat[label_col].nunique() < 2:
        print(f"\nSkipping probe for {label_col}: fewer than 2 classes")
        return None

    class_counts = dat[label_col].value_counts()

    min_class = class_counts.min()

    n_splits = min(5, int(min_class))

    if n_splits < 2:
        print(f"\nSkipping probe for {label_col}: smallest class has <2 samples")
        return None

    X = dat[feature_cols].to_numpy()
    y = dat[label_col].astype(str).to_numpy(dtype=str)

    X = StandardScaler().fit_transform(X)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
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
    print(class_counts)

    result = {
        "Model": name,
        "Task": label_col,
        "BalancedAccuracyMean": scores.mean(),
        "BalancedAccuracySD": scores.std(),
        "ChanceApprox": 1 / len(pd.unique(y)),
        "N": len(y),
        "NClasses": len(pd.unique(y)),
        "NFeatures": len(feature_cols),
        "NSplits": n_splits,
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--feature_prefix", default="emb_")
    parser.add_argument("--name", default="danum_model")
    parser.add_argument(
        "--method",
        choices=["pca"],
        default="pca"
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.embedding_csv)

    if "ClipID" not in df.columns:
        raise ValueError("Expected a ClipID column in embedding CSV.")

    # ------------------------------------------------------------
    # Parse Danum metadata from ClipID
    # Example:
    # T0010DV01_016K_S01_SWIFT01_20180423_160000Z
    # ------------------------------------------------------------

    df["Location"] = df["ClipID"].apply(extract_location)

    dt_parts = df["ClipID"].apply(extract_datetime_parts)

    df["DateRaw"] = [x[0] for x in dt_parts]
    df["TimeRaw"] = [x[1] for x in dt_parts]

    df["Date"] = pd.to_datetime(
        df["DateRaw"],
        format="%Y%m%d",
        errors="coerce"
    )

    df["Hour"] = (
        df["TimeRaw"]
        .str.slice(0, 2)
        .astype("float")
    )

    df["TimeOfDay"] = df["Hour"].apply(classify_time)

    print("\nParsed metadata preview:")
    print(df[["ClipID", "Location", "Date", "Hour", "TimeOfDay"]].head())

    print("\nLocation counts:")
    print(df["Location"].value_counts(dropna=False))

    print("\nTime-of-day counts:")
    print(df["TimeOfDay"].value_counts(dropna=False))

    print("\nDate range:")
    print(df["Date"].min(), "to", df["Date"].max())

    # ------------------------------------------------------------
    # Features
    # ------------------------------------------------------------

    feature_cols = [
        c for c in df.columns
        if c.startswith(args.feature_prefix)
    ]

    if len(feature_cols) == 0:
        raise ValueError(f"No columns found with prefix: {args.feature_prefix}")

    X = df[feature_cols].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    # ------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    scores_csv = outdir / f"{args.name}_danum_pca_scores_metadata.csv"
    df.to_csv(scores_csv, index=False)

    # ------------------------------------------------------------
    # Plot by Location
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 7))

    for loc, sub in df.groupby("Location"):
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            s=12,
            alpha=0.65,
            label=loc
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title(f"{args.name}: Danum PCA by recording location")
    plt.legend(frameon=False, fontsize=8, markerscale=2)
    plt.tight_layout()

    location_plot = outdir / f"{args.name}_danum_pca_location.png"
    plt.savefig(location_plot, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Plot by Time of Day
    # ------------------------------------------------------------

    plt.figure(figsize=(9, 7))

    for tod in ["Dawn", "Day", "Dusk", "Night"]:
        sub = df[df["TimeOfDay"] == tod]
        if len(sub) == 0:
            continue

        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            s=12,
            alpha=0.65,
            label=tod
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title(f"{args.name}: Danum PCA by time of day")
    plt.legend(frameon=False, fontsize=8, markerscale=2)
    plt.tight_layout()

    tod_plot = outdir / f"{args.name}_danum_pca_timeofday.png"
    plt.savefig(tod_plot, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Probe tasks
    # ------------------------------------------------------------

    results = []

    for label_col in ["Location", "TimeOfDay"]:
        res = run_probe(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            outdir=outdir,
            name=args.name
        )
        if res is not None:
            results.append(res)

    if len(results) > 0:
        summary = pd.DataFrame(results)
        summary_path = outdir / f"{args.name}_danum_probe_summary.csv"
        summary.to_csv(summary_path, index=False)

        print("\nSaved summary:")
        print(summary_path)

    print("\nSaved plots:")
    print(location_plot)
    print(tod_plot)
    print(scores_csv)


if __name__ == "__main__":
    main()
