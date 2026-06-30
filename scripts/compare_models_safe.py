import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import confusion_matrix


out_root = Path("/data/ssl_v1/safe_soundscape_outputs")

models = {
    "SimCLR_full":
        out_root / "simclr_full_outputs/simclr_embeddings.csv",

    "SimCLR_danum":
        out_root / "simclr_danum_outputs/simclr_embeddings.csv",

    "SimCLR_tropical_weighted_r18":
        out_root / "simclr_tropical_weighted_outputs/simclr_embeddings.csv",

    "SimCLR_tropical_weighted_r50":
        out_root / "simclr_tropical_weighted_resnet50_outputs/simclr_embeddings.csv",

    "Masked_autoencoder_full":
        out_root / "masked_autoencoder_full_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_danum":
        out_root / "masked_autoencoder_danum_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_tropical_weighted":
        out_root / "masked_autoencoder_tropical_weighted_outputs/masked_autoencoder_embeddings.csv",

    "NeMo_base":
        out_root / "nemo_outputs/nemo_embeddings.csv",

    "NeMo_adapted_full":
        out_root / "nemo_adapt_full_outputs/nemo_adapt_embeddings.csv",

    "BirdNET":
        out_root / "birdnet_outputs/birdnet_embeddings.csv",

    "Perch_v2":
        out_root / "perch_outputs/perch_embeddings.csv",

    "BEATs":
        out_root / "beats_outputs/beats_embeddings.csv",

    "Supervised_ecology":
        out_root / "supervised_ecology_outputs/supervised_ecology_embeddings.csv",

    "Supervised_ecology_rainfall":
        out_root / "supervised_ecology_rainfall_outputs/supervised_ecology_rainfall_embeddings.csv",

    "Student_BirdNET_resnet18":
        out_root / "student_birdnet_outputs/student_birdnet_embeddings.csv",

        "BigVGAN":
        out_root / "bigvgan_outputs/bigvgan_embeddings.csv",
}

summary_path = out_root / "safe_model_comparison_summary.csv"

min_clips_per_class = 5
max_probe_rows = 20000

results = []


def get_safe_class(row):

    if "ClipID" in row and pd.notna(row["ClipID"]):
        stem = str(row["ClipID"])

    elif "OutputPath" in row and pd.notna(row["OutputPath"]):
        stem = Path(str(row["OutputPath"])).stem

    elif "FilePath" in row and pd.notna(row["FilePath"]):
        stem = Path(str(row["FilePath"])).stem

    else:
        return None

    return stem.split("_")[0]


for model_name, embedding_csv in models.items():

    print(f"\nProcessing: {model_name}")

    embedding_csv = Path(embedding_csv)

    if not embedding_csv.exists():
        print(f"Skipping {model_name}: file not found")
        print(embedding_csv)
        continue

    df = pd.read_csv(embedding_csv)

    feature_cols = [
        c for c in df.columns
        if c.startswith("emb_")
    ]

    if len(feature_cols) == 0:
        print(f"Skipping {model_name}: no embedding columns")
        continue

    # ------------------------------------------------------------
    # SAFE class labels
    # ------------------------------------------------------------

    df["Class"] = df.apply(get_safe_class, axis=1)

    # Combine old-growth classes

    df["Class"] = df["Class"].replace({
        "OG1": "OG",
        "OG2": "OG",
    })

    dat = df.dropna(subset=["Class"]).copy()

    counts = dat["Class"].value_counts()

    keep_classes = counts[
        counts >= min_clips_per_class
    ].index

    dat = dat[
        dat["Class"].isin(keep_classes)
    ].copy()

    print(f"Keeping classes with >= {min_clips_per_class} clips")

    print("N clips:", len(dat))
    print("N classes:", dat["Class"].nunique())

    print("Class counts:")
    print(dat["Class"].value_counts())

    if dat["Class"].nunique() < 2:
        print(f"Skipping {model_name}: fewer than 2 classes")
        continue

    probe_dat = dat.copy()

    if len(probe_dat) > max_probe_rows:

        probe_dat = (
            probe_dat
            .groupby("Class", group_keys=False)
            .apply(
                lambda x: x.sample(
                    min(
                        len(x),
                        max(
                            1,
                            int(
                                max_probe_rows *
                                len(x) /
                                len(dat)
                            )
                        )
                    ),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )

        if len(probe_dat) > max_probe_rows:

            probe_dat = probe_dat.sample(
                max_probe_rows,
                random_state=42
            ).copy()

    X = probe_dat[feature_cols].to_numpy()

    y = probe_dat["Class"].astype(str).to_numpy(
        dtype=str
    )

    X = StandardScaler().fit_transform(X)

    class_counts = pd.Series(y).value_counts()

    n_splits = min(
        3,
        int(class_counts.min())
    )

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    # ------------------------------------------------------------
    # Balanced accuracy
    # ------------------------------------------------------------

    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy"
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    y_pred = cross_val_predict(
        clf,
        X,
        y,
        cv=cv
    )

    labels = sorted(pd.unique(y))

    cm = confusion_matrix(
        y,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("\nConfusion matrix:")
    print(cm_df)

    # ------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------

    result = {
        "Model": model_name,
        "Task": "SAFE_ForestGradient",
        "BalancedAccuracyMean": scores.mean(),
        "BalancedAccuracySD": scores.std(),
        "ChanceApprox": 1 / len(pd.unique(y)),
        "N": len(y),
        "NClasses": len(pd.unique(y)),
        "NFeatures": len(feature_cols),
        "MinClipsPerClass": min_clips_per_class,
        "MaxProbeRows": max_probe_rows,
        "EmbeddingCSV": str(embedding_csv),
    }

    results.append(result)

    summary = pd.DataFrame(results).sort_values(
        "BalancedAccuracyMean",
        ascending=False
    )

    summary.to_csv(summary_path, index=False)

    print("\nLinear probe: SAFE forest gradient")
    print(result)

    print("\nIncrementally saved:")
    print(summary_path)


summary = pd.DataFrame(results).sort_values(
    "BalancedAccuracyMean",
    ascending=False
)

summary.to_csv(summary_path, index=False)

print("\nFinal SAFE model comparison summary:")
print(summary)

print("\nSaved:")
print(summary_path)