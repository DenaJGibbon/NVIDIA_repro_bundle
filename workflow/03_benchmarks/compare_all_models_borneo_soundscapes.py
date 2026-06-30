import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


DATA_ROOT = Path("/home/nvidia/data/ssl_v1/ambient_acoustics_flat")
out_root = Path("/data/ssl_v1/borneo_soundscape_outputs")

models = {
    "SimCLR_full":
        out_root / "simclr_full_outputs/simclr_embeddings.csv",

    "SimCLR_danum":
        out_root / "simclr_danum_outputs/simclr_embeddings.csv",

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

    "Masked_autoencoder_full":
        out_root / "masked_autoencoder_full_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_danum":
        out_root / "masked_autoencoder_danum_outputs/masked_autoencoder_embeddings.csv",

    "Supervised_ecology":
        out_root / "supervised_ecology_outputs/supervised_ecology_embeddings.csv",

    "Student_BirdNET_resnet18":
    out_root / "student_birdnet_outputs/student_birdnet_embeddings.csv",

    "Supervised_ecology_rainfall":
        out_root / "supervised_ecology_rainfall_outputs/supervised_ecology_rainfall_embeddings.csv",
    
    "SimCLR_tropical_weighted_r18":
        out_root / "simclr_tropical_weighted_outputs/simclr_embeddings.csv",
    
    "SimCLR_tropical_weighted_r50":
        out_root / "simclr_tropical_weighted_resnet50_outputs/simclr_embeddings.csv",
    
    "Masked_autoencoder_tropical_weighted":
        out_root / "masked_autoencoder_tropical_weighted_outputs/masked_autoencoder_embeddings.csv",
     
    "BigVGAN":
        out_root / "bigvgan_outputs/bigvgan_embeddings.csv",
    
}


umap_root = out_root / "umap_outputs"
umap_root.mkdir(parents=True, exist_ok=True)

min_clips_per_class = 5
max_probe_rows = 20000
max_umap_rows = 5000

results = []


def get_class_from_path(row):
    path_col = None

    if "OutputPath" in row and pd.notna(row["OutputPath"]):
        path_col = row["OutputPath"]
    elif "FilePath" in row and pd.notna(row["FilePath"]):
        path_col = row["FilePath"]

    if path_col is None:
        return None

    file_stem = Path(str(path_col)).stem

    return file_stem.split("_")[0]
    
def make_umap_class_plot(dat, feature_cols, model_name):

    outdir = umap_root / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    plot_df = dat.copy()

    if len(plot_df) > max_umap_rows:
        plot_df = (
            plot_df
            .groupby("Class", group_keys=False)
            .apply(
                lambda x: x.sample(
                    min(
                        len(x),
                        max(
                            1,
                            int(max_umap_rows * len(x) / len(dat))
                        )
                    ),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )

        if len(plot_df) > max_umap_rows:
            plot_df = plot_df.sample(max_umap_rows, random_state=42).copy()

    X = plot_df[feature_cols].to_numpy()
    X = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    coords = reducer.fit_transform(X)

    plot_df["UMAP1"] = coords[:, 0]
    plot_df["UMAP2"] = coords[:, 1]

    score_csv = outdir / f"{model_name}_umap_scores_detection.csv"
    plot_df.to_csv(score_csv, index=False)

    plt.figure(figsize=(10, 8))

    for cls, sub in plot_df.groupby("Class"):
        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=10,
            alpha=0.65,
            label=str(cls),
        )

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"{model_name}: UMAP colored by detection class")

    if plot_df["Class"].nunique() <= 20:
        plt.legend(
            frameon=False,
            fontsize=8,
            markerscale=2,
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

    plt.tight_layout()

    plot_path = outdir / f"{model_name}_umap_detection_class.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("Saved UMAP:", plot_path)
    print("Saved UMAP scores:", score_csv)


for model_name, embedding_csv in models.items():

    print(f"\nProcessing: {model_name}")

    embedding_csv = Path(embedding_csv)

    if not embedding_csv.exists():
        print(f"Skipping {model_name}: file not found")
        continue

    df = pd.read_csv(embedding_csv)

    feature_cols = [
        c for c in df.columns
        if c.startswith("emb_")
    ]

    if len(feature_cols) == 0:
        print(f"Skipping {model_name}: no embedding columns")
        continue

    df["Class"] = df.apply(get_class_from_path, axis=1)

    dat = df.dropna(subset=["Class"]).copy()

    counts = dat["Class"].value_counts()
    keep_classes = counts[counts >= min_clips_per_class].index

    dat = dat[dat["Class"].isin(keep_classes)].copy()

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
                            int(max_probe_rows * len(x) / len(dat))
                        )
                    ),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )

        if len(probe_dat) > max_probe_rows:
            probe_dat = probe_dat.sample(max_probe_rows, random_state=42).copy()

    X = probe_dat[feature_cols].to_numpy()
    y = probe_dat["Class"].astype(str).to_numpy(dtype=str)

    X = StandardScaler().fit_transform(X)

    class_counts = pd.Series(y).value_counts()
    n_splits = min(5, int(class_counts.min()))

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

    result = {
        "Model": model_name,
        "BalancedAccuracyMean": scores.mean(),
        "BalancedAccuracySD": scores.std(),
        "ChanceApprox": 1 / len(pd.unique(y)),
        "N": len(y),
        "NClasses": len(pd.unique(y)),
        "NFeatures": len(feature_cols),
        "MinClipsPerClass": min_clips_per_class,
        "MaxProbeRows": max_probe_rows,
    }

    results.append(result)

    print("\nLinear probe: detection class")
    print(f"Balanced accuracy mean: {scores.mean():.3f}")
    print(f"Balanced accuracy sd:   {scores.std():.3f}")
    print(f"Chance level approx:    {1 / len(pd.unique(y)):.3f}")

    make_umap_class_plot(
        dat=dat,
        feature_cols=feature_cols,
        model_name=model_name,
    )


summary = pd.DataFrame(results)

summary = summary.sort_values(
    "BalancedAccuracyMean",
    ascending=False
)

print("\nModel comparison summary:")
print(summary)

summary_path = out_root / "model_comparison_summary.csv"

summary.to_csv(
    summary_path,
    index=False
)

print("\nSaved summary:")
print(summary_path)