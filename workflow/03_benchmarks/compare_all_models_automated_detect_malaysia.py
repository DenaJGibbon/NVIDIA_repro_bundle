import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


models = {
    "SimCLR_full":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/simclr_full_outputs/simclr_embeddings.csv",

    "SimCLR_danum":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/simclr_danum_outputs/simclr_embeddings.csv",

    "SimCLR_tropical_weighted_r18":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/simclr_tropical_weighted_outputs/simclr_embeddings.csv",

    "SimCLR_tropical_weighted_r50":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/simclr_tropical_weighted_resnet50_outputs/simclr_embeddings.csv",

    "NeMo_base":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/nemo_outputs/nemo_embeddings.csv",

    "NeMo_adapted_full":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/nemo_adapt_full_outputs/nemo_adapt_embeddings.csv",

    "BirdNET":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/birdnet_outputs/birdnet_embeddings.csv",

    "Student_BirdNET_resnet18":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/student_birdnet_outputs/student_birdnet_embeddings.csv",

    "Perch_v2":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/perch_outputs/perch_embeddings.csv",

    "BEATs":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/beats_outputs/beats_embeddings.csv",

    "Masked_autoencoder_full":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/masked_autoencoder_full_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_danum":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/masked_autoencoder_danum_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_tropical_weighted":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/masked_autoencoder_tropical_weighted_outputs/masked_autoencoder_embeddings.csv",

    "Supervised_ecology":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/supervised_ecology_outputs/supervised_ecology_embeddings.csv",

    "Supervised_ecology_rainfall":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/supervised_ecology_rainfall_outputs/supervised_ecology_rainfall_embeddings.csv",

    "BigVGAN":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/bigvgan_outputs/bigvgan_embeddings.csv",
}

out_root = Path("/home/nvidia/test_run/gibbon_detection_outputs_malaysia")
umap_root = out_root / "umap_outputs"
umap_root.mkdir(parents=True, exist_ok=True)

min_clips_per_class = 5
max_probe_rows = 20000
max_umap_rows = 5000

results = []


def get_class_from_path(row):

    if "ClipID" in row and pd.notna(row["ClipID"]):
        return str(row["ClipID"]).split("_")[0]

    path_col = None

    if "OutputPath" in row and pd.notna(row["OutputPath"]):
        path_col = row["OutputPath"]
    elif "FilePath" in row and pd.notna(row["FilePath"]):
        path_col = row["FilePath"]

    if path_col is None:
        return None

    file_stem = Path(str(path_col)).stem

    return file_stem.split("_")[0]




for model_name, embedding_csv in models.items():

    print(f"\nProcessing: {model_name}")

    if not Path(embedding_csv).exists():
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