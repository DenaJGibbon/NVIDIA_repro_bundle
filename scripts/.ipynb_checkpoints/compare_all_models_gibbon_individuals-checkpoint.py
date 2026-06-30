import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


models = {

    "SimCLR_full":
        "/home/nvidia/test_run/gibbon_individual_outputs/simclr_full_outputs/simclr_embeddings.csv",

    "SimCLR_danum":
        "/home/nvidia/test_run/gibbon_individual_outputs/simclr_danum_outputs/simclr_embeddings.csv",

    "SimCLR_tropical_weighted_r18":
        "/home/nvidia/test_run/gibbon_individual_outputs/simclr_tropical_weighted_outputs/simclr_embeddings.csv",

    "SimCLR_tropical_weighted_r50":
        "/home/nvidia/test_run/gibbon_individual_outputs/simclr_tropical_weighted_resnet50_outputs/simclr_embeddings.csv",

    "NeMo_base":
        "/home/nvidia/test_run/gibbon_individual_outputs/nemo_outputs/nemo_embeddings.csv",

    "NeMo_adapted_full":
        "/home/nvidia/test_run/gibbon_individual_outputs/nemo_adapt_full_outputs/nemo_adapt_embeddings.csv",

    "BirdNET":
        "/home/nvidia/test_run/gibbon_individual_outputs/birdnet_outputs/birdnet_embeddings.csv",

    "Student_BirdNET_resnet18":
        "/home/nvidia/test_run/gibbon_individual_outputs/student_birdnet_outputs/student_birdnet_embeddings.csv",

    "Perch_v2":
        "/home/nvidia/test_run/gibbon_individual_outputs/perch_outputs/perch_embeddings.csv",

    "BEATs":
        "/home/nvidia/test_run/gibbon_individual_outputs/beats_outputs/beats_embeddings.csv",

    "Masked_autoencoder_full":
        "/home/nvidia/test_run/gibbon_individual_outputs/masked_autoencoder_full_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_danum":
        "/home/nvidia/test_run/gibbon_individual_outputs/masked_autoencoder_danum_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_tropical_weighted":
        "/home/nvidia/test_run/gibbon_individual_outputs/masked_autoencoder_tropical_weighted_outputs/masked_autoencoder_embeddings.csv",

    "Supervised_ecology":
        "/home/nvidia/test_run/gibbon_individual_outputs/supervised_ecology_outputs/supervised_ecology_embeddings.csv",

    "Supervised_ecology_rainfall":
        "/home/nvidia/test_run/gibbon_individual_outputs/supervised_ecology_rainfall_outputs/supervised_ecology_rainfall_embeddings.csv",

    "BigVGAN":
        "/home/nvidia/test_run/gibbon_individual_outputs/bigvgan_outputs/bigvgan_embeddings.csv",
}
min_clips_per_individual = 5

out_root = Path("/home/nvidia/test_run/gibbon_individual_outputs")


results = []




for model_name, embedding_csv in models.items():

    print(f"\nProcessing: {model_name}")

    if not Path(embedding_csv).exists():
        print(f"Skipping {model_name}: file not found")
        continue

    df = pd.read_csv(embedding_csv)

    df["IndividualID"] = (
        df["ClipID"]
        .str.extract(r"^([A-Za-z]+_\d{2})")
    )

    feature_cols = [
        c for c in df.columns
        if c.startswith("emb_")
    ]

    if len(feature_cols) == 0:
        print(f"Skipping {model_name}: no embedding columns")
        continue

    dat = df.dropna(subset=["IndividualID"]).copy()

    counts = dat["IndividualID"].value_counts()
    keep_ids = counts[counts >= min_clips_per_individual].index

    dat = dat[dat["IndividualID"].isin(keep_ids)].copy()

    print(f"Keeping individuals with >= {min_clips_per_individual} clips")
    print("N clips:", len(dat))
    print("N individuals:", dat["IndividualID"].nunique())

    if dat["IndividualID"].nunique() < 2:
        print(f"Skipping {model_name}: fewer than 2 individuals")
        continue

    X = dat[feature_cols].to_numpy()

    y = (
        dat["IndividualID"]
        .astype(str)
        .to_numpy(dtype=str)
    )

    X = StandardScaler().fit_transform(X)

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
        "MinClipsPerIndividual": min_clips_per_individual,
    }

    results.append(result)

    print("\nLinear probe: IndividualID")
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