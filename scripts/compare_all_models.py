import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


habitat_csv = "/home/nvidia/test_run/habitat_metadata.csv"

models = {
    "SimCLR_local":
        "/home/nvidia/test_run/simclr_outputs/simclr_embeddings.csv",

    "NeMo_base":
        "/home/nvidia/test_run/nemo_embeddings.csv",

    "NeMo_adapted":
        "/home/nvidia/test_run/nemo_adapt_outputs_10ep/nemo_adapt_embeddings.csv",

    "BirdNET":
        "/home/nvidia/test_run/birdnet_outputs/birdnet_embeddings.csv",

    "Perch_v2":
        "/home/nvidia/test_run/perch_outputs/perch_embeddings.csv",

    "BEATs":
        "/home/nvidia/test_run/beats_outputs/beats_embeddings.csv",

    "Masked_autoencoder":
        "/home/nvidia/test_run/masked_autoencoder_outputs/masked_autoencoder_embeddings.csv",
}

out_csv = "/home/nvidia/test_run/model_comparison_summary.csv"


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

    return scores.mean(), scores.std(), 1 / len(pd.unique(y)), len(y)


habitat = pd.read_csv(habitat_csv)

results = []

for model_name, embedding_csv in models.items():

    print(f"\nProcessing: {model_name}")

    df = pd.read_csv(embedding_csv)

    df["Plot"] = df["ClipID"].str.extract(r"(WA-T\d{2})")

    df["Hour"] = (
        df["ClipID"]
        .str.extract(r"_(\d{2})\d{4}\+")[0]
        .astype(int)
    )

    df["TimeOfDay"] = df["Hour"].apply(classify_time)

    df = df.merge(habitat, on="Plot", how="left")

    feature_cols = [c for c in df.columns if c.startswith("emb_")]

    for label_col in ["HabitatType", "TimeOfDay"]:

        mean_acc, sd_acc, chance, n = run_probe(
            df,
            feature_cols,
            label_col
        )

        results.append({
            "Model": model_name,
            "Task": label_col,
            "BalancedAccuracyMean": mean_acc,
            "BalancedAccuracySD": sd_acc,
            "ChanceApprox": chance,
            "N": n,
            "NFeatures": len(feature_cols)
        })

summary = pd.DataFrame(results)

summary = summary.sort_values(
    ["Task", "BalancedAccuracyMean"],
    ascending=[True, False]
)

summary.to_csv(out_csv, index=False)

print("\nModel comparison summary:")
print(summary)

print("\nSaved:")
print(out_csv)
