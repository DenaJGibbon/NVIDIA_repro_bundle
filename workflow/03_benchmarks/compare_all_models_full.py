import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

habitat_csv = "/home/nvidia/test_run/habitat_metadata.csv"

models = {

    "SimCLR_local":
        "/data/ssl_v1/SSL_full_experiments/simclr_outputs/simclr_embeddings.csv",

    "SimCLR_tropical_weighted_r18_epoch20":
        "/data/ssl_v1/SSL_combined_experiments/simclr_tropical_weighted_outputs/simclr_tropical_weighted_embeddings.csv",

    "SimCLR_tropical_weighted_r50_epoch19":
        "/data/ssl_v1/SSL_combined_experiments/simclr_tropical_weighted_resnet50_outputs/simclr_resnet50_tropical_weighted_embeddings.csv",

    "Masked_autoencoder":
        "/data/ssl_v1/SSL_full_experiments/masked_autoencoder_outputs/masked_autoencoder_embeddings.csv",

    "Masked_autoencoder_tropical_weighted_epoch30":
        "/data/ssl_v1/SSL_combined_experiments/masked_autoencoder_tropical_weighted_outputs/masked_autoencoder_tropical_weighted_embeddings.csv",

    "NeMo_base":
        "/data/ssl_v1/SSL_full_experiments/nemo_outputs/nemo_embeddings.csv",

    "NeMo_adapted":
        "/data/ssl_v1/SSL_full_experiments/nemo_adapt_outputs/nemo_adapt_embeddingsepoch30.csv",

    "BirdNET":
        "/data/ssl_v1/SSL_full_experiments/birdnet_outputs/birdnet_embeddings.csv",

    "Perch_v2":
        "/data/ssl_v1/SSL_full_experiments/perch_outputs/perch_embeddings.csv",

    "BEATs":
        "/data/ssl_v1/SSL_full_experiments/beats_outputs/beats_embeddings.csv",

    "Supervised_ecology":
        "/data/ssl_v1/SSL_full_experiments/supervised_ecology_outputs/supervised_ecology_embeddings.csv",

    "Supervised_ecology_rainfall":
        "/data/ssl_v1/SSL_full_experiments/supervised_ecology_rainfall_outputs/supervised_ecology_rainfall_embeddings.csv",
    
    "Student_BirdNET_resnet18":
    "/data/ssl_v1/SSL_combined_experiments/student_birdnet_resnet18_outputs/student_birdnet_embeddings.csv",

    "BigVGAN":
    "/data/ssl_v1/SSL_full_experiments/bigvgan_outputs/bigvgan_embeddings.csv",
}

out_root = Path("/data/ssl_v1/SSL_full_experiments")

out_csv = out_root / "model_comparison_summary.csv"


# ------------------------------------------------------------
# Remove models already completed in output CSV
# ------------------------------------------------------------

expected_task_names = [
    "HabitatType",
    "TimeOfDay",
    "Habitat_TimeOfDay",
]

if out_csv.exists():

    existing_summary = pd.read_csv(out_csv)

    completed = set(
        zip(
            existing_summary["Model"],
            existing_summary["Task"]
        )
    )

    results = existing_summary.to_dict("records")

    models_to_run = {}

    for model_name, embedding_csv in models.items():

        expected_tasks = {
            (model_name, task)
            for task in expected_task_names
        }

        if expected_tasks.issubset(completed):

            print(
                f"Already complete, skipping model: "
                f"{model_name}"
            )

        else:

            models_to_run[model_name] = embedding_csv

    models = models_to_run

    print("\nModels remaining to run:")

    print(list(models.keys()))

else:

    completed = set()
    results = []

    print(
        "\nNo existing summary found. "
        "Running all models."
    )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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

    if dat[label_col].nunique() < 2:
        print(f"Skipping {label_col}: fewer than 2 classes")
        return None

    class_counts = dat[label_col].value_counts()

    if class_counts.min() < 2:
        print(f"Skipping {label_col}: smallest class has <2 samples")
        return None

    n_splits = 3

    X = dat[feature_cols].to_numpy()

    y = dat[label_col].astype(str).to_numpy(dtype=str)

    X = StandardScaler().fit_transform(X)

    clf = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
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

    return {
        "BalancedAccuracyMean": scores.mean(),
        "BalancedAccuracySD": scores.std(),
        "ChanceApprox": 1 / len(pd.unique(y)),
        "N": len(y),
        "NClasses": len(pd.unique(y)),
        "NSplits": n_splits,
    }


# ------------------------------------------------------------
# Load habitat metadata
# ------------------------------------------------------------

habitat = pd.read_csv(habitat_csv)


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

for model_name, embedding_csv in models.items():

    print(f"\n================================================")
    print(f"Processing model: {model_name}")
    print(f"================================================")

    if not Path(embedding_csv).exists():

        print(f"Skipping {model_name}: file not found")

        continue

    print(f"Reading embeddings:")

    df = pd.read_csv(embedding_csv)

    print(f"Rows loaded: {len(df)}")

    # ------------------------------------------------------------
    # Metadata parsing
    # ------------------------------------------------------------

    df["Plot"] = (
        df["ClipID"]
        .str.extract(r"(WA-T\d{2})")
    )

    hour_raw = (
        df["ClipID"]
        .str.extract(r"_(\d{2})\d{4}\+")[0]
    )

    df["Hour"] = pd.to_numeric(
        hour_raw,
        errors="coerce"
    )

    df["TimeOfDay"] = df["Hour"].apply(
        lambda x: (
            classify_time(int(x))
            if pd.notna(x)
            else None
        )
    )

    df = df.merge(
        habitat,
        on="Plot",
        how="left"
    )

    df["Habitat_TimeOfDay"] = (
        df["HabitatType"].astype(str)
        + "_"
        + df["TimeOfDay"].astype(str)
    )

    df.loc[
        df["HabitatType"].isna() |
        df["TimeOfDay"].isna(),
        "Habitat_TimeOfDay"
    ] = None

    # ------------------------------------------------------------
    # Feature columns
    # ------------------------------------------------------------

    feature_cols = [
        c for c in df.columns
        if c.startswith("emb_")
    ]

    if len(feature_cols) == 0:

        print(
            f"Skipping {model_name}: no embedding columns"
        )

        continue

    print("Rows:", len(df))
    print("Features:", len(feature_cols))

    # ------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------

    for label_col in [
        "HabitatType",
        "TimeOfDay",
        "Habitat_TimeOfDay",
    ]:

        if (model_name, label_col) in completed:

            print(
                f"Skipping completed task: "
                f"{model_name} / {label_col}"
            )

            continue

        print(f"\nRunning probe: {label_col}")

        probe_result = run_probe(
            df,
            feature_cols,
            label_col
        )

        if probe_result is None:
            continue

        row = {
            "Model": model_name,
            "Task": label_col,
            "EmbeddingCSV": embedding_csv,
            "NFeatures": len(feature_cols),
        }

        row.update(probe_result)

        results.append(row)

        completed.add((model_name, label_col))

        print(row)

        # ------------------------------------------------------------
        # Incremental save
        # ------------------------------------------------------------

        summary = pd.DataFrame(results)

        summary = summary.sort_values(
            ["Task", "BalancedAccuracyMean"],
            ascending=[True, False]
        )

        summary.to_csv(
            out_csv,
            index=False
        )

        print("\nIncrementally saved:")
        print(out_csv)


# ------------------------------------------------------------
# Final save
# ------------------------------------------------------------

summary = pd.DataFrame(results)

summary = summary.sort_values(
    ["Task", "BalancedAccuracyMean"],
    ascending=[True, False]
)

summary.to_csv(
    out_csv,
    index=False
)

print("\nFinal model comparison summary:")
print(summary)

print("\nSaved:")
print(out_csv)