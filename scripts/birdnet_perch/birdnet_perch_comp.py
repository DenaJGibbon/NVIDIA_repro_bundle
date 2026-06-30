# ============================================================
# SAMPLE CLIPS + BIRDNET + PERCH + UMAP
# ============================================================

from pathlib import Path
import random

import birdnet
import numpy as np
import pandas as pd

import umap
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


# ============================================================
# SETTINGS
# ============================================================

DATA_ROOT = Path(
    "/data/ssl_v1/training_kswsspecies"
)

OUT_DIR = Path(
    "/data/ssl_v1/ksws_agile_species_search/species_embedding_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAX_PER_CLASS = 100
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ============================================================
# SAMPLE FILES
# ============================================================

rows = []

for species_dir in sorted(DATA_ROOT.iterdir()):

    if not species_dir.is_dir():
        continue

    files = []

    for ext in [
        "*.wav","*.WAV",
        "*.flac","*.FLAC",
        "*.mp3","*.MP3"
    ]:
        files.extend(
            species_dir.glob(ext)
        )

    files = sorted(files)

    if len(files) == 0:
        continue

    if len(files) > MAX_PER_CLASS:
        files = random.sample(
            files,
            MAX_PER_CLASS
        )

    print(
        f"{species_dir.name}: {len(files)}"
    )

    for f in files:

        rows.append({
            "Class": species_dir.name,
            "File": str(f)
        })

sample_df = pd.DataFrame(rows)

print(
    "\nTotal clips:",
    len(sample_df)
)

sample_df.to_csv(
    OUT_DIR / "sampled_clips.csv",
    index=False
)

# ============================================================
# FUNCTION
# ============================================================

def run_umap(
    embeddings,
    metadata,
    model_name
):

    emb_df = metadata.copy()

    for j in range(
        embeddings.shape[1]
    ):

        emb_df[
            f"emb_{j}"
        ] = embeddings[:, j]

    emb_df.to_csv(
        OUT_DIR /
        f"{model_name}_embeddings.csv",
        index=False
    )

    X = normalize(
        embeddings
    )

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    )

    coords = reducer.fit_transform(X)

    emb_df["UMAP1"] = coords[:,0]
    emb_df["UMAP2"] = coords[:,1]

    sil = silhouette_score(
        X,
        emb_df["Class"]
    )

    print(
        f"{model_name} silhouette:"
        f" {sil:.3f}"
    )

    plt.figure(
        figsize=(12,10)
    )

    for cls in sorted(
        emb_df["Class"].unique()
    ):

        sub = emb_df[
            emb_df["Class"] == cls
        ]

        plt.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=20,
            alpha=0.7,
            label=cls
        )

    plt.title(
        f"{model_name}\n"
        f"Silhouette={sil:.3f}"
    )

    plt.legend(
        bbox_to_anchor=(1.05,1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR /
        f"{model_name}_umap.png",
        dpi=300,
        bbox_inches="tight"
    )

    emb_df.to_csv(
        OUT_DIR /
        f"{model_name}_umap_coords.csv",
        index=False
    )

    plt.close()


# ============================================================
# BIRDNET
# ============================================================

print("\nLoading BirdNET")

birdnet_model = birdnet.load(
    "acoustic",
    "2.4",
    "tf"
)

birdnet_result = birdnet_model.encode(
    sample_df["File"].tolist(),
    batch_size=128,
    n_workers=16,
    show_stats="progress"
)

birdnet_embeddings = np.asarray(
    birdnet_result.embeddings
)

if birdnet_embeddings.ndim == 3:

    birdnet_embeddings = (
        birdnet_embeddings.mean(
            axis=1
        )
    )

print(
    "BirdNET shape:",
    birdnet_embeddings.shape
)

run_umap(
    birdnet_embeddings,
    sample_df,
    "birdnet"
)

# ============================================================
# PERCH
# ============================================================

print("\nLoading Perch")

perch_model = birdnet.load_perch_v2(
    device="CPU"
)

perch_result = perch_model.encode(
    sample_df["File"].tolist(),
    batch_size=16,
    n_workers=16,
    n_producers=1,
    device="CPU",
    show_stats="progress"
)

if hasattr(perch_result, "embeddings"):
    perch_embeddings = perch_result.embeddings
elif hasattr(perch_result, "embedding"):
    perch_embeddings = perch_result.embedding
elif hasattr(perch_result, "data"):
    perch_embeddings = perch_result.data
else:
    raise ValueError(
        "Could not find Perch embeddings."
    )

perch_embeddings = np.asarray(
    perch_embeddings
)

if perch_embeddings.ndim == 3:

    perch_embeddings = (
        perch_embeddings.mean(
            axis=1
        )
    )

print(
    "Perch shape:",
    perch_embeddings.shape
)

run_umap(
    perch_embeddings,
    sample_df,
    "perch"
)

print("\nDone.")
