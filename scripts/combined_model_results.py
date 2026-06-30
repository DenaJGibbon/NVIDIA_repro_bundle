import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

summary_files = {
    "Cambodia_Detection":
        "/home/nvidia/test_run/gibbon_detection_outputs/model_comparison_summary.csv",

    "Malaysia_Detection":
        "/home/nvidia/test_run/gibbon_detection_outputs_malaysia/model_comparison_summary.csv",

    "Ecological_Tasks":
        "/data/ssl_v1/SSL_full_experiments/model_comparison_summary.csv",

    "Grey_Gibbon_ID":
        "/home/nvidia/test_run/gibbon_individual_outputs/model_comparison_summary.csv",

    "Crested_Gibbon_ID":
        "/home/nvidia/test_run/gibbon_individual_outputs_cambodia/model_comparison_summary.csv",

    "Ambient_Soundscapes":
        "/data/ssl_v1/borneo_soundscape_outputs/model_comparison_summary.csv",

    "Ecological_Gradients":
        "/data/ssl_v1/safe_soundscape_outputs/safe_model_comparison_summary.csv",
}

results_root = Path("/home/nvidia/results")
results_root.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Model and benchmark label cleanup
# ------------------------------------------------------------

def clean_model_name(x):

    x = str(x).strip()

    model_name_map = {
        "BirdNET": "BirdNET",
        "Perch_v2": "Perch_v2",

        "Student_BirdNET": "Student_BirdNET",
        "Student_BirdNET_resnet18": "Student_BirdNET",

        "Supervised_ecology": "Eco_supervised",
        "Supervised_ecology_rainfall": "Eco_supervised_rainfall",
        "Supervised_ecologyrainfall": "Eco_supervised_rainfall",

        "SimCLR_local": "SimCLR_Cambodia",
        "SimCLR_full": "SimCLR_Cambodia",
        "SimCLR_danum": "SimCLR_Danum",
        "SimCLR_tropical_weighted_r18": "SimCLR_tropical_R18",
        "SimCLR_tropical_weighted_r18_epoch20": "SimCLR_tropical_R18",
        "SimCLR_tropical_weighted_r50": "SimCLR_tropical_R50",
        "SimCLR_tropical_weighted_r50_epoch19": "SimCLR_tropical_R50",

        "Masked_autoencoder": "MAE_Cambodia",
        "Masked_autoencoder_full": "MAE_Cambodia",
        "Masked_autoencoder_danum": "MAE_Danum",
        "Masked_autoencoder_tropical_weighted": "MAE_tropical",
        "Masked_autoencoder_tropical_weighted_epoch30": "MAE_tropical",

        "BEATs": "BEATs",
        "NeMo_base": "NeMo_base",
        "NeMo_adapted": "NeMo_adapted",
        "NeMo_adapted_full": "NeMo_adapted",
    }

    return model_name_map.get(x, x)


preferred_order = [
    "BirdNET",
    "Perch_v2",
    "Student_BirdNET",
    "Eco_supervised_rainfall",
    "Eco_supervised",
    "SimCLR_Cambodia",
    "SimCLR_Danum",
    "SimCLR_tropical_R18",
    "SimCLR_tropical_R50",
    "MAE_Cambodia",
    "MAE_Danum",
    "MAE_tropical",
    "BEATs",
    "NeMo_base",
    "NeMo_adapted",
]


# ------------------------------------------------------------
# Load all summaries
# ------------------------------------------------------------

all_results = []

for benchmark_name, csv_path in summary_files.items():

    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    if "Model" not in df.columns:
        print(f"Skipping {csv_path}: no Model column")
        continue

    if "BalancedAccuracyMean" not in df.columns:
        print(f"Skipping {csv_path}: no BalancedAccuracyMean column")
        continue

    df["Benchmark"] = benchmark_name

    if "Task" not in df.columns:
        df["Task"] = benchmark_name
    else:
        df["Task"] = df["Task"].fillna(benchmark_name)

    all_results.append(df)


if len(all_results) == 0:
    raise RuntimeError("No benchmark summary files were loaded.")


combined = pd.concat(
    all_results,
    ignore_index=True
)


# ------------------------------------------------------------
# Clean model labels
# ------------------------------------------------------------

combined["OriginalModel"] = (
    combined["Model"]
    .astype(str)
    .str.strip()
)

combined["CleanModel"] = (
    combined["OriginalModel"]
    .apply(clean_model_name)
)

combined["BalancedAccuracyMean"] = pd.to_numeric(
    combined["BalancedAccuracyMean"],
    errors="coerce"
).round(3)

if "BalancedAccuracySD" in combined.columns:
    combined["BalancedAccuracySD"] = pd.to_numeric(
        combined["BalancedAccuracySD"],
        errors="coerce"
    ).round(3)


print("\nModel name check:")
print(
    combined[["OriginalModel", "CleanModel"]]
    .drop_duplicates()
    .sort_values(["CleanModel", "OriginalModel"])
    .to_string(index=False)
)


# ------------------------------------------------------------
# Save combined long table
# ------------------------------------------------------------

combined_csv = (
    results_root /
    "all_model_benchmarks_combined.csv"
)

combined.to_csv(
    combined_csv,
    index=False
)

print("\nSaved combined table:")
print(combined_csv)


# ------------------------------------------------------------
# Collapse duplicate cleaned names before pivoting
# ------------------------------------------------------------

combined_collapsed = (
    combined
    .groupby(
        ["Benchmark", "Task", "CleanModel"],
        as_index=False
    )
    .agg(
        BalancedAccuracyMean=("BalancedAccuracyMean", "max"),
        BalancedAccuracySD=("BalancedAccuracySD", "mean")
        if "BalancedAccuracySD" in combined.columns
        else ("BalancedAccuracyMean", "count"),
    )
)

collapsed_csv = (
    results_root /
    "all_model_benchmarks_combined_collapsed.csv"
)

combined_collapsed.to_csv(
    collapsed_csv,
    index=False
)

print("\nSaved collapsed table:")
print(collapsed_csv)


# ------------------------------------------------------------
# Heatmap pivot
# ------------------------------------------------------------

heatmap_pivot = combined_collapsed.pivot_table(
    index=["Benchmark", "Task"],
    columns="CleanModel",
    values="BalancedAccuracyMean",
    aggfunc="max"
)

ordered_cols = [
    c for c in preferred_order
    if c in heatmap_pivot.columns
]

remaining_cols = [
    c for c in heatmap_pivot.columns
    if c not in ordered_cols
]

heatmap_pivot = (
    heatmap_pivot[
        ordered_cols + remaining_cols
    ]
    .round(3)
)

pivot_csv = (
    results_root /
    "all_model_benchmarks_pivot.csv"
)

heatmap_pivot.to_csv(pivot_csv)

print("\nSaved pivot table:")
print(pivot_csv)

print("\nFinal heatmap columns:")
print(list(heatmap_pivot.columns))


# ------------------------------------------------------------
# Heatmap plot
# ------------------------------------------------------------

plt.figure(
    figsize=(
        max(15, 0.85 * len(heatmap_pivot.columns)),
        max(7, 0.65 * len(heatmap_pivot))
    )
)

im = plt.imshow(
    heatmap_pivot,
    aspect="auto",
    cmap="viridis",
    vmin=0.5,
    vmax=1
)

cbar = plt.colorbar(im)
cbar.set_label("Balanced accuracy")

plt.xticks(
    range(len(heatmap_pivot.columns)),
    heatmap_pivot.columns,
    rotation=45,
    ha="right",
    fontsize=11
)

plt.yticks(
    range(len(heatmap_pivot.index)),
    [
        " / ".join(map(str, idx))
        if isinstance(idx, tuple)
        else str(idx)
        for idx in heatmap_pivot.index
    ],
    fontsize=11
)

plt.title(
    "Model performance across acoustic embedding benchmarks",
    fontsize=16,
    pad=15
)

plt.xlabel("Model", fontsize=13)
plt.ylabel("Benchmark / Task", fontsize=13)


# ------------------------------------------------------------
# Annotate cells and mark rounded-best models
# ------------------------------------------------------------

ax = plt.gca()

for i in range(heatmap_pivot.shape[0]):

    row_vals = heatmap_pivot.iloc[i, :]

    if row_vals.dropna().empty:
        continue

    # Compare based on exactly what is displayed in the cell
    displayed_vals = row_vals.apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else None
    )

    row_max = row_vals.max(skipna=True)
    displayed_best = f"{row_max:.2f}"

    for j in range(heatmap_pivot.shape[1]):

        val = heatmap_pivot.iloc[i, j]

        if pd.notna(val):

            label = f"{val:.2f}"
            is_best = label == displayed_best

            text_color = (
                "white" if val < 0.6 else "black"
            )
            #text_color = "darkgrey"

            if is_best:
                label = f"★ {label}"

            plt.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=10 if is_best else 9,
                fontweight="bold" if is_best else "normal",
                color=text_color
            )

            if is_best:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="purple",
                        linewidth=2.5,
                        zorder=10
                    )
                )

plt.tight_layout()

heatmap_path = (
    results_root /
    "all_model_benchmarks_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved heatmap:")
print(heatmap_path)