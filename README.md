# NVIDIA SSL Bioacoustics Workflow Reproducibility Bundle

This repository contains scripts, workflow logs, metadata, environment records, and lightweight reproducibility materials for the NVIDIA acoustic representation benchmarking project.

## Project overview

This project benchmarks acoustic embedding models for conservation bioacoustics across detection, individual identification, habitat, ecological-gradient, and soundscape classification tasks.

Models evaluated include BirdNET, Perch v2, Student BirdNET, SimCLR, masked autoencoders, supervised ecological encoders, BEATs, NeMo, and BigVGAN.

## Repository structure

- `workflow/00_cli_logs/`: command-line workflows used to run experiments
- `workflow/01_training/`: model training scripts
- `workflow/02_embedding_extraction/`: embedding extraction scripts
- `workflow/03_benchmarks/`: benchmark and linear-probe scripts
- `workflow/04_summary_figures/`: scripts for combined results and heatmaps
- `workflow/05_bigvgan/`: BigVGAN-related training/extraction code
- `workflow/metadata/`: metadata and configuration files
- `workflow/environment/`: package/environment records
- `scripts/`: original collected scripts
- `configs/`: metadata and model configuration files

## Key external assets

Large checkpoints, embeddings, and audio-derived outputs are not stored in GitHub. They are archived separately on the project backup drive.

Important external assets include:

- BigVGAN checkpoint `g_00004000`
- BEATs checkpoint
- SSL experiment output folders
- Combined/tropical experiment output folders
- Danum experiment output folders
- benchmark embedding CSV folders
- final results and figures

## Reproduction order

1. Recreate the Python environments using the files in `workflow/environment/`.
2. Prepare metadata from `workflow/metadata/`.
3. Run training workflows from `workflow/01_training/` or the CLI logs.
4. Extract embeddings using `workflow/02_embedding_extraction/`.
5. Run benchmark scripts in `workflow/03_benchmarks/`.
6. Generate combined tables and heatmaps using `workflow/04_summary_figures/`.

## Main benchmark tasks

- Cambodia gibbon detection
- Malaysia gibbon detection
- Grey gibbon individual identification
- Crested gibbon individual identification
- Habitat classification
- Time-of-day classification
- Habitat/time-of-day classification
- SAFE ecological gradient classification
- Borneo ambient soundscape classification

## Notes

This repository is intended to document and reproduce the workflow. Large binary files and raw/derived audio datasets are intentionally excluded from Git tracking.
