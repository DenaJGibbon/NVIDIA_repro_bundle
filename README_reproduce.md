# NVIDIA SSL Bioacoustics Workflow Reproducibility Bundle

Author: Dena Clink  
Project: NVIDIA Conservation Bioacoustics SSL Benchmarking  
Date: June 2026

## Overview

This bundle contains the code, configuration files, metadata, checkpoints, environment specifications, command-line workflows, and benchmark summaries used for the NVIDIA-funded evaluation of acoustic representations for conservation bioacoustics.

Primary embedding models evaluated:

- BirdNET
- Perch v2
- Student BirdNET
- SimCLR (Cambodia)
- SimCLR (Danum)
- SimCLR Tropical Weighted ResNet18
- SimCLR Tropical Weighted ResNet50
- Masked Autoencoder (Cambodia)
- Masked Autoencoder (Danum)
- Masked Autoencoder Tropical Weighted
- Supervised Ecology
- Supervised Ecology + Rainfall
- BEATs
- NeMo Base
- NeMo Adapted
- BigVGAN

## Workflow Structure

### 00_cli_logs
Command-line workflows used to execute the experiments.

### 01_training
Training scripts for SSL, ecological, and distillation models.

### 02_embedding_extraction
Embedding extraction pipelines for all evaluated models.

### 03_benchmarks
Benchmarking, probing, and evaluation scripts.

### 04_summary_figures
Scripts used to generate final summary tables and heatmaps.

### 05_bigvgan
BigVGAN training, inference, resynthesis, and feature extraction code.

## Metadata

Located in `workflow/metadata/`.

Key files:

- Dep01Dep02SoundscapeMetadata.csv
- satellite_embeddings_T01_T40_2025.csv
- habitat_metadata.csv
- BigVGAN configuration JSON files

## Environment Specifications

Located in `workflow/environment/`.

- requirements_ssl_soundscapes.txt
- requirements_birdnet_env.txt

## Checkpoints

### BigVGAN

Primary checkpoint used for benchmarking:

- g_00004000
- config.json

### BEATs

- BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt

## Major Benchmark Tasks

- Cambodia Automated Detection
- Malaysia Automated Detection
- Grey Gibbon Individual Identification
- Crested Gibbon Individual Identification
- Ecological Tasks
- SAFE Forest Gradient
- Ambient Soundscapes

## Recommended Reproduction Order

1. Install environments
2. Prepare metadata files
3. Train models
4. Extract embeddings
5. Run benchmark scripts
6. Generate summary tables
7. Generate combined figures and heatmaps

Workflow order:

CLI Logs → Training → Embedding Extraction → Benchmarks → Summary Figures

## Key Outputs

- model_comparison_summary.csv
- all_model_benchmarks_combined.csv
- all_model_benchmarks_combined_collapsed.csv
- all_model_benchmarks_pivot.csv
- all_model_benchmarks_heatmap.png

## Notes

The original project was executed on NVIDIA Brev infrastructure using A100 GPUs.

Training data were derived from multiple tropical passive acoustic monitoring projects including Cambodia, Danum Valley, SAFE, Maliau Basin, Tangkoko, and Sumatra.

The benchmark suite evaluates representation quality across species detection, individual identification, habitat classification, ecological gradients, and soundscape characterization.
