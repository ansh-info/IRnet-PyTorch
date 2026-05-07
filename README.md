# IRnet Reproduce

PyTorch + PyG reproduction of [IRnet](https://www.sciencedirect.com/science/article/pii/S2090123224003205) -- Immunotherapy Response Prediction using Pathway Knowledge-Informed Graph Neural Network.

## Problem

Immune checkpoint inhibitors (ICIs) like anti-PD1, anti-PD-L1, and anti-CTLA4 have revolutionized cancer treatment, but only 20-40% of patients respond to therapy. Identifying responders before treatment would spare non-responders from costly side effects and allow clinicians to select alternative therapies earlier.

Current approaches to predicting ICI response from gene expression face two challenges:

1. **Small sample sizes** -- clinical trial cohorts typically have 50-200 patients, making deep learning prone to overfitting
2. **Black-box predictions** -- clinicians need interpretable models that explain *why* a patient is predicted to respond, not just a probability score

## Approach

IRnet addresses both challenges by incorporating **biological pathway knowledge** as an inductive bias:

- Instead of treating 8,080 genes as independent features, it maps them onto **344 KEGG pathways** using known gene-pathway membership
- This reduces trainable parameters by **98.8%** (33K vs 2.7M for a dense layer), preventing overfitting on small cohorts
- A **Graph Attention Network** then processes pathway interactions, learning which pathway cross-talk patterns predict response
- **Pathway-level importance scores** provide biological interpretability -- clinicians can see which signaling pathways drive each prediction

The model uses a 2-phase training strategy: pre-train on large TCGA survival data (1,162 patients), then fine-tune on small clinical ICI cohorts with transfer learning.

## Original Paper & Data

- **Original repository:** [github.com/yuexujiang/IRnet](https://github.com/yuexujiang/IRnet)
- **Paper:** Jiang Y, et al. "IRnet: Immunotherapy Response Prediction Using Pathway Knowledge-Informed Graph Neural Network." (2024)
- **Training data source:** TCGA (via original repo) + clinical ICI cohorts (Gide, Liu, IMvigor210, Kim, Riaz, Auslander)

All training data in this repo was obtained from the [original IRnet repository](https://github.com/yuexujiang/IRnet).

## Overview

Predicts whether a cancer patient will respond to immune checkpoint inhibitor (ICI) therapy by encoding gene expression data onto a KEGG pathway interaction graph and processing it with Graph Attention Networks.

### Architecture

```
Gene Expression (8080 genes)
    |
    v PathwayMappingLayer (biologically-masked: only 33K of 2.7M weights are trainable)
    |
    v 344 KEGG Pathway Nodes
    |
    v GATConv (4 heads, 4 channels) -- message passing over pathway graph
    |
    v GATConv (1 head, 4 channels) -- returns attention for interpretability
    |
    v GlobalAttention Pooling -- 344 nodes -> 1 graph embedding
    |
    v Dense(8) -> Dense(2) -- Responder vs Non-responder
```

### Key Innovation

The `PathwayMappingLayer` constrains the first layer using KEGG pathway membership. A gene can only influence a pathway if it biologically belongs to that pathway. This reduces parameters by 98.8% and prevents overfitting on small clinical datasets.

## Setup

```bash
# Requires Python 3.12+
uv sync

# Or with pip
pip install torch torch-geometric numpy pandas scipy scikit-learn
```

## Usage

### Full Reproduction Pipeline

```bash
# Step 1: Verify setup
uv run python run.py info

# Step 2: Phase 1 -- Pre-train on TCGA data (SKCM + BLCA + STAD, 1162 patients)
uv run python run.py --data-dir data train --epochs 400 --output checkpoints/pretrain

# Step 3: Process clinical cohorts into npz format
uv run python scripts/prepare_clinical.py --data-dir data --all --output-dir data/clinical/

# Step 4: Phase 2 -- Fine-tune on clinical cohort (transfer learning)
uv run python run.py --data-dir data train --npz data/clinical/clinical_Gide.npz \
    --pretrained checkpoints/pretrain/fold_0_best.pt \
    --output checkpoints/finetune_gide

# Step 5: Predict on new patients
uv run python run.py --data-dir data predict --input data/example_expression.txt \
    --checkpoint-dir checkpoints/finetune_gide --output results/

# Step 6: Generate pathway-level explanations
uv run python run.py --data-dir data explain --input data/example_expression.txt \
    --checkpoint-dir checkpoints/finetune_gide --output results/
```

### Individual Commands

```bash
# Show model info and available data
uv run python run.py info

# Train on specific cohort
uv run python run.py --data-dir data train --npz path/to/cohort.npz --output checkpoints/cohort

# Process a single clinical cohort
uv run python scripts/prepare_clinical.py --data-dir data --cohort Liu2019 --output data/clinical/clinical_Liu2019.npz

# Predict on new patients
uv run python run.py --data-dir data predict --input data/expression_matrix.txt \
    --checkpoint-dir checkpoints/finetune_gide --output results/

# Generate pathway-level explanations
uv run python run.py --data-dir data explain --input data/expression_matrix.txt \
    --checkpoint-dir checkpoints/finetune_gide --output results/
```

## Input Format

Expression matrix (TSV, genes in rows, patients in columns):

```
gene        Patient1    Patient2    Patient3
A1BG        0.0         150.3       733.78
A1CF        0.0         6.0         1.0
A2M         3407.0      11659.79    82844.09
```

## Output

**Predictions** (`prediction_results.txt`):
```
Patient_ID    P(Responder)    Prediction
Patient107    0.7234          Responder
Patient163    0.4512          Non-responder
```

**Pathway Importance** (`pathway_importance.csv`): Per-patient importance scores for all 344 KEGG pathways.

**Pathway Relations** (`pathway_relation/<patient>_pathway_relation.csv`): Per-patient 344x344 attention matrices showing pathway cross-talk from GAT2.

## Project Structure

```
.
├── run.py                  # CLI entry point
├── scripts/
│   ├── prepare_clinical.py # Convert raw clinical cohorts to npz format
│   └── generate_plots.py   # Generate presentation plots from results
├── docs/
│   └── architecture.svg    # Model architecture diagram
├── src/irnet/
│   ├── data.py             # Data loading, KEGG graph construction, preprocessing
│   ├── model.py            # PathwayMappingLayer + GAT + GlobalAttention + FocalLoss
│   └── train.py            # 5-fold CV training, early stopping, ensemble prediction
├── data/
│   ├── Kegg/               # KEGG pathway data (344 pathways, 3152 edges)
│   ├── genelist_8080.txt   # Reference gene list (fixed ordering)
│   ├── example_expression.txt  # Example input file (121 patients)
│   ├── training files/     # TCGA pre-training npz (SKCM, BLCA, STAD)
│   ├── Gide/               # Clinical cohort: Melanoma (PD1)
│   ├── Liu2019_Melanoma_RNAseq/  # Clinical cohort: Melanoma
│   ├── IMvigor210/         # Clinical cohort: Bladder
│   ├── Kim2018_PD1_Gastric_RNASeq/  # Clinical cohort: Gastric
│   ├── Riaz2017_PD1_Melanoma_RNASeq_Ipi.Naive/  # Clinical cohort: Melanoma
│   └── Auslander/          # Clinical cohort: Melanoma
└── pyproject.toml          # Dependencies (torch, torch-geometric, etc.)
```

## Training Strategy

1. **Pre-train on TCGA** (SKCM + BLCA + STAD, 1162 patients, 2-year survival as pseudo-label)
2. **Fine-tune on clinical ICI cohorts** (transfer learning from pre-trained weights)

Both phases use: 5-fold stratified CV, bootstrap oversampling (2x), focal loss (gamma=2), Adam (lr=1e-4).

## Results

### Phase 1: Pre-training on TCGA (1,162 patients)

| Fold | F1 | AUC | ACC | MCC |
|------|------|------|------|------|
| 0 | 0.620 | 0.752 | 0.743 | 0.426 |
| 1 | 0.554 | 0.699 | 0.717 | 0.349 |
| 2 | 0.652 | 0.828 | 0.733 | 0.452 |
| 3 | 0.667 | 0.774 | 0.741 | 0.473 |
| 4 | 0.534 | 0.691 | 0.677 | 0.287 |
| **Mean** | **0.605** | **0.749** | **0.722** | **0.397** |

### Phase 2: Fine-tuning on Gide (91 patients, transfer learning)

| Fold | F1 | AUC | ACC | MCC |
|------|------|------|------|------|
| 0 | 0.947 | 1.000 | 0.947 | 0.900 |
| 1 | 0.947 | 0.963 | 0.944 | 0.894 |
| 2 | 0.800 | 0.700 | 0.778 | 0.550 |
| 3 | 0.667 | 0.663 | 0.722 | 0.555 |
| 4 | 0.667 | 0.675 | 0.667 | 0.350 |
| **Mean** | **0.806** | **0.800** | **0.812** | **0.650** |

### Ensemble Evaluation (5-fold ensemble on Gide)

| Metric | Value |
|--------|-------|
| AUC | 0.993 |
| Average Precision | 0.994 |

## Data

### TCGA Pre-training Data (Phase 1)

Pre-processed npz files from the [original repo](https://github.com/yuexujiang/IRnet):

| Cohort | Patients | Responders | Non-responders | Pseudo-label |
|--------|----------|------------|----------------|--------------|
| TCGA-SKCM (Melanoma) | 435 | 251 | 184 | 2-year survival |
| TCGA-BLCA (Bladder) | 389 | 96 | 293 | 2-year survival |
| TCGA-STAD (Stomach) | 338 | 42 | 296 | 2-year survival |
| **Total** | **1,162** | **389** | **773** | |

### Clinical ICI Cohorts (Phase 2)

Raw RNA-seq counts + clinical response labels from the [original repo](https://github.com/yuexujiang/IRnet):

| Cohort | Cancer | Treatment | Files |
|--------|--------|-----------|-------|
| Gide | Melanoma | PD1 + CTLA4 | counts + clinical |
| Liu2019 | Melanoma | PD1 | RNA-seq + clinical |
| IMvigor210 | Bladder | PD-L1 | counts + clinical |
| Kim2018 | Gastric | PD1 | counts + clinical |
| Riaz2017 | Melanoma | PD1 | counts + clinical |
| Auslander | Melanoma | PD1 | counts + clinical |

## Differences from Original

| Aspect | Original | This Reproduction |
|--------|----------|-------------------|
| Framework | TensorFlow 2.5 + Spektral | PyTorch 2.11 + PyG 2.7 |
| Python | 3.7 | 3.12 |
| Package manager | Conda | uv |
| Focal loss | tensorflow-addons (deprecated) | Custom FocalLoss module |
| Graph format | Dense adjacency (344x344) | Sparse edge_index (2x3152) |
| Batching | Spektral MixedLoader | Manual edge_index replication |
| Code style | Notebook + monolithic script | Modular .py files with types |

## Citation

Original paper:
```
Jiang Y, et al. "IRnet: Immunotherapy Response Prediction Using Pathway
Knowledge-Informed Graph Neural Network." (2024)
```

## License

MIT
