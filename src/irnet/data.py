"""
Data loading and preprocessing for IRnet (PyTorch Geometric version).

WHAT THIS MODULE DOES:
=====================
Transforms raw gene expression data into graph-structured inputs for the GNN.

THE KEY IDEA:
============
Each patient's gene expression (8080 genes) gets mapped onto a graph of
344 KEGG pathways. The graph structure (which pathways connect to which)
is FIXED — it comes from biological knowledge. What changes per patient
is the expression values that flow through the graph.

DATA FLOW:
=========
  Raw expression (genes x patients TSV)
      |
      v
  Align to reference gene list (8080 genes, fixed order)
      |
      v
  Z-score normalize per patient (removes batch effects)
      |
      v
  Package into PyG Data objects with:
    - x: patient expression vector (used by SparseTF layer in model)
    - edge_index: pathway-pathway edges (from KEGG, shared across patients)
    - y: response label (0 or 1) if training

PyG vs Spektral difference:
- Spektral uses dense adjacency matrix (344x344)
- PyG uses edge_index in COO format: tensor of shape (2, num_edges)
  Row 0 = source nodes, Row 1 = target nodes
  This is more memory-efficient for sparse graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import stats


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatArray = NDArray[np.floating]


# ---------------------------------------------------------------------------
# Immutable data containers
# ---------------------------------------------------------------------------
class PathwayGraph(NamedTuple):
    """
    Holds the STATIC pathway graph structure (same for ALL patients).

    Think of this as the "skeleton" of our GNN — the biological prior knowledge:
    - gene_pathway_mask: tells the model which genes feed into which pathways
    - edge_index: tells the GAT layers which pathways can talk to each other
    """

    gene_pathway_mask: FloatArray  # (n_genes, n_pathways) binary, e.g. (8080, 344)
    edge_index: torch.Tensor  # (2, n_edges) long tensor — PyG COO format
    gene_names: list[str]  # ordered gene symbols
    pathway_names: list[str]  # ordered pathway IDs
    n_genes: int
    n_pathways: int
    n_edges: int


class PatientData(NamedTuple):
    """Holds processed patient expression + optional labels."""

    expression: FloatArray  # (n_patients, n_genes) z-score normalized float32
    patient_ids: NDArray  # (n_patients,) string array
    labels: FloatArray | None  # (n_patients,) int labels: 0 or 1, or None


# ---------------------------------------------------------------------------
# 1. Load the reference gene list
# ---------------------------------------------------------------------------
def load_gene_list(filepath: str | Path) -> list[str]:
    """
    Load the fixed 8080-gene reference list.

    WHY THIS EXISTS:
    The SparseTF (pathway mapping) layer has a weight for each gene-pathway
    pair where the gene belongs to that pathway. The weights are stored in a
    fixed order — gene index 0 is always "AKR1A1", index 1 is always "ADH1A",
    etc. If you shuffle the gene order, the model breaks silently.

    All patient expression data MUST be reordered to match this list.
    """
    filepath = Path(filepath)
    genes = [line.strip() for line in open(filepath) if line.strip()]
    return genes


# ---------------------------------------------------------------------------
# 2. Parse KEGG GMT file -> gene-pathway binary mask
# ---------------------------------------------------------------------------
def load_pathway_gene_mapping(
    gmt_filepath: str | Path,
    gene_list: list[str],
) -> tuple[FloatArray, list[str]]:
    """
    Build the gene-pathway membership mask from a GMT file.

    GMT FORMAT (tab-separated):
        path:hsa00010   AKR1A1  ADH1A  ADH1B  ...
        path:hsa00020   CS      DLAT   DLD    ...

    Each line = one pathway, followed by all genes in that pathway.

    OUTPUT:
        mask[i, j] = 1.0 if gene_list[i] belongs to pathway j, else 0.0

    WHY THIS MATTERS:
    This mask is the biological prior knowledge that constrains our model.
    Instead of a fully-connected first layer (8080 x 344 = 2.7M weights),
    we only learn weights where mask=1 (~50K non-zero entries).
    This is a MASSIVE reduction that prevents overfitting and encodes
    that "gene X only influences pathway Y if it's actually in pathway Y."
    """
    gmt_filepath = Path(gmt_filepath)

    pathway_genes: dict[str, set[str]] = {}
    pathway_order: list[str] = []

    with open(gmt_filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            pathway_id = parts[0]
            genes_in_pathway = set(parts[1:])
            pathway_genes[pathway_id] = genes_in_pathway
            pathway_order.append(pathway_id)

    n_genes = len(gene_list)
    n_pathways = len(pathway_order)
    mask = np.zeros((n_genes, n_pathways), dtype=np.float32)

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    for pathway_idx, pathway_id in enumerate(pathway_order):
        for gene in pathway_genes[pathway_id]:
            gene_idx = gene_to_idx.get(gene)
            if gene_idx is not None:
                mask[gene_idx, pathway_idx] = 1.0

    return mask, pathway_order


# ---------------------------------------------------------------------------
# 3. Parse KEGG pathway network -> edge_index (PyG format)
# ---------------------------------------------------------------------------
def load_pathway_adjacency(
    net_filepath: str | Path,
    pathway_names: list[str],
) -> torch.Tensor:
    """
    Build edge_index from KEGG pathway interaction file.

    FILE FORMAT (tab-separated):
        path:hsa00010   path:hsa00020
        path:hsa00020   path:hsa00010
        ...

    PyG EDGE FORMAT:
    Instead of a dense 344x344 matrix, PyG uses edge_index: a (2, num_edges)
    tensor where:
        edge_index[0] = source node indices
        edge_index[1] = target node indices

    Example: if pathway 0 connects to pathway 5:
        edge_index = [[..., 0, 5, ...],
                      [..., 5, 0, ...]]  (undirected = both directions)

    This is called COO (Coordinate) format — standard in sparse matrix world.
    """
    net_filepath = Path(net_filepath)
    pathway_to_idx = {p: i for i, p in enumerate(pathway_names)}

    sources: list[int] = []
    targets: list[int] = []

    with open(net_filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            p1, p2 = parts[0], parts[1]
            if p1 == p2:
                continue
            idx1 = pathway_to_idx.get(p1)
            idx2 = pathway_to_idx.get(p2)
            if idx1 is not None and idx2 is not None:
                sources.append(idx1)
                targets.append(idx2)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return edge_index


# ---------------------------------------------------------------------------
# 4. Load full pathway graph (convenience function)
# ---------------------------------------------------------------------------
def load_pathway_graph(data_dir: str | Path) -> PathwayGraph:
    """
    Load all KEGG reference data and return the complete pathway graph.

    This is the entry point you'll use most — call once at startup,
    then reuse the PathwayGraph for all patients.
    """
    data_dir = Path(data_dir)

    gene_list = load_gene_list(data_dir / "genelist_8080.txt")
    mask, pathway_names = load_pathway_gene_mapping(
        data_dir / "Kegg" / "human_KeggPathwayGene.gmt",
        gene_list,
    )
    edge_index = load_pathway_adjacency(
        data_dir / "Kegg" / "human_KeggPathwayNet.txt",
        pathway_names,
    )

    return PathwayGraph(
        gene_pathway_mask=mask,
        edge_index=edge_index,
        gene_names=gene_list,
        pathway_names=pathway_names,
        n_genes=len(gene_list),
        n_pathways=len(pathway_names),
        n_edges=edge_index.shape[1],
    )


# ---------------------------------------------------------------------------
# 5. Process patient expression data (inference — no labels)
# ---------------------------------------------------------------------------
def process_expression(
    filepath: str | Path,
    gene_list: list[str],
) -> PatientData:
    """
    Load and preprocess a gene expression matrix.

    INPUT FORMAT (TSV):
        gene      Patient1    Patient2    Patient3
        A1BG      0.0         150.3       733.78
        A1CF      0.0         6.0         1.0
        A2M       3407.0      11659.79    82844.09

    PROCESSING STEPS:
    1. Read matrix — rows are genes, columns are patients
    2. Reorder genes to match the 8080-gene reference list
       - Genes not in reference: ignored
       - Reference genes not in data: filled with 0
    3. Transpose to (patients, genes) — each row is one patient
    4. Z-score normalize EACH PATIENT independently (across their genes)

    WHY Z-SCORE PER PATIENT:
    Different patients may have been sequenced at different depths or on
    different platforms. Z-scoring removes this "scale" effect so the model
    sees relative gene importance, not absolute counts.
    """
    import pandas as pd

    filepath = Path(filepath)

    df = pd.read_csv(filepath, sep="\t", index_col=0)
    patient_ids = np.array(df.columns.tolist())

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)
    n_patients = len(patient_ids)

    # Build aligned expression matrix
    expression = np.zeros((n_genes, n_patients), dtype=np.float64)
    for gene_symbol, row_values in df.iterrows():
        idx = gene_to_idx.get(str(gene_symbol))
        if idx is not None:
            expression[idx, :] = row_values.values

    # Transpose: (genes, patients) -> (patients, genes)
    expression = expression.T

    # Z-score per patient (axis=1 = across genes for each patient)
    expression = stats.zscore(expression, axis=1, nan_policy="omit")
    expression = np.nan_to_num(expression, nan=0.0)
    expression = expression.astype(np.float32)

    return PatientData(expression=expression, patient_ids=patient_ids, labels=None)


# ---------------------------------------------------------------------------
# 6. Process labeled training data
# ---------------------------------------------------------------------------
def process_training_data(
    expression_filepath: str | Path,
    labels_dict: dict[str, int],
    gene_list: list[str],
) -> PatientData:
    """
    Load expression + clinical labels for training.

    labels_dict maps patient_id -> 0 (non-responder) or 1 (responder).
    Only patients present in BOTH expression file AND labels_dict are kept.
    """
    patient_data = process_expression(expression_filepath, gene_list)

    keep_idx = []
    labels = []
    for i, pid in enumerate(patient_data.patient_ids):
        if pid in labels_dict:
            keep_idx.append(i)
            labels.append(labels_dict[pid])

    keep_idx_arr = np.array(keep_idx)
    return PatientData(
        expression=patient_data.expression[keep_idx_arr],
        patient_ids=patient_data.patient_ids[keep_idx_arr],
        labels=np.array(labels, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# 7. Bootstrap oversampling for class balance
# ---------------------------------------------------------------------------
def bootstrap_balance(
    expression: FloatArray,
    labels: NDArray,
    fold: int = 2,
    seed: int = 996,
) -> tuple[FloatArray, NDArray]:
    """
    Oversample minority class to balance the dataset.

    WHY: ICI response datasets are imbalanced (typically 30% responders,
    70% non-responders). Without balancing, the model learns to always
    predict "non-responder" and gets ~70% accuracy while being useless.

    HOW: We resample (with replacement) the minority class until both
    classes have `fold * majority_count` samples.

    Args:
        expression: (n_samples, n_features)
        labels: (n_samples,) integer labels 0 or 1
        fold: target = fold * max_class_count
        seed: for reproducibility
    """
    rng = np.random.default_rng(seed)

    class_0_mask = labels == 0
    class_1_mask = labels == 1

    x0, y0 = expression[class_0_mask], labels[class_0_mask]
    x1, y1 = expression[class_1_mask], labels[class_1_mask]

    max_count = max(len(y0), len(y1))
    target_count = fold * max_count

    def _resample(x: FloatArray, y: NDArray, target: int) -> tuple[FloatArray, NDArray]:
        n = len(y)
        if n >= target:
            return x[:target], y[:target]
        indices = rng.choice(n, size=target, replace=True)
        return x[indices], y[indices]

    x0_bal, y0_bal = _resample(x0, y0, target_count)
    x1_bal, y1_bal = _resample(x1, y1, target_count)

    out_x = np.concatenate([x0_bal, x1_bal], axis=0)
    out_y = np.concatenate([y0_bal, y1_bal], axis=0)

    perm = rng.permutation(len(out_y))
    return out_x[perm], out_y[perm]
