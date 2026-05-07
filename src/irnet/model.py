"""
IRnet model architecture in PyTorch + PyG.

ARCHITECTURE OVERVIEW:
====================
    Patient Expression (8080 genes)
        |
        v Dropout
        |
        v PathwayMappingLayer (biologically-masked linear: 8080 -> 344)
        |   Only ~33K of 2.7M weights exist -- rest are permanently zero
        |
        v Each patient now has 344 pathway "activation" values
        |   These become NODE FEATURES on the pathway graph
        |
        v GATConv #1 (multi-head graph attention, 344 nodes x channels)
        |   Each pathway aggregates info from its neighboring pathways
        |
        v BatchNorm + GATConv #2 (returns attention weights for interpretability)
        |
        v BatchNorm + GlobalAttention pooling (344 nodes -> 1 graph embedding)
        |
        v Dense -> 2-class output (non-responder vs responder)

KEY DIFFERENCES FROM ORIGINAL TF VERSION:
=========================================
- SparseTF layer -> PathwayMappingLayer (same math, PyTorch idioms)
- Spektral GATConv -> PyG GATConv (edge_index instead of dense adjacency)
- Spektral GlobalAttentionPool -> PyG GlobalAttention
- tf focal loss (tfa) -> custom FocalLoss module
- Mixed batching -> PyG Batch (replicated graph per patient in batch)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation


# ---------------------------------------------------------------------------
# 1. PathwayMappingLayer (replaces SparseTF)
# ---------------------------------------------------------------------------
class PathwayMappingLayer(nn.Module):
    """
    Biologically-constrained linear layer: genes -> pathways.

    WHAT IT DOES:
    Maps 8080 gene expression values to 344 pathway activation scores,
    BUT only through connections that exist in biology (KEGG).

    HOW IT WORKS:
    - Normal linear layer: y = xW + b, where W is (8080, 344) = 2.7M params
    - This layer: same equation, but W is SPARSE -- only ~33K entries are
      non-zero (where gene actually belongs to pathway)
    - We store only the non-zero weights as a flat vector (saves memory)
    - Each forward pass: reconstruct the sparse matrix, then multiply

    WHY THIS MATTERS:
    Without this constraint, the model could learn nonsensical connections
    like "hemoglobin gene -> Wnt signaling pathway". The mask prevents this.
    It's a form of INDUCTIVE BIAS -- we're telling the model "only look at
    connections that make biological sense."

    PYTORCH IMPLEMENTATION:
    In TF they used tf.scatter_nd to place learned weights into a full matrix.
    In PyTorch we use sparse tensors for efficiency:
    - Store nonzero positions as a (2, nnz) index tensor
    - Store a nn.Parameter of shape (nnz,) for the learnable weights
    - Build a sparse COO tensor each forward pass, then matmul with input
    """

    def __init__(
        self,
        n_genes: int,
        n_pathways: int,
        mask: NDArray,
        activation: str = "elu",
        bias: bool = True,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_pathways = n_pathways

        # Find nonzero positions in the mask
        nonzero = mask.nonzero()  # returns (row_indices, col_indices) tuple
        import numpy as np
        nonzero_indices = torch.from_numpy(
            np.array([nonzero[0], nonzero[1]], dtype=np.int64)
        )  # shape (2, nnz)
        self.register_buffer("nonzero_indices", nonzero_indices)

        n_nonzero = nonzero_indices.shape[1]

        # Only these weights are trainable -- everything else stays zero
        self.kernel_vector = nn.Parameter(torch.empty(n_nonzero))
        nn.init.normal_(self.kernel_vector, std=0.05)

        if bias:
            self.bias = nn.Parameter(torch.zeros(n_pathways))
        else:
            self.bias = None

        # Activation function
        activations = {"elu": F.elu, "relu": F.relu, "tanh": torch.tanh}
        self.activation_fn = activations.get(activation, F.elu)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, n_genes) -- raw expression per patient
        returns: (batch_size, n_pathways) -- pathway activations
        """
        # Reconstruct the sparse weight matrix from learned values
        weight = torch.sparse_coo_tensor(
            self.nonzero_indices,
            self.kernel_vector,
            size=(self.n_genes, self.n_pathways),
            device=x.device,
        )

        # Matrix multiply: (batch, 8080) @ (8080, 344) -> (batch, 344)
        output = torch.sparse.mm(weight.t(), x.t()).t()

        if self.bias is not None:
            output = output + self.bias

        output = self.activation_fn(output)
        return output


# ---------------------------------------------------------------------------
# 2. IRnet Model
# ---------------------------------------------------------------------------
class IRnet(nn.Module):
    """
    Full IRnet model: expression -> graph neural network -> response prediction.

    BATCHING IN PyG:
    ================
    PyG doesn't use dense adjacency matrices. Instead, when we batch B patients:
    - We create ONE big graph with B * 344 nodes
    - Patient 0 owns nodes 0-343, patient 1 owns nodes 344-687, etc.
    - edge_index is replicated + offset for each patient
    - A "batch" vector tells GlobalAttention which nodes belong to which patient

    This is handled by our forward() method -- it takes a flat expression batch
    and constructs the batched graph internally.
    """

    def __init__(
        self,
        n_genes: int = 8080,
        n_pathways: int = 344,
        gene_pathway_mask: NDArray | None = None,
        edge_index: Tensor | None = None,
        # Hyperparameters (defaults match original paper's best config)
        x_dropout: float = 0.5,
        mapping_activation: str = "elu",
        gat1_channels: int = 4,
        gat1_heads: int = 4,
        gat1_dropout: float = 0.4,
        gat2_channels: int = 4,
        gat2_heads: int = 1,
        gat2_dropout: float = 0.4,
        pool_channels: int = 8,
        dense_channels: int = 8,
        n_classes: int = 2,
    ):
        super().__init__()

        # Store graph structure as buffers (not trainable, move with model)
        if edge_index is not None:
            self.register_buffer("edge_index", edge_index)
        else:
            self.register_buffer("edge_index", torch.zeros(2, 0, dtype=torch.long))

        self.n_pathways = n_pathways

        # --- Layer 1: Input dropout ---
        self.input_dropout = nn.Dropout(p=x_dropout)

        # --- Layer 2: Gene -> Pathway mapping (biologically constrained) ---
        if gene_pathway_mask is not None:
            self.pathway_mapping = PathwayMappingLayer(
                n_genes=n_genes,
                n_pathways=n_pathways,
                mask=gene_pathway_mask,
                activation=mapping_activation,
            )
        else:
            raise ValueError("gene_pathway_mask is required to build the model")

        # --- Layer 3: GAT #1 (multi-head graph attention) ---
        # Input: 1 feature per node (the pathway activation score)
        # Output: gat1_channels features per node (after averaging heads)
        # concat=False means we AVERAGE across heads (not concatenate)
        self.gat1 = GATConv(
            in_channels=1,
            out_channels=gat1_channels,
            heads=gat1_heads,
            concat=False,  # average heads -> output is (N, gat1_channels)
            dropout=gat1_dropout,
        )
        self.bn1 = nn.BatchNorm1d(gat1_channels, momentum=0.01)

        # --- Layer 4: GAT #2 (single head, returns attention for interpretability) ---
        self.gat2 = GATConv(
            in_channels=gat1_channels,
            out_channels=gat2_channels,
            heads=gat2_heads,
            concat=True,
            dropout=gat2_dropout,
        )
        self.gat2_out_channels = gat2_channels * gat2_heads
        self.bn2 = nn.BatchNorm1d(self.gat2_out_channels, momentum=0.01)

        # --- Layer 5: Global Attention Pooling ---
        # Collapses all 344 pathway nodes into one graph-level vector
        # HOW: learns a "gate" that scores each node's importance,
        # then takes a weighted sum of node features
        gate_nn = nn.Sequential(
            nn.Linear(self.gat2_out_channels, pool_channels),
            nn.Tanh(),
            nn.Linear(pool_channels, 1),
        )
        transform_nn = nn.Sequential(
            nn.Linear(self.gat2_out_channels, pool_channels),
        )
        self.global_pool = AttentionalAggregation(gate_nn=gate_nn, nn=transform_nn)

        # --- Layer 6: Classification head ---
        self.fc1 = nn.Linear(pool_channels, dense_channels)
        self.classifier = nn.Linear(dense_channels, n_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: (batch_size, n_genes) -- raw expression matrix
            edge_index: (2, E) -- override stored edge_index (optional)
            return_attention: if True, also return pathway embeddings and GAT attention

        Returns:
            logits: (batch_size, 2) -- raw scores (use CrossEntropyLoss, not softmax)
            If return_attention=True, also returns:
                pathway_embeddings: (batch_size, n_pathways, gat2_out)
                attention_weights: edge attention from GAT2
        """
        if edge_index is None:
            edge_index = self.edge_index

        batch_size = x.shape[0]

        # Step 1: Dropout on raw expression
        x = self.input_dropout(x)

        # Step 2: Map genes -> pathway activations
        # (batch_size, 8080) -> (batch_size, 344)
        x = self.pathway_mapping(x)

        # Step 3: Reshape for graph processing
        # Each patient's 344 values become node features on the pathway graph
        # (batch_size, 344) -> (batch_size * 344, 1)
        x = x.reshape(batch_size * self.n_pathways, 1)

        # Step 4: Build batched edge_index
        batched_edge_index = self._batch_edge_index(edge_index, batch_size)

        # Build batch vector: [0,0,...0, 1,1,...1, ..., B-1,B-1,...B-1]
        batch_vec = torch.arange(batch_size, device=x.device).repeat_interleave(
            self.n_pathways
        )

        # Step 5: GAT layer 1
        x = self.gat1(x, batched_edge_index)
        x = self.bn1(x)

        # Step 6: GAT layer 2 (optionally return attention)
        if return_attention:
            x, (edge_idx_att, attention_weights) = self.gat2(
                x, batched_edge_index, return_attention_weights=True
            )
        else:
            x = self.gat2(x, batched_edge_index)
            attention_weights = None

        x = self.bn2(x)

        # Save pathway embeddings before pooling (for interpretability)
        pathway_embeddings = (
            x.view(batch_size, self.n_pathways, -1) if return_attention else None
        )

        # Step 7: Global attention pooling (344 nodes -> 1 vector per patient)
        x = self.global_pool(x, batch_vec)  # (batch_size, pool_channels)

        # Step 8: Classification
        x = F.elu(self.fc1(x))
        logits = self.classifier(x)  # (batch_size, 2)

        if return_attention:
            return logits, pathway_embeddings, attention_weights
        return logits

    def _batch_edge_index(self, edge_index: Tensor, batch_size: int) -> Tensor:
        """
        Replicate edge_index for B patients, offsetting node indices.

        If edge_index connects nodes within a single 344-node graph,
        this creates B disconnected copies in one big graph.

        Example with 3 nodes, 2 edges, batch_size=2:
            Original: [[0,1], [1,2]]
            Batched:  [[0,1,3,4], [1,2,4,5]]  (second copy offset by 3)
        """
        if edge_index.shape[1] == 0:
            return edge_index

        # Repeat edge_index batch_size times
        repeated = edge_index.repeat(1, batch_size)  # (2, E * batch_size)

        # Create offsets: [0, 0, ..., n_pathways, n_pathways, ..., 2*n_pathways, ...]
        n_edges = edge_index.shape[1]
        offsets = (
            torch.arange(batch_size, device=edge_index.device) * self.n_pathways
        )
        offsets = offsets.repeat_interleave(n_edges)  # (E * batch_size,)

        repeated = repeated + offsets.unsqueeze(0)
        return repeated


# ---------------------------------------------------------------------------
# 3. Focal Loss (replaces tensorflow-addons)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    WHAT: A modified cross-entropy that down-weights "easy" examples and
    focuses training on "hard" ones.

    WHY: In ICI response prediction, ~70% are non-responders. Standard
    cross-entropy lets the model get lazy -- predicting "non-responder"
    for everyone gives 70% accuracy. Focal loss penalizes this by
    reducing the loss for confident correct predictions.

    HOW: FL(p) = -alpha * (1 - p)^gamma * log(p)
    - When gamma=0: standard cross-entropy
    - When gamma=2 (our default): well-classified examples (p close to 1)
      contribute almost nothing to loss. Misclassified examples (p close to 0)
      contribute normally.

    PAPER: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: (batch_size, n_classes) raw model output
            targets: (batch_size,) integer class labels
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = torch.where(
                targets == 1,
                torch.tensor(self.alpha, device=logits.device),
                torch.tensor(1 - self.alpha, device=logits.device),
            )
            focal_weight = focal_weight * alpha_weight

        return (focal_weight * ce_loss).mean()


# ---------------------------------------------------------------------------
# 4. Factory function
# ---------------------------------------------------------------------------
def build_model(
    gene_pathway_mask: NDArray,
    edge_index: Tensor,
    n_genes: int = 8080,
    n_pathways: int = 344,
    **kwargs,
) -> IRnet:
    """
    Convenience function to build IRnet with pathway graph data.

    Usage:
        graph = load_pathway_graph("path/to/data")
        model = build_model(
            gene_pathway_mask=graph.gene_pathway_mask,
            edge_index=graph.edge_index,
            n_genes=graph.n_genes,
            n_pathways=graph.n_pathways,
        )
    """
    return IRnet(
        n_genes=n_genes,
        n_pathways=n_pathways,
        gene_pathway_mask=gene_pathway_mask,
        edge_index=edge_index,
        **kwargs,
    )
