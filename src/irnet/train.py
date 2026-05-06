"""
Training pipeline for IRnet (PyTorch version).

TRAINING STRATEGY (from the paper):
====================================
1. Pre-train on TCGA data (SKCM + BLCA + STAD) using survival as pseudo-label
2. Fine-tune on real clinical ICI cohort (Liu, IMvigor, Gide, etc.)

Both phases use:
- 5-fold stratified cross-validation
- Bootstrap oversampling (minority class upsampled to 2x majority)
- Focal loss (gamma=2) to handle remaining imbalance
- Adam optimizer (lr=1e-4)
- Early stopping on validation F1

THIS SCRIPT SUPPORTS BOTH PHASES:
- Phase 1: call train() with TCGA data, save weights
- Phase 2: call train() with clinical data + pretrained_weights path

PyTorch TRAINING LOOP EXPLAINED:
=================================
Unlike TF where model.fit() hides everything, in PyTorch we write the loop:

    for epoch in range(n_epochs):
        model.train()                    # enable dropout + batchnorm training mode
        for batch_x, batch_y in loader:
            optimizer.zero_grad()        # clear old gradients
            logits = model(batch_x)      # forward pass
            loss = criterion(logits, y)  # compute loss
            loss.backward()              # backpropagate (compute gradients)
            optimizer.step()             # update weights

        model.eval()                     # disable dropout, use running stats for BN
        with torch.no_grad():            # no gradient computation (saves memory)
            val_logits = model(val_x)    # validation forward pass
            # compute metrics...

This is more verbose than TF but gives full control over the training process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .data import PathwayGraph, bootstrap_balance
from .model import FocalLoss, build_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """
    All hyperparameters in one place.

    WHY A DATACLASS:
    Instead of passing 20 arguments everywhere, we group them here.
    Frozen=False so you can tweak individual params before training.
    """

    # Cross-validation
    n_folds: int = 5
    random_seed: int = 996

    # Training
    n_epochs: int = 400
    batch_size: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Bootstrap oversampling
    bootstrap_fold: int = 2

    # Focal loss
    focal_gamma: float = 2.0
    focal_alpha: float | None = None

    # Model hyperparameters
    x_dropout: float = 0.5
    gat1_channels: int = 4
    gat1_heads: int = 4
    gat1_dropout: float = 0.4
    gat2_channels: int = 4
    gat2_heads: int = 1
    gat2_dropout: float = 0.4
    pool_channels: int = 8
    dense_channels: int = 8

    # Early stopping
    patience: int = 50
    min_delta: float = 0.001

    # Output
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"

    # Regularization (L1 for mapping layer, L2 for GAT)
    mapping_l1: float = 2.5e-3
    gat_l2: float = 2.5e-3


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------
@dataclass
class EpochMetrics:
    """Stores metrics for one epoch (train or validation)."""

    loss: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    mcc: float = 0.0


# ---------------------------------------------------------------------------
# Training results
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Results from one fold of cross-validation."""

    fold: int
    best_epoch: int
    best_val_f1: float
    best_val_auc: float
    best_val_acc: float
    best_val_mcc: float
    checkpoint_path: str


@dataclass
class TrainResult:
    """Results from full cross-validation training."""

    fold_results: list[FoldResult] = field(default_factory=list)

    @property
    def mean_val_f1(self) -> float:
        return float(np.mean([r.best_val_f1 for r in self.fold_results]))

    @property
    def mean_val_auc(self) -> float:
        return float(np.mean([r.best_val_auc for r in self.fold_results]))

    def summary(self) -> str:
        lines = ["=== Training Results ==="]
        for r in self.fold_results:
            lines.append(
                f"  Fold {r.fold}: F1={r.best_val_f1:.4f} AUC={r.best_val_auc:.4f} "
                f"ACC={r.best_val_acc:.4f} MCC={r.best_val_mcc:.4f} (epoch {r.best_epoch})"
            )
        lines.append(f"  Mean F1: {self.mean_val_f1:.4f}")
        lines.append(f"  Mean AUC: {self.mean_val_auc:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------
def train(
    expression: NDArray,
    labels: NDArray,
    pathway_graph: PathwayGraph,
    config: TrainConfig | None = None,
    pretrained_weights: str | None = None,
) -> TrainResult:
    """
    Train IRnet with stratified k-fold cross-validation.

    Args:
        expression: (n_patients, n_genes) float32 z-score normalized
        labels: (n_patients,) int64 labels (0=non-resp, 1=resp)
        pathway_graph: PathwayGraph from load_pathway_graph()
        config: training hyperparameters (uses defaults if None)
        pretrained_weights: path to pretrained checkpoint for transfer learning

    Returns:
        TrainResult with per-fold metrics and checkpoint paths

    WHAT HAPPENS:
    1. Split data into 5 stratified folds
    2. For each fold:
       a. Split into train/val
       b. Bootstrap-balance training set
       c. Build fresh model (optionally load pretrained weights)
       d. Train with focal loss + Adam
       e. Track best validation F1, save checkpoint
    3. Return results across all folds
    """
    if config is None:
        config = TrainConfig()

    device = _get_device(config.device)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result = TrainResult()

    # Stratified K-Fold ensures each fold has similar class ratios
    skf = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_seed,
    )

    print(f"Training on device: {device}")
    print(f"Dataset: {len(labels)} patients ({(labels == 1).sum()} responders, "
          f"{(labels == 0).sum()} non-responders)")
    print(f"Config: {config.n_folds}-fold CV, {config.n_epochs} epochs, "
          f"lr={config.learning_rate}, batch={config.batch_size}")
    print("-" * 60)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(expression, labels)):
        print(f"\n--- Fold {fold_idx} ---")

        # Split data
        train_x, train_y = expression[train_idx], labels[train_idx]
        val_x, val_y = expression[val_idx], labels[val_idx]

        print(f"  Train: {len(train_y)} ({(train_y == 1).sum()} resp)")
        print(f"  Val:   {len(val_y)} ({(val_y == 1).sum()} resp)")

        # Bootstrap balance training set
        train_x, train_y = bootstrap_balance(
            train_x, train_y,
            fold=config.bootstrap_fold,
            seed=config.random_seed + fold_idx,
        )
        print(f"  After bootstrap: {len(train_y)} ({(train_y == 1).sum()} resp)")

        # Build model
        model = build_model(
            gene_pathway_mask=pathway_graph.gene_pathway_mask,
            edge_index=pathway_graph.edge_index,
            n_genes=pathway_graph.n_genes,
            n_pathways=pathway_graph.n_pathways,
            x_dropout=config.x_dropout,
            gat1_channels=config.gat1_channels,
            gat1_heads=config.gat1_heads,
            gat1_dropout=config.gat1_dropout,
            gat2_channels=config.gat2_channels,
            gat2_heads=config.gat2_heads,
            gat2_dropout=config.gat2_dropout,
            pool_channels=config.pool_channels,
            dense_channels=config.dense_channels,
        )

        # Load pretrained weights if available (transfer learning)
        if pretrained_weights is not None:
            state = torch.load(pretrained_weights, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"  Loaded pretrained weights from {pretrained_weights}")

        model = model.to(device)

        # Train this fold
        fold_result = _train_fold(
            model=model,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            config=config,
            device=device,
            fold_idx=fold_idx,
            checkpoint_dir=checkpoint_dir,
        )
        result.fold_results.append(fold_result)

        print(f"  Best: F1={fold_result.best_val_f1:.4f} AUC={fold_result.best_val_auc:.4f} "
              f"@ epoch {fold_result.best_epoch}")

    print("\n" + result.summary())
    return result


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------
def _train_fold(
    model: nn.Module,
    train_x: NDArray,
    train_y: NDArray,
    val_x: NDArray,
    val_y: NDArray,
    config: TrainConfig,
    device: torch.device,
    fold_idx: int,
    checkpoint_dir: Path,
) -> FoldResult:
    """Train a single fold and return the best metrics."""

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Validation data as full tensors (small enough to fit in memory)
    val_x_t = torch.tensor(val_x, dtype=torch.float32).to(device)
    val_y_t = torch.tensor(val_y, dtype=torch.long).to(device)

    # Optimizer and loss
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)

    # Tracking best validation metrics
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_mcc = 0.0
    best_epoch = 0
    patience_counter = 0

    checkpoint_path = str(checkpoint_dir / f"fold_{fold_idx}_best.pt")

    for epoch in range(config.n_epochs):
        # --- Training phase ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # Add L1 regularization on pathway mapping weights
            if config.mapping_l1 > 0:
                l1_loss = config.mapping_l1 * model.pathway_mapping.kernel_vector.abs().mean()
                loss = loss + l1_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation phase ---
        val_metrics = _evaluate(model, val_x_t, val_y_t, criterion)

        # Check for improvement
        if val_metrics.f1 > best_val_f1 + config.min_delta:
            best_val_f1 = val_metrics.f1
            best_val_auc = val_metrics.auc
            best_val_acc = val_metrics.accuracy
            best_val_mcc = val_metrics.mcc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f} "
                f"val_f1={val_metrics.f1:.4f} val_auc={val_metrics.auc:.4f} "
                f"val_acc={val_metrics.accuracy:.4f} "
                f"{'*' if patience_counter == 0 else ''}"
            )

        # Early stopping
        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {config.patience} epochs)")
            break

    return FoldResult(
        fold=fold_idx,
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        best_val_auc=best_val_auc,
        best_val_acc=best_val_acc,
        best_val_mcc=best_val_mcc,
        checkpoint_path=checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def _evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
) -> EpochMetrics:
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_np = y.cpu().numpy()
        preds_np = preds.cpu().numpy()
        probs_np = probs[:, 1].cpu().numpy()

    # Handle edge case where only one class in validation set
    try:
        auc = roc_auc_score(y_np, probs_np)
    except ValueError:
        auc = 0.5

    return EpochMetrics(
        loss=loss,
        accuracy=accuracy_score(y_np, preds_np),
        f1=f1_score(y_np, preds_np, zero_division=0),
        auc=auc,
        mcc=matthews_corrcoef(y_np, preds_np),
    )


# ---------------------------------------------------------------------------
# Ensemble inference (matches original's 5-fold ensemble)
# ---------------------------------------------------------------------------
def predict_ensemble(
    expression: NDArray,
    pathway_graph: PathwayGraph,
    checkpoint_paths: list[str],
    device: str = "auto",
    config: TrainConfig | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Run ensemble inference across multiple fold checkpoints.

    HOW ENSEMBLE WORKS:
    1. Load each fold's best checkpoint
    2. Run forward pass, get softmax probabilities
    3. Average probabilities across all folds
    4. Threshold at 0.5 for final prediction

    WHY ENSEMBLE:
    Each fold sees different train/val splits, so learns slightly different
    patterns. Averaging reduces variance and improves generalization.
    Typically gains 1-3% AUC over single model.

    Returns:
        probabilities: (n_patients, 2) averaged across folds
        predictions: (n_patients,) boolean -- True=responder
    """
    if config is None:
        config = TrainConfig()

    dev = _get_device(device)
    x_tensor = torch.tensor(expression, dtype=torch.float32).to(dev)

    all_probs = []

    for ckpt_path in checkpoint_paths:
        model = build_model(
            gene_pathway_mask=pathway_graph.gene_pathway_mask,
            edge_index=pathway_graph.edge_index,
            n_genes=pathway_graph.n_genes,
            n_pathways=pathway_graph.n_pathways,
            x_dropout=config.x_dropout,
            gat1_channels=config.gat1_channels,
            gat1_heads=config.gat1_heads,
            gat1_dropout=config.gat1_dropout,
            gat2_channels=config.gat2_channels,
            gat2_heads=config.gat2_heads,
            gat2_dropout=config.gat2_dropout,
            pool_channels=config.pool_channels,
            dense_channels=config.dense_channels,
        )
        state = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model.load_state_dict(state)
        model = model.to(dev)
        model.eval()

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    # Average across folds
    avg_probs = np.mean(all_probs, axis=0)
    predictions = avg_probs[:, 1] > 0.5

    return avg_probs, predictions


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
