"""
Generate presentation plots from IRnet results.

Usage:
    uv run python scripts/generate_plots.py --results-dir results/finetune_gide --data-dir data --output plots/
    uv run python scripts/generate_plots.py --results-dir results/finetune_liu --data-dir data --output plots/

    # With ROC/PR evaluation
    uv run python scripts/generate_plots.py --results-dir results/finetune_gide --data-dir data \
        --checkpoint-dir checkpoints/finetune_gide --eval-npz data/clinical/clinical_Gide.npz --output plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
})


def plot_prediction_distribution(results_dir: Path, output_dir: Path) -> None:
    """Histogram of P(responder) scores."""
    pred_file = results_dir / "prediction_results.txt"
    if not pred_file.exists():
        print(f"  Skipping: {pred_file} not found")
        return

    df = pd.read_csv(pred_file, sep="\t")
    fig, ax = plt.subplots(figsize=(6, 4))

    resp = df[df["Prediction"] == "Responder"]["P(Responder)"]
    non_resp = df[df["Prediction"] == "Non-responder"]["P(Responder)"]

    ax.hist(resp, bins=20, alpha=0.7, color="#2563eb",
            label=f"Responder (n={len(resp)})", edgecolor="white", linewidth=0.5)
    ax.hist(non_resp, bins=20, alpha=0.7, color="#dc2626",
            label=f"Non-responder (n={len(non_resp)})", edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="#333", linestyle="--", linewidth=1, label="Threshold (0.5)")
    ax.set_xlabel("P(Responder)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Score Distribution")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "prediction_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  prediction_distribution.png")


def plot_top_pathways(results_dir: Path, output_dir: Path, top_n: int = 20) -> None:
    """Bar chart of top N most important pathways."""
    imp_file = results_dir / "pathway_importance.csv"
    if not imp_file.exists():
        print(f"  Skipping: {imp_file} not found")
        return

    imp_df = pd.read_csv(imp_file, index_col=0)
    mean_imp = imp_df.mean(axis=1).sort_values(ascending=False)
    top = mean_imp.head(top_n)

    names = [n.replace("path:", "") for n in top.index]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(top)), top.values, color="#2563eb", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Importance Score")
    ax.set_title(f"Top {top_n} Pathways by Importance")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "top_pathways.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  top_pathways.png")


def plot_pathway_heatmap(results_dir: Path, output_dir: Path, top_n: int = 30) -> None:
    """Heatmap of top pathways x patients."""
    imp_file = results_dir / "pathway_importance.csv"
    if not imp_file.exists():
        print(f"  Skipping: {imp_file} not found")
        return

    imp_df = pd.read_csv(imp_file, index_col=0)
    mean_imp = imp_df.mean(axis=1).sort_values(ascending=False)
    top_names = mean_imp.head(top_n).index
    heatmap_data = imp_df.loc[top_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels([n.replace("path:", "") for n in top_names], fontsize=7)
    ax.set_xlabel("Patients")
    ax.set_title(f"Pathway Importance Heatmap (top {top_n})")
    plt.colorbar(im, ax=ax, label="Importance Score", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "pathway_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  pathway_heatmap.png")


def plot_pathway_relation(results_dir: Path, output_dir: Path) -> None:
    """344x344 pathway cross-talk attention matrix for a sample patient."""
    relation_dir = results_dir / "pathway_relation"
    if not relation_dir.exists():
        print(f"  Skipping: {relation_dir} not found")
        return

    files = sorted(relation_dir.glob("*_pathway_relation.csv"))
    if not files:
        print("  Skipping: no pathway_relation files")
        return

    sample_file = files[0]
    relation_df = pd.read_csv(sample_file, index_col=0)
    patient_name = sample_file.stem.replace("_pathway_relation", "")

    fig, ax = plt.subplots(figsize=(8, 7))
    rel_matrix = relation_df.values

    im = ax.imshow(rel_matrix, cmap="Blues", interpolation="nearest", aspect="equal")
    ax.set_title(f"Pathway Cross-talk Attention ({patient_name})")
    ax.set_xlabel("Target Pathway")
    ax.set_ylabel("Source Pathway")
    plt.colorbar(im, ax=ax, label="Attention Weight", shrink=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_dir / "pathway_relation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  pathway_relation.png")


def plot_roc_pr(
    data_dir: Path,
    checkpoint_dir: Path,
    eval_npz: Path,
    output_dir: Path,
) -> None:
    """ROC and Precision-Recall curves from ensemble evaluation."""
    from irnet.data import load_pathway_graph
    from irnet.train import predict_ensemble

    if not eval_npz.exists():
        print(f"  Skipping: {eval_npz} not found")
        return

    ckpts = sorted(checkpoint_dir.glob("fold_*_best.pt"))
    if not ckpts:
        print(f"  Skipping: no checkpoints in {checkpoint_dir}")
        return

    graph = load_pathway_graph(str(data_dir))
    data = np.load(str(eval_npz), allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32).argmax(axis=1)

    probs, preds = predict_ensemble(x, graph, [str(c) for c in ckpts], device="cpu")

    fpr, tpr, _ = roc_curve(y, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, probs[:, 1])
    ap = average_precision_score(y, probs[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].plot(recall, precision, color="#059669", linewidth=2, label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  roc_pr_curve.png (AUC={roc_auc:.3f}, AP={ap:.3f})")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y, preds)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-resp", "Responder"])
    ax.set_yticklabels(["Non-resp", "Responder"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  confusion_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation plots from IRnet results")
    parser.add_argument("--results-dir", required=True, help="Results directory")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory for evaluation")
    parser.add_argument("--eval-npz", help="NPZ file to evaluate (for ROC/PR curves)")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots from: {results_dir}")
    print(f"Output: {output_dir}/\n")

    plot_prediction_distribution(results_dir, output_dir)
    plot_top_pathways(results_dir, output_dir)
    plot_pathway_heatmap(results_dir, output_dir)
    plot_pathway_relation(results_dir, output_dir)

    if args.checkpoint_dir and args.eval_npz:
        plot_roc_pr(
            data_dir=Path(args.data_dir),
            checkpoint_dir=Path(args.checkpoint_dir),
            eval_npz=Path(args.eval_npz),
            output_dir=output_dir,
        )

    print(f"\nDone! All plots in: {output_dir}/")


if __name__ == "__main__":
    main()
