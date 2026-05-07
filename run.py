"""
IRnet CLI - Immunotherapy Response Prediction using Pathway-Informed GNN.

USAGE:
======
# Show model and data info
python run.py info

# Train on TCGA data (pre-training phase)
python run.py train --data-dir data --output checkpoints/pretrain

# Train on specific cohort npz file
python run.py train --data-dir data --npz "training files/cohort.npz" --output checkpoints/cohort

# Train with transfer learning (fine-tune)
python run.py train --data-dir data --npz "training files/cohort.npz" \
    --pretrained checkpoints/pretrain/fold_0_best.pt --output checkpoints/finetune

# Predict on new patients
python run.py predict --data-dir data --input example_expression.txt \
    --checkpoint-dir checkpoints/pretrain --output results/

# Explain predictions (pathway importance)
python run.py explain --data-dir data --input example_expression.txt \
    --checkpoint-dir checkpoints/pretrain --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from irnet.data import load_pathway_graph, process_expression
from irnet.model import build_model
from irnet.train import TrainConfig, predict_ensemble, train


# ---------------------------------------------------------------------------
# Load TCGA npz data (original format from the paper)
# ---------------------------------------------------------------------------
def load_npz_dataset(
    npz_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load preprocessed .npz dataset (TCGA or clinical cohort).

    The npz files contain:
        x: (n_samples, 8080) float16 -- z-score normalized expression
        y: (n_samples, 2) float16 -- one-hot labels [non-resp, resp]
        info: (n_samples,) -- patient IDs

    Returns: (expression, labels_int, patient_ids)
    """
    data = np.load(npz_path, allow_pickle=True)
    expression = data["x"].astype(np.float32)
    labels = data["y"].astype(np.float32).argmax(axis=1).astype(np.int64)
    patient_ids = data["info"]
    return expression, labels, patient_ids


# ---------------------------------------------------------------------------
# CLI: train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    """Train IRnet on TCGA or clinical data."""
    graph = load_pathway_graph(args.data_dir)

    if args.npz:
        print(f"Loading data from: {args.npz}")
        expression, labels, patient_ids = load_npz_dataset(args.npz)
    else:
        training_dir = Path(args.data_dir) / "training files"
        npz_files = sorted(training_dir.glob("immunotherapy_tcga_*.npz"))
        if not npz_files:
            print(f"ERROR: No TCGA npz files found in {training_dir}")
            sys.exit(1)

        print(f"Combining {len(npz_files)} TCGA cohorts for pre-training:")
        all_x, all_y, all_ids = [], [], []
        for npz_file in npz_files:
            x, y, ids = load_npz_dataset(npz_file)
            cohort_name = npz_file.stem.split("pathgraph_")[1].split("_")[0]
            print(f"  {cohort_name}: {len(y)} patients "
                  f"({(y == 1).sum()} resp, {(y == 0).sum()} non-resp)")
            all_x.append(x)
            all_y.append(y)
            all_ids.append(ids)

        expression = np.concatenate(all_x, axis=0)
        labels = np.concatenate(all_y, axis=0)
        patient_ids = np.concatenate(all_ids, axis=0)

    print(f"\nTotal dataset: {len(labels)} patients "
          f"({(labels == 1).sum()} resp, {(labels == 0).sum()} non-resp)")

    config = TrainConfig(
        n_folds=args.n_folds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=args.output,
        device=args.device,
    )

    result = train(
        expression=expression,
        labels=labels,
        pathway_graph=graph,
        config=config,
        pretrained_weights=args.pretrained,
    )

    print(f"\nCheckpoints saved to: {args.output}")
    print(f"Use for prediction: python run.py predict --checkpoint-dir {args.output} --input <file>")


# ---------------------------------------------------------------------------
# CLI: predict
# ---------------------------------------------------------------------------
def cmd_predict(args: argparse.Namespace) -> None:
    """Predict ICI response for new patients."""
    graph = load_pathway_graph(args.data_dir)
    patient_data = process_expression(args.input, graph.gene_names)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("fold_*_best.pt"))
    if not ckpts:
        print(f"ERROR: No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    print(f"Loaded {len(ckpts)} fold checkpoints from {ckpt_dir}")
    print(f"Processing {len(patient_data.patient_ids)} patients...")

    probs, preds = predict_ensemble(
        expression=patient_data.expression,
        pathway_graph=graph,
        checkpoint_paths=[str(c) for c in ckpts],
        device=args.device,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "prediction_results.txt"

    with open(results_file, "w") as f:
        f.write("Patient_ID\tP(Responder)\tPrediction\n")
        for i, pid in enumerate(patient_data.patient_ids):
            label = "Responder" if preds[i] else "Non-responder"
            f.write(f"{pid}\t{probs[i, 1]:.4f}\t{label}\n")

    print(f"\nResults written to: {results_file}")
    print(f"  Predicted responders: {preds.sum()}")
    print(f"  Predicted non-responders: {(~preds).sum()}")

    print("\nTop 5 most likely responders:")
    sorted_idx = np.argsort(probs[:, 1])[::-1]
    for i in sorted_idx[:5]:
        print(f"  {patient_data.patient_ids[i]}: P(resp)={probs[i, 1]:.4f}")


# ---------------------------------------------------------------------------
# CLI: explain
# ---------------------------------------------------------------------------
def cmd_explain(args: argparse.Namespace) -> None:
    """
    Generate pathway-level explanations for predictions.

    Produces:
    1. pathway_importance.csv -- which pathways drive each patient's prediction
    2. pathway_relation/ -- per-patient 344x344 attention matrices (pathway cross-talk)
    3. prediction_results.txt -- predictions alongside explanations
    """
    import pandas as pd
    from scipy.special import softmax

    graph = load_pathway_graph(args.data_dir)
    patient_data = process_expression(args.input, graph.gene_names)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("fold_*_best.pt"))
    if not ckpts:
        print(f"ERROR: No checkpoints found in {ckpt_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating explanations for {len(patient_data.patient_ids)} patients...")
    print(f"Using {len(ckpts)} fold checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    x_tensor = torch.tensor(patient_data.expression, dtype=torch.float32).to(device)

    n_patients = len(patient_data.patient_ids)
    n_pathways = graph.n_pathways
    n_edges = graph.edge_index.shape[1]

    all_embeddings = []
    all_probs = []
    all_attn = []

    for ckpt_path in ckpts:
        model = build_model(
            gene_pathway_mask=graph.gene_pathway_mask,
            edge_index=graph.edge_index,
            n_genes=graph.n_genes,
            n_pathways=graph.n_pathways,
        )
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            logits, pathway_emb, attn_weights = model(
                x_tensor, return_attention=True
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_embeddings.append(pathway_emb.cpu().numpy())

            attn_np = attn_weights.cpu().numpy().squeeze()
            all_attn.append(attn_np)

    avg_probs = np.mean(all_probs, axis=0)
    avg_embeddings = np.mean(all_embeddings, axis=0)
    avg_attn = np.mean(all_attn, axis=0)

    # Pathway importance = L2 norm of embedding, then softmax normalize
    pathway_importance = np.linalg.norm(avg_embeddings, axis=2)
    pathway_importance_norm = softmax(pathway_importance, axis=1)

    # Write pathway importance CSV
    importance_df = pd.DataFrame(
        pathway_importance_norm,
        index=patient_data.patient_ids,
        columns=graph.pathway_names,
    )
    importance_file = output_dir / "pathway_importance.csv"
    importance_df.T.to_csv(importance_file)
    print(f"\nPathway importance saved to: {importance_file}")

    # Write pathway relation matrices (344x344 attention per patient)
    edge_index_np = graph.edge_index.numpy()
    relation_dir = output_dir / "pathway_relation"
    relation_dir.mkdir(parents=True, exist_ok=True)

    for i, pid in enumerate(patient_data.patient_ids):
        patient_attn = avg_attn[i * n_edges:(i + 1) * n_edges]
        relation_matrix = np.zeros((n_pathways, n_pathways), dtype=np.float32)
        for edge_idx in range(n_edges):
            src = edge_index_np[0, edge_idx]
            dst = edge_index_np[1, edge_idx]
            relation_matrix[src, dst] = patient_attn[edge_idx]

        relation_df = pd.DataFrame(
            relation_matrix,
            index=graph.pathway_names,
            columns=graph.pathway_names,
        )
        relation_df.to_csv(relation_dir / f"{pid}_pathway_relation.csv")

    print(f"Pathway relations saved to: {relation_dir}/ ({n_patients} files)")

    # Write predictions
    preds = avg_probs[:, 1] > 0.5
    pred_file = output_dir / "prediction_results.txt"
    with open(pred_file, "w") as f:
        f.write("Patient_ID\tP(Responder)\tPrediction\n")
        for i, pid in enumerate(patient_data.patient_ids):
            label = "Responder" if preds[i] else "Non-responder"
            f.write(f"{pid}\t{avg_probs[i, 1]:.4f}\t{label}\n")

    # Top pathways summary
    mean_importance = pathway_importance_norm.mean(axis=0)
    top_idx = np.argsort(mean_importance)[::-1][:20]

    print("\nTop 20 most important pathways (averaged across patients):")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {graph.pathway_names[idx]}: {mean_importance[idx]:.4f}")

    print(f"\nPredictions saved to: {pred_file}")
    print(f"  Responders: {preds.sum()} / {len(preds)}")


# ---------------------------------------------------------------------------
# CLI: info
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> None:
    """Show information about the model and data."""
    graph = load_pathway_graph(args.data_dir)
    model = build_model(
        gene_pathway_mask=graph.gene_pathway_mask,
        edge_index=graph.edge_index,
        n_genes=graph.n_genes,
        n_pathways=graph.n_pathways,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== IRnet Model Info ===")
    print(f"  Genes: {graph.n_genes}")
    print(f"  Pathways: {graph.n_pathways}")
    print(f"  Pathway edges: {graph.n_edges}")
    print(f"  Gene-pathway connections: {int((graph.gene_pathway_mask > 0).sum())}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter reduction vs dense: "
          f"{1 - trainable_params / (graph.n_genes * graph.n_pathways):.1%}")
    print(f"\n  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Available training data
    training_dir = Path(args.data_dir) / "training files"
    npz_files = sorted(training_dir.glob("immunotherapy_tcga_*.npz"))
    if npz_files:
        print(f"\n=== Available Training Data ===")
        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            cohort = npz_file.stem.split("pathgraph_")[1].split("_")[0]
            n = data["x"].shape[0]
            resp = int((data["y"][:, 1] == 1).sum())
            print(f"  {cohort}: {n} patients ({resp} resp, {n - resp} non-resp)")

    # Available checkpoints
    for ckpt_dir in sorted(Path(args.data_dir).glob("checkpoints*")):
        ckpts = list(ckpt_dir.glob("fold_*_best.pt"))
        if ckpts:
            print(f"\n=== Checkpoints: {ckpt_dir.name} ===")
            print(f"  {len(ckpts)} fold(s) available")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="IRnet: Immunotherapy Response Prediction using "
                    "Pathway Knowledge-Informed Graph Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory with KEGG data (default: data)"
    )
    parser.add_argument(
        "--device", default="auto", help="Device: auto, cpu, or cuda"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- train ---
    train_p = subparsers.add_parser("train", help="Train the model")
    train_p.add_argument("--npz", help="Specific .npz file (default: combine all TCGA)")
    train_p.add_argument("--pretrained", help="Pretrained checkpoint for transfer learning")
    train_p.add_argument("--output", default="checkpoints", help="Output directory")
    train_p.add_argument("--n-folds", type=int, default=5)
    train_p.add_argument("--epochs", type=int, default=400)
    train_p.add_argument("--batch-size", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--patience", type=int, default=400)

    # --- predict ---
    pred_p = subparsers.add_parser("predict", help="Predict ICI response")
    pred_p.add_argument("--input", required=True, help="Expression matrix (TSV)")
    pred_p.add_argument("--checkpoint-dir", required=True, help="Model checkpoints")
    pred_p.add_argument("--output", default="results", help="Output directory")

    # --- explain ---
    exp_p = subparsers.add_parser("explain", help="Pathway-level explanations")
    exp_p.add_argument("--input", required=True, help="Expression matrix (TSV)")
    exp_p.add_argument("--checkpoint-dir", required=True, help="Model checkpoints")
    exp_p.add_argument("--output", default="results", help="Output directory")

    # --- info ---
    subparsers.add_parser("info", help="Show model and data info")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "explain": cmd_explain,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
