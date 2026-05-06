"""
Prepare clinical cohort data for Phase 2 fine-tuning.

Converts raw RNA-seq counts + clinical response labels into npz format
compatible with run.py train.

Usage:
    uv run python scripts/prepare_clinical.py --data-dir data --cohort Gide --output data/clinical/clinical_Gide.npz
    uv run python scripts/prepare_clinical.py --data-dir data --all --output-dir data/clinical/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

COHORT_CONFIG = {
    "Gide": {
        "counts": "counts.txt",
        "clinical": "clinic.txt",
        "pt_col": "Pt",
        "response_col": "Response",
        "gene_col": "Geneid",
    },
    "Liu2019": {
        "dir": "Liu2019_Melanoma_RNAseq",
        "counts": "rnaseq_rawcounts.txt",
        "clinical": "clinic.txt",
        "pt_col": "Pt",
        "response_col": "Response",
        "gene_col": "gene",
    },
    "IMvigor210": {
        "counts": "gene_counts.txt",
        "clinical": "clinical.txt",
        "pt_col": "Pt",
        "response_col": "Response",
        "gene_col": None,
    },
    "Kim2018": {
        "dir": "Kim2018_PD1_Gastric_RNASeq",
        "counts": "counts.txt",
        "clinical": "clinic.txt",
        "pt_col": "Pt",
        "response_col": "Response",
        "gene_col": None,
        "strip_counts_suffix": "_RNA",
    },
    "Riaz2017": {
        "dir": "Riaz2017_PD1_Melanoma_RNASeq_Ipi.Naive",
        "counts": "geneName_counts.txt",
        "clinical": "clinical.txt",
        "pt_col": "PT",
        "response_col": "Response",
        "gene_col": None,
    },
    "Auslander": {
        "counts": "gene_counts.txt",
        "clinical": "clinical.txt",
        "pt_col": "Pt",
        "response_col": "Response",
        "gene_col": "Geneid",
        "skip_reason": "Patient IDs don't match between counts and clinical files",
    },
}


def load_gene_list(data_dir: Path) -> list[str]:
    filepath = data_dir / "genelist_8080.txt"
    return [line.strip() for line in open(filepath) if line.strip()]


def process_cohort(
    data_dir: Path,
    cohort_name: str,
    gene_list: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a clinical cohort into aligned, z-scored expression + labels."""
    import pandas as pd

    config = COHORT_CONFIG[cohort_name]
    cohort_dir = data_dir / config.get("dir", cohort_name)

    counts_file = cohort_dir / config["counts"]
    clinical_file = cohort_dir / config["clinical"]

    clinical_df = pd.read_csv(clinical_file, sep="\t")
    pt_col = config["pt_col"]
    response_col = config["response_col"]
    labels_dict = dict(zip(clinical_df[pt_col], clinical_df[response_col]))

    gene_col = config.get("gene_col")
    if gene_col:
        counts_df = pd.read_csv(counts_file, sep="\t", index_col=gene_col)
    else:
        counts_df = pd.read_csv(counts_file, sep="\t", index_col=0)

    strip_suffix = config.get("strip_counts_suffix")
    if strip_suffix:
        counts_df.columns = [c.removesuffix(strip_suffix) for c in counts_df.columns]

    common_patients = [p for p in counts_df.columns if p in labels_dict]
    if not common_patients:
        print(f"  ERROR: No patients overlap between counts and clinical for {cohort_name}")
        sys.exit(1)

    counts_df = counts_df[common_patients]

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)
    n_patients = len(common_patients)

    expression = np.zeros((n_genes, n_patients), dtype=np.float64)
    for gene_symbol, row_values in counts_df.iterrows():
        idx = gene_to_idx.get(str(gene_symbol))
        if idx is not None:
            expression[idx, :] = row_values.values

    expression = expression.T

    expression = stats.zscore(expression, axis=1, nan_policy="omit")
    expression = np.nan_to_num(expression, nan=0.0)
    expression = expression.astype(np.float16)

    labels = np.array([labels_dict[p] for p in common_patients])
    one_hot = np.zeros((n_patients, 2), dtype=np.float16)
    one_hot[labels == 0, 0] = 1.0
    one_hot[labels == 1, 1] = 1.0

    patient_ids = np.array(common_patients)

    return expression, one_hot, patient_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clinical cohort npz files")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--cohort", help="Single cohort name (e.g., Gide, Liu2019)")
    parser.add_argument("--all", action="store_true", help="Process all cohorts")
    parser.add_argument("--output", help="Output npz file (for single cohort)")
    parser.add_argument("--output-dir", default="data/clinical", help="Output dir (for --all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    gene_list = load_gene_list(data_dir)

    if args.all:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cohorts = list(COHORT_CONFIG.keys())
    elif args.cohort:
        cohorts = [args.cohort]
    else:
        print("ERROR: Specify --cohort <name> or --all")
        sys.exit(1)

    for cohort_name in cohorts:
        if cohort_name not in COHORT_CONFIG:
            print(f"ERROR: Unknown cohort '{cohort_name}'. Available: {list(COHORT_CONFIG.keys())}")
            sys.exit(1)

        if COHORT_CONFIG[cohort_name].get("skip_reason"):
            print(f"Skipping {cohort_name}: {COHORT_CONFIG[cohort_name]['skip_reason']}")
            continue

        print(f"Processing {cohort_name}...")
        expression, one_hot, patient_ids = process_cohort(data_dir, cohort_name, gene_list)

        resp = int((one_hot[:, 1] == 1).sum())
        non_resp = int((one_hot[:, 0] == 1).sum())
        print(f"  {len(patient_ids)} patients ({resp} resp, {non_resp} non-resp)")
        print(f"  Expression shape: {expression.shape}")

        if args.all:
            output_path = Path(args.output_dir) / f"clinical_{cohort_name}.npz"
        else:
            output_path = Path(args.output)

        np.savez(output_path, x=expression, y=one_hot, info=patient_ids)
        print(f"  Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
