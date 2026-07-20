"""Prepare a confidence-filtered variant of the processed BSD dataset."""

import argparse
from pathlib import Path

import pandas as pd

from utils import get_subconfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter processed examples using annotation confidence."
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=4,
        help="Minimum confidence value to retain (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/processed_dataset_confidence_geN.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = get_subconfig("active_dataset")
    dataset_config = get_subconfig("datasets")[dataset_name]
    output_dir = Path(get_subconfig("output_path"))
    processed_path = output_dir / get_subconfig("processed_dataset_csv")
    metadata_path = Path(dataset_config["metadata_csv"])
    output_path = args.output or (
        output_dir / f"processed_dataset_confidence_ge{args.min_confidence}.csv"
    )
    distribution_path = output_path.with_name(
        f"{output_path.stem}_class_distribution.csv"
    )

    processed = pd.read_csv(processed_path)
    metadata = pd.read_csv(metadata_path, usecols=["sound_id", "confidence"])

    if "index" not in processed.columns:
        raise ValueError(f"Expected an 'index' column in {processed_path}")
    if metadata["sound_id"].duplicated().any():
        raise ValueError(f"Duplicate sound_id values found in {metadata_path}")

    merged = processed.merge(
        metadata,
        left_on="index",
        right_on="sound_id",
        how="left",
        validate="one_to_one",
    )
    if merged["confidence"].isna().any():
        missing = int(merged["confidence"].isna().sum())
        raise ValueError(f"Confidence is missing for {missing} processed examples")

    subset = merged[merged["confidence"] >= args.min_confidence].copy()
    subset = subset.drop(columns="sound_id")
    subset.to_csv(output_path, index=False)

    full_counts = processed["class"].value_counts().rename("full_count")
    subset_counts = subset["class"].value_counts().rename("subset_count")
    distribution = pd.concat([full_counts, subset_counts], axis=1).fillna(0)
    distribution = distribution.astype(int).sort_index()
    distribution["retained_pct"] = (
        100 * distribution["subset_count"] / distribution["full_count"]
    ).round(2)
    distribution.to_csv(distribution_path, index_label="class")

    print(f"Dataset: {dataset_name}")
    print(f"Minimum confidence: {args.min_confidence}")
    print(f"Full dataset: {len(processed)} examples")
    print(
        f"Filtered dataset: {len(subset)} examples "
        f"({100 * len(subset) / len(processed):.2f}%)"
    )
    print(f"Classes retained: {subset['class'].nunique()}")
    print(f"Smallest class: {subset['class'].value_counts().min()} examples")
    print(f"Saved dataset to {output_path}")
    print(f"Saved class distribution to {distribution_path}")


if __name__ == "__main__":
    main()
