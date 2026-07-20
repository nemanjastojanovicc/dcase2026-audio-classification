"""Create controlled datasets that share one fixed test set."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils import get_subconfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-confidence", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1821)
    parser.add_argument("--output-dir", type=Path, default=Path("data/controlled"))
    return parser.parse_args()


def stratified_partition(dataframe, validation_size, seed):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=seed
    )
    labels = dataframe["class_idx"].to_numpy()
    train_pos, val_pos = next(splitter.split(np.zeros(len(labels)), labels))
    train = dataframe.iloc[train_pos].copy().assign(split="train")
    val = dataframe.iloc[val_pos].copy().assign(split="val")
    return train, val


def sample_matched_by_class(pool, target, seed):
    sampled = []
    for class_name, target_group in target.groupby("class", sort=True):
        candidates = pool[pool["class"] == class_name]
        sampled.append(
            candidates.sample(n=len(target_group), random_state=seed, replace=False)
        )
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)


def save_variant(name, trainval_pool, fixed_test, args):
    train, val = stratified_partition(
        trainval_pool, args.validation_size, args.seed
    )
    test = fixed_test.copy().assign(split="test")
    output = pd.concat([train, val, test], ignore_index=True)
    output_path = args.output_dir / f"{name}.csv"
    output.to_csv(output_path, index=False)
    print(
        f"{name}: train={len(train)}, val={len(val)}, test={len(test)}, "
        f"total={len(output)}"
    )
    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(get_subconfig("output_path"))
    processed_path = data_dir / get_subconfig("processed_dataset_csv")
    dataset_name = get_subconfig("active_dataset")
    metadata_path = Path(get_subconfig("datasets")[dataset_name]["metadata_csv"])

    processed = pd.read_csv(processed_path)
    metadata = pd.read_csv(metadata_path, usecols=["sound_id", "confidence"])
    full = processed.merge(
        metadata,
        left_on="index",
        right_on="sound_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="sound_id")
    if full["confidence"].isna().any():
        raise ValueError("Some processed examples have no confidence value")

    test_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    labels = full["class_idx"].to_numpy()
    pool_pos, test_pos = next(test_splitter.split(np.zeros(len(labels)), labels))
    pool = full.iloc[pool_pos].reset_index(drop=True)
    fixed_test = full.iloc[test_pos].reset_index(drop=True)

    confident_pool = pool[pool["confidence"] >= args.min_confidence].reset_index(drop=True)
    random_matched_pool = sample_matched_by_class(
        pool, confident_pool, args.seed
    )

    variants = {
        "full_train": pool,
        f"confidence_ge{args.min_confidence}_train": confident_pool,
        "random_matched_train": random_matched_pool,
    }
    outputs = {
        name: save_variant(name, variant, fixed_test, args)
        for name, variant in variants.items()
    }

    test_ids = [
        set(output.loc[output["split"] == "test", "index"])
        for output in outputs.values()
    ]
    if not all(ids == test_ids[0] for ids in test_ids[1:]):
        raise RuntimeError("Controlled variants do not share the same test set")

    manifest = pd.DataFrame(
        [
            {
                "variant": name,
                "train": int((output["split"] == "train").sum()),
                "validation": int((output["split"] == "val").sum()),
                "test": int((output["split"] == "test").sum()),
                "unique_classes": output["class"].nunique(),
                "seed": args.seed,
            }
            for name, output in outputs.items()
        ]
    )
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    fixed_test[["index", "class", "class_idx", "confidence"]].to_csv(
        args.output_dir / "fixed_test_ids.csv", index=False
    )
    print(f"Saved controlled datasets to {args.output_dir}")


if __name__ == "__main__":
    main()
