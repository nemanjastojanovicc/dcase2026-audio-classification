"""Run all controlled training variants on their shared fixed test set."""

import os
from pathlib import Path
import subprocess
import sys


VARIANTS = {
    "full": "full_train.csv",
    "confidence_ge4": "confidence_ge4_train.csv",
    "random_matched": "random_matched_train.csv",
}


def main():
    controlled_dir = Path("data/controlled")
    output_root = Path(os.getenv("DCASE_CONTROLLED_OUTPUT", "model_output_controlled"))

    for name, filename in VARIANTS.items():
        dataset_path = controlled_dir / filename
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Missing {dataset_path}; run prepare_controlled_splits.py first"
            )

        environment = os.environ.copy()
        environment.update(
            {
                "DCASE_MODES": os.getenv("DCASE_MODES", "audio,both"),
                "DCASE_MODEL_OUTPUT": str(output_root / name),
                "DCASE_NUM_EPOCHS": os.getenv("DCASE_NUM_EPOCHS", "100"),
                "DCASE_NUM_WORKERS": os.getenv("DCASE_NUM_WORKERS", "0"),
                "DCASE_PROCESSED_DATASET": str(dataset_path),
                "DCASE_EXPERIMENT_NAME": f"Controlled {name}",
            }
        )
        print(f"\n######## Controlled variant: {name} ########", flush=True)
        subprocess.run(
            [sys.executable, "train_test.py"], env=environment, check=True
        )

    print(f"\nControlled experiment completed: {output_root}")


if __name__ == "__main__":
    main()
