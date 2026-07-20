"""Summarize controlled variants evaluated on the shared test set."""

from pathlib import Path
import re

import pandas as pd


ROOT = Path("model_output_controlled")
PATTERN = re.compile(r"([a-zA-Z0-9_]+):\s*([\d.]+)%")


def main():
    rows = []
    prediction_ids = {}

    for variant in ("full", "confidence_ge4", "random_matched"):
        for mode in ("audio", "both"):
            evaluation_dir = ROOT / variant / mode / "fold_0" / "evaluation"
            result_path = evaluation_dir / "results.txt"
            prediction_path = evaluation_dir / "predictions.csv"
            metrics = {
                name: float(value)
                for name, value in PATTERN.findall(result_path.read_text())
            }
            rows.append({"variant": variant, "mode": mode, **metrics})
            prediction_ids[(variant, mode)] = pd.read_csv(prediction_path)[
                "sound_id"
            ].tolist()

    reference_ids = next(iter(prediction_ids.values()))
    if not all(ids == reference_ids for ids in prediction_ids.values()):
        raise RuntimeError("Predictions do not use the same ordered test examples")

    results = pd.DataFrame(rows)
    output_path = ROOT / "controlled_results.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"\nAll models share {len(reference_ids)} ordered test examples.")
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
