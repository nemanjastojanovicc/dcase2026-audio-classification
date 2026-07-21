# DCASE 2026: uticaj pouzdanosti oznaka

Projekat se zasniva na zvaničnom baseline rešenju za DCASE 2026 Task 1. Rešenje je osposobljeno za klasifikaciju zvuka u audio i multimodalnom režimu, a kao modifikacija ispitan je uticaj pouzdanosti oznaka u skupu podataka.

Detaljan opis postupka i rezultata nalazi se u [`izvestaj.ipynb`](izvestaj.ipynb), dok su zbirni rezultati sačuvani u direktorijumu [`rezultati`](rezultati/).

## Pokretanje

Potrebni su Python 3 i skup [BSD10k-v1.2](https://zenodo.org/records/17233904), koji treba raspakovati u `dcase2026_task1_baseline/data/BSD10k-v1.2/`.

```bash
cd dcase2026_task1_baseline
pip install -r requirements.txt
python build_dataset.py
python train_test.py
python summarize_results.py
```

Eksperiment sa pouzdanijim oznakama pokreće se na sledeći način:

```bash
python prepare_confident_subset.py --min-confidence 4
DCASE_PROCESSED_DATASET=./data/processed_dataset_confidence_ge4.csv \
DCASE_MODEL_OUTPUT=./model_output_conf_ge4_full \
DCASE_EXPERIMENT_NAME=BSD10k-confidence-ge4 \
python train_test.py
```

Poređenje modela na istom test skupu pokreće se komandama:

```bash
python prepare_controlled_splits.py
python run_controlled_experiment.py
python summarize_controlled.py
```
