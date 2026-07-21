# DCASE 2026: uticaj pouzdanosti anotacija

Projekat istražuje da li manji, ali pouzdanije anotiran skup daje bolje rezultate u heterogenoj klasifikaciji zvuka. Korišćen je zvanični baseline za **DCASE 2026 Task 1** i skup **BSD10k-v1.2**.

Reprodukovani su audio-only i multimodalni baseline rezultati. Glavno poređenje koristi zajednički zaključani test skup za kompletan trening skup, podskup sa `confidence >= 4` i nasumični podskup iste veličine i klasne raspodele.

| Trening skup | Audio | Audio + tekst |
|---|---:|---:|
| Kompletan | **76,83%** | **79,78%** |
| `confidence >= 4` | 75,91% | 77,11% |
| Nasumični podskup | 71,95% | 77,63% |

Vrednosti predstavljaju hierarchical F1 na istih 2.192 test primera. Kompletan trening skup daje najbolji rezultat, dok je pouzdano filtriranje u audio režimu znatno bolje od nasumičnog smanjenja.

Detaljan opis postupka, rezultata i ograničenja nalazi se u [`izvestaj.ipynb`](izvestaj.ipynb). Zbirne tabele su sačuvane u direktorijumu [`rezultati`](rezultati/), a izvorni kod u [`dcase2026_task1_baseline`](dcase2026_task1_baseline/).

## Rezultati

Direktorijum [`rezultati`](rezultati/) sadrži:

- `baseline_5fold.csv` — reprodukciju baseline-a nad kompletnim BSD10k skupom;
- `confidence_ge4_5fold.csv` — 5-fold evaluaciju unutar filtriranog skupa;
- `controlled_results.csv` — glavno kontrolisano poređenje na zajedničkih 2.192 test primera.

Sve vrednosti metrika izražene su u procentima. Za odgovor na istraživačko pitanje merodavna je tabela `controlled_results.csv`, jer svi modeli u njoj koriste isti test skup.

## Podaci i instalacija

Potrebni su Python 3 i paketi navedeni u `requirements.txt`:

```bash
cd dcase2026_task1_baseline
pip install -r requirements.txt
```

Skup [BSD10k-v1.2](https://zenodo.org/records/17233904) treba raspakovati u `dcase2026_task1_baseline/data/BSD10k-v1.2/`. Dataset, CLAP embedding fajlovi, modeli i lokalni eksperimentalni izlazi nisu uključeni u repozitorijum.

## Reprodukcija eksperimenata

Priprema podataka i puni baseline:

```bash
python build_dataset.py
python train_test.py
python summarize_results.py
```

5-fold eksperiment sa anotacijama `confidence >= 4`:

```bash
python prepare_confident_subset.py --min-confidence 4
DCASE_PROCESSED_DATASET=./data/processed_dataset_confidence_ge4.csv \
DCASE_MODEL_OUTPUT=./model_output_conf_ge4_full \
DCASE_EXPERIMENT_NAME=BSD10k-confidence-ge4 \
python train_test.py
```

Kontrolisani eksperiment sa zajedničkim test skupom:

```bash
python prepare_controlled_splits.py
python run_controlled_experiment.py
python summarize_controlled.py
```

Kontrolisani eksperiment poredi kompletan, pouzdano filtriran i nasumični trening podskup, dok svi modeli koriste identičan test skup.
