# DCASE 2026: uticaj pouzdanosti anotacija

Projekat istražuje da li manji, ali pouzdanije anotiran skup daje bolje rezultate u heterogenoj klasifikaciji zvuka. Korišćen je zvanični baseline za **DCASE 2026 Task 1** i skup **BSD10k-v1.2**.

Reprodukovani su audio-only i multimodalni baseline rezultati. Glavno poređenje koristi zajednički zaključani test skup za kompletan trening skup, podskup sa `confidence >= 4` i nasumični podskup iste veličine.

| Trening skup | Audio | Audio + tekst |
|---|---:|---:|
| Kompletan | **76,83%** | **79,78%** |
| `confidence >= 4` | 75,91% | 77,11% |
| Nasumični podskup | 71,95% | 77,63% |

Vrednosti predstavljaju hierarchical F1 na istih 2.192 test primera. Kompletan trening skup daje najbolji rezultat, dok je pouzdano filtriranje u audio režimu znatno bolje od nasumičnog smanjenja.

Detaljan opis postupka, rezultata i ograničenja nalazi se u [`izvestaj.ipynb`](izvestaj.ipynb). Izvorni kod i uputstvo za baseline nalaze se u direktorijumu [`dcase2026_task1_baseline`](dcase2026_task1_baseline/).

## Pokretanje

```bash
cd dcase2026_task1_baseline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Dataset i unapred izračunati CLAP embedding fajlovi nisu uključeni u repozitorijum.
