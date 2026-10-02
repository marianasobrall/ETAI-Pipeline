# Baseline Predictive Pipeline -- ETAI

20260489 Mariana Sobral

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. 

## Project structure
```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Environment Setup

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to run
```bash
python main.py
```

## Dataset

See `data/README.md`.


# Analysis
## Week 2

**Decision Tree** — train accuracy: 0.829, test accuracy: 0.629
**Logistic Regression** — train accuracy: 0.679, test accuracy: 0.677

**Best Model:** Logistic Regression. The Decision Tree has a higher train accuracy, but
that's because it is overfitting — the large gap between train and test accuracy shows
it memorized the training data instead of actually learning, while Logistic Regression
generalizes better and achieves a higher test accuracy.

## Week 3

- **Decision Tree** — train accuracy: 0.792, test accuracy: 0.611
- **Logistic Regression** — train accuracy: 0.673, test accuracy: 0.665

**Before vs after cleaning:**
- *Decision Tree*: train accuracy dropped from 0.829 to 0.792, test accuracy dropped
  from 0.629 to 0.611. The gap shrank (+0.201 → +0.181), so cleaning reduced
  overfitting a bit, but the tree still memorizes training data more than it generalizes.
- *Logistic Regression*: both train and test accuracy dropped slightly (0.679 → 0.673
  train, 0.677 → 0.665 test), and the gap stayed small (+0.008). The drop is minor and
  the model still generalizes well — likely reflects fewer rows being thrown away as
  missing/invalid, changing the exact split slightly, rather than the model getting
  worse at the task.

**Best Model:** Logistic Regression. The Decision Tree still overfits badly even after
cleaning — its train/test gap (+0.181) is more than 20x Logistic Regression's (+0.008)
— while Logistic Regression keeps a small gap and a higher test accuracy (0.665 vs 0.611).

## Week 4

| Model | Holdout test | CV mean (std) |
|---|---|---|
| Logistic Regression | 0.665 | 0.670 (0.015) |
| Decision Tree | 0.611 | 0.619 (0.010) |
| Random Forest (new) | 0.633 | 0.627 (0.009) |
| Dummy (new, baseline) | 0.550 | 0.549 (0.000) |

**Week 3 vs Week 4:** LR and DT numbers are basically unchanged — moving preprocessing
into a `Pipeline` didn't change the results, just unlocked CV.

**Holdout vs CV:** all three real models land within ~1pp of their holdout score, with
low std — confirms the single holdout number was trustworthy, not lucky.

**New models:** Dummy just predicts the majority class every time, no features involved
— that's the 0.550 floor, and every real model clears it without much effort. Random
Forest is a bunch of Decision Trees trained on random subsets, votes averaged out,
which is why its gap (+0.158) is smaller than one tree alone (+0.180) and both its
scores come in higher.

**Best Model:** Logistic Regression — highest test accuracy, smallest gap. Random
Forest is the best of the tree-based models, but still behind LR, and its FPR gap by
race (0.36 vs 0.24) is actually a bit worse than LR or DT.