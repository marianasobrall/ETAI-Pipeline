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
Week 2: 
Decision Tree — train accuracy: 0.829, test accuracy: 0.629
Logistic Regression — train accuracy: 0.679, test accuracy: 0.677

Best Model: Logistic Regression. The Decision Tree has a higher train accuracy, but that's because it is overfitting, the large gap between train and test accuracy shows it memorized the training data instead of actually learning, while Logistic Regression generalizes better and achieves a higher test accuracy.

Week 3:
Decision Tree — train accuracy: 0.792, test accuracy: 0.611
Logistic Regression — train accuracy: 0.673, test accuracy: 0.665

Before and after cleaning:
Decision Tree: train accuracy dropped from 0.829 to 0.792, and test accuracy dropped from 0.629 to 0.611. The gap did shrink (from +0.201 to +0.181), so the cleaning pipeline reduced overfitting a bit, but the tree is still  memorizing the training data rather than generalizing.

Logistic Regression: both train and test accuracy dropped slightly (0.679 → 0.673 train, 0.677 → 0.665 test), and the gap stayed small (+0.008). The drop is minor and the model generalizes well, the slightly lower numbers probably reflect fewer rows being thrown away for encoding as missing/invalid, which changes the exact split slightly, rather than the model getting worse at the task.

Best Model: Logistic Regression. The Decision Tree still overfits badly even after cleaning — its train/test gap (+0.181) is more than 20x the Logistic Regression's (+0.008) — while Logistic Regression keeps a small gap and a higher test accuracy (0.665 vs 0.611).


Week 4:

Logistic Regression — holdout test 0.665 | CV mean 0.670 (std 0.015)
Decision Tree — holdout test 0.611 | CV mean 0.619 (std 0.010)
Random Forest (new) — holdout test 0.633 | CV mean 0.627 (std 0.009)
Dummy (new, baseline) — holdout test 0.550 | CV mean 0.549 (std 0.000)

Week 3 vs Week 4: LR and DT numbers are basically unchanged: moving preprocessing into a Pipeline didn't change the results, just unlocked CV.

Holdout vs CV: all three real models land within ~1pp of their holdout score, with low std : confirms the single holdout number was trustworthy, not lucky.

New models: Dummy just predicts the majority class every time, no features involved — that's the 0.550 floor, and every real model clears it without much effort. Random Forest is a bunch of Decision Trees trained on random subsets, votes averaged out, which is why its gap (+0.158) is smaller than one tree alone (+0.180) and both its scores come in higher.

Best Model: Logistic Regression because it has the highest test accuracy and smallest gap. Random Forest is the best of the tree-based models, but still behind LR, and its FPR gap by race (0.36 vs 0.24) is actually a bit worse than LR or DT.

-------

goals
1. add preprocessing to pipeline using HOLDOUT(train/validation unique set -> current one), compare results with week 3.

challenge 2
2. Add CV to your pipeline, compare BEFORE VS AFTER CV 
note - 1+2 can be done at home. Add the new models to the pipeline (Dumies and RF)
3. Change something in your pipeline (Imputer -> KNN,encoder,etc) Can you get a getter modeling pipeline?