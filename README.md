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

