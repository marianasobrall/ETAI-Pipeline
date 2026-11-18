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
│   ├── preprocessing.py    # cleaning, features/target, locked dev/test split, preprocessor
│   ├── model.py            # model registry + build_pipeline (prep + model)
│   ├── evaluate.py         # scorers, CV with out-of-fold probabilities, reports, fairness, paired t-test
│   ├── tuning.py           # Optuna tuning + nested CV
│   └── results.py          # run report (.txt) + scoreboard (runs.csv)
├── results/                # created automatically -- run_<id>.txt per run + runs.csv scoreboard (not tracked in git)
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
Set `model.type` (and its `params`) in `config.yaml` to choose the model, and `tuning.enabled` to
turn Optuna + nested CV on or off. Each run prints every metric, adds one row to
`results/runs.csv` and prints the scoreboard. The locked test set is never used.

## Dataset

See `data/README.md`.


# Analysis
## Week 2

**Decision Tree** - train accuracy: 0.829, test accuracy: 0.629
**Logistic Regression** - train accuracy: 0.679, test accuracy: 0.677

**Best Model:** Logistic Regression. The Decision Tree has a higher train accuracy, but
that's because it is overfitting - the gap between train and test accuracy shows
it memorized the training data instead of  learning, while Logistic Regression
generalizes better and achieves a higher test accuracy.

## Week 3

- **Decision Tree** - train accuracy: 0.792, test accuracy: 0.611
- **Logistic Regression** - train accuracy: 0.673, test accuracy: 0.665

**Before vs after cleaning:**
- *Decision Tree*: train accuracy dropped from 0.829 to 0.792, test accuracy dropped
  from 0.629 to 0.611. The gap shrank (+0.201 → +0.181), so cleaning reduced
  overfitting a bit, but the tree still memorizes training data more than it generalizes.
- *Logistic Regression*: both train and test accuracy dropped slightly (0.679 → 0.673
  train, 0.677 → 0.665 test), and the gap stayed small (+0.008). The drop is minor and
  the model still generalizes well - likely reflects fewer rows being thrown away as
  missing/invalid, changing the exact split slightly, rather than the model getting
  worse at the task.

**Best Model:** Logistic Regression. The Decision Tree still overfits badly even after
cleaning - its train/test gap (+0.181) is more than 20x Logistic Regression's (+0.008)
- while Logistic Regression keeps a small gap and a higher test accuracy (0.665 vs 0.611).

## Week 4

| Model | Holdout test | CV mean (std) |
|---|---|---|
| Logistic Regression | 0.665 | 0.670 (0.015) |
| Decision Tree | 0.611 | 0.619 (0.010) |
| Random Forest (new) | 0.633 | 0.627 (0.009) |
| Dummy (new, baseline) | 0.550 | 0.549 (0.000) |

**Week 3 vs Week 4:** LR and DT numbers are basically unchanged - moving preprocessing
into a `Pipeline` didn't change the results, just unlocked CV.

**Holdout vs CV:** all three real models land within ~1pp of their holdout score, with
low std — confirms the single holdout number was trustworthy, not lucky.

**New models:** Dummy just predicts the majority class every time, no features involved
- that's the 0.550 floor, and every real model clears it without much effort. Random
Forest is a bunch of Decision Trees trained on random subsets, votes averaged out,
which is why its gap (+0.158) is smaller than one tree alone (+0.180) and both its
scores come in higher.

**Best Model:** Logistic Regression - highest test accuracy, smallest gap. Random
Forest is the best of the tree-based models, but still behind LR, and its FPR gap by
race (0.36 vs 0.24) is actually a bit worse than LR or DT.


## Week 5
### Ensembles

| Model | CV mean ± std | Gap (train − test) |
|---|---|---|
| Logistic regression | 0.670 ± 0.015 | +0.008 |
| Decision tree | 0.619 ± 0.010 | +0.180 |
| Random forest | 0.629 ± 0.006 | +0.153 |
| Random forest (min_samples_leaf=20) | 0.679 ± 0.009 | +0.018 |
| Oblique random forest | 0.638 ± 0.007 | +0.145 |
| AdaBoost (stumps) | 0.680 ± 0.009 | +0.011 |

Note: the pruned random forest row was computed with shuffled folds, so it is not exactly comparable with the others.

AdaBoost has the best CV mean (0.680), but the difference to logistic regression is smaller than the std, so it is a tie. The forests with deep trees overfit (gaps around +0.15), while AdaBoost uses stumps and doesn't. Limiting leaf size took the random forest from 0.629 to 0.679, a bigger gain than any feature tried. The oblique forest is a bit better than the normal one, but within one std.

### Feature engineering

Seven features were added one at a time and compared on the same 5 folds. They are fixed formulas, so they are built before the split. None improved any model by more than the std. The best was priors × young, +0.0016 on the random forest, which is noise. Most of the signal is already in age and prior count, so combining them adds little. The original features were kept.

## Week 6 and 7

ROC-AUC, same 5 folds for every model. *Tuned* = nested CV with Optuna (30 trials).

| Model | Untuned | Tuned mean ± std |
|---|---|---|
| HistGradientBoosting (new) | 0.717 | 0.728 ± 0.016 |
| Gradient boosting (new) | 0.727 | 0.728 ± 0.014 |
| AdaBoost | 0.723 | 0.727 ± 0.015 |
| Oblique random forest | 0.683 | 0.726 ± 0.014 |
| Random forest | 0.683 | 0.725 ± 0.015 |
| Logistic regression | 0.718 | 0.718 ± 0.014 |
| Decision tree | 0.608 | 0.717 ± 0.017 |
| Naive Bayes (new) | 0.685 | 0.710 ± 0.012 |
| Dummy | 0.500 | - |

**Why ROC-AUC:** COMPAS is used to rank people by risk, and ROC-AUC measures exactly
that without picking a 0.5 threshold. It is also less noisy than accuracy.

**Tuning:** helped most where the defaults overfit - decision tree +0.109, forests +0.04.
Tuning picked large `min_samples_leaf` values, i.e. it mostly stopped the trees from
memorizing. The boosting models were already good with default settings.

**Best Model:** HistGradientBoosting (0.728), but the five ensembles are within 0.003 of
each other, much less than the std, so it is a tie. The paired t-test confirms it beats
logistic regression (p = 0.02), the decision tree and Naive Bayes, but not the other ensembles.

## Week 9 - Beyond COMPAS

Option A: same pipeline, new dataset (Adult income, predict if a person earns more than 50K).
COMPAS still runs exactly as before.

**Choices in `config_adult.yaml`:** `sex` is the sensitive attribute for the fairness check, and `race`
is also left out of the features, so the model uses neither. `fnlwgt` is a sampling weight, not a
characteristic of the person, so it is dropped. `education` is the same as `education_num`, so only
`education_num` is used. `workclass` and `occupation` get a missingness flag, because people with
these missing earn > 50K about half as often. `native_country` and `v01`-`v06` are only imputed.
The 99999 in `capital_gain` is kept, it is the census top code and not an error.

### What broke and how I fixed it

| What broke | Why | Fix |
|---|---|---|
| `python main.py` crashed (`KeyError: 'test_set'`) | `config.yaml` was still an old version | updated `config.yaml` |
| COMPAS rule `age: {min: 18}` | would delete the 74 valid 17-year-olds | Adult has its own validity rules in its config |
| Predicting `adult_new.csv` crashed (`KeyError: 'income_over_50k'`) | `split_features_target()` assumed the target is always there | `y` is `None` when there is no target |
| No way to predict new rows | the pipeline only evaluated | `predict` section in the config, `main.py` predicts after the final refit |
| Fairness report talked about "reoffending" | text was written for COMPAS | generic text (class 0 / class 1) |

Check after cleaning: `workclass`, `occupation` and `native_country` have 379, 381 and 86 missing
values, and no `"?"` category is left.

### Results

5-fold CV on the development set.

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Dummy | 0.500 | 0.761 | 0.000 |
| Decision tree | 0.731 | 0.803 | 0.590 |
| Logistic regression | 0.894 | 0.845 | 0.635 |
| HistGradientBoosting (default) | 0.899 | - | - |
| **HistGradientBoosting (tuned, nested CV)** | **0.906 ± 0.014** | **0.861 ± 0.010** | **0.669 ± 0.032** |
| Target | >= 0.90 | >= 0.855 | >= 0.66 |

The tuned HistGradientBoosting reaches all three targets in the same run (max_iter 84,
learning_rate 0.085, max_leaf_nodes 5, min_samples_leaf 95). All features are kept (no feature
selection), including the 25 `v` columns. Predictions for the 2,000 people in `adult_new.csv` are in
`predictions/adult_new_predictions.csv` (id, probability, 0/1 prediction).

The hardest assumption to find was that the data always has a target column, it only broke when
predicting the new file. Most COMPAS-specific things were in the config, not in the code, so most
of the work was writing `config_adult.yaml`. Boosting plus tuning is what got the scores over the
targets. Accuracy alone is misleading here: predicting "<= 50K" for everyone already gets 0.76.



## Week 10 - Missing values and predicting new data

| Week | Added to the pipeline |
|---|---|
| 10 | the numeric imputer is a setting in `config.yaml` (`median`, `mean`, `knn`, `iterative`, `random_forest`); numeric columns are scaled first, then imputed; `main.py` saves the final model; new `predict.py` |

### Missing values

COMPAS, gradient boosting with default params (tuning off), same 5 folds. Each imputer is
compared with the median using the corrected paired t-test.

| Imputer | ROC-AUC | Difference vs median | p-value |
|---|---|---|---|
| median | 0.727 ± 0.015 | - | - |
| mean | 0.727 ± 0.014 | 0.000 | 0.83 |
| knn | 0.725 ± 0.015 | -0.001 | 0.02 |
| iterative (MICE) | 0.726 ± 0.015 | -0.001 | 0.44 |
| random_forest (missForest) | 0.728 ± 0.013 | +0.001 | 0.52 |

I kept the **median**. No imputer is clearly better: random forest is +0.001 but the test says
it is noise, and it is the slowest, while KNN is a bit worse (p = 0.02). The other imputers only
see the 5 numeric columns, which say little about each other, so they can't guess better than the median.

### Predicting new data

```bash
python main.py                                                # trains and saves models/model.joblib
python predict.py data/compas_new_people.csv predictions.csv  # predicts with the saved model
```

`predict.py` loads the saved model, cleans the new file with the same rules as training (without
dropping any row) and writes `predictions.csv` with `id`, `probability` (of reoffending) and
`prediction` (0/1 at 0.5). 20 rows in, 20 rows out: the `"?"` and empty values are imputed, the
unseen charge degree `"X"` doesn't crash, and the person who appears twice gets the same prediction twice.