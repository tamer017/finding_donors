# Finding Donors — Charitable Giving Prediction

> **Binary classification pipeline predicting whether a Census respondent earns >$50K/year (proxy for charity donation likelihood), benchmarking Naive Bayes, SVM, and Random Forest with GridSearchCV optimization.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Census-green.svg)](https://archive.ics.uci.edu/ml/datasets/census+income)

---

## Overview

This project addresses a real-world **charitable organization problem**: identifying which individuals from the UCI Adult Census dataset are likely to earn above $50,000/year, making them strong donation candidates. The project walks through the full supervised ML pipeline — EDA, preprocessing, algorithm benchmarking, and hyperparameter optimization — as part of the Udacity ML Nanodegree.

---

## Dataset — UCI Adult Census Income

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Samples | 45,222 |
| Features | 13 (age, education, occupation, hours-per-week, etc.) |
| Target | Income: `<=50K` (0) or `>50K` (1) |
| Class imbalance | ~76% ≤$50K / ~24% >$50K |

---

## Preprocessing Pipeline

```python
# 1. Handle skewed continuous features
for col in ['capital-gain', 'capital-loss']:
    data[col] = np.log1p(data[col])  # log-transform right-skewed

# 2. Normalize continuous features
scaler = MinMaxScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# 3. One-hot encode categorical features
data = pd.get_dummies(data, columns=categorical_cols)

# Result: 103 features after encoding
```

---

## Algorithm Benchmark

All models evaluated with **F-beta score (β=0.5)** — precision-weighted, since false positives (contacting non-donors) cost resources.

| Model | F-beta (Train) | F-beta (Test) | Training Time |
|---|---|---|---|
| Naive Bayes | 0.476 | 0.472 | **0.02s** |
| SVM (RBF) | 0.866 | 0.841 | 142s |
| **Random Forest** | **0.999** | **0.863** | 8.3s |

**Random Forest selected** — best test F-beta with reasonable training time.

---

## GridSearchCV Optimization

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
# Best: n_estimators=200, max_depth=20, min_samples_split=5
```

**Optimized F-beta: 0.871** (+0.8pp over default Random Forest)

---

## Feature Importance

Top 5 features by Random Forest importance:
1. `capital-gain` (log-transformed) — 28.4%
2. `age` — 14.2%
3. `hours-per-week` — 10.8%
4. `education-num` — 9.1%
5. `capital-loss` (log-transformed) — 7.3%

---

## Naive Predictor Baseline

A naive model that predicts everyone earns >$50K:
- Accuracy: 24.78% (class frequency)
- F-beta: 0.292

Random Forest achieves **3× better F-beta** than the naive baseline.

---

## Installation

```bash
git clone https://github.com/tamer017/finding_donors.git
cd finding_donors
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook finding_donors.ipynb
```

---

## Skills & Concepts

`Supervised Learning` `Binary Classification` `Random Forest` `SVM` `Naive Bayes` `GridSearchCV` `F-beta Score` `Feature Importance` `Log Transformation` `Class Imbalance` `Udacity ML Nanodegree`

---

## Author

**Ahmed Tamer Assy** — [GitHub](https://github.com/tamer017) | Machine Learning Researcher @ Volkswagen AG
