# Finding Donors — Census Income Binary Classification

> A supervised learning project predicting whether an individual earns **>$50K/year** from 1994 U.S. Census data, to help non-profit organizations optimize their donor outreach strategies.

[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Census%20Income-blue?style=flat-square)](https://archive.ics.uci.edu/ml/datasets/adult)
[![Language](https://img.shields.io/badge/Language-Python%2FJupyter-green?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)]()

---

## Overview

This project builds a **binary income classifier** using a variant of the [UCI Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult). The trained model predicts whether a given individual earns more or less than $50,000 per year based on demographic and occupational attributes.

The primary application is **non-profit donor targeting**: by identifying high-income individuals, organizations can tailor their donation requests appropriately — asking high earners for larger contributions while not alienating lower-income supporters.

---

## Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository (1994 U.S. Census) |
| Samples | 14,809 |
| Features | 14 (age, education, workclass, occupation, marital status, etc.) |
| Target | `income`: `<=50K` (~76%) or `>50K` (~24%) |
| Class balance | Imbalanced — 3:1 ratio |

### Feature Overview

| Feature | Type | Example |
|---|---|---|
| `age` | Numeric | 39 |
| `workclass` | Categorical | Private, Self-emp, Gov |
| `education_level` | Categorical | Bachelors, Masters |
| `education-num` | Numeric | 13 |
| `marital-status` | Categorical | Married-civ-spouse |
| `occupation` | Categorical | Tech-support, Sales |
| `capital-gain` | Numeric | 2174 |
| `hours-per-week` | Numeric | 40 |
| `income` | Target (binary) | `>50K` |

---

## Planned ML Pipeline

```
[Census CSV: 14,809 samples × 14 features]
              |
              v
   [Preprocessing]
   • OneHotEncoder for categorical features
   • StandardScaler for numeric features
   • Handle imbalance: class_weight='balanced' or SMOTE
              |
              v
   [Model Training & Comparison]
   ├── Naive Bayes (baseline)
   ├── Decision Tree
   ├── Support Vector Machine (SVM)
   ├── AdaBoost / Gradient Boosting
              |
              v
   [Model Selection via fbeta_score (β=0.5)]
   Priority: Precision > Recall (avoid wasting outreach budget)
              |
              v
   [Feature Importance Analysis]
   [Final Model Evaluation on test set]
```

### Why Fβ (beta=0.5)?
For donor targeting, **false positives** (predicting high income for a low earner) are more costly than false negatives. The Fβ metric with β=0.5 weights precision twice as heavily as recall.

---

## Project Status

> ⚠️ **Work in Progress** — Data loading and exploration complete. Model training pipeline is under development.

- [x] Dataset loading into Pandas DataFrame
- [x] Initial data exploration (shape, dtypes, sample rows)
- [ ] Preprocessing pipeline (encoding, scaling)
- [ ] Baseline model implementation
- [ ] Full model comparison (DT, SVM, AdaBoost)
- [ ] Feature importance visualization
- [ ] Final evaluation report

---

## Getting Started

```bash
git clone https://github.com/tamer017/finding_donors.git
cd finding_donors
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook finding_donors.ipynb
```

---

## Skills Demonstrated

- **Supervised Learning:** Binary classification, imbalanced dataset handling, model selection
- **Feature Engineering:** One-Hot Encoding, StandardScaler, log-transform for skewed features
- **Model Evaluation:** Accuracy, Fβ-score, confusion matrix, ROC-AUC
- **Domain Application:** Non-profit donor targeting, socioeconomic data analysis
- **Python:** Pandas, NumPy, Scikit-learn, Matplotlib

---

## References

- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid*. KDD.
