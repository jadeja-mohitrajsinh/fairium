# Kaggle Datasets For FairSight

These are the two primary datasets selected for live fairness validation:

## 1. Hiring Dataset
- Name: `IBM HR Analytics Employee Attrition & Performance`
- Kaggle: `pavansubhasht/ibm-hr-analytics-attrition-dataset`
- Link: <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>
- Why:
  - useful for hiring and workforce-risk style fairness checks
  - includes demographic and work-related features
  - good fit for policy and explainability demos

## 2. Lending Dataset
- Name: `German Credit Risk`
- Kaggle: `uciml/german-credit`
- Link: <https://www.kaggle.com/datasets/uciml/german-credit>
- Why:
  - strong credit decision benchmark
  - includes age and sex features for fairness evaluation
  - useful for disparate impact and equal opportunity analysis

## Download

1. Add your Kaggle API credentials to `~/.kaggle/kaggle.json`
2. Run:

```bash
python scripts/download_kaggle_datasets.py
```

## Current Blocker

Kaggle CLI is installed locally, but credentials are not configured yet. The downloader script is ready and will work as soon as the Kaggle API key is added.
