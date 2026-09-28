# Telco Customer Churn — Data Mining

Analysis of **customer churn** at a telecommunications company: understanding which customers leave and predicting it. The dataset (7,043 customers) comes from the IBM sample files and combines demographics, subscribed services, pricing plan and customer status.

University project for the Data Mining course, developed with Vincenzo Napoli. All the work is in the notebook [`ProgettoDM.ipynb`](ProgettoDM.ipynb); [`ProgettoDM.html`](ProgettoDM.html) is its export, readable without installing anything. The notebook text is in Italian.

## What the notebook covers

1. **Exploration and preprocessing**: building a single dataset from the IBM files, analysis of demographic attributes, services and pricing plan, correlation analysis.
2. **Clustering**: K-Means and DBSCAN on two versions of the dataset (original and reduced to 15 attributes).
3. **Classification**: Decision Tree, AdaBoost, XGBoost, Random Forest, Naive Bayes, SVM (linear, polynomial, RBF and sigmoid kernels) and a neural network (MLP).
4. **Model comparison** on the test set.

## Evaluation protocol

The data is split **60% train / 20% validation / 20% test** (stratified, `random_state=42`):

- the **train set** is used for fitting;
- the **validation set** is used *only* to choose hyperparameters (one grid per model, selection metric: accuracy);
- the **test set** is used *once* per model, after the choice, and drives no decision.

The selected model stays trained on the train set only, so every algorithm sees the same data. The SVMs and the neural network have `StandardScaler` inside the pipeline, fitted on the train set only.

## Test-set results

| Model | Validation acc. | Test acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost | 0.8268 | **0.8176** | 0.6686 | 0.6203 | 0.6436 |
| AdaBoost | 0.8204 | 0.8105 | 0.6524 | 0.6123 | 0.6317 |
| SVM sigmoid | 0.8077 | 0.8105 | 0.6551 | 0.6043 | 0.6287 |
| SVM polynomial | 0.8062 | 0.8091 | 0.6756 | 0.5401 | 0.6003 |
| SVM linear | 0.8091 | 0.8070 | 0.6509 | 0.5882 | 0.6180 |
| Random Forest | 0.8148 | 0.8020 | 0.6537 | 0.5401 | 0.5915 |
| Decision Tree | 0.8098 | 0.7991 | 0.6608 | 0.5000 | 0.5693 |
| SVM RBF | 0.8126 | 0.7963 | 0.6212 | 0.5963 | 0.6085 |
| MLP | 0.8126 | 0.7942 | 0.6180 | 0.5882 | 0.6027 |
| Naive Bayes | 0.7502 | 0.7388 | 0.5053 | 0.7701 | 0.6102 |

26.5% of customers churn, so always predicting "no churn" would give about 73.5% accuracy: accuracy should be read together with recall and F1. Excluding Naive Bayes, the models are within about 2 points of accuracy; with a single split there is no clear winner.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ProgettoDM.ipynb
```

The data files are in `sample_data/`, where the notebook looks for them (path relative to the project folder). A full run takes a few minutes. Exporting the decision tree to PDF also requires [Graphviz](https://graphviz.org/download/); otherwise that cell is skipped.

## Data

The files in `sample_data/` are the IBM sample datasets on telecom customer churn.
