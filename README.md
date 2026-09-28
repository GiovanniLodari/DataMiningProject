# Telco Customer Churn — Data Mining

Analisi del **churn** di un'azienda di telecomunicazioni: capire quali clienti lasciano il servizio e prevederlo. Il dataset (7043 clienti) proviene dai file di esempio IBM e unisce dati demografici, servizi sottoscritti, piano tariffario e stato del cliente.

Progetto per il corso di Data Mining. Tutto il lavoro è nel notebook [`ProgettoDM.ipynb`](ProgettoDM.ipynb); [`ProgettoDM.html`](ProgettoDM.html) ne è l'esportazione, leggibile senza installare nulla.

## Cosa contiene il notebook

1. **Esplorazione e preprocessing**: costruzione di un unico dataset a partire dai file IBM, analisi degli attributi demografici, dei servizi e del piano tariffario, analisi delle correlazioni.
2. **Clustering**: K-Means e DBSCAN su due versioni del dataset (originale e ridotta a 15 attributi).
3. **Classificazione**: Decision Tree, AdaBoost, XGBoost, Random Forest, Naive Bayes, SVM (kernel lineare, polinomiale, RBF, sigmoide) e rete neurale (MLP).
4. **Confronto dei modelli** sul test set.

## Protocollo di valutazione

I dati sono divisi in **train 60% / validation 20% / test 20%** (split stratificato, `random_state=42`):

- il **train set** serve ad addestrare;
- il **validation set** serve *solo* a scegliere gli iperparametri (una griglia per modello, metrica di selezione: accuracy);
- il **test set** viene usato *una sola volta* per ogni modello, dopo la scelta, e non influenza nessuna decisione.

Il modello scelto resta addestrato sul solo train set, così tutti gli algoritmi vedono gli stessi dati. Le SVM e la rete neurale hanno lo `StandardScaler` dentro la pipeline, calcolato sul solo train.

## Risultati sul test set

| Modello | Acc. validation | Acc. test | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost | 0.8268 | **0.8176** | 0.6686 | 0.6203 | 0.6436 |
| AdaBoost | 0.8204 | 0.8105 | 0.6524 | 0.6123 | 0.6317 |
| SVM sigmoide | 0.8077 | 0.8105 | 0.6551 | 0.6043 | 0.6287 |
| SVM polinomiale | 0.8062 | 0.8091 | 0.6756 | 0.5401 | 0.6003 |
| SVM lineare | 0.8091 | 0.8070 | 0.6509 | 0.5882 | 0.6180 |
| Random Forest | 0.8148 | 0.8020 | 0.6537 | 0.5401 | 0.5915 |
| Decision Tree | 0.8098 | 0.7991 | 0.6608 | 0.5000 | 0.5693 |
| SVM RBF | 0.8126 | 0.7963 | 0.6212 | 0.5963 | 0.6085 |
| MLP | 0.8126 | 0.7942 | 0.6180 | 0.5882 | 0.6027 |
| Naive Bayes | 0.7502 | 0.7388 | 0.5053 | 0.7701 | 0.6102 |

Il 26,5% dei clienti fa churn, quindi predire sempre "nessun churn" darebbe circa il 73,5% di accuracy: l'accuracy va letta insieme a recall e F1. Escluso Naive Bayes, i modelli stanno in circa 2 punti di accuracy; con uno split solo non c'è un vincitore netto.

## Esecuzione

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ProgettoDM.ipynb
```

I file dati sono in `sample_data/`, dove il notebook li cerca (percorso relativo alla cartella del progetto). L'esecuzione completa richiede alcuni minuti. Per esportare l'albero decisionale in PDF serve anche [Graphviz](https://graphviz.org/download/) installato; altrimenti quella cella viene saltata.

## Dati

I file in `sample_data/` sono i dataset di esempio IBM sul churn di un'azienda di telecomunicazioni.
