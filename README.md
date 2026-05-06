Expense Forecasting with Machine Learning
Modello che analizza spese giornaliere e prova a stimare i valori futuri usando feature temporali base.

- Obiettivi del progetto:
  analizzare andamento spese personali
  costruire un modello predittivo semplice
  valutare performance su dati temporali.

Dataset iniziale dotato di 3 colonne: data, prodotto, prezzo.
Per addestrare il modello sono state utilizzate, come features, le colonne create: 
  "weekday" (in cui i giorni della settimana vengono indicati con giorni da 0 a 6)
  "is_weekend" (in cui i giorni feriali vengono indicati con 0 e quelli del weekend con 1).

Come modello per l'addestramento è stato utilizzato il RandomForestRegressor.
L'errore medio calcolato con il cross-validation si aggira intorno ai 13€, un valore non indifferente.
Questo è soltanto un esempio di approccio, sicuramente lavorando su un database più grande (fornendo quindi più dati da studiare)
e lavorando su delle features aggiuntive il valore potrebbe scendere.
