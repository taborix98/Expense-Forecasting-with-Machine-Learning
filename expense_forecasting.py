import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

# dati
df = pd.read_csv("expenses.csv")
df["date"] = pd.to_datetime(df["date"])

# spese giornaliere
daily = df.groupby("date")["amount"].sum().reset_index()
daily = daily.sort_values("date")

# grafico veloce per vedere trend
plt.plot(daily["date"], daily["amount"], marker="o")
plt.title("Spese giornaliere")
plt.xticks(rotation=45)
plt.show()

# feature base
daily["weekday"] = daily["date"].dt.weekday
daily["is_weekend"] = (daily["weekday"] > 4).astype(int)

X = daily[["weekday", "is_weekend"]]
y = daily["amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# valutazione
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

cv = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

print(f"MAE test: {mae:.2f}€")
print(f"MAE CV: {-cv.mean():.2f}€")

# previsione futuro
last_date = daily["date"].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

future_df = pd.DataFrame({"date": future_dates})
future_df["weekday"] = future_df["date"].dt.weekday
future_df["is_weekend"] = (future_df["weekday"] > 4).astype(int)

future_df["prediction"] = model.predict(future_df[["weekday", "is_weekend"]])

print(future_df)