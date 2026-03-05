import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv("data/sales.csv")

# Convert month to datetime
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)

# ARIMA model
model = ARIMA(df['sales'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast next 3 months
forecast = model_fit.forecast(steps=3)

print("Next 3 months sales forecast:")
print(forecast)

# Plot
plt.figure()
plt.plot(df['sales'], label="Actual Sales")
plt.plot(forecast, label="Forecast")
plt.legend()
plt.title("Sales Forecasting")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()