import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json


# 1. Загрузка данных о курсе Эфира
def load_data(ticker="ETH-USD", period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    data.index = pd.to_datetime(data.index)  # Конвертируем индекс в формат datetime
    data = data.asfreq('D')  # Устанавливаем частоту данных на дневную
    return data['Close']


# 2. Деление на тренировочный и тестовый набор
def split_data(data, test_size=30):
    train, test = data[:-test_size], data[-test_size:]
    return train, test


# 3. Обучение модели Хольта-Уинтерса с возможностью выбора тренда и сезонности
def holt_winters_forecast(train, seasonal_periods=30, forecast_periods=30, trend='add', seasonal='add'):
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()

    forecast = model.forecast(forecast_periods)
    params = model.params
    return forecast, params


# 4. Сохранение параметров модели в файл
def save_model_params(params, filename="holt_winters_params.json"):
    serializable_params = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()}

    with open(filename, 'w') as f:
        json.dump(serializable_params, f, indent=4)


# 5. Построение прогноза и визуализация
def plot_forecast(train, test, forecast):
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test', color='orange')
    plt.plot(test.index, forecast, label='Forecast', color='green')
    plt.legend()
    plt.show()


def main():
    data = load_data()
    train, test = split_data(data)
    period = 65
    trend = 'add'
    seasonal = 'mul'
    forecast, params = holt_winters_forecast(train, seasonal_periods=period, trend=trend, seasonal=seasonal)

    save_model_params(params)

    plot_forecast(train, test, forecast)

if __name__ == "__main__":
    main()
