import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

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


# 3. Инициализация уровня, тренда и сезонности
def initialize_components(data, seasonal_periods):
    level = np.mean(data[:seasonal_periods])
    trend = (np.mean(data[seasonal_periods:2 * seasonal_periods]) - np.mean(data[:seasonal_periods])) / seasonal_periods
    seasonals = [data[i] - level for i in range(seasonal_periods)]
    return level, trend, seasonals


# 4. Обновление уровня, тренда и сезонности
def holt_winters_additive(data, seasonal_periods, alpha, beta, gamma, forecast_periods):
    # Инициализация
    level, trend, seasonals = initialize_components(data, seasonal_periods)

    # Списки для хранения результатов
    levels = [level]
    trends = [trend]
    seasonality = seasonals[:]
    forecast = []

    for i in range(len(data)):
        if i >= seasonal_periods:
            # Вычисляем прогноз на один шаг вперед
            forecast.append(level + trend + seasonality[i % seasonal_periods])

            # Обновляем уровень
            last_level = level
            level = alpha * (data[i] - seasonality[i % seasonal_periods]) + (1 - alpha) * (level + trend)

            # Обновляем тренд
            trend = beta * (level - last_level) + (1 - beta) * trend

            # Обновляем сезонность
            seasonality[i % seasonal_periods] = gamma * (data[i] - level) + (1 - gamma) * seasonality[
                i % seasonal_periods]

            # Сохраняем уровень и тренд
            levels.append(level)
            trends.append(trend)
        else:
            forecast.append(data[i])  # Используем фактические данные для первых точек

    # Прогноз на будущее
    for i in range(forecast_periods):
        forecast.append(level + (i + 1) * trend + seasonality[(len(data) + i) % seasonal_periods])

    return forecast, levels, trends, seasonality


# 5. Построение прогноза и визуализация
def plot_forecast(train, test, forecast, forecast_periods):
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test', color='orange')
    plt.plot(test.index, forecast[-forecast_periods:], label='Forecast',
             color='green')  # Берем последние значения прогноза
    plt.legend()
    plt.show()


def main():
    data = load_data()
    train, test = split_data(data)

    seasonal_periods = 40  # Сезонность дней
    alpha = 0.17  # Коэффициент сглаживания уровня
    beta = 0.11  # Коэффициент сглаживания тренда
    gamma = 0.12  # Коэффициент сглаживания сезонности
    forecast_periods = len(test)  # Прогнозируем для тестового набора

    # Вычисление модели Хольта-Уинтерса
    forecast, levels, trends, seasonality = holt_winters_additive(train, seasonal_periods, alpha, beta, gamma,
                                                                  forecast_periods)

    # Построение графика
    plot_forecast(train, test, forecast, forecast_periods)


if __name__ == "__main__":
    main()
