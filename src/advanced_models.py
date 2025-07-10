import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def forecast_lstm(df, days=30, lookback=14, epochs=10):
    """
    Forecast future radiation levels using LSTM.
    - df: DataFrame with columns 'date', 'radiation'
    - days: number of days to forecast
    - lookback: number of days used as input for each forecast
    - epochs: number of training epochs
    Returns forecast DataFrame (columns 'date', 'yhat_lstm')
    """
    data = df[['date', 'radiation']].copy().dropna()
    data = data.sort_values('date')
    values = data['radiation'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(32, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    # Forecast future
    last_seq = scaled[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(days):
        pred = model.predict(last_seq, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[:,1:,:], [[[pred]]], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    future_dates = pd.date_range(data['date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({'date': future_dates, 'yhat_lstm': preds})

def forecast_xgboost(df, days=30, lookback=14):
    """
    Forecast future radiation levels using XGBoost.
    - df: DataFrame with columns 'date', 'radiation'
    - days: number of days to forecast
    - lookback: number of days used as input for each forecast
    Returns forecast DataFrame (columns 'date', 'yhat_xgb')
    """
    data = df[['date', 'radiation']].copy().dropna()
    data = data.sort_values('date')
    for i in range(1, lookback+1):
        data[f'lag_{i}'] = data['radiation'].shift(i)
    data = data.dropna()
    X = data[[f'lag_{i}' for i in range(lookback,0,-1)]].values
    y = data['radiation'].values
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    last_lags = X[-1].reshape(1, -1)
    preds = []
    for _ in range(days):
        pred = model.predict(last_lags)[0]
        preds.append(pred)
        last_lags = np.roll(last_lags, -1)
        last_lags[0, -1] = pred
    future_dates = pd.date_range(data['date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({'date': future_dates, 'yhat_xgb': preds}) 