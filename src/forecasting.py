from prophet import Prophet
import pandas as pd

def forecast_radiation_prophet(df, days=30):
    """
    Forecast future radiation levels using Prophet.
    Parameters:
        - df: DataFrame with columns 'date' (datetime), 'radiation'
        - days: number of days to forecast
    Returns:
        - forecast: forecast DataFrame (columns ds, yhat, yhat_lower, yhat_upper)
        - model: fitted Prophet object
    """
    data = df[['date', 'radiation']].copy()
    data = data.rename(columns={'date': 'ds', 'radiation': 'y'})
    data = data.dropna()
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast, model 