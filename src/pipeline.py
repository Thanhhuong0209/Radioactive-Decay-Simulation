from data_loader import load_opr_measurements
from analysis import detect_geospatial_peaks, analyze_time_series_opr
import pandas as pd

def radioactive_analysis_pipeline(
    nrows=50000,
    country=None,
    date_range=None,
    grid_size=0.5,
    who_threshold=0.2
):
    """
    OpenRadiation radioactive data analysis pipeline.
    Parameters:
        - nrows: number of data rows to read
        - country: filter by country (None = all)
        - date_range: tuple (start_date, end_date) as string or pd.Timestamp
        - grid_size: hotspot grid size (degree)
        - who_threshold: WHO threshold (μSv/h)
    Returns dict:
        - filtered: filtered DataFrame
        - hotspot: hotspot DataFrame by grid
        - time_series: DataFrame with rolling mean, anomaly peaks
    """
    # Load data
    df = load_opr_measurements(nrows=nrows)
    # Filter by country
    if country and country != "All":
        df = df[df['country'] == country]
    # Filter by time
    if date_range and 'date' in df.columns:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['date'] >= start) & (df['date'] <= end)]
    # Hotspot analysis
    hotspot = detect_geospatial_peaks(df, grid_size=grid_size, threshold=who_threshold)
    # Rolling mean, anomaly peak analysis
    time_series = analyze_time_series_opr(df, threshold=who_threshold)
    return {
        "filtered": df,
        "hotspot": hotspot,
        "time_series": time_series
    } 