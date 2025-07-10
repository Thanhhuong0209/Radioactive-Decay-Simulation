import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

def quick_overview(df, value_col="radiation", time_col="date", location_col="country", n_head=5):
    """
    Display quick overview information about the DataFrame:
    - info(), head(), describe()
    - Distribution of measurement values (histogram)
    - Distribution over time (overall line plot)
    - Distribution by region (bar plot)
    """
    print("\n--- Data Info ---")
    print(df.info())
    print("\n--- Head ---")
    print(df.head(n_head))
    print("\n--- Describe ---")
    print(df.describe())
    # Distribution of measurement values
    plt.figure(figsize=(6,3))
    df[value_col].hist(bins=50)
    plt.title("Distribution of radiation measurement values")
    plt.xlabel(value_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    # Distribution over time
    if time_col in df.columns:
        plt.figure(figsize=(8,3))
        df.groupby(df[time_col].dt.date)[value_col].mean().plot()
        plt.title("Average radiation trend by day")
        plt.xlabel("Date")
        plt.ylabel(f"{value_col} (mean)")
        plt.tight_layout()
        plt.show()
    # Distribution by region
    if location_col in df.columns:
        plt.figure(figsize=(8,3))
        df.groupby(location_col)[value_col].mean().sort_values(ascending=False).head(20).plot(kind="bar")
        plt.title("Top 20 regions with highest average radiation")
        plt.xlabel(location_col)
        plt.ylabel(f"{value_col} (mean)")
        plt.tight_layout()
        plt.show() 

def analyze_time_series_opr(df, value_col="radiation", time_col="date", window=7, threshold=0.2):
    """
    Analyze radiation trend over time:
    - Calculate rolling mean (default 7 days)
    - Detect anomaly peaks (z-score > 3)
    - Mark above WHO threshold (0.2 μSv/h)
    Return DataFrame with columns: rolling_mean, is_peak, is_above_who
    """
    df = df.copy()
    df = df.sort_values(time_col)
    df["rolling_mean"] = df[value_col].rolling(window, min_periods=1).mean()
    # Z-score for anomaly peak detection
    df["zscore"] = (df[value_col] - df[value_col].mean()) / df[value_col].std()
    df["is_peak"] = df["zscore"].abs() > 3
    # Mark above WHO threshold
    df["is_above_who"] = df[value_col] > threshold
    return df

def plot_radiation_peaks(df, value_col="radiation", time_col="date", window=7, threshold=0.2):
    """
    Plot radiation trend over time:
    - Line plot of measured values
    - Rolling mean
    - Mark anomaly peaks (z-score > 3)
    - Mark above WHO threshold
    """
    plt.figure(figsize=(12,5))
    plt.plot(df[time_col], df[value_col], label="Measured Value", alpha=0.5)
    plt.plot(df[time_col], df["rolling_mean"], label=f"Rolling mean ({window} days)", color="orange")
    plt.scatter(df.loc[df["is_peak"], time_col], df.loc[df["is_peak"], value_col], color="red", label="Anomaly Peak")
    plt.axhline(threshold, color="purple", linestyle="--", label="WHO Threshold (0.2 μSv/h)")
    plt.title("Radiation Trend Over Time (OpenRadiation)")
    plt.xlabel("Time")
    plt.ylabel(f"{value_col} (μSv/h)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmap_opr(df, value_col="radiation", location_col="country", time_col="date", freq="M"):
    """
    Plot heatmap of average radiation by region and time (by month).
    If data is empty or only has 1 row/column, show warning and do not plot.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["period"] = df[time_col].dt.to_period(freq)
    pivot = df.pivot_table(index=location_col, columns="period", values=value_col, aggfunc="mean")
    st.write("Pivot table shape:", pivot.shape)
    st.dataframe(pivot.head())
    if pivot.shape[0] < 2 or pivot.shape[1] < 2 or pivot.isnull().all().all():
        st.warning("Not enough data to plot heatmap (at least 2 regions and 2 time points, not all NaN).")
        return
    plt.figure(figsize=(14,8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5)
    plt.title(f"Heatmap of average radiation by {location_col} and time")
    plt.xlabel("Time")
    plt.ylabel(location_col)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

def plot_geospatial_heatmap_opr(df, value_col="radiation", lat_col="latitude", lon_col="longitude", gridsize=100):
    """
    Plot geospatial heatmap of radiation on map (using hexbin or scatter density).
    """
    plt.figure(figsize=(10,7))
    hb = plt.hexbin(df[lon_col], df[lat_col], C=df[value_col], gridsize=gridsize, reduce_C_function=np.mean, cmap="YlOrRd", mincnt=5)
    plt.colorbar(hb, label=f"Average radiation ({value_col})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geospatial Heatmap of Radiation (OpenRadiation)")
    plt.tight_layout()
    plt.show()

def detect_geospatial_peaks(df, value_col="radiation", lat_col="latitude", lon_col="longitude", grid_size=0.5, threshold=0.2):
    """
    Detect radiation hotspots by coordinate grid:
    - Divide grid by grid_size degree (default 0.5)
    - Calculate mean for each grid cell
    - Mark cells above WHO threshold
    Return hotspot DataFrame with columns: lat_bin, lon_bin, mean_radiation, is_above_who
    """
    df = df.copy()
    df = df.dropna(subset=[lat_col, lon_col, value_col])
    df['lat_bin'] = (df[lat_col] // grid_size) * grid_size
    df['lon_bin'] = (df[lon_col] // grid_size) * grid_size
    hotspot = df.groupby(['lat_bin', 'lon_bin'])[value_col].mean().reset_index()
    hotspot = hotspot.rename(columns={value_col: 'mean_radiation'})
    hotspot['is_above_who'] = hotspot['mean_radiation'] > threshold
    return hotspot 