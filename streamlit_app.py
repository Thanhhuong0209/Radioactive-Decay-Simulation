import streamlit as st
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from data_loader import load_opr_measurements
from analysis import detect_geospatial_peaks, analyze_time_series_opr, plot_heatmap_opr
from visualization import plot_interactive_hotspot_map
from forecasting import forecast_radiation_prophet
from anomaly import detect_anomalies_isolation_forest
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

# Page config
st.set_page_config(layout="wide")
st.title("Radioactive Decay Simulation Dashboard")

st.markdown("""
### User Guide
- Select number of data rows, grid size, time range, WHO threshold, filter radiation level, or crawl new data from OpenRadiation.
- Switch tabs to view hotspot map, trend, heatmap, forecasting, anomaly detection.
- If the dataset is large, reduce the number of rows to increase speed.
""")

# --- Sidebar filter ---
st.sidebar.header("Data Options")

# Real-time polling
auto_update = st.sidebar.checkbox("Auto-refresh data (5s)")
if auto_update:
    st_autorefresh(interval=5000, key="datarefresh")

nrows = st.sidebar.slider("Number of data rows", 10000, 200000, 50000, step=10000)
grid_size = st.sidebar.selectbox("Grid size (degree)", [0.1, 0.25, 0.5, 1.0], index=2)
who_threshold = st.sidebar.number_input("WHO threshold (μSv/h)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

# --- Session state for button ---
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False

if st.sidebar.button("Load & Analyze Data"):
    st.session_state['run_analysis'] = True

# --- Load data ---
@st.cache_data(show_spinner=True)
def get_data(nrows):
    return load_opr_measurements(nrows=nrows)

df_opr = None
if st.session_state['run_analysis']:
    with st.spinner("Loading data..."):
        df_opr = get_data(nrows)
    # Always sample if data is large, before any processing
    if df_opr is not None and len(df_opr) > 100000:
        st.warning("Dataset has more than 100,000 rows. Automatically sampling 100,000 random rows to improve speed and avoid memory errors.")
        df_opr = df_opr.sample(100000, random_state=42)
    if df_opr is None or len(df_opr) == 0:
        st.error("No data left after filtering or invalid upload file! Please adjust filters or select another file.")
    else:
        st.success(f"Loaded {len(df_opr)} data rows!")
        st.write("Number of data rows after filtering:", len(df_opr))
        st.dataframe(df_opr.head())
        
        # Time filtering
        if 'date' in df_opr.columns:
            min_date = pd.to_datetime(df_opr['date']).min()
            max_date = pd.to_datetime(df_opr['date']).max()
            date_range = st.sidebar.date_input("Time range", 
                                             value=(min_date.date(), max_date.date()),
                                             min_value=min_date.date(), 
                                             max_date=max_date.date())
            if len(date_range) == 2:
                df_opr['date'] = pd.to_datetime(df_opr['date'])
                start = pd.to_datetime(date_range[0])
                end = pd.to_datetime(date_range[1])
                df_opr = df_opr[(df_opr['date'] >= start) & (df_opr['date'] <= end)]
        
        # Radiation level filter
        min_rad = float(df_opr['radiation'].min())
        max_rad = float(df_opr['radiation'].max())
        rad_range = st.sidebar.slider("Radiation level range (μSv/h)", min_rad, max_rad, (min_rad, max_rad), step=0.01)
        df_opr = df_opr[(df_opr['radiation'] >= rad_range[0]) & (df_opr['radiation'] <= rad_range[1])]
        st.write("Number of data rows after time & radiation filtering:", len(df_opr))
        st.dataframe(df_opr.head())
        
        if len(df_opr) == 0:
            st.error("No data left after time & radiation filtering! Please broaden your filters.")
        else:
            # If country column exists, allow filtering, otherwise skip
            if 'country' in df_opr.columns:
                countries = sorted(df_opr['country'].dropna().unique())
                country = st.sidebar.selectbox("Select country", ["All"] + list(countries))
                if country != "All":
                    df_opr = df_opr[df_opr['country'] == country]
            else:
                st.sidebar.info("Data does not have a 'country' column, skipping country filter.")
            
            # Hotspot analysis
            hotspot = detect_geospatial_peaks(df_opr, grid_size=grid_size, threshold=who_threshold)
            # Rolling mean, peak analysis
            df_opr_analyzed = analyze_time_series_opr(df_opr, threshold=who_threshold)
            
            # --- Tabs ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Hotspot Map", "Trend", "Heatmap", "Forecasting", "Anomaly"])
            
            with tab1:
                try:
                    st.subheader("Radiation Hotspot Map")
                    st.write("Number of hotspots:", len(hotspot))
                    st.dataframe(hotspot.head())
                    if len(hotspot) == 0:
                        st.warning("No hotspots to display on the map!")
                    else:
                        m = plot_interactive_hotspot_map(hotspot, is_above_col='is_above_who')
                        st_folium(m, width=900, height=600)
                    st.markdown("- <span style='color:red'>Red dot</span>: hotspot above WHO threshold. <span style='color:blue'>Blue dot</span>: below threshold.", unsafe_allow_html=True)
                    st.dataframe(hotspot[hotspot['is_above_who']].sort_values('mean_radiation', ascending=False), use_container_width=True)
                except Exception as e:
                    st.error(f"Hotspot Map tab error: {e}")
            
            with tab2:
                try:
                    st.subheader("Radiation Trend Over Time")
                    fig, ax = plt.subplots(figsize=(12,5))
                    ax.plot(df_opr_analyzed['date'], df_opr_analyzed['radiation'], label="Measured Value", alpha=0.5)
                    ax.plot(df_opr_analyzed['date'], df_opr_analyzed['rolling_mean'], label="Rolling mean", color="orange")
                    ax.scatter(df_opr_analyzed.loc[df_opr_analyzed['is_peak'], 'date'], df_opr_analyzed.loc[df_opr_analyzed['is_peak'], 'radiation'], color="red", label="Anomaly Peak")
                    ax.axhline(who_threshold, color="purple", linestyle="--", label=f"WHO Threshold ({who_threshold} μSv/h)")
                    ax.set_title("Radiation Trend Over Time")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("μSv/h")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Trend tab error: {e}")
            
            with tab3:
                try:
                    st.subheader("Radiation Heatmap by Region and Time")
                    fig2 = plt.figure(figsize=(14,8))
                    if 'country' in df_opr_analyzed.columns:
                        plot_heatmap_opr(df_opr_analyzed, value_col="radiation", location_col="country", time_col="date", freq="M")
                    else:
                        plot_heatmap_opr(df_opr_analyzed, value_col="radiation", location_col="latitude", time_col="date", freq="M")
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Heatmap tab error: {e}")
            
            with tab4:
                try:
                    st.subheader("Radiation Forecasting (Prophet)")
                    days = st.number_input("Number of forecast days", min_value=7, max_value=90, value=30)
                    if len(df_opr) > 30:
                        forecast, model = forecast_radiation_prophet(df_opr, days=days)
                        fig, ax = plt.subplots(figsize=(12,5))
                        ax.plot(df_opr['date'], df_opr['radiation'], label="Observed", alpha=0.5)
                        ax.plot(forecast['ds'], forecast['yhat'], label="Prophet Forecast", color="green")
                        ax.set_title(f"Radiation Forecast for Next {days} Days")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("μSv/h")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.info("At least 30 data points are required for forecasting.")
                except Exception as e:
                    st.error(f"Forecasting tab error: {e}")
            
            with tab5:
                try:
                    st.subheader("Anomaly Detection (Isolation Forest)")
                    if len(df_opr) > 10:
                        df_anom = detect_anomalies_isolation_forest(df_opr)
                        fig4, ax4 = plt.subplots(figsize=(12,5))
                        ax4.plot(df_anom['date'], df_anom['radiation'], label="Measured Value", alpha=0.5)
                        ax4.scatter(df_anom.loc[df_anom['anomaly'], 'date'], df_anom.loc[df_anom['anomaly'], 'radiation'], color="red", label="Anomaly")
                        ax4.set_title("Anomaly Detection in Radiation Levels")
                        ax4.set_xlabel("Time")
                        ax4.set_ylabel("μSv/h")
                        ax4.legend()
                        st.pyplot(fig4)
                        st.dataframe(df_anom[df_anom['anomaly']], use_container_width=True)
                    else:
                        st.info("At least 10 data points are required for anomaly detection.")
                except Exception as e:
                    st.error(f"Anomaly tab error: {e}") 