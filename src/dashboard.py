import streamlit as st
import os
import requests
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
st.write("PID:", os.getpid())
import data_loader
import analysis
import visualization
import forecasting
import anomaly
import matplotlib.pyplot as plt
import pandas as pd

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
    return data_loader.load_opr_measurements(nrows=nrows)

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
            date_range = st.sidebar.date_input("Time range", [min_date, max_date], min_value=min_date, max_value=max_date)
            if len(date_range) == 2:
                df_opr['date'] = pd.to_datetime(df_opr['date']).dt.tz_localize(None)
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
                country = st.sidebar.selectbox("Select country", ["All"] + countries)
                if country != "All":
                    df_opr = df_opr[df_opr['country'] == country]
            else:
                st.sidebar.info("Data does not have a 'country' column, skipping country filter.")
            # Hotspot analysis
            hotspot = analysis.detect_geospatial_peaks(df_opr, grid_size=grid_size, threshold=who_threshold)
            # Rolling mean, peak analysis
            df_opr_analyzed = analysis.analyze_time_series_opr(df_opr, threshold=who_threshold)
            # --- Tabs ---
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Hotspot Map", "Trend", "Heatmap", "Forecasting", "Anomaly", "Region Comparison", "📄 Export Report"])
            with tab1:
                try:
                    st.subheader("Radiation Hotspot Map")
                    st.write("Number of hotspots:", len(hotspot))
                    st.dataframe(hotspot.head())
                    if len(hotspot) == 0:
                        st.warning("No hotspots to display on the map!")
                    else:
                        m = visualization.plot_interactive_hotspot_map(hotspot, is_above_col='is_above_who')
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
                        visualization.plot_heatmap_opr(df_opr_analyzed, value_col="radiation", location_col="country", time_col="date", freq="M")
                    else:
                        visualization.plot_heatmap_opr(df_opr_analyzed, value_col="radiation", location_col="latitude", time_col="date", freq="M")
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Heatmap tab error: {e}")
            with tab4:
                try:
                    st.subheader("Radiation Forecasting (Prophet, LSTM, XGBoost)")
                    algo = st.selectbox("Select forecasting algorithm", ["Prophet", "LSTM", "XGBoost"], index=0)
                    days = st.number_input("Number of forecast days", min_value=7, max_value=90, value=30)
                    results = {}
                    if len(df_opr) > 30:
                        # Prophet
                        if algo == "Prophet" or algo == "All":
                            forecast, model = forecasting.forecast_radiation_prophet(df_opr, days=days)
                            results['Prophet'] = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'yhat_prophet'})
                        # LSTM
                        if algo == "LSTM" or algo == "All":
                            forecast_lstm_df = forecasting.forecast_lstm(df_opr, days=days)
                            results['LSTM'] = forecast_lstm_df
                        # XGBoost
                        if algo == "XGBoost" or algo == "All":
                            forecast_xgb_df = forecasting.forecast_xgboost(df_opr, days=days)
                            results['XGBoost'] = forecast_xgb_df
                        # Plot comparison
                        fig, ax = plt.subplots(figsize=(12,5))
                        ax.plot(df_opr['date'], df_opr['radiation'], label="Observed", alpha=0.5)
                        if 'Prophet' in results:
                            ax.plot(results['Prophet']['date'], results['Prophet']['yhat_prophet'], label="Prophet", color="green")
                        if 'LSTM' in results:
                            ax.plot(results['LSTM']['date'], results['LSTM']['yhat_lstm'], label="LSTM", color="orange")
                        if 'XGBoost' in results:
                            ax.plot(results['XGBoost']['date'], results['XGBoost']['yhat_xgb'], label="XGBoost", color="purple")
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
                        df_anom = anomaly.detect_anomalies_isolation_forest(df_opr)
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
            with tab6:
                try:
                    st.subheader("Compare Radiation Levels Across Regions")
                    # Preset regions: Chernobyl, Vietnam, Fukushima
                    presets = {
                        "Chernobyl": {
                            "lat_min": 51.2, "lat_max": 51.5, "lon_min": 29.0, "lon_max": 30.5
                        },
                        "Vietnam": {
                            "lat_min": 8.0, "lat_max": 24.0, "lon_min": 102.0, "lon_max": 110.0
                        },
                        "Fukushima": {
                            "lat_min": 36.8, "lat_max": 37.8, "lon_min": 139.5, "lon_max": 141.5
                        }
                    }
                    selected = st.multiselect("Select regions to compare", list(presets.keys()), default=["Chernobyl", "Vietnam"])
                    custom = st.checkbox("Enter custom region")
                    if custom:
                        lat_min = st.number_input("Lat min", value=10.0)
                        lat_max = st.number_input("Lat max", value=11.0)
                        lon_min = st.number_input("Lon min", value=105.0)
                        lon_max = st.number_input("Lon max", value=106.0)
                        custom_label = st.text_input("Custom region name", value="Custom")
                    # Plot comparison line chart
                    fig, ax = plt.subplots(figsize=(12,5))
                    colors = ["red", "blue", "green", "orange", "purple"]
                    any_data = False
                    for i, region in enumerate(selected):
                        p = presets[region]
                        df_region = df_opr[(df_opr['latitude'] >= p['lat_min']) & (df_opr['latitude'] <= p['lat_max']) & (df_opr['longitude'] >= p['lon_min']) & (df_opr['longitude'] <= p['lon_max'])]
                        st.write(f"Data for {region}: {len(df_region)} rows")
                        if len(df_region) > 0:
                            any_data = True
                            df_region = df_region.copy()
                            df_region['date'] = pd.to_datetime(df_region['date']).dt.tz_localize(None)
                            ts = df_region.groupby(df_region['date'].dt.date)['radiation'].mean()
                            ax.plot(ts.index, ts.values, label=region, color=colors[i%len(colors)])
                        else:
                            st.warning(f"No data for region {region}")
                    if custom:
                        df_region = df_opr[(df_opr['latitude'] >= lat_min) & (df_opr['latitude'] <= lat_max) & (df_opr['longitude'] >= lon_min) & (df_opr['longitude'] <= lon_max)]
                        st.write(f"Data for {custom_label}: {len(df_region)} rows")
                        if len(df_region) > 0:
                            any_data = True
                            df_region = df_region.copy()
                            df_region['date'] = pd.to_datetime(df_region['date']).dt.tz_localize(None)
                            ts = df_region.groupby(df_region['date'].dt.date)['radiation'].mean()
                            ax.plot(ts.index, ts.values, label=custom_label, color="black")
                        else:
                            st.warning(f"No data for custom region {custom_label}")
                    if not any_data:
                        st.error("No data for any selected region! Please check your filters or select another region.")
                    else:
                        ax.set_title("Compare Radiation Trends Across Regions")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("μSv/h")
                        ax.legend()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Region Comparison tab error: {e}")
            with tab7:
                try:
                    st.subheader("Export HTML Report")
                    st.markdown("The report includes overview tables, trend images, heatmaps, and region comparisons.")
                    # Generate simple HTML report
                    from io import BytesIO
                    import base64
                    html = f"""
                    <h2>Radioactive Data Analysis Report</h2>
                    <h3>Data Overview Table</h3>
                    {df_opr.head(20).to_html(index=False)}
                    <h3>Radiation Trend</h3>
                    """
                    # Save trend image to buffer
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(df_opr_analyzed['date'], df_opr_analyzed['radiation'], label="Measured Value", alpha=0.5)
                    ax.plot(df_opr_analyzed['date'], df_opr_analyzed['rolling_mean'], label="Rolling mean", color="orange")
                    ax.set_title("Radiation Trend")
                    ax.legend()
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    html += f'<img src="data:image/png;base64,{img_str}" width="700">'
                    # Heatmap (if available)
                    html += "<h3>Radiation Heatmap</h3>"
                    try:
                        fig2 = plt.figure(figsize=(10,4))
                        if 'country' in df_opr_analyzed.columns:
                            visualization.plot_heatmap_opr(df_opr_analyzed, value_col="radiation", location_col="country", time_col="date", freq="M")
                        else:
                            visualization.plot_heatmap_opr(df_opr_analyzed, value_col="radiation", location_col="latitude", time_col="date", freq="M")
                        buf2 = BytesIO()
                        fig2.savefig(buf2, format="png")
                        plt.close(fig2)
                        img_str2 = base64.b64encode(buf2.getvalue()).decode()
                        html += f'<img src="data:image/png;base64,{img_str2}" width="700">'
                    except Exception as e:
                        html += f"<p>Could not generate heatmap: {e}</p>"
                    # Region comparison (if available)
                    html += "<h3>Region Comparison</h3>"
                    try:
                        fig3, ax3 = plt.subplots(figsize=(10,4))
                        presets = {
                            "Chernobyl": {"lat_min": 51.2, "lat_max": 51.5, "lon_min": 29.0, "lon_max": 30.5},
                            "Vietnam": {"lat_min": 8.0, "lat_max": 24.0, "lon_min": 102.0, "lon_max": 110.0},
                            "Fukushima": {"lat_min": 36.8, "lat_max": 37.8, "lon_min": 139.5, "lon_max": 141.5}
                        }
                        for i, region in enumerate(["Chernobyl", "Vietnam", "Fukushima"]):
                            p = presets[region]
                            df_region = df_opr[(df_opr['latitude'] >= p['lat_min']) & (df_opr['latitude'] <= p['lat_max']) & (df_opr['longitude'] >= p['lon_min']) & (df_opr['longitude'] <= p['lon_max'])]
                            if len(df_region) > 0:
                                df_region = df_region.copy()
                                df_region['date'] = pd.to_datetime(df_region['date']).dt.tz_localize(None)
                                ts = df_region.groupby(df_region['date'].dt.date)['radiation'].mean()
                                ax3.plot(ts.index, ts.values, label=region)
                        ax3.set_title("Compare Radiation Trends Across Regions")
                        ax3.set_xlabel("Date")
                        ax3.set_ylabel("μSv/h")
                        ax3.legend()
                        buf3 = BytesIO()
                        fig3.savefig(buf3, format="png")
                        plt.close(fig3)
                        img_str3 = base64.b64encode(buf3.getvalue()).decode()
                        html += f'<img src="data:image/png;base64,{img_str3}" width="700">'
                    except Exception as e:
                        html += f"<p>Could not generate region comparison chart: {e}</p>"
                    # Download report button
                    b64 = base64.b64encode(html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="radioactive_data_analysis_report.html">Download HTML Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Export Report tab error: {e}")
else:
    st.info("Select number of rows, grid size, time range, WHO threshold, radiation filter, and click 'Load & Analyze Data' in the sidebar.") 