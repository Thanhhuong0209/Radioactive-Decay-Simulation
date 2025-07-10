# 🚀 [Access the deployed Radioactive Decay Simulation app on Streamlit Cloud](https://radioactive-decay-simulation-nwpuanbbx3t9eh8gnmj58y.streamlit.app/)

# Radioactive Decay Data Analysis & Interactive Dashboard

## Project Overview
This project provides a comprehensive platform for analyzing real-world radioactive decay data, including simulation, parameter estimation, anomaly detection, forecasting, and interactive visualization. The solution is designed to handle large datasets efficiently and offers a modern Streamlit dashboard for exploration, comparison, and reporting.

## Key Features
- **Real Data Analysis:** Supports large datasets from OpenRadiation and Chernobyl Kaggle datasets. Includes robust data cleaning, normalization, and outlier removal.
- **Parameter Estimation:** Estimate decay parameters (λ) using both classical and Bayesian (PyMC) methods.
- **Smoothing & Filtering:** Compare smoothing techniques such as Moving Average and Kalman Filter.
- **Anomaly Detection:** Detect peaks and anomalies using Isolation Forest and statistical methods.
- **Forecasting:** Time series forecasting with Prophet, LSTM, and XGBoost. Visual comparison of model results.
- **Comparative Study:** Compare radiation trends across preset regions (Chernobyl, Vietnam, Fukushima) or custom bounding boxes.
- **Interactive Dashboard:** Modern Streamlit dashboard with multiple tabs (hotspot map, trend, heatmap, forecasting, anomaly, region comparison, report export). Flexible filters and real-time data refresh.
- **Automatic Report Export:** Generate and download HTML reports based on current analysis and visualizations.
- **Performance Optimization:** Automatic sampling for large datasets (>100,000 rows), Dask integration for fast data loading, and user warnings for large data.
- **Robust Error Handling:** Handles missing libraries, empty data, and common dashboard errors gracefully.

## Folder Structure
```
Radioactive Decay Simulation/
├── data/                  # Raw and processed datasets
├── src/
│   ├── dashboard.py       # Main Streamlit dashboard
│   ├── radioactive_decay_simulation.py # Simulation & analysis scripts
│   ├── data_loader.py     # Data loading & cleaning
│   ├── analysis.py        # Analysis & anomaly detection
│   ├── visualization.py   # Plotting & mapping
│   ├── pipeline.py        # Data processing pipeline
│   ├── advanced_models.py # LSTM, XGBoost, advanced forecasting
│   └── ...
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```

## Installation
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) For large data, install Dask:
   ```bash
   pip install dask[dataframe]
   ```

## Usage
- **Run the interactive dashboard:**
  ```bash
  streamlit run src/dashboard.py
  ```
- **Run simulation/analysis scripts:**
  ```bash
  python src/radioactive_decay_simulation.py
  ```

## How the Dashboard Works
- **Sidebar:** Select data size, grid size, time range, WHO threshold, and filters. Option for real-time auto-refresh.
- **Tabs:**
  - **Hotspot Map:** Interactive map of radiation hotspots (Folium).
  - **Trend:** Time series plots with smoothing and anomaly peaks.
  - **Heatmap:** Radiation heatmap by region and time.
  - **Forecasting:** Predict future radiation levels with Prophet, LSTM, XGBoost.
  - **Anomaly:** Detect and visualize anomalies (Isolation Forest).
  - **Region Comparison:** Compare trends across preset or custom regions.
  - **Report Export:** Generate and download HTML reports with tables and charts.
- **Performance:**
  - For datasets >100,000 rows, the system automatically samples 100,000 rows and warns the user.
  - Dask is used for fast loading of very large files.

## Highlights
- **Scalable:** Handles large real-world datasets efficiently.
- **Extensible:** Easy to add new regions, algorithms, or data sources.
- **User-Friendly:** Modern UI, clear warnings, and robust error handling.
- **Ready for Demo/Portfolio:** Suitable for academic reports, demos, or as a foundation for real-world products.

## Credits
- OpenRadiation, Kaggle Chernobyl datasets
- Streamlit, Dask, Prophet, XGBoost, TensorFlow, Folium, PyMC, and other open-source libraries

---
Feel free to use, extend, or adapt this project for your own research, portfolio, or product development! 